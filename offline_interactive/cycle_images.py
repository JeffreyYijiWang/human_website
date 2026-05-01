import re
import numpy as np
from PIL import Image
from pathlib import Path

import moderngl
import moderngl_window as mglw


VERTEX_SHADER = """
#version 330
in vec2 in_pos;
out vec2 v_uv;

void main() {
    v_uv = (in_pos * 0.5) + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FRAGMENT_SHADER = """
#version 330
uniform sampler2D tex0;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);
    fragColor = texture(tex0, uv);
}
"""


_vm_re = re.compile(r"_vm(\d+)\.png$", re.IGNORECASE)

def sorted_pngs(folder: Path):
    paths = list(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNGs found in {folder}")

    def key(p: Path):
        m = _vm_re.search(p.name)
        return int(m.group(1)) if m else 10**12

    return sorted(paths, key=key)


class ClickToAdvance(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Click to advance abdomen_png (GPU)"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        base = Path(__file__).resolve().parent
        folder = base / "abdomen_png"
        self.paths = sorted_pngs(folder)

        self.idx = 0

        first = Image.open(self.paths[0]).convert("RGBA")
        self.img_w, self.img_h = first.size
        first_pixels = np.array(first, dtype=np.uint8)

        print("Frames:", len(self.paths))
        print("First :", self.paths[0].name)
        print("Last  :", self.paths[-1].name)
        print("Loaded:", self.paths[0].name)
        print("Click: left mouse to advance. Right mouse to go back.")

        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        quad = np.array([
            -1.0, -1.0,
             1.0, -1.0,
            -1.0,  1.0,
            -1.0,  1.0,
             1.0, -1.0,
             1.0,  1.0,
        ], dtype="f4")

        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_pos")

        self.tex = self.ctx.texture((self.img_w, self.img_h), 4, first_pixels.tobytes())
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.use(location=0)
        self.prog["tex0"].value = 0

    def _load_into_texture(self, path: Path):
        im = Image.open(path).convert("RGBA")
        if im.size != (self.img_w, self.img_h):
            im = im.resize((self.img_w, self.img_h), Image.BILINEAR)
        pixels = np.array(im, dtype=np.uint8)
        self.tex.write(pixels.tobytes())

    def on_mouse_position_event(self, x, y, dx, dy):
        # uncomment if you want to spam-test that events are firing
        print("mouse move", x, y)
        pass
    def on_mouse_press_event(self, x, y, button):
        # pyglet: left=1 right=4, glfw: often left=1 right=2
        if button in (1,):                   # left
            self.idx = (self.idx + 1) % len(self.paths)
        elif button in (4, 2):               # right (pyglet=4, glfw=2)
            self.idx = (self.idx - 1) % len(self.paths)
        else:
            return
        self._load_into_texture(self.paths[self.idx])

    def on_key_event(self, key, action, modifiers):
        # We only care about the 'PRESS' action (not release or repeat)
        if action == self.wnd.keys.ACTION_PRESS:
            if key == self.wnd.keys.RIGHT:
                self.idx = (self.idx + 1) % len(self.paths)
            elif key == self.wnd.keys.LEFT:
                self.idx = (self.idx - 1) % len(self.paths)
            else:
                return # Ignore other keys

            # Update the texture with the new image
            self._load_into_texture(self.paths[self.idx])
            print(f"Loaded: {self.paths[self.idx].name} (Index: {self.idx})")

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render()


if __name__ == "__main__":
    mglw.run_window_config(ClickToAdvance)
