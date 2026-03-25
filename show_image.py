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


class ShowOneImage(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Show one PNG (GPU)"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        base = Path(__file__).resolve().parent          # ...\venv
        folder = base / "abdomen_png"                   # ...\venv\abdomen_png

        # Pick ONE image to show:
        # Option A: hardcode a filename
        img_path = folder / "a_vm1455.png"

        # Option B: if you don't want to hardcode, uncomment this:
        # img_path = sorted(folder.glob("*.png"))[0]

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGBA")
        self.img_w, self.img_h = img.size
        pixels = np.array(img, dtype=np.uint8)

        print("Loaded:", img_path.name, "size:", self.img_w, "x", self.img_h)

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

        self.tex = self.ctx.texture((self.img_w, self.img_h), 4, pixels.tobytes())
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex.use(location=0)
        self.prog["tex0"].value = 0

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render()


if __name__ == "__main__":
    mglw.run_window_config(ShowOneImage)
