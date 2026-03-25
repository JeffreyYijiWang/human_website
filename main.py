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

# "heap map" here = a soft scalar field around the cursor (brush falloff)
# used as a mask to blend tex0 -> tex1.
FRAGMENT_SHADER = """
#version 330

uniform sampler2D tex0;
uniform sampler2D tex1;

uniform vec2  u_mouse;     // [0,1], origin bottom-left
uniform float u_radius;    // UV units
uniform float u_softness;  // feather width
uniform float u_mix_amt;   // global strength [0,1]

in vec2 v_uv;
out vec4 fragColor;


float soft_circle(vec2 p, vec2 c, float r, float s) {
    float d = distance(p, c);
    return 1.0 - smoothstep(r - s, r + s, d);
}

void main() {
    // flip Y because images are typically top-left origin
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);

    vec4 c0 = texture(tex0, uv);
    vec4 c1 = texture(tex1, uv);

    float m = soft_circle(v_uv, u_mouse, u_radius, u_softness);
    m *= u_mix_amt;

    fragColor = mix(c0, c1, m);
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


class HeapMaskClickAdvance(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Heap-mask temporal blend (GPU) — click/keys to advance"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        base = Path(__file__).resolve().parent
        folder = base / "abdomen_png"
        self.paths = sorted_pngs(folder)

        self.idx = 0  # base frame: tex0 = idx, tex1 = idx+1

        # Brush / heap-mask params
        self.radius = 0.12
        self.softness = 0.04
        self.mix_amt = 1.0
        self.mouse_uv = (0.5, 0.5)

        # Load first image to set size
        first = Image.open(self.paths[0]).convert("RGBA")
        self.img_w, self.img_h = first.size

        print("Frames:", len(self.paths))
        print("First :", self.paths[0].name)
        print("Last  :", self.paths[-1].name)
        print("Controls:")
        print("  Move mouse: move mask")
        print("  Left click / Right arrow: next frame")
        print("  Right click / Left arrow: prev frame")
        print("  Up/Down: radius +/-")
        print("  [ and ]: strength +/-")
        print("  , and . : softness -/+")
        print("  ESC: quit")

        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)

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

        # Two textures (current and next)

        a0 = np.ascontiguousarray(self._load_rgba_resized(self.paths[0]))
        a1 = np.ascontiguousarray(self._load_rgba_resized(self.paths[1]))

        self.tex0 = self.ctx.texture((self.img_w, self.img_h), 4, data=a0.tobytes())
        self.tex1 = self.ctx.texture((self.img_w, self.img_h), 4, data=a1.tobytes())

        self.tex0.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.tex1.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self.tex0.use(location=0)
        self.tex1.use(location=1)
        self.prog["tex0"].value = 0
        self.prog["tex1"].value = 1

        self._upload_pair()
        self._push_uniforms()

    def _load_rgba_resized(self, path: Path) -> np.ndarray:
        im = Image.open(path).convert("RGBA")
        if im.size != (self.img_w, self.img_h):
            im = im.resize((self.img_w, self.img_h), Image.BILINEAR)
        arr = np.array(im, dtype=np.uint8)
        arr[..., 3] = 255  # FORCE opaque
        return arr

    def _upload_pair(self):
        i0 = self.idx % len(self.paths)
        i1 = (self.idx + 1) % len(self.paths)

        a = self._load_rgba_resized(self.paths[i0])
        b = self._load_rgba_resized(self.paths[i1])

        print("upload", self.paths[i0].name, a.shape, a.dtype, "bytes", a.nbytes)
        print(
            "CPU a:", self.paths[i0].name,
            "min", int(a[..., :3].min()),
            "max", int(a[..., :3].max()),
            "alpha_min", int(a[..., 3].min()),
            "alpha_max", int(a[..., 3].max()),
        )
        print(
            "CPU b:", self.paths[i1].name,
            "min", int(b[..., :3].min()),
            "max", int(b[..., :3].max()),
            "alpha_min", int(b[..., 3].min()),
            "alpha_max", int(b[..., 3].max()),
        )

        self.tex0.write(a.tobytes())
        if not hasattr(self, "_checked"):
            self._checked = True
            raw = self.tex0.read()  # bytes
            print("tex0 readback nonzero?", any(b != 0 for b in raw[:4096]))
        self.tex1.write(b.tobytes())

    def _push_uniforms(self):
        self.prog["u_mouse"].value = self.mouse_uv
        self.prog["u_radius"].value = float(self.radius)
        self.prog["u_softness"].value = float(self.softness)
        self.prog["u_mix_amt"].value = float(self.mix_amt)

    # --- events (moderngl-window 3.x uses on_* names) ---

    def on_mouse_position_event(self, x, y, dx, dy):
        u = x / max(1, self.wnd.width)
        v = 1.0 - (y / max(1, self.wnd.height))
        self.mouse_uv = (float(u), float(v))
        self.prog["u_mouse"].value = self.mouse_uv

    def on_mouse_press_event(self, x, y, button):
        # pyglet: left=1 right=4; glfw often right=2
        if button in (1,):          # left -> next
            self.idx = (self.idx + 1) % len(self.paths)
        elif button in (4, 2):      # right -> prev
            self.idx = (self.idx - 1) % len(self.paths)
        else:
            return
        self._upload_pair()

    def on_key_event(self, key, action, modifiers):
        if action != self.wnd.keys.ACTION_PRESS:
            return

        k = self.wnd.keys

        if key == k.ESCAPE:
            self.wnd.close()
            return

        # frame stepping
        if key == k.RIGHT:
            self.idx = (self.idx + 1) % len(self.paths)
            self._upload_pair()
            return
        if key == k.LEFT:
            self.idx = (self.idx - 1) % len(self.paths)
            self._upload_pair()
            return

        # brush params
        if key == k.UP:
            self.radius = min(0.5, self.radius + 0.01)
        elif key == k.DOWN:
            self.radius = max(0.01, self.radius - 0.01)
        elif key == k.LEFT_BRACKET:
            self.mix_amt = max(0.0, self.mix_amt - 0.05)
        elif key == k.RIGHT_BRACKET:
            self.mix_amt = min(1.0, self.mix_amt + 0.05)
        elif key == k.COMMA:
            self.softness = max(0.0, self.softness - 0.005)
        elif key == k.PERIOD:
            self.softness = min(0.25, self.softness + 0.005)
        else:
            return

        self._push_uniforms()
        print(f"radius={self.radius:.3f} softness={self.softness:.3f} mix={self.mix_amt:.2f}")

    def on_render(self, time: float, frame_time: float):
        self.ctx.viewport = (0, 0, self.wnd.width, self.wnd.height)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.clear(0.2, 0.0, 0.2, 1.0)  # purple background so you know it’s drawing


        self.tex0.use(location=0)
        self.tex1.use(location=1)
    
        self.vao.render()


if __name__ == "__main__":
    mglw.run_window_config(HeapMaskClickAdvance)
