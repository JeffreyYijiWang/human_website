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

uniform sampler2DArray tex_array;   // all N layers loaded
uniform int   u_num_layers;         // total loaded layers
uniform int   u_active_layers;      // how many layers participate (<= u_num_layers)

uniform vec2  u_mouse;              // [0,1] UV, origin bottom-left
uniform float u_radius;             // radius of the full brush in UV units
uniform float u_softness;           // feather at the outer edge

uniform float u_layer_stretch;      // >1 = more time near surface, <1 = dive faster
uniform int   u_blend_layers;       // 1 = blend between layers, 0 = pick nearest layer

in vec2  v_uv;
out vec4 fragColor;

void main() {
    // flip Y: images are top-left origin, UVs are bottom-left
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);

    float d = distance(v_uv, u_mouse);

    // t = 0 at mouse center, 1 at brush edge (and beyond)
    float t = clamp(d / u_radius, 0.0, 1.0);

    // soft feather at the rim
    float rim0 = 1.0 - (u_softness / max(u_radius, 1e-6));
    float edge_fade = 1.0 - smoothstep(rim0, 1.0, t);

    // apply "stretch" shaping in radius space
    float shaped_t = pow(t, max(u_layer_stretch, 1e-6));

    // clamp active layers
    int active_layers = clamp(u_active_layers, 1, u_num_layers);
    float n = float(active_layers - 1);

    // map center -> deepest, edge -> layer 0
    float layer_f = (1.0 - shaped_t) * n;

    // base (outside brush) is surface
    vec4 outside = texture(tex_array, vec3(uv, 0.0));

    vec4 inside;
    if (u_blend_layers != 0) {
        int lo = int(floor(layer_f));
        int hi = min(lo + 1, active_layers - 1);
        float ft = fract(layer_f);

        vec4 col_lo = texture(tex_array, vec3(uv, float(lo)));
        vec4 col_hi = texture(tex_array, vec3(uv, float(hi)));
        inside = mix(col_lo, col_hi, ft);
    } else {
        int li = int(floor(layer_f + 0.5)); // nearest layer index
        li = clamp(li, 0, active_layers - 1);
        inside = texture(tex_array, vec3(uv, float(li)));
    }

    fragColor = mix(outside, inside, edge_fade);
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


MAX_LAYERS = 100


class HeapMaskArray(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Heap-mask texture array — mouse center=deepest layer"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        base   = Path(__file__).resolve().parent
        folder = base / "abdomen_png"
        all_paths = sorted_pngs(folder)

        self.paths = all_paths[:MAX_LAYERS]


        # # Take up to MAX_LAYERS evenly spaced frames (or all if fewer)
        # if len(all_paths) <= MAX_LAYERS:
        #     self.paths = all_paths
        # else:
        #     indices    = np.linspace(0, len(all_paths) - 1, MAX_LAYERS, dtype=int)
        #     self.paths = [all_paths[i] for i in indices]

        self.num_layers = len(self.paths)

        # Brush params
        self.radius   = 0.15
        self.softness = 0.05
        self.mouse_uv = (0.5, 0.5)
        self.active_layers = self.num_layers   # how many layers participate
        self.layer_stretch = 1.0               # 1.0 = linear
        self.blend_layers  = 1                 # 1 = blend between layers, 0 = discrete
        self.blend_layers  = 1                 # 1 = blend between layers, 0 = discrete

        # Pixel sampling: nearest avoids "pixel distortion"/blur
        self.nearest_sampling = True

        # Determine texture size from first image
        first          = Image.open(self.paths[0]).convert("RGBA")
        self.img_w, self.img_h = first.size

        print(f"Loaded {self.num_layers} layers  ({self.img_w}x{self.img_h})")
        print("Controls:")
        print("  Move mouse  : shift depth reveal")
        print("  Up/Down     : radius +/-")
        print("  [ ]         : softness -/+")
        print("  ESC         : quit")
        print("  K / L      : active layers -/+")
        print("  , / .      : layer stretch -/+")
        print("  B          : toggle blend vs discrete layers")
        print("  F          : toggle nearest vs linear sampling")

        # Compile shader
        self.prog = self.ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=FRAGMENT_SHADER,
        )

        # Full-screen quad
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

        # Build texture array  (width, height, layers)
        self.tex_array = self.ctx.texture_array(
            (self.img_w, self.img_h, self.num_layers),
            components=4,
        )
        self.tex_array.filter = (moderngl.LINEAR, moderngl.LINEAR)

        # Upload all layers
        for layer_idx, path in enumerate(self.paths):
            arr = self._load_rgba(path)
            # write() with offset for texture arrays:
            #   write(data, viewport=(x, y, layer, w, h, 1))
            self.tex_array.write(
                arr.tobytes(),
                viewport=(0, 0, layer_idx, self.img_w, self.img_h, 1),
            )
            print(f"  layer {layer_idx:02d} ← {path.name}")

        self.tex_array.use(location=0)
        self.prog["tex_array"].value   = 0
        self.prog["u_num_layers"].value = self.num_layers
        self.prog["u_active_layers"].value = self.active_layers
        self.prog["u_layer_stretch"].value = float(self.layer_stretch)
        self.prog["u_blend_layers"].value  = int(self.blend_layers)

        self._push_uniforms()

    # ------------------------------------------------------------------ #

    def _load_rgba(self, path: Path) -> np.ndarray:
        im  = Image.open(path).convert("RGBA")
        if im.size != (self.img_w, self.img_h):
            im = im.resize((self.img_w, self.img_h), Image.BILINEAR)
        arr = np.ascontiguousarray(np.array(im, dtype=np.uint8))
        arr[..., 3] = 255   # force opaque
        return arr

    def _push_uniforms(self):
        self.prog["u_mouse"].value    = self.mouse_uv
        self.prog["u_radius"].value   = float(self.radius)
        self.prog["u_softness"].value = float(self.softness)
        self.prog["u_active_layers"].value = int(self.active_layers)
        self.prog["u_layer_stretch"].value = float(self.layer_stretch)
        self.prog["u_blend_layers"].value  = int(self.blend_layers)

    # ------------------------------------------------------------------ #
    #  Events
    # ------------------------------------------------------------------ #

    def on_mouse_position_event(self, x, y, dx, dy):
        u = x / max(1, self.wnd.width)
        v = 1.0 - (y / max(1, self.wnd.height))
        self.mouse_uv = (float(u), float(v))
        self.prog["u_mouse"].value = self.mouse_uv

    def on_key_event(self, key, action, modifiers):
        if action != self.wnd.keys.ACTION_PRESS:
            return
        k = self.wnd.keys

        if key == k.ESCAPE:
            self.wnd.close()
            return
        elif key == k.UP:
            self.radius = min(0.6, self.radius + 0.01)
        elif key == k.DOWN:
            self.radius = max(0.01, self.radius - 0.01)
        elif key == k.LEFT_BRACKET:
            self.softness = max(0.0, self.softness - 0.005)
        elif key == k.RIGHT_BRACKET:
            self.softness = min(0.3, self.softness + 0.005)
        elif key == k.K:  # fewer active layers
            self.active_layers = max(1, self.active_layers - 1)
        elif key == k.L:  # more active layers
            self.active_layers = min(self.num_layers, self.active_layers + 1)
        elif key == k.COMMA:  # stretch down
            self.layer_stretch = max(0.1, self.layer_stretch - 0.1)
        elif key == k.PERIOD:  # stretch up
            self.layer_stretch = min(10.0, self.layer_stretch + 0.1)
        elif key == k.B:  # toggle layer blending
            self.blend_layers = 0 if self.blend_layers else 1
        elif key == k.F:  # toggle nearest/linear sampling
            self.nearest_sampling = not self.nearest_sampling
            self.tex_array.filter = (
                (moderngl.NEAREST, moderngl.NEAREST)
                if self.nearest_sampling
                else (moderngl.LINEAR, moderngl.LINEAR)
            )
        else:
            return

        self._push_uniforms()
        print(
            f"radius={self.radius:.3f}  softness={self.softness:.3f}  "
            f"active_layers={self.active_layers}/{self.num_layers}  "
            f"stretch={self.layer_stretch:.2f}  "
            f"blend_layers={self.blend_layers}  "
            f"nearest={self.nearest_sampling}"
        )

    def on_render(self, time: float, frame_time: float):
        self.ctx.viewport = (0, 0, self.wnd.width, self.wnd.height)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.clear(0.1, 0.0, 0.1, 1.0)

        self.tex_array.use(location=0)
        self.vao.render()


if __name__ == "__main__":
    mglw.run_window_config(HeapMaskArray)