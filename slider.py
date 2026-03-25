import ctypes, os

# Tell Windows to use the NVIDIA GPU for this process
try:
    ctypes.windll.ntdll.NtSetInformationProcess(
        ctypes.windll.kernel32.GetCurrentProcess(),
        0x27, ctypes.byref(ctypes.c_ulong(1)), ctypes.sizeof(ctypes.c_ulong)
    )
except Exception:
    pass

# Also set NVIDIA env vars as a fallback
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"

import moderngl
ctx = moderngl.create_standalone_context()
print(ctx.info["GL_RENDERER"])
ctx.release()

"""
heap_mask_array.py  –  GPU texture-array depth-reveal viewer
─────────────────────────────────────────────────────────────
Controls
  Mouse move        – depth brush
  Hover HUD bar     – thumbnail tooltip (on-screen, scales with window)
  Drag HUD handles  – white line = surface layer, gold line = deep layer
  T                 – toggle settings overlay
  Up / Down         – radius ±
  [ / ]             – softness ±
  , / .             – layer stretch ±
  B                 – toggle bilinear blend
  F                 – toggle nearest/linear sampling
  ESC               – quit
"""

import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

import moderngl
import moderngl_window as mglw


# ── SHADER 1 – main image ────────────────────────────────────────────────────

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
uniform sampler2DArray tex_array;
uniform vec2  u_mouse;
uniform float u_radius;
uniform float u_softness;
uniform float u_layer_lo;
uniform float u_layer_hi;
uniform int   u_num_layers;
uniform float u_layer_stretch;
uniform int   u_blend_layers;

in vec2  v_uv;
out vec4 fragColor;

void main() {
    vec2 uv = vec2(v_uv.x, 1.0 - v_uv.y);

    float d         = clamp(distance(v_uv, u_mouse) / max(u_radius, 1e-6), 0.0, 1.0);
    float rim0      = 1.0 - (u_softness / max(u_radius, 1e-6));
    float edge_fade = 1.0 - smoothstep(rim0, 1.0, d);
    float shaped_t  = pow(d, max(u_layer_stretch, 1e-6));

    float n       = float(u_num_layers - 1);
    float lo_idx  = u_layer_lo * n;
    float hi_idx  = u_layer_hi * n;
    float layer_f = clamp(mix(hi_idx, lo_idx, shaped_t), 0.0, n);

    vec4 outside = texture(tex_array, vec3(uv, lo_idx));
    vec4 inside;
    if (u_blend_layers != 0) {
        int   loi = int(floor(layer_f));
        int   hii = min(loi + 1, u_num_layers - 1);
        inside = mix(texture(tex_array, vec3(uv, float(loi))),
                     texture(tex_array, vec3(uv, float(hii))),
                     fract(layer_f));
    } else {
        inside = texture(tex_array, vec3(uv, float(int(layer_f + 0.5))));
    }
    fragColor = mix(outside, inside, edge_fade);
}
"""

# ── SHADER 2 – HUD bar (background gradient + ticks, NO handle lines) ────────

HUD_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

HUD_FRAG = """
#version 330
uniform float u_lo;
uniform float u_hi;
uniform int   u_num_layers;
in  vec2 v_uv;
out vec4 fragColor;

vec3 heatmap(float t) {
    vec3 c = mix(vec3(0.1,0.3,1.0), vec3(0.1,1.0,0.5), smoothstep(0.00,0.33,t));
         c = mix(c, vec3(1.0,1.0,0.1),  smoothstep(0.33,0.66,t));
         c = mix(c, vec3(1.0,0.15,0.1), smoothstep(0.66,1.00,t));
    return c;
}

void main() {
    float x = v_uv.x;
    float y = v_uv.y;

    vec3 col = heatmap(x) * 0.75;

    // brighten the active range
    float active = step(u_lo, x) * step(x, u_hi);
    col = mix(col * 0.35, col * 1.3, active);

    // subtle per-layer tick at the bottom half
    float tick_w = 1.0 / float(u_num_layers);
    float phase  = mod(x, tick_w) / tick_w;
    col += step(0.93, phase) * 0.15 * (1.0 - step(0.5, y));

    // top border
    col = mix(col, vec3(0.55), step(0.93, y) * 0.7);

    fragColor = vec4(clamp(col, 0.0, 1.0), 0.94);
}
"""

# ── SHADER 3 – handle lines (two separate thin quads drawn per handle) ────────
# Each handle is a thin vertical rectangle with colour passed as uniform.

HANDLE_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

HANDLE_FRAG = """
#version 330
uniform vec4 u_color;   // RGBA
in  vec2 v_uv;          // 0..1 within the handle quad
out vec4 fragColor;

void main() {
    // Solid core with soft antialiased edges on left/right (x-axis)
    float edge_dist = abs(v_uv.x - 0.5) * 2.0;       // 0 = centre, 1 = edge
    float aa        = 1.0 - smoothstep(0.75, 1.0, edge_dist);

    // Diamond / arrow cap at the top 20% of the handle
    float cap_y     = smoothstep(0.78, 0.82, v_uv.y);
    float cap_taper = 1.0 - abs(v_uv.x - 0.5) * 2.0 * (1.0 + cap_y * 1.5);
    float cap_alpha = clamp(cap_taper, 0.0, 1.0) * cap_y;

    float a = max(aa, cap_alpha) * u_color.a;
    fragColor = vec4(u_color.rgb, a);
}
"""

# ── SHADER 4 – popup / overlay (plain 2D texture sampled over a quad) ─────────

TEX_QUAD_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

TEX_QUAD_FRAG = """
#version 330
uniform sampler2D u_tex;
in  vec2 v_uv;
out vec4 fragColor;
void main() { fragColor = texture(u_tex, v_uv); }
"""


# ── Pillow helpers ────────────────────────────────────────────────────────────

def _font(size: int):
    for path in [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def build_popup(layer_rgba: np.ndarray, layer_i: int,
                total: int, name: str,
                px_w: int, px_h: int) -> np.ndarray:
    """
    Compose an on-screen tooltip:
      upper area  – scaled layer thumbnail
      lower strip – layer index + filename
    px_w / px_h are computed from the current window size each call.
    """
    text_h     = max(70, px_h // 4)   # more room for two lines of text
    img_area_h = px_h - text_h

    src = Image.fromarray(layer_rgba, "RGBA")
    src.thumbnail((px_w - 8, img_area_h - 8), Image.BILINEAR)

    canvas = Image.new("RGBA", (px_w, px_h), (18, 20, 35, 230))
    draw   = ImageDraw.Draw(canvas)

    # border
    draw.rectangle([0, 0, px_w - 1, px_h - 1],
                   outline=(80, 140, 255, 255), width=2)

    # thumbnail centred in image area
    ox = (px_w      - src.width)  // 2
    oy = (img_area_h - src.height) // 2 + 2
    canvas.paste(src, (ox, oy))

    # separator
    draw.line([(2, img_area_h + 1), (px_w - 3, img_area_h + 1)],
              fill=(60, 80, 140, 200), width=1)

    # font sizes — generous so they never collide
    fnt_lg   = _font(max(13, min(18, text_h // 4)))
    fnt_sm   = _font(max(11, min(14, text_h // 5)))
    line1_y  = img_area_h + 8
    line2_y  = line1_y + fnt_lg.size + 8   # use .size for correct spacing

    draw.text((8, line1_y),
              f"Layer  {layer_i}  /  {total - 1}",
              font=fnt_lg, fill=(200, 220, 255, 255))
    short = name if len(name) <= 34 else "…" + name[-32:]
    draw.text((8, line2_y),
              short, font=fnt_sm, fill=(130, 155, 200, 200))

    return np.ascontiguousarray(np.array(canvas, dtype=np.uint8))


def build_overlay(radius, softness, stretch, blend,
                  lo_i, hi_i, total, nearest,
                  px_w: int, px_h: int) -> np.ndarray:
    canvas = Image.new("RGBA", (px_w, px_h), (14, 16, 32, 218))
    draw   = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, px_w - 1, px_h - 1],
                   outline=(80, 140, 255, 240), width=2)

    title_h = max(30, px_h // 10)
    draw.rectangle([2, 2, px_w - 3, title_h + 2], fill=(30, 40, 80, 255))
    draw.text((10, 6), "⚙  Settings",
              font=_font(max(10, title_h - 8)), fill=(180, 210, 255, 255))

    rows = [
        ("Radius",        f"{radius:.3f}",              "Up / Down"),
        ("Softness",      f"{softness:.3f}",             "[ / ]"),
        ("Stretch",       f"{stretch:.2f}",              ", / ."),
        ("Layer range",   f"{lo_i}  →  {hi_i}",          "drag bar"),
        ("Total layers",  str(total),                    ""),
        ("Blend",         "ON" if blend   else "OFF",    "B"),
        ("Nearest samp",  "ON" if nearest else "OFF",    "F"),
    ]

    row_h   = max(22, (px_h - title_h - 30) // len(rows))
    fnt_lbl = _font(max(9,  row_h - 10))
    fnt_val = _font(max(9,  row_h - 10))
    col_val = int(px_w * 0.50)
    col_key = int(px_w * 0.72)

    y = title_h + 6
    for idx, (label, value, hint) in enumerate(rows):
        bg = (22, 28, 52, 180) if idx % 2 == 0 else (18, 22, 44, 180)
        draw.rectangle([3, y, px_w - 4, y + row_h - 2], fill=bg)
        draw.text((8,       y + 4), label, font=fnt_lbl, fill=(160, 180, 220, 255))
        draw.text((col_val, y + 4), value, font=fnt_val, fill=(255, 230, 100, 255))
        if hint:
            draw.text((col_key, y + 4), hint, font=fnt_val, fill=(100, 130, 180, 200))
        y += row_h

    draw.text((10, px_h - 20), "Press  T  to close",
              font=_font(max(8, px_h // 20)), fill=(80, 100, 160, 200))

    return np.ascontiguousarray(np.array(canvas, dtype=np.uint8))


# ── misc ──────────────────────────────────────────────────────────────────────

_vm_re = re.compile(r"_vm(\d+)\.png$", re.IGNORECASE)

def sorted_pngs(folder: Path):
    paths = list(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNGs in {folder}")
    def key(p):
        m = _vm_re.search(p.name)
        return int(m.group(1)) if m else 10**12
    return sorted(paths, key=key)


MAX_LAYERS = 100
HUD_HEIGHT = 52    # px – base height of the slider bar
THUMB_PAD  = 6     # px – gap between HUD top and popup bottom

# Popup / overlay as fraction of window size
POPUP_W_FRAC   = 0.28   # 28% of window width  (wider so text doesn't overlap)
POPUP_H_FRAC   = 0.38   # 38% of window height
OVERLAY_W_FRAC = 0.38
OVERLAY_H_FRAC = 0.52

HANDLE_W_PX    = 10   # half-width of each handle strip in pixels
HANDLE_GRAB_PX = 22  # grab tolerance – how close you need to click to a handle


def _quad_verts(x0, y0, x1, y1, u0=0., v0=0., u1=1., v1=1.):
    """Return 6-vertex interleaved (x,y,u,v) float32 array for a quad."""
    return np.array([
        x0, y0, u0, v0,
        x1, y0, u1, v0,
        x0, y1, u0, v1,
        x0, y1, u0, v1,
        x1, y0, u1, v0,
        x1, y1, u1, v1,
    ], dtype="f4")


class HeapMaskArray(mglw.WindowConfig):
    gl_version   = (3, 3)
    title        = "Heap-mask  |  hover bar=tooltip  |  T=settings"
    window_size  = (1280, 720)
    aspect_ratio = None
    resizable    = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        base      = Path(__file__).resolve().parent
        folder    = base / "abdomen_png"
        all_paths = sorted_pngs(folder)
        self.paths      = all_paths[:MAX_LAYERS]
        self.num_layers = len(self.paths)

        # params
        self.radius        = 0.15
        self.softness      = 0.05
        self.mouse_uv      = (0.5, 0.5)
        self.layer_stretch = 1.0
        self.blend_layers  = 1
        self.nearest_sampling = False
        self.layer_lo      = 0.0
        self.layer_hi      = 1.0

        # UI state
        self._drag          = None
        self._show_overlay  = False
        self._hud_hover     = False
        self._hover_layer_i = 0
        self._hover_px_x    = 0
        self._last_popup_i  = -1
        self._last_win_size = (0, 0)
        self._held_keys     = set()
        self._mouse_px_x    = 0     # raw pixel X updated on every mouse event
        self._mouse_px_y    = 0     # raw pixel Y (top-down from window)
        self._mouse_btn1    = False  # is left button currently held down

        # load layers
        first = Image.open(self.paths[0]).convert("RGBA")
        self.img_w, self.img_h = first.size
        print(f"Loading {self.num_layers} layers …")
        self._layer_arrays = []
        for path in self.paths:
            im = Image.open(path).convert("RGBA")
            if im.size != (self.img_w, self.img_h):
                im = im.resize((self.img_w, self.img_h), Image.BILINEAR)
            arr = np.ascontiguousarray(np.array(im, dtype=np.uint8))
            arr[..., 3] = 255
            self._layer_arrays.append(arr)
        print("  done.")

        # ── shaders ───────────────────────────────────────────────────────
        self.prog        = self.ctx.program(vertex_shader=VERTEX_SHADER,
                                             fragment_shader=FRAGMENT_SHADER)
        self.hud_prog    = self.ctx.program(vertex_shader=HUD_VERT,
                                             fragment_shader=HUD_FRAG)
        self.handle_prog = self.ctx.program(vertex_shader=HANDLE_VERT,
                                             fragment_shader=HANDLE_FRAG)
        self.quad_prog   = self.ctx.program(vertex_shader=TEX_QUAD_VERT,
                                             fragment_shader=TEX_QUAD_FRAG)

        # ── geometry buffers ──────────────────────────────────────────────
        fsq = np.array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1], dtype="f4")
        self.vbo     = self.ctx.buffer(fsq)
        self.vao     = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_pos")

        self.hud_vbo = self.ctx.buffer(reserve=6*4*4)
        self.hud_vao = self.ctx.simple_vertex_array(
            self.hud_prog, self.hud_vbo, "in_pos", "in_uv")

        # Each handle is a separate quad buffer (rebuilt when fraction changes)
        self.lo_handle_vbo = self.ctx.buffer(reserve=6*4*4)
        self.lo_handle_vao = self.ctx.simple_vertex_array(
            self.handle_prog, self.lo_handle_vbo, "in_pos", "in_uv")

        self.hi_handle_vbo = self.ctx.buffer(reserve=6*4*4)
        self.hi_handle_vao = self.ctx.simple_vertex_array(
            self.handle_prog, self.hi_handle_vbo, "in_pos", "in_uv")

        self.popup_vbo   = self.ctx.buffer(reserve=6*4*4)
        self.popup_vao   = self.ctx.simple_vertex_array(
            self.quad_prog, self.popup_vbo, "in_pos", "in_uv")

        self.overlay_vbo = self.ctx.buffer(reserve=6*4*4)
        self.overlay_vao = self.ctx.simple_vertex_array(
            self.quad_prog, self.overlay_vbo, "in_pos", "in_uv")

        # ── textures ──────────────────────────────────────────────────────
        self.tex_array = self.ctx.texture_array(
            (self.img_w, self.img_h, self.num_layers), components=4)
        self.tex_array.filter = (moderngl.LINEAR, moderngl.LINEAR)
        for i, arr in enumerate(self._layer_arrays):
            self.tex_array.write(arr.tobytes(),
                                 viewport=(0, 0, i, self.img_w, self.img_h, 1))
            print(f"  layer {i:03d} → GPU")

        self.tex_array.use(location=0)
        self.prog["tex_array"].value    = 0
        self.prog["u_num_layers"].value = self.num_layers

        # Popup and overlay textures are created lazily at first use
        self.popup_tex   = None
        self.overlay_tex = None
        self._popup_px   = (0, 0)   # track size so we recreate if window resizes
        self._overlay_px = (0, 0)

        self._push_uniforms()
        self._rebuild_hud_quad()
        self._rebuild_handle_quads()

        print("Ready.  T = settings  |  hover bottom bar = tooltip  |  ESC = quit")

    # ── helpers ────────────────────────────────────────────────────────────

    def _push_uniforms(self):
        p = self.prog
        p["u_mouse"].value         = self.mouse_uv
        p["u_radius"].value        = float(self.radius)
        p["u_softness"].value      = float(self.softness)
        p["u_layer_lo"].value      = float(self.layer_lo)
        p["u_layer_hi"].value      = float(self.layer_hi)
        p["u_layer_stretch"].value = float(self.layer_stretch)
        p["u_blend_layers"].value  = int(self.blend_layers)

    # ── window-size helpers ────────────────────────────────────────────────

    def _W(self): return max(self.wnd.width,  1)
    def _H(self): return max(self.wnd.height, 1)

    def _frac_to_ndc_x(self, frac):
        """Slider fraction 0..1  →  NDC x  (-1..+1)."""
        return frac * 2.0 - 1.0

    def _px_to_ndc_x(self, px):
        return (px / self._W()) * 2.0 - 1.0

    def _px_to_ndc_y(self, py_from_bottom):
        return (py_from_bottom / self._H()) * 2.0 - 1.0

    def _ndc_size(self, px_w, px_h):
        return 2.0 * px_w / self._W(), 2.0 * px_h / self._H()

    # ── geometry rebuild ───────────────────────────────────────────────────

    def _rebuild_hud_quad(self):
        H = self._H()
        y0_ndc = -1.0
        y1_ndc = -1.0 + 2.0 * HUD_HEIGHT / H
        self.hud_vbo.write(_quad_verts(-1.0, y0_ndc, 1.0, y1_ndc).tobytes())

    def _handle_ndc_bounds(self, frac):
        """Return (x0, y0, x1, y1) NDC for a handle at slider fraction."""
        W, H = self._W(), self._H()
        hw_ndc = 2.0 * HANDLE_W_PX / W           # half-width in NDC
        cx_ndc = frac * 2.0 - 1.0                 # centre x
        x0 = cx_ndc - hw_ndc
        x1 = cx_ndc + hw_ndc
        y0 = -1.0                                  # bottom of screen
        y1 = -1.0 + 2.0 * HUD_HEIGHT / H          # top of HUD
        return x0, y0, x1, y1 

    def _rebuild_handle_quads(self):
        x0, y0, x1, y1 = self._handle_ndc_bounds(self.layer_lo)
        self.lo_handle_vbo.write(_quad_verts(x0, y0, x1, y1).tobytes())
        x0, y0, x1, y1 = self._handle_ndc_bounds(self.layer_hi)
        self.hi_handle_vbo.write(_quad_verts(x0, y0, x1, y1).tobytes())

    def _popup_px_size(self):
        W, H = self._W(), self._H()
        pw = max(120, int(W * POPUP_W_FRAC))
        ph = max(100, int(H * POPUP_H_FRAC))
        return pw, ph

    def _overlay_px_size(self):
        W, H = self._W(), self._H()
        pw = max(280, int(W * OVERLAY_W_FRAC))
        ph = max(200, int(H * OVERLAY_H_FRAC))
        return pw, ph

    def _rebuild_popup_quad(self, cursor_px_x):
        W, H = self._W(), self._H()
        pw, ph = self._popup_px_size()
        pw_ndc, ph_ndc = self._ndc_size(pw, ph)

        cx_ndc = self._px_to_ndc_x(cursor_px_x)
        cx_ndc = float(np.clip(cx_ndc,
                               -1.0 + pw_ndc/2 + 0.01,
                                1.0 - pw_ndc/2 - 0.01))

        hud_top_ndc = -1.0 + 2.0 * HUD_HEIGHT / H
        pad_ndc     =  2.0 * THUMB_PAD  / H
        y0 = hud_top_ndc + pad_ndc
        y1 = min(y0 + ph_ndc, 1.0)
        y0 = y1 - ph_ndc
        x0 = cx_ndc - pw_ndc / 2
        x1 = cx_ndc + pw_ndc / 2

        # v flipped so Pillow image (top-left origin) renders right-way up
        self.popup_vbo.write(_quad_verts(x0, y0, x1, y1,
                                         u0=0., v0=1., u1=1., v1=0.).tobytes())

    def _rebuild_overlay_quad(self):
        pw, ph = self._overlay_px_size()
        pw_ndc, ph_ndc = self._ndc_size(pw, ph)
        pw_ndc = min(pw_ndc, 1.7)
        ph_ndc = min(ph_ndc, 1.7)
        x0, x1 = -pw_ndc/2, pw_ndc/2
        y0, y1 = -ph_ndc/2, ph_ndc/2
        self.overlay_vbo.write(_quad_verts(x0, y0, x1, y1,
                                            u0=0., v0=1., u1=1., v1=0.).tobytes())

    # ── texture helpers ────────────────────────────────────────────────────

    def _ensure_popup_tex(self, pw, ph):
        """Create or recreate popup texture if size changed."""
        if self.popup_tex is None or self._popup_px != (pw, ph):
            if self.popup_tex:
                self.popup_tex.release()
            self.popup_tex = self.ctx.texture((pw, ph), 4)
            self.popup_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._popup_px   = (pw, ph)
            self._last_popup_i = -1   # force rebake

    def _ensure_overlay_tex(self, pw, ph):
        if self.overlay_tex is None or self._overlay_px != (pw, ph):
            if self.overlay_tex:
                self.overlay_tex.release()
            self.overlay_tex = self.ctx.texture((pw, ph), 4)
            self.overlay_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._overlay_px = (pw, ph)

    def _update_popup_tex(self, layer_i: int):
        pw, ph = self._popup_px_size()
        self._ensure_popup_tex(pw, ph)
        if layer_i == self._last_popup_i and self._popup_px == (pw, ph):
            return
        self._last_popup_i = layer_i
        img = build_popup(self._layer_arrays[layer_i],
                          layer_i, self.num_layers,
                          self.paths[layer_i].name, pw, ph)
        self.popup_tex.write(img.tobytes())

    def _update_overlay_tex(self):
        pw, ph = self._overlay_px_size()
        self._ensure_overlay_tex(pw, ph)
        lo_i = int(round(self.layer_lo * (self.num_layers - 1)))
        hi_i = int(round(self.layer_hi * (self.num_layers - 1)))
        img = build_overlay(self.radius, self.softness, self.layer_stretch,
                            self.blend_layers, lo_i, hi_i, self.num_layers,
                            self.nearest_sampling, pw, ph)
        self.overlay_tex.write(img.tobytes())

    # ── event helpers ──────────────────────────────────────────────────────

    def _in_hud(self, y_bot):
        return y_bot <= HUD_HEIGHT

    def _x_frac(self, px):
        return float(np.clip(px / self._W(), 0.0, 1.0))

    def _nearest_handle(self, px_x):
        """Return 'lo', 'hi', or None depending on which handle is within grab distance."""
        W = self._W()
        lo_px = self.layer_lo * W
        hi_px = self.layer_hi * W
        dist_lo = abs(px_x - lo_px)
        dist_hi = abs(px_x - hi_px)
        # must be within HANDLE_GRAB_PX of a handle to grab it
        best = "lo" if dist_lo <= dist_hi else "hi"
        best_dist = min(dist_lo, dist_hi)
        return best if best_dist <= HANDLE_GRAB_PX else None

    def _set_handle(self, handle, frac):
        if handle == "lo":
            self.layer_lo = float(np.clip(frac, 0.0, self.layer_hi - 0.01))
        else:
            self.layer_hi = float(np.clip(frac, self.layer_lo + 0.01, 1.0))
        self._push_uniforms()
        self._rebuild_handle_quads()
        if self._show_overlay:
            self._update_overlay_tex()

    # ── events ─────────────────────────────────────────────────────────────

    def _update_mouse(self, x, y):
        """Central mouse state update — called from every mouse event."""
        W, H = self._W(), self._H()
        self._mouse_px_x = x
        self._mouse_px_y = y
        y_bot = H - y
        in_hud = self._in_hud(y_bot)

        # ── Brush UV: always update when outside the HUD ──────────────────
        # This runs even during a handle drag so the depth reveal follows
        # the cursor continuously (the "draw while held" behaviour).
        if not in_hud:
            self.mouse_uv = (x / W, y_bot / H)
            self.prog["u_mouse"].value = self.mouse_uv

        # ── Handle drag ───────────────────────────────────────────────────
        if self._drag is not None:
            frac    = self._x_frac(x)
            raw_val = frac

            # ── Swap handles if lo crosses hi (real range-slider behaviour) ──
            if self._drag == "lo" and raw_val > self.layer_hi:
                # lo has crossed hi → swap identities
                self.layer_lo  = self.layer_hi
                self.layer_hi  = float(np.clip(raw_val, 0.0, 1.0))
                self._drag     = "hi"
            elif self._drag == "hi" and raw_val < self.layer_lo:
                self.layer_hi  = self.layer_lo
                self.layer_lo  = float(np.clip(raw_val, 0.0, 1.0))
                self._drag     = "lo"
            else:
                self._set_handle(self._drag, frac)

            self._push_uniforms()
            self._rebuild_handle_quads()

            li = int(np.clip(round(frac * (self.num_layers - 1)),
                             0, self.num_layers - 1))
            self._hover_layer_i = li
            self._hover_px_x    = x
            self._hud_hover     = True
            self._rebuild_popup_quad(x)
            self._update_popup_tex(li)
            if self._show_overlay:
                self._update_overlay_tex()
            return

        # ── Hover (no drag) ───────────────────────────────────────────────
        self._hud_hover = in_hud
        if in_hud:
            frac = self._x_frac(x)
            li   = int(np.clip(round(frac * (self.num_layers - 1)),
                               0, self.num_layers - 1))
            self._hover_layer_i = li
            self._hover_px_x    = x
            self._rebuild_popup_quad(x)
            self._update_popup_tex(li)

    def on_mouse_position_event(self, x, y, dx, dy):
        self._update_mouse(x, y)

    def on_mouse_press_event(self, x, y, button):
        if button != 1:
            return
        self._mouse_btn1 = True
        H = self._H()
        y_bot = H - y
        if self._in_hud(y_bot):
            handle = self._nearest_handle(x)
            if handle is not None:
                self._drag = handle
            else:
                frac = self._x_frac(x)
                dist_lo = abs(frac - self.layer_lo)
                dist_hi = abs(frac - self.layer_hi)
                self._drag = "lo" if dist_lo <= dist_hi else "hi"
                self._set_handle(self._drag, frac)
        # always sync mouse state on press
        self._update_mouse(x, y)

    def on_mouse_release_event(self, x, y, button):
        if button == 1:
            self._mouse_btn1 = False
            self._drag = None
            y_bot = self._H() - y
            self._hud_hover = self._in_hud(y_bot)
            self._update_mouse(x, y)

    def on_key_event(self, key, action, modifiers):
        k = self.wnd.keys

        # Only care about PRESS and RELEASE (pyglet has no ACTION_REPEAT)
        if action not in (k.ACTION_PRESS, k.ACTION_RELEASE):
            return

        pressing = (action == k.ACTION_PRESS)

        # ── Hold-to-change keys (UP/DOWN/brackets/comma/period) ───────────
        hold_keys = {k.UP, k.DOWN, k.LEFT_BRACKET, k.RIGHT_BRACKET,
                     k.COMMA, k.PERIOD}
        if key in hold_keys:
            if pressing:
                self._held_keys.add(key)
            else:
                self._held_keys.discard(key)
            return

        # ── Arrow keys: TOGGLE on press, cancel on press again ────────────
        # Press LEFT once → start moving left handle; press LEFT again → stop
        if key == k.LEFT and pressing:
            if k.LEFT in self._held_keys:
                self._held_keys.discard(k.LEFT)
            else:
                self._held_keys.add(k.LEFT)
                self._held_keys.discard(k.RIGHT)   # cancel opposite direction
            return
        if key == k.RIGHT and pressing:
            if k.RIGHT in self._held_keys:
                self._held_keys.discard(k.RIGHT)
            else:
                self._held_keys.add(k.RIGHT)
                self._held_keys.discard(k.LEFT)
            return
        # release of arrow keys does nothing (they're toggle, not hold)
        if key in (k.LEFT, k.RIGHT):
            return

        # ── One-shot keys (only on press) ─────────────────────────────────
        if not pressing:
            return

        if   key == k.ESCAPE:   self.wnd.close(); return
        elif key == k.T:
            self._show_overlay = not self._show_overlay
            if self._show_overlay:
                self._update_overlay_tex()
                self._rebuild_overlay_quad()
            return
        elif key == k.B:
            self.blend_layers = 0 if self.blend_layers else 1
        elif key == k.F:
            self.nearest_sampling = not self.nearest_sampling
            self.tex_array.filter = (
                (moderngl.NEAREST, moderngl.NEAREST) if self.nearest_sampling
                else (moderngl.LINEAR, moderngl.LINEAR))
        else:
            return

        self._push_uniforms()
        if self._show_overlay:
            self._update_overlay_tex()

    def _apply_held_keys(self, dt: float):
        """Called every frame; smoothly applies held/toggled key changes."""
        if not self._held_keys:
            return
        k = self.wnd.keys
        changed        = False
        handles_moved  = False

        RADIUS_RATE  = 0.15
        SOFT_RATE    = 0.08
        STRETCH_RATE = 2.0
        HANDLE_RATE  = 0.40   # fraction/sec

        if k.UP            in self._held_keys:
            self.radius        = min(0.6,   self.radius        + RADIUS_RATE  * dt); changed = True
        if k.DOWN          in self._held_keys:
            self.radius        = max(0.01,  self.radius        - RADIUS_RATE  * dt); changed = True
        if k.LEFT_BRACKET  in self._held_keys:
            self.softness      = max(0.0,   self.softness      - SOFT_RATE    * dt); changed = True
        if k.RIGHT_BRACKET in self._held_keys:
            self.softness      = min(0.3,   self.softness      + SOFT_RATE    * dt); changed = True
        if k.COMMA         in self._held_keys:
            self.layer_stretch = max(0.1,   self.layer_stretch - STRETCH_RATE * dt); changed = True
        if k.PERIOD        in self._held_keys:
            self.layer_stretch = min(10.0,  self.layer_stretch + STRETCH_RATE * dt); changed = True

        # ── Arrow keys move the handle closest to the mouse ───────────────
        if k.LEFT in self._held_keys or k.RIGHT in self._held_keys:
            W       = self._W()
            lo_px   = self.layer_lo * W
            hi_px   = self.layer_hi * W
            dist_lo = abs(self._mouse_px_x - lo_px)
            dist_hi = abs(self._mouse_px_x - hi_px)
            target  = "lo" if dist_lo <= dist_hi else "hi"
            delta   = HANDLE_RATE * dt * (1.0 if k.RIGHT in self._held_keys else -1.0)

            if target == "lo":
                new_lo = self.layer_lo + delta
                # swap if lo crosses hi
                if new_lo > self.layer_hi:
                    self.layer_lo, self.layer_hi = self.layer_hi, float(np.clip(new_lo, 0.0, 1.0))
                    # switch which arrow direction is active so motion continues naturally
                    if k.LEFT  in self._held_keys: self._held_keys.discard(k.LEFT);  self._held_keys.add(k.RIGHT)
                    elif k.RIGHT in self._held_keys: self._held_keys.discard(k.RIGHT); self._held_keys.add(k.LEFT)
                else:
                    self.layer_lo = float(np.clip(new_lo, 0.0, self.layer_hi - 0.001))
            else:
                new_hi = self.layer_hi + delta
                # swap if hi crosses lo
                if new_hi < self.layer_lo:
                    self.layer_lo, self.layer_hi = float(np.clip(new_hi, 0.0, 1.0)), self.layer_lo
                    if k.LEFT  in self._held_keys: self._held_keys.discard(k.LEFT);  self._held_keys.add(k.RIGHT)
                    elif k.RIGHT in self._held_keys: self._held_keys.discard(k.RIGHT); self._held_keys.add(k.LEFT)
                else:
                    self.layer_hi = float(np.clip(new_hi, self.layer_lo + 0.001, 1.0))

            handles_moved = True
            changed       = True

        if handles_moved:
            self._rebuild_handle_quads()

        if changed:
            self._push_uniforms()
            if self._show_overlay:
                self._update_overlay_tex()

    def on_resize(self, width, height):
        self._rebuild_hud_quad()
        self._rebuild_handle_quads()
        if self._show_overlay:
            self._update_overlay_tex()
            self._rebuild_overlay_quad()
        if self._hud_hover or self._drag is not None:
            self._rebuild_popup_quad(self._hover_px_x)
            self._last_popup_i = -1   # force texture rebake at new size
            self._update_popup_tex(self._hover_layer_i)

    # ── render ─────────────────────────────────────────────────────────────

    def on_render(self, time: float, frame_time: float):
        W, H = self._W(), self._H()
        self.ctx.viewport = (0, 0, W, H)

        # apply held-key repeats each frame
        self._apply_held_keys(frame_time)

        # Re-push brush UV every frame while left button is held.
        # on_mouse_position_event only fires on movement, so without this
        # the brush freezes the moment you stop moving while holding click.
        if self._mouse_btn1 and self._drag is None:
            y_bot = H - self._mouse_px_y
            if not self._in_hud(y_bot):
                self.mouse_uv = (self._mouse_px_x / W, y_bot / H)
                self.prog["u_mouse"].value = self.mouse_uv
        self.ctx.disable(moderngl.BLEND)
        self.ctx.clear(0.08, 0.0, 0.10, 1.0)

        # 1. main image
        self.tex_array.use(location=0)
        self.vao.render()

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # 2. HUD bar background
        self.hud_prog["u_lo"].value         = float(self.layer_lo)
        self.hud_prog["u_hi"].value         = float(self.layer_hi)
        self.hud_prog["u_num_layers"].value = int(self.num_layers)
        self.hud_vao.render()

        # 3. handle lines  (separate geometry so they always sit at the right x)
        self.handle_prog["u_color"].value = (1.0, 1.0, 1.0, 1.0)   # white = lo
        self.lo_handle_vao.render()
        self.handle_prog["u_color"].value = (1.0, 0.82, 0.1, 1.0)  # gold  = hi
        self.hi_handle_vao.render()

        # 4. tooltip popup
        if (self._hud_hover or self._drag is not None) and self.popup_tex:
            self.popup_tex.use(location=1)
            self.quad_prog["u_tex"].value = 1
            self.popup_vao.render()

        # 5. settings overlay
        if self._show_overlay and self.overlay_tex:
            self.overlay_tex.use(location=2)
            self.quad_prog["u_tex"].value = 2
            self.overlay_vao.render()

        self.ctx.disable(moderngl.BLEND)


if __name__ == "__main__":
    mglw.run_window_config(HeapMaskArray)