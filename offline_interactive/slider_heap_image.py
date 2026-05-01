import ctypes, os
from pathlib import Path

import numpy as np
import moderngl
import moderngl_window as mglw
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Best-effort: prefer NVIDIA GPU on Windows
# ============================================================
try:
    ctypes.windll.ntdll.NtSetInformationProcess(
        ctypes.windll.kernel32.GetCurrentProcess(),
        0x27, ctypes.byref(ctypes.c_ulong(1)), ctypes.sizeof(ctypes.c_ulong)
    )
except Exception:
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"


# ============================================================
# Slice shaders (full-screen view) + HEAP BRUSH
# ============================================================

SLICE_VERT = r"""
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = (in_pos * 0.5) + 0.5;   // 0..1, bottom-left origin
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

# Key idea:
# - We compute the oblique plane sample position p in volume coords [0,1]^3
# - Then apply a "heap brush" that offsets p along the plane normal n
#   based on mouse distance: center digs deepest, outside digs none.
# - u_heap_depth is in normalized volume units (like 0.25 means 25% of box)
SLICE_FRAG = r"""
#version 330

uniform sampler2DArray tex_array;
uniform int   u_num_layers;     // Z layers in texture array
uniform vec2  u_slice_px;       // viewport (w,h) in pixels

uniform vec3  u_center;         // plane center in [0,1]^3
uniform vec3  u_axis_u;         // plane axis U (unit)
uniform vec3  u_axis_v;         // plane axis V (unit)
uniform vec3  u_axis_n;         // plane normal N (unit)
uniform float u_scale;          // half-width in normalized volume units

// heap brush
uniform int   u_heap_enable;    // 0/1
uniform vec2  u_mouse;          // [0,1], bottom-left origin
uniform float u_radius;         // brush radius in UV
uniform float u_softness;       // feather size (UV)
uniform float u_layer_stretch;  // shaping in radius space
uniform float u_heap_depth;     // max offset along +/-N (normalized volume units)
uniform float u_heap_dir;       // +1 or -1 (direction along N)

// color controls
uniform int   u_flip_y;         // 1 if texture rows are top-left origin (most PNG stacks), else 0
uniform int   u_bgr_input;      // 1 if stored as BGR in RGB channels, else 0

in vec2 v_uv;
out vec4 fragColor;

vec4 sample_volume(vec3 p) {
    // p in [0,1]^3
    float zf = clamp(p.z, 0.0, 1.0) * float(u_num_layers - 1);
    int   z0 = int(floor(zf));
    int   z1 = min(z0 + 1, u_num_layers - 1);
    float t  = fract(zf);

    vec2 uv = p.xy;
    if (u_flip_y != 0) uv.y = 1.0 - uv.y;

    vec4 a = texture(tex_array, vec3(uv, float(z0)));
    vec4 b = texture(tex_array, vec3(uv, float(z1)));
    return mix(a, b, t);
}

float heap_edge_fade(float d) {
    float r = max(u_radius, 1e-6);
    float t = clamp(d / r, 0.0, 1.0);

    // feather band near rim
    float rim0 = 1.0 - (u_softness / r);
    rim0 = clamp(rim0, 0.0, 1.0);

    float edge_fade = 1.0 - smoothstep(rim0, 1.0, t);

    // radius->depth shaping
    float shaped_t = pow(t, max(u_layer_stretch, 1e-6));

    // Return BOTH: fade controls blend; shaped_t controls depth
    // We'll recompute shaped_t in main to avoid packing.
    return edge_fade;
}

void main() {
    // map to [-1,1]
    vec2 s = (v_uv * 2.0 - 1.0);

    // keep pixels approximately square
    float aspect = u_slice_px.x / max(u_slice_px.y, 1.0);
    s.x *= aspect;

    // base plane position:
    vec3 p0 = u_center + (u_axis_u * (s.x * u_scale)) + (u_axis_v * (s.y * u_scale));

    // quick reject if outside before heap:
    if (any(lessThan(p0, vec3(0.0))) || any(greaterThan(p0, vec3(1.0)))) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 p = p0;

    if (u_heap_enable != 0) {
        float d = distance(v_uv, u_mouse);

        float r = max(u_radius, 1e-6);
        float t = clamp(d / r, 0.0, 1.0);
        float shaped_t = pow(t, max(u_layer_stretch, 1e-6));

        // center -> deepest, rim -> none
        float depth = (1.0 - shaped_t) * u_heap_depth;

        // offset along normal (dir = +/-1)
        p = p0 + (u_axis_n * (depth * u_heap_dir));

        // if dug outside volume, clamp by discarding (hard edge)
        if (any(lessThan(p, vec3(0.0))) || any(greaterThan(p, vec3(1.0)))) {
            // still show the undug plane (feels better than black):
            p = p0;
        }

        // blend based on rim fade (soft brush)
        vec4 outside = sample_volume(p0);
        vec4 inside  = sample_volume(p);

        float rim0 = 1.0 - (u_softness / r);
        rim0 = clamp(rim0, 0.0, 1.0);
        float edge_fade = 1.0 - smoothstep(rim0, 1.0, t);

        vec4 c = mix(outside, inside, edge_fade);

        if (u_bgr_input != 0) c.rgb = c.bgr;
        fragColor = vec4(c.rgb, 1.0);
        return;
    }

    // no heap
    vec4 c = sample_volume(p);
    if (u_bgr_input != 0) c.rgb = c.bgr;
    fragColor = vec4(c.rgb, 1.0);
}
"""


# ============================================================
# Gizmo shaders (3D box + 3D plane slab + normal arrow)
# ============================================================

GIZMO_VERT = r"""
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

GIZMO_FRAG = r"""
#version 330
uniform vec4 u_color;
out vec4 fragColor;
void main() { fragColor = u_color; }
"""


# ============================================================
# HUD textured quad shaders (for help + gizmo UI overlay)
# ============================================================

HUD_TEX_VERT = r"""
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() { v_uv = in_uv; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

HUD_TEX_FRAG = r"""
#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 fragColor;
void main() { fragColor = texture(u_tex, v_uv); }
"""


# ============================================================
# Math helpers
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return v / n

def orthonormal_basis_from_normal(n: np.ndarray):
    n = normalize(n)
    if abs(float(n[2])) < 0.9:
        a = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = normalize(np.cross(a, n))
    v = normalize(np.cross(n, u))
    return u, v

def yaw_pitch_to_normal(yaw: float, pitch: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    x = sy * cp
    y = cy * cp
    z = sp
    return normalize(np.array([x, y, z], dtype=np.float32))

def look_at(eye, target, up):
    eye = np.array(eye, np.float32)
    target = np.array(target, np.float32)
    up = np.array(up, np.float32)

    f = normalize(target - eye)
    s = normalize(np.cross(f, up))
    u = np.cross(s, f)

    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = -eye
    return M @ T

def perspective(fovy_deg, aspect, znear, zfar):
    fovy = np.deg2rad(fovy_deg)
    f = 1.0 / np.tan(fovy / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / max(aspect, 1e-8)
    M[1, 1] = f
    M[2, 2] = (zfar + znear) / (znear - zfar)
    M[2, 3] = (2 * zfar * znear) / (znear - zfar)
    M[3, 2] = -1.0
    return M


# ============================================================
# Main app
# ============================================================

class MPRPlaneUI(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "MPR Plane UI — LMB rotate | MMB pan | wheel zoom | heap brush (H) | WASDQE move"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ---------- load volume ----------
        # Expect (Z,H,W,3) uint8 BGR
        vol_path = Path("threshold_images") / "volume_uint8.npy"
        if not vol_path.exists():
            raise FileNotFoundError(f"Missing {vol_path}. Put your prebuilt volume there.")

        V = np.load(vol_path, mmap_mode="r")
        if V.dtype != np.uint8 or V.ndim != 4 or V.shape[3] != 3:
            raise ValueError(f"Unexpected volume: shape={V.shape} dtype={V.dtype} (expected (Z,H,W,3) uint8)")

        self.Z, self.H, self.W, _ = V.shape
        self.V = V

        print(f"[volume] Z={self.Z} H={self.H} W={self.W} dtype=uint8 BGR")

        # ---------- programs ----------
        self.slice_prog = self.ctx.program(vertex_shader=SLICE_VERT, fragment_shader=SLICE_FRAG)
        self.gizmo_prog = self.ctx.program(vertex_shader=GIZMO_VERT, fragment_shader=GIZMO_FRAG)

        # ---------- fullscreen quad ----------
        fsq = np.array([-1,-1,  1,-1,  -1, 1,
                        -1, 1,  1,-1,   1, 1], dtype="f4")
        self.slice_vbo = self.ctx.buffer(fsq.tobytes())
        self.slice_vao = self.ctx.simple_vertex_array(self.slice_prog, self.slice_vbo, "in_pos")

        # ---------- upload texture array (RGBA8) ----------
        # NOTE: sampler2DArray reads normalized floats from uint8 when dtype="f1"
        self.tex = self.ctx.texture_array((self.W, self.H, self.Z), components=4, dtype="f1")
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        alpha = np.full((self.H, self.W, 1), 255, dtype=np.uint8)
        for z in range(self.Z):
            slab = np.ascontiguousarray(self.V[z])              # (H,W,3) BGR
            slab_rgba = np.concatenate([slab, alpha], axis=2)   # (H,W,4)
            self.tex.write(slab_rgba.tobytes(), viewport=(0, 0, z, self.W, self.H, 1))
            if z % 25 == 0 or z == self.Z - 1:
                print(f"  uploaded {z+1}/{self.Z}")

        self.tex.use(location=0)
        self.slice_prog["tex_array"].value = 0
        self.slice_prog["u_num_layers"].value = int(self.Z)

        # ---------- plane state ----------
        self.yaw = 0.0
        self.pitch = 0.0
        self.center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.scale  = 0.55

        self._update_plane_axes()

        # ---------- heap brush state ----------
        self.heap_enable = True
        self.mouse_uv = (0.5, 0.5)     # bottom-left UV
        self.heap_radius = 0.18
        self.heap_softness = 0.06
        self.heap_depth = 0.22         # normalized volume units
        self.heap_stretch = 1.0
        self.heap_dir = -1.0           # -1 digs along -N; +1 digs along +N

        # flip rules
        # If your stack is PNG-origin top-left and you DID NOT flip on upload, set flip_y=1.
        # If you already fixed orientation by upload/format, set flip_y=0.
        self.flip_y = 1
        self.bgr_input = 1

        self._push_slice_uniforms()

        # ---------- input state ----------
        self._drag_plane = False
        self._drag_pan   = False
        self._held_keys = set()

        # ---------- gizmo camera state ----------
        self.gizmo_yaw = 0.8
        self.gizmo_pitch = 0.5
        self.gizmo_radius = 2.4

        # ---------- gizmo geometry ----------
        def to_gizmo(p01):
            return np.array(p01, np.float32) - 0.5

        corners = [
            to_gizmo([0,0,0]), to_gizmo([1,0,0]),
            to_gizmo([0,1,0]), to_gizmo([1,1,0]),
            to_gizmo([0,0,1]), to_gizmo([1,0,1]),
            to_gizmo([0,1,1]), to_gizmo([1,1,1]),
        ]
        edges = [
            (0,1),(0,2),(1,3),(2,3),
            (4,5),(4,6),(5,7),(6,7),
            (0,4),(1,5),(2,6),(3,7),
        ]
        box_lines = []
        for a,b in edges:
            box_lines.append(corners[a]); box_lines.append(corners[b])
        box_lines = np.array(box_lines, dtype="f4")  # (24,3)

        self.box_vbo = self.ctx.buffer(box_lines.tobytes())
        self.box_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.box_vbo, "in_pos")

        # Plane slab (extruded quad)
        self.plane_vbo = self.ctx.buffer(reserve=36 * 3 * 4)
        self.plane_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.plane_vbo, "in_pos")

        self.n_vbo = self.ctx.buffer(reserve=2 * 3 * 4)
        self.n_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.n_vbo, "in_pos")

        self._update_gizmo_geometry()

        print("Ready.")
        print("  LMB drag (main): rotate plane")
        print("  MMB drag: pan plane (U/V)")
        print("  Wheel: zoom plane size")
        print("  WASDQE: move plane; Shift = faster")
        print("  Heap brush: move mouse (H toggles)")
        print("    I/K = depth +/-    J/L = radius +/-    [ ] = softness -/+")
        print("    , / . = stretch -/+    N = flip dig direction")
        print("  Y = toggle flip_y (if upside-down)")
        print("  R: reset   ESC: quit")

    # ------------------------------------------------------------
    # Plane + uniforms
    # ------------------------------------------------------------

    def _update_plane_axes(self):
        n = yaw_pitch_to_normal(self.yaw, self.pitch)
        u, v = orthonormal_basis_from_normal(n)
        self.n, self.u, self.v = n, u, v

    def _push_slice_uniforms(self):
        self.slice_prog["u_center"].value = tuple(float(x) for x in self.center)
        self.slice_prog["u_axis_u"].value = tuple(float(x) for x in self.u)
        self.slice_prog["u_axis_v"].value = tuple(float(x) for x in self.v)
        self.slice_prog["u_axis_n"].value = tuple(float(x) for x in self.n)
        self.slice_prog["u_scale"].value  = float(self.scale)
        self.slice_prog["u_slice_px"].value = (float(self.wnd.width), float(self.wnd.height))

        # heap uniforms
        self.slice_prog["u_heap_enable"].value = int(self.heap_enable)
        self.slice_prog["u_mouse"].value = (float(self.mouse_uv[0]), float(self.mouse_uv[1]))
        self.slice_prog["u_radius"].value = float(self.heap_radius)
        self.slice_prog["u_softness"].value = float(self.heap_softness)
        self.slice_prog["u_layer_stretch"].value = float(self.heap_stretch)
        self.slice_prog["u_heap_depth"].value = float(self.heap_depth)
        self.slice_prog["u_heap_dir"].value = float(self.heap_dir)

        # orientation/color
        self.slice_prog["u_flip_y"].value = int(self.flip_y)
        self.slice_prog["u_bgr_input"].value = int(self.bgr_input)

    # ------------------------------------------------------------
    # Gizmo helpers
    # ------------------------------------------------------------

    def _gizmo_eye(self):
        cy, sy = np.cos(self.gizmo_yaw), np.sin(self.gizmo_yaw)
        cp, sp = np.cos(self.gizmo_pitch), np.sin(self.gizmo_pitch)
        x = self.gizmo_radius * sy * cp
        y = self.gizmo_radius * cy * cp
        z = self.gizmo_radius * sp
        return [x, y, z]

    def _gizmo_viewport(self):
        W, H = self.wnd.width, self.wnd.height
        giz_px = int(min(W, H) * 0.28)
        pad = 12
        gx0 = W - giz_px - pad
        gy0 = H - giz_px - pad
        return gx0, gy0, giz_px

    def _update_gizmo_geometry(self):
        # plane corners in gizmo space (volume coords minus 0.5)
        c = self.center
        u = self.u
        v = self.v
        n = self.n
        s = self.scale

        p00 = (c - u*s - v*s) - 0.5
        p10 = (c + u*s - v*s) - 0.5
        p01 = (c - u*s + v*s) - 0.5
        p11 = (c + u*s + v*s) - 0.5

        # extrude to a slab so it reads as a 3D object
        t = 0.02
        off = n * (t * 0.5)
        a00, a10, a01, a11 = p00+off, p10+off, p01+off, p11+off
        b00, b10, b01, b11 = p00-off, p10-off, p01-off, p11-off

        tris = []
        # top
        tris += [a00,a10,a01,  a01,a10,a11]
        # bottom (reverse)
        tris += [b00,b01,b10,  b01,b11,b10]
        # sides
        tris += [a00,b00,a10,  a10,b00,b10]
        tris += [a10,b10,a11,  a11,b10,b11]
        tris += [a11,b11,a01,  a01,b11,b01]
        tris += [a01,b01,a00,  a00,b01,b00]

        plane = np.array(tris, dtype=np.float32)
        self.plane_vbo.orphan(size=plane.nbytes)
        self.plane_vbo.write(plane.tobytes())

        # normal arrow
        start = (c - 0.5)
        end   = (c + self.n * 0.35 - 0.5)
        arrow = np.array([start, end], dtype=np.float32)
        self.n_vbo.write(arrow.tobytes())

    # ------------------------------------------------------------
    # Input: mouse
    # ------------------------------------------------------------

    def on_mouse_position_event(self, x, y, dx, dy):
        # update heap brush mouse UV continuously (bottom-left origin)
        u = x / max(1, self.wnd.width)
        v = 1.0 - (y / max(1, self.wnd.height))
        self.mouse_uv = (float(u), float(v))
        # avoid writing uniforms every pixel if you want, but this is fine:
        self.slice_prog["u_mouse"].value = self.mouse_uv

    def on_mouse_press_event(self, x, y, button):
        LEFT   = self.wnd.mouse.left
        MIDDLE = self.wnd.mouse.middle

        if button == LEFT:
            self._drag_plane = True
        if button == MIDDLE:
            self._drag_pan = True

    def on_mouse_release_event(self, x, y, button):
        LEFT   = self.wnd.mouse.left
        MIDDLE = self.wnd.mouse.middle
        if button == LEFT:
            self._drag_plane = False
        if button == MIDDLE:
            self._drag_pan = False

    def on_mouse_drag_event(self, x, y, dx, dy):
        # rotate plane
        if self._drag_plane:
            self.yaw   += dx * 0.005
            self.pitch += -dy * 0.005
            self.pitch = float(np.clip(self.pitch, -1.55, 1.55))
            self._update_plane_axes()
            self._push_slice_uniforms()
            self._update_gizmo_geometry()

        # pan plane center
        if self._drag_pan:
            W, H = max(self.wnd.width, 1), max(self.wnd.height, 1)
            aspect = W / max(H, 1)

            du = (dx / W) * (2.0 * self.scale) * aspect
            dv = (-dy / H) * (2.0 * self.scale)

            self.center += self.u * du + self.v * dv
            self.center[:] = np.clip(self.center, 0.0, 1.0)

            self._push_slice_uniforms()
            self._update_gizmo_geometry()

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.scale *= float(0.92 ** y_offset)
        self.scale = float(np.clip(self.scale, 0.05, 2.0))
        self._push_slice_uniforms()
        self._update_gizmo_geometry()

    # ------------------------------------------------------------
    # Input: keys (continuous while holding + heap controls)
    # ------------------------------------------------------------

    def on_key_event(self, key, action, modifiers):
        k = self.wnd.keys
        if action == k.ACTION_PRESS and key == k.ESCAPE:
            self.wnd.close()
            return

        # one-shot toggles + heap params
        if action == k.ACTION_PRESS:
            if key == k.H:
                self.heap_enable = not self.heap_enable
                self.slice_prog["u_heap_enable"].value = int(self.heap_enable)
                print(f"heap_enable={self.heap_enable}")
                return

            if key == k.N:
                self.heap_dir *= -1.0
                self.slice_prog["u_heap_dir"].value = float(self.heap_dir)
                print(f"heap_dir={self.heap_dir:+.0f}")
                return

            if key == k.Y:
                self.flip_y = 0 if self.flip_y else 1
                self.slice_prog["u_flip_y"].value = int(self.flip_y)
                print(f"flip_y={self.flip_y}")
                return

            # heap tweak keys
            if key == k.J:
                self.heap_radius = max(0.01, self.heap_radius - 0.01)
                self.slice_prog["u_radius"].value = float(self.heap_radius)
                return
            if key == k.L:
                self.heap_radius = min(0.9, self.heap_radius + 0.01)
                self.slice_prog["u_radius"].value = float(self.heap_radius)
                return

            if key == k.LEFT_BRACKET:
                self.heap_softness = max(0.0, self.heap_softness - 0.005)
                self.slice_prog["u_softness"].value = float(self.heap_softness)
                return
            if key == k.RIGHT_BRACKET:
                self.heap_softness = min(0.5, self.heap_softness + 0.005)
                self.slice_prog["u_softness"].value = float(self.heap_softness)
                return

            if key == k.I:
                self.heap_depth = min(1.0, self.heap_depth + 0.02)
                self.slice_prog["u_heap_depth"].value = float(self.heap_depth)
                return
            if key == k.K:
                self.heap_depth = max(0.0, self.heap_depth - 0.02)
                self.slice_prog["u_heap_depth"].value = float(self.heap_depth)
                return

            if key == k.COMMA:
                self.heap_stretch = max(0.1, self.heap_stretch - 0.1)
                self.slice_prog["u_layer_stretch"].value = float(self.heap_stretch)
                return
            if key == k.PERIOD:
                self.heap_stretch = min(10.0, self.heap_stretch + 0.1)
                self.slice_prog["u_layer_stretch"].value = float(self.heap_stretch)
                return

            if key == k.R:
                self.yaw = 0.0
                self.pitch = 0.0
                self.center[:] = (0.5, 0.5, 0.5)
                self.scale = 0.55 
                self.heap_enable = True
                self.heap_radius = 0.18
                self.heap_softness = 0.06
                self.heap_depth = 0.22
                self.heap_stretch = 1.0
                self.heap_dir = -1.0
                self.flip_y = 1
                self._update_plane_axes()
                self._push_slice_uniforms()
                self._update_gizmo_geometry()
                return

        # held movement keys
        move_keys = {k.W, k.S, k.A, k.D, k.Q, k.E}
        if key in move_keys:
            if action == k.ACTION_PRESS:
                self._held_keys.add((key, bool(modifiers.shift)))
            elif action == k.ACTION_RELEASE:
                self._held_keys = {km for km in self._held_keys if km[0] != key}

    def _apply_held_keys(self, dt: float):
        if not self._held_keys:
            return
        k = self.wnd.keys
        base = 0.22  # normalized units per second

        for key, is_shift in list(self._held_keys):
            step = base * dt * (3.0 if is_shift else 1.0)
            if key == k.W: self.center += self.n * step
            if key == k.S: self.center -= self.n * step
            if key == k.A: self.center -= self.u * step
            if key == k.D: self.center += self.u * step
            if key == k.Q: self.center -= self.v * step
            if key == k.E: self.center += self.v * step

        self.center[:] = np.clip(self.center, 0.0, 1.0)
        self._push_slice_uniforms()
        self._update_gizmo_geometry()

    # ------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------

    def resize(self, width, height):
        self._push_slice_uniforms()

    # ------------------------------------------------------------
    # Render
    # ------------------------------------------------------------

    def on_render(self, time: float, frame_time: float):
        W, H = self.wnd.width, self.wnd.height

        # smooth movement while held
        self._apply_held_keys(frame_time)

        # ---- main slice
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.tex.use(location=0)
        self.slice_vao.render()

        # ---- gizmo (top-right)
        gx0, gy0, giz_px = self._gizmo_viewport()
        self.ctx.viewport = (gx0, gy0, giz_px, giz_px)
        self.ctx.enable(moderngl.DEPTH_TEST)

        P = perspective(45.0, 1.0, 0.05, 10.0)
        V = look_at(eye=self._gizmo_eye(), target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0])
        MVP = (P @ V).astype(np.float32)
        self.gizmo_prog["u_mvp"].write(MVP.tobytes())

        # box wireframe
        self.gizmo_prog["u_color"].value = (0.85, 0.90, 0.98, 1.0)
        self.box_vao.render(mode=moderngl.LINES)
https://chatgpt.com/g/g-2fkFE8rbu-dall-e
        # plane slab
        self.gizmo_prog["u_color"].value = (1.0, 0.15, 0.15, 0.80)
        self.plane_vao.render(mode=moderngl.TRIANGLES)

        # normal arrow
        self.gizmo_prog["u_color"].value = (1.0, 0.90, 0.25, 1.0)
        self.n_vao.render(mode=moderngl.LINES)https://chatgpt.com/gpts

        self.ctx.disable(moderngl.DEPTH_TEST)


if __name__ == "__main__":
    mglw.run_window_config(MPRPlaneUI)
