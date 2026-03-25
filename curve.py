import ctypes, os
from pathlib import Path

import numpy as np
import moderngl
import moderngl_window as mglw


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

// color/orientation controls
uniform int   u_flip_y;         // 1 if texture rows are top-left origin (most PNG stacks), else 0
uniform int   u_bgr_input;      // 1 if stored as BGR in RGB channels, else 0

in vec2 v_uv;
out vec4 fragColor;

vec4 sample_volume(vec3 p) {
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

void main() {
    vec2 s = (v_uv * 2.0 - 1.0);

    float aspect = u_slice_px.x / max(u_slice_px.y, 1.0);
    s.x *= aspect;

    vec3 p0 = u_center + (u_axis_u * (s.x * u_scale)) + (u_axis_v * (s.y * u_scale));

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

        float depth = (1.0 - shaped_t) * u_heap_depth;
        p = p0 + (u_axis_n * (depth * u_heap_dir));

        if (any(lessThan(p, vec3(0.0))) || any(greaterThan(p, vec3(1.0)))) {
            p = p0;
        }

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

    vec4 c = sample_volume(p);
    if (u_bgr_input != 0) c.rgb = c.bgr;
    fragColor = vec4(c.rgb, 1.0);
}
"""


# ============================================================
# Gizmo shaders (3D box + deformed 3D plane slab + normal arrow)
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

        # Deformed slab VBO (dynamic)
        self.plane_vbo = self.ctx.buffer(reserve=1)  # will orphan to correct size
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

        self.slice_prog["u_heap_enable"].value = int(self.heap_enable)
        self.slice_prog["u_mouse"].value = (float(self.mouse_uv[0]), float(self.mouse_uv[1]))
        self.slice_prog["u_radius"].value = float(self.heap_radius)
        self.slice_prog["u_softness"].value = float(self.heap_softness)
        self.slice_prog["u_layer_stretch"].value = float(self.heap_stretch)
        self.slice_prog["u_heap_depth"].value = float(self.heap_depth)
        self.slice_prog["u_heap_dir"].value = float(self.heap_dir)

        self.slice_prog["u_flip_y"].value = int(self.flip_y)
        self.slice_prog["u_bgr_input"].value = int(self.bgr_input)

    # ------------------------------------------------------------
    # Gizmo deformation helpers
    # ------------------------------------------------------------

    def _heap_depth_at_uv(self, uv):
        """uv in [0,1]^2, bottom-left origin, same convention as u_mouse/v_uv."""
        if not self.heap_enable:
            return 0.0

        mx, my = self.mouse_uv
        ux, uy = float(uv[0]), float(uv[1])
        d = float(np.hypot(ux - mx, uy - my))

        r = max(float(self.heap_radius), 1e-6)
        t = np.clip(d / r, 0.0, 1.0)
        shaped_t = float(t ** max(float(self.heap_stretch), 1e-6))

        return (1.0 - shaped_t) * float(self.heap_depth)

    def _build_deformed_plane_slab(self, grid=40, slab_thickness=0.02):
        """
        Builds a deformed plane slab in gizmo space (volume coords minus 0.5).
        Deformation matches heap brush in plane-local UV.
        """
        c = self.center
        u = self.u
        v = self.v
        n = self.n
        s = float(self.scale)

        N = int(grid)
        verts_top = np.zeros(((N + 1) * (N + 1), 3), dtype=np.float32)

        idx = 0
        for j in range(N + 1):
            for i in range(N + 1):
                uv = np.array([i / N, j / N], dtype=np.float32)
                sx = (uv[0] * 2.0 - 1.0)
                sy = (uv[1] * 2.0 - 1.0)

                p0 = c + u * (sx * s) + v * (sy * s)

                depth = self._heap_depth_at_uv(uv)
                p = p0 + n * (depth * float(self.heap_dir))

                # match shader behavior: if out of bounds, fall back to base plane
                if np.any(p < 0.0) or np.any(p > 1.0):
                    p = p0

                verts_top[idx] = (p - 0.5).astype(np.float32)
                idx += 1

        tris = []
        def vid(i, j): return j * (N + 1) + i

        for j in range(N):
            for i in range(N):
                a = verts_top[vid(i, j)]
                b = verts_top[vid(i + 1, j)]
                c0 = verts_top[vid(i, j + 1)]
                d = verts_top[vid(i + 1, j + 1)]
                tris += [a, b, c0,  c0, b, d]

        top = np.array(tris, dtype=np.float32)

        off = (n * (slab_thickness * 0.5)).astype(np.float32)
        top_a = top + off
        bot_b = top - off

        # bottom reverse winding
        bot = bot_b.reshape(-1, 3)[::-1].copy()

        # side walls around boundary ring
        ring = []
        for i in range(N + 1): ring.append(vid(i, 0))
        for j in range(1, N + 1): ring.append(vid(N, j))
        for i in range(N - 1, -1, -1): ring.append(vid(i, N))
        for j in range(N - 1, 0, -1): ring.append(vid(0, j))

        ring_pts = verts_top[np.array(ring, dtype=np.int32)]
        ring_pts2 = np.vstack([ring_pts, ring_pts[:1]])

        side_tris = []
        for k in range(len(ring_pts2) - 1):
            p0 = ring_pts2[k]
            p1 = ring_pts2[k + 1]
            a0 = p0 + off
            a1 = p1 + off
            b0 = p0 - off
            b1 = p1 - off
            side_tris += [a0, b0, a1,  a1, b0, b1]

        sides = np.array(side_tris, dtype=np.float32)

        slab = np.vstack([top_a, bot, sides]).astype(np.float32)
        return slab

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
        plane = self._build_deformed_plane_slab(grid=40, slab_thickness=0.02)

        self.plane_vbo.orphan(size=plane.nbytes)
        self.plane_vbo.write(plane.tobytes())

        start = (self.center - 0.5)
        end   = (self.center + self.n * 0.35 - 0.5)
        arrow = np.array([start, end], dtype=np.float32)
        self.n_vbo.write(arrow.tobytes())

    # ------------------------------------------------------------
    # Input: mouse
    # ------------------------------------------------------------

    def on_mouse_position_event(self, x, y, dx, dy):
        u = x / max(1, self.wnd.width)
        v = 1.0 - (y / max(1, self.wnd.height))
        self.mouse_uv = (float(u), float(v))
        self.slice_prog["u_mouse"].value = self.mouse_uv

        # Update gizmo deformation as mouse moves
        if self.heap_enable:
            self._update_gizmo_geometry()

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
        if self._drag_plane:
            self.yaw   += dx * 0.005
            self.pitch += -dy * 0.005
            self.pitch = float(np.clip(self.pitch, -1.55, 1.55))
            self._update_plane_axes()
            self._push_slice_uniforms()
            self._update_gizmo_geometry()

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

        if action == k.ACTION_PRESS:
            if key == k.H:
                self.heap_enable = not self.heap_enable
                self.slice_prog["u_heap_enable"].value = int(self.heap_enable)
                self._update_gizmo_geometry()
                print(f"heap_enable={self.heap_enable}")
                return

            if key == k.N:
                self.heap_dir *= -1.0
                self.slice_prog["u_heap_dir"].value = float(self.heap_dir)
                self._update_gizmo_geometry()
                print(f"heap_dir={self.heap_dir:+.0f}")
                return

            if key == k.Y:
                self.flip_y = 0 if self.flip_y else 1
                self.slice_prog["u_flip_y"].value = int(self.flip_y)
                print(f"flip_y={self.flip_y}")
                return

            if key == k.J:
                self.heap_radius = max(0.01, self.heap_radius - 0.01)
                self.slice_prog["u_radius"].value = float(self.heap_radius)
                self._update_gizmo_geometry()
                return
            if key == k.L:
                self.heap_radius = min(0.9, self.heap_radius + 0.01)
                self.slice_prog["u_radius"].value = float(self.heap_radius)
                self._update_gizmo_geometry()
                return

            if key == k.LEFT_BRACKET:
                self.heap_softness = max(0.0, self.heap_softness - 0.005)
                self.slice_prog["u_softness"].value = float(self.heap_softness)
                self._update_gizmo_geometry()
                return
            if key == k.RIGHT_BRACKET:
                self.heap_softness = min(0.5, self.heap_softness + 0.005)
                self.slice_prog["u_softness"].value = float(self.heap_softness)
                self._update_gizmo_geometry()
                return

            if key == k.I:
                self.heap_depth = min(1.0, self.heap_depth + 0.02)
                self.slice_prog["u_heap_depth"].value = float(self.heap_depth)
                self._update_gizmo_geometry()
                return
            if key == k.K:
                self.heap_depth = max(0.0, self.heap_depth - 0.02)
                self.slice_prog["u_heap_depth"].value = float(self.heap_depth)
                self._update_gizmo_geometry()
                return

            if key == k.COMMA:
                self.heap_stretch = max(0.1, self.heap_stretch - 0.1)
                self.slice_prog["u_layer_stretch"].value = float(self.heap_stretch)
                self._update_gizmo_geometry()
                return
            if key == k.PERIOD:
                self.heap_stretch = min(10.0, self.heap_stretch + 0.1)
                self.slice_prog["u_layer_stretch"].value = float(self.heap_stretch)
                self._update_gizmo_geometry()
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
        base = 0.22

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
    # Resize (moderngl_window calls this)
    # ------------------------------------------------------------

    def on_resize(self, width: int, height: int):
        self._push_slice_uniforms()

    # ------------------------------------------------------------
    # Render
    # ------------------------------------------------------------

    def on_render(self, time: float, frame_time: float):
        W, H = self.wnd.width, self.wnd.height
        self._apply_held_keys(frame_time)

        # main slice
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.tex.use(location=0)
        self.slice_vao.render()

        # gizmo (top-right)
        gx0, gy0, giz_px = self._gizmo_viewport()
        self.ctx.viewport = (gx0, gy0, giz_px, giz_px)
        self.ctx.enable(moderngl.DEPTH_TEST)

        P = perspective(45.0, 1.0, 0.05, 10.0)
        V = look_at(eye=self._gizmo_eye(), target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0])
        MVP = (P @ V).astype(np.float32)
        self.gizmo_prog["u_mvp"].write(MVP.tobytes())

        self.gizmo_prog["u_color"].value = (0.85, 0.90, 0.98, 1.0)
        self.box_vao.render(mode=moderngl.LINES)

        self.gizmo_prog["u_color"].value = (1.0, 0.15, 0.15, 0.80)
        self.plane_vao.render(mode=moderngl.TRIANGLES)

        self.gizmo_prog["u_color"].value = (1.0, 0.90, 0.25, 1.0)
        self.n_vao.render(mode=moderngl.LINES)

        self.ctx.disable(moderngl.DEPTH_TEST)


if __name__ == "__main__":
    mglw.run_window_config(MPRPlaneUI)