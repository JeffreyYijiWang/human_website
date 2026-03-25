import ctypes, os

# Tell Windows to use the NVIDIA GPU for this process (best-effort)
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

import numpy as np
from pathlib import Path

import moderngl
import moderngl_window as mglw
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Slice shaders (full-screen view)
# ============================================================

SLICE_VERT = r"""
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = (in_pos * 0.5) + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

SLICE_FRAG = r"""
#version 330

uniform sampler2DArray tex_array;
uniform int   u_num_layers;     // Z
uniform vec2  u_slice_px;       // viewport (w,h) in pixels

uniform vec3  u_center;         // plane center in normalized volume coords [0,1]^3
uniform vec3  u_axis_u;         // plane axis U (unit)
uniform vec3  u_axis_v;         // plane axis V (unit)
uniform float u_scale;          // half-width in normalized volume units

in vec2 v_uv;
out vec4 fragColor;

vec4 sample_volume(vec3 p) {
    float zf = clamp(p.z, 0.0, 1.0) * float(u_num_layers - 1);
    int   z0 = int(floor(zf));
    int   z1 = min(z0 + 1, u_num_layers - 1);
    float t  = fract(zf);

    // If your stack is top-left origin, keep this flip. If wrong, remove.
    vec2 uv = vec2(p.x, 1.0 - p.y);

    vec4 a = texture(tex_array, vec3(uv, float(z0)));
    vec4 b = texture(tex_array, vec3(uv, float(z1)));
    return mix(a, b, t);
}

void main() {
    // map to [-1,1]
    vec2 s = (v_uv * 2.0 - 1.0);

    // keep pixels approximately square
    float aspect = u_slice_px.x / max(u_slice_px.y, 1.0);
    s.x *= aspect;

    vec3 p = u_center + (u_axis_u * (s.x * u_scale)) + (u_axis_v * (s.y * u_scale));

    if (any(lessThan(p, vec3(0.0))) || any(greaterThan(p, vec3(1.0)))) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec4 c = sample_volume(p);

    // Input volume stored as BGR -> show as RGB
    c.rgb = c.bgr;

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

def inv(m: np.ndarray) -> np.ndarray:
    return np.linalg.inv(m).astype(np.float32)

def gizmo_ray_from_ndc(ndc_x: float, ndc_y: float, P: np.ndarray, V: np.ndarray):
    """
    World-space ray from gizmo camera through NDC in [-1,1]^2.
    """
    PV_inv = inv(P @ V)

    p_near = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
    p_far  = np.array([ndc_x, ndc_y,  1.0, 1.0], dtype=np.float32)

    w_near = PV_inv @ p_near
    w_far  = PV_inv @ p_far
    w_near /= max(float(w_near[3]), 1e-8)
    w_far  /= max(float(w_far[3]), 1e-8)

    origin = w_near[:3]
    direction = normalize(w_far[:3] - w_near[:3])
    return origin, direction

def ray_aabb_intersect(ro, rd, bmin, bmax):
    """
    Slab method. Returns (hit, tmin, tmax).
    """
    ro = np.array(ro, np.float32)
    rd = np.array(rd, np.float32)
    bmin = np.array(bmin, np.float32)
    bmax = np.array(bmax, np.float32)

    invd = np.where(np.abs(rd) > 1e-8, 1.0 / rd, 1e8)
    t0 = (bmin - ro) * invd
    t1 = (bmax - ro) * invd
    tmin = float(np.max(np.minimum(t0, t1)))
    tmax = float(np.min(np.maximum(t0, t1)))
    return (tmax >= max(tmin, 0.0)), tmin, tmax

def hit_face_from_point(p, eps=0.03):
    """
    p is on cube surface in gizmo space [-0.5,0.5]^3.
    Returns one of '+x','-x','+y','-y','+z','-z' or None.
    """
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    if abs(x - 0.5) < eps:  return '+x'
    if abs(x + 0.5) < eps:  return '-x'
    if abs(y - 0.5) < eps:  return '+y'
    if abs(y + 0.5) < eps:  return '-y'
    if abs(z - 0.5) < eps:  return '+z'
    if abs(z + 0.5) < eps:  return '-z'
    return None


# ============================================================
# Main app
# ============================================================

class MPRPlaneUI(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "MPR Plane UI — LMB rotate | MMB pan | wheel zoom | T help | WASDQE move | click gizmo faces/buttons"
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
        self.hud_prog   = self.ctx.program(vertex_shader=HUD_TEX_VERT, fragment_shader=HUD_TEX_FRAG)

        # ---------- fullscreen quad ----------
        fsq = np.array([-1,-1,  1,-1,  -1, 1,
                        -1, 1,  1,-1,   1, 1], dtype="f4")
        self.slice_vbo = self.ctx.buffer(fsq.tobytes())
        self.slice_vao = self.ctx.simple_vertex_array(self.slice_prog, self.slice_vbo, "in_pos")

        # ---------- upload texture array (uint8 RGBA) ----------
        self.tex = self.ctx.texture_array((self.W, self.H, self.Z), components=4, dtype="f1")
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        alpha = np.full((self.H, self.W, 1), 255, dtype=np.uint8)
        for z in range(self.Z):
            slab = np.ascontiguousarray(self.V[z])  # (H,W,3) BGR
            slab_rgba = np.concatenate([slab, alpha], axis=2)  # (H,W,4)
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
        self._push_slice_uniforms()

        # ---------- input state ----------
        self._drag_plane = False
        self._drag_gizmo = False
        self._drag_pan   = False
        self._last_mouse = None
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

        # Plane as a *3D slab* (extruded)
        self.plane_vbo = self.ctx.buffer(reserve=36 * 3 * 4)  # up to 36 verts
        self.plane_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.plane_vbo, "in_pos")

        self.n_vbo = self.ctx.buffer(reserve=2 * 3 * 4)
        self.n_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.n_vbo, "in_pos")

        self._update_gizmo_geometry()

        # ---------- help tooltip overlay ----------
        self.show_help = False
        self.help_vbo = self.ctx.buffer(reserve=6 * 4 * 4)
        self.help_vao = self.ctx.simple_vertex_array(self.hud_prog, self.help_vbo, "in_pos", "in_uv")
        self.help_tex = None
        self._help_px = (0, 0)

        # ---------- gizmo UI overlay (buttons inside gizmo viewport) ----------
        self.gizmo_ui_vbo = self.ctx.buffer(reserve=6 * 4 * 4)
        self.gizmo_ui_vao = self.ctx.simple_vertex_array(self.hud_prog, self.gizmo_ui_vbo, "in_pos", "in_uv")
        self.gizmo_ui_tex = None
        self._gizmo_ui_px = (0, 0)

        print("Ready.")
        print("  Main view: LMB drag rotates plane; MMB pans; wheel zooms plane size")
        print("  Gizmo: click faces to snap (1..6), click X/Y/Z buttons to snap axis planes")
        print("  Gizmo: LMB drag (empty gizmo area) orbits gizmo camera")
        print("  WASDQE: move plane (hold to move continuously); Shift speeds up")
        print("  T: toggle help overlay; R: reset; ESC: quit")

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
        self.slice_prog["u_scale"].value  = float(self.scale)
        self.slice_prog["u_slice_px"].value = (float(self.wnd.width), float(self.wnd.height))

    def _snap_plane_to_axis(self, axis: str):
        if axis == 'x': self.n = np.array([1,0,0], np.float32)
        if axis == 'y': self.n = np.array([0,1,0], np.float32)
        if axis == 'z': self.n = np.array([0,0,1], np.float32)

        self.u, self.v = orthonormal_basis_from_normal(self.n)
        self.yaw = float(np.arctan2(float(self.n[0]), float(self.n[1])))
        self.pitch = float(np.arcsin(np.clip(float(self.n[2]), -1.0, 1.0)))

        self._push_slice_uniforms()
        self._update_gizmo_geometry()

    def _snap_gizmo_camera_to_face(self, face: str):
        # "Unity-like" view cube snap
        if face == '+x': self.gizmo_yaw, self.gizmo_pitch = (np.pi/2, 0.0)
        if face == '-x': self.gizmo_yaw, self.gizmo_pitch = (-np.pi/2, 0.0)
        if face == '+y': self.gizmo_yaw, self.gizmo_pitch = (0.0, 0.0)
        if face == '-y': self.gizmo_yaw, self.gizmo_pitch = (np.pi, 0.0)
        if face == '+z': self.gizmo_yaw, self.gizmo_pitch = (0.0, np.pi/2)
        if face == '-z': self.gizmo_yaw, self.gizmo_pitch = (0.0, -np.pi/2)

    def _snap_plane_to_face(self, face: str):
        d = {
            '+x': np.array([ 1, 0, 0], np.float32),
            '-x': np.array([-1, 0, 0], np.float32),
            '+y': np.array([ 0, 1, 0], np.float32),
            '-y': np.array([ 0,-1, 0], np.float32),
            '+z': np.array([ 0, 0, 1], np.float32),
            '-z': np.array([ 0, 0,-1], np.float32),
        }[face]

        self.n = d
        self.u, self.v = orthonormal_basis_from_normal(self.n)

        self.yaw = float(np.arctan2(float(self.n[0]), float(self.n[1])))
        self.pitch = float(np.arcsin(np.clip(float(self.n[2]), -1.0, 1.0)))

        self._snap_gizmo_camera_to_face(face)
        self._push_slice_uniforms()
        self._update_gizmo_geometry()

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

    def _in_gizmo(self, x, y):
        W, H = self.wnd.width, self.wnd.height
        gx0, gy0, giz_px = self._gizmo_viewport()
        y_bot = H - y
        return (gx0 <= x <= gx0 + giz_px) and (gy0 <= y_bot <= gy0 + giz_px)

    def _pick_gizmo_face(self, x, y):
        gx0, gy0, giz_px = self._gizmo_viewport()
        W, H = self.wnd.width, self.wnd.height

        y_bot = H - y
        lx = (x - gx0) / max(giz_px, 1)
        ly = (y_bot - gy0) / max(giz_px, 1)

        ndc_x = lx * 2.0 - 1.0
        ndc_y = ly * 2.0 - 1.0

        P = perspective(45.0, 1.0, 0.05, 10.0)
        V = look_at(eye=self._gizmo_eye(), target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0])

        ro, rd = gizmo_ray_from_ndc(ndc_x, ndc_y, P, V)
        hit, tmin, _ = ray_aabb_intersect(ro, rd, [-0.5,-0.5,-0.5], [0.5,0.5,0.5])
        if not hit:
            return None

        p = ro + rd * tmin
        return hit_face_from_point(p, eps=0.04)

    def _update_gizmo_geometry(self):
        # Plane quad in volume coords, then map to gizmo space by subtracting 0.5
        c = self.center
        u = self.u
        v = self.v
        n = self.n
        s = self.scale

        p00 = (c - u*s - v*s) - 0.5
        p10 = (c + u*s - v*s) - 0.5
        p01 = (c - u*s + v*s) - 0.5
        p11 = (c + u*s + v*s) - 0.5

        # extrude the plane into a 3D slab
        t = 0.02
        off = n * (t * 0.5)

        a00, a10, a01, a11 = p00+off, p10+off, p01+off, p11+off
        b00, b10, b01, b11 = p00-off, p10-off, p01-off, p11-off

        tris = []

        # top
        tris += [a00,a10,a01,  a01,a10,a11]
        # bottom (reverse winding)
        tris += [b00,b01,b10,  b01,b11,b10]

        # sides
        tris += [a00,b00,a10,  a10,b00,b10]
        tris += [a10,b10,a11,  a11,b10,b11]
        tris += [a11,b11,a01,  a01,b11,b01]
        tris += [a01,b01,a00,  a00,b01,b00]

        plane = np.array(tris, dtype=np.float32)
        self.plane_vbo.orphan(size=plane.nbytes)
        self.plane_vbo.write(plane.tobytes())

        # Normal arrow
        start = (c - 0.5)
        end   = (c + self.n * 0.35 - 0.5)
        arrow = np.array([start, end], dtype=np.float32)
        self.n_vbo.write(arrow.tobytes())

    # ------------------------------------------------------------
    # Fonts + Help HUD
    # ------------------------------------------------------------

    def _font(self, size: int):
        for p in [
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]:
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
        return ImageFont.load_default()

    def _build_help_image(self, w, h):
        img = Image.new("RGBA", (w, h), (18, 20, 35, 220))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, w-1, h-1], outline=(90, 150, 255, 255), width=2)

        lines = [
            "Controls",
            "",
            "Main view:",
            "  LMB drag: rotate plane",
            "  MMB drag: pan plane (U/V)",
            "  Wheel: zoom plane size",
            "",
            "Gizmo (top-right):",
            "  Click cube face: snap to 1..6 views (±X ±Y ±Z)",
            "  Click X/Y/Z: snap axis-aligned plane",
            "  LMB drag empty area: orbit gizmo camera",
            "",
            "Hold keys (smooth):",
            "  W/S: move along normal",
            "  A/D: move along U",
            "  Q/E: move along V",
            "",
            "R: reset   T: toggle help   ESC: quit",
        ]

        f_title = self._font(max(12, h//10))
        f_line  = self._font(max(10, h//18))

        y = 10
        d.text((12, y), lines[0], font=f_title, fill=(220, 235, 255, 255))
        y += f_title.size + 8

        for s in lines[1:]:
            d.text((12, y), s, font=f_line, fill=(200, 220, 255, 235))
            y += f_line.size + 6

        return np.ascontiguousarray(np.array(img, dtype=np.uint8))

    def _ensure_help_tex(self):
        W, H = max(self.wnd.width, 1), max(self.wnd.height, 1)
        pw = max(360, int(W * 0.38))
        ph = max(220, int(H * 0.36))

        if self.help_tex is None or self._help_px != (pw, ph):
            if self.help_tex:
                self.help_tex.release()
            self.help_tex = self.ctx.texture((pw, ph), 4)
            self.help_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._help_px = (pw, ph)

            rgba = self._build_help_image(pw, ph)
            self.help_tex.write(rgba.tobytes())

            # quad in NDC top-left
            pad = 12
            x0 = -1.0 + 2.0 * pad / W
            y1 =  1.0 - 2.0 * pad / H
            x1 = x0 + 2.0 * pw / W
            y0 = y1 - 2.0 * ph / H

            quad = np.array([
                x0,y0, 0,1,
                x1,y0, 1,1,
                x0,y1, 0,0,
                x0,y1, 0,0,
                x1,y0, 1,1,
                x1,y1, 1,0,
            ], dtype="f4")
            self.help_vbo.write(quad.tobytes())

    # ------------------------------------------------------------
    # Gizmo UI overlay (X/Y/Z + 1..6 buttons)
    # ------------------------------------------------------------

    def _gizmo_ui_buttons(self):
        """
        Button rects in *event coords* (top-left origin).
        Returns list of: ((x0,y0,x1,y1), payload)
        payload: ('axis','x'|'y'|'z') or ('face','+x'..)
        """
        gx0, gy0, giz_px = self._gizmo_viewport()
        W, H = self.wnd.width, self.wnd.height

        # gizmo top-left in event coords
        top = H - (gy0 + giz_px)
        left = gx0

        pad = int(max(6, giz_px * 0.04))
        bw = int(max(34, giz_px * 0.18))
        bh = int(max(24, giz_px * 0.12))
        gap = int(max(5, giz_px * 0.03))

        btns = []

        # Axis buttons down left
        bx = left + pad
        by = top + pad
        btns.append(((bx, by + 0*(bh+gap), bx+bw, by + 0*(bh+gap) + bh), ('axis','x')))
        btns.append(((bx, by + 1*(bh+gap), bx+bw, by + 1*(bh+gap) + bh), ('axis','y')))
        btns.append(((bx, by + 2*(bh+gap), bx+bw, by + 2*(bh+gap) + bh), ('axis','z')))

        # Face buttons (1..6) along bottom inside gizmo
        # Map numbers to faces for your "1..6 game facing":
        # 1:+X 2:-X 3:+Y 4:-Y 5:+Z 6:-Z
        labels = [('1','+x'), ('2','-x'), ('3','+y'), ('4','-y'), ('5','+z'), ('6','-z')]

        fw = int(max(28, giz_px * 0.12))
        fh = int(max(22, giz_px * 0.10))
        fy = top + giz_px - pad - fh
        fx = left + pad
        for i, (lab, face) in enumerate(labels):
            x0 = fx + i * (fw + gap)
            x1 = x0 + fw
            if x1 > left + giz_px - pad:
                break
            btns.append(((x0, fy, x1, fy+fh), ('face', face)))

        return btns

    def _hit_gizmo_ui_button(self, x, y):
        for (x0,y0,x1,y1), payload in self._gizmo_ui_buttons():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return payload
        return None

    def _build_gizmo_ui_image(self, w, h):
        """
        Builds an RGBA image sized to the gizmo viewport. We draw buttons directly into it.
        """
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        # translate button rects into gizmo-local coords
        gx0, gy0, giz_px = self._gizmo_viewport()
        W, H = self.wnd.width, self.wnd.height
        top = H - (gy0 + giz_px)
        left = gx0

        f = self._font(max(12, int(giz_px * 0.10)))

        def draw_button(rect_evt, text):
            x0,y0,x1,y1 = rect_evt
            # to gizmo-local
            rx0 = int(x0 - left)
            ry0 = int(y0 - top)
            rx1 = int(x1 - left)
            ry1 = int(y1 - top)

            d.rectangle([rx0, ry0, rx1, ry1], fill=(18, 20, 35, 185), outline=(90, 150, 255, 235), width=2)
            
            bbox = d.textbbox((0, 0), text, font=f)   # (l,t,r,b)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            cx = (rx0 + rx1 - tw) // 2
            cy = (ry0 + ry1 - th) // 2

        for (rect, payload) in self._gizmo_ui_buttons():
            kind, val = payload
            if kind == 'axis':
                draw_button(rect, val.upper())
            elif kind == 'face':
                # show 1..6 label based on mapping
                mapping = {'+x':'1', '-x':'2', '+y':'3', '-y':'4', '+z':'5', '-z':'6'}
                draw_button(rect, mapping.get(val, '?'))

        # Optional: tiny legend
        legend = "X/Y/Z: planes   1..6: faces"
        lf = self._font(max(10, int(giz_px * 0.07)))
        d.text((8, 8), legend, font=lf, fill=(220, 235, 255, 210))

        return np.ascontiguousarray(np.array(img, dtype=np.uint8))

    def _ensure_gizmo_ui_tex(self):
        gx0, gy0, giz_px = self._gizmo_viewport()
        w = max(1, giz_px)
        h = max(1, giz_px)

        if self.gizmo_ui_tex is None or self._gizmo_ui_px != (w, h):
            if self.gizmo_ui_tex:
                self.gizmo_ui_tex.release()
            self.gizmo_ui_tex = self.ctx.texture((w, h), 4)
            self.gizmo_ui_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._gizmo_ui_px = (w, h)

        rgba = self._build_gizmo_ui_image(w, h)
        self.gizmo_ui_tex.write(rgba.tobytes())

        # Build a quad that covers the gizmo viewport in *screen NDC*
        W, H = max(self.wnd.width, 1), max(self.wnd.height, 1)

        # viewport is (gx0,gy0,giz_px,giz_px) in bottom-left coords
        # Convert to NDC box:
        x0_px = gx0
        x1_px = gx0 + giz_px
        y0_px = gy0
        y1_px = gy0 + giz_px

        # NDC:
        x0 = (x0_px / W) * 2.0 - 1.0
        x1 = (x1_px / W) * 2.0 - 1.0
        y0 = (y0_px / H) * 2.0 - 1.0
        y1 = (y1_px / H) * 2.0 - 1.0

        quad = np.array([
            x0,y0, 0,0,
            x1,y0, 1,0,
            x0,y1, 0,1,
            x0,y1, 0,1,
            x1,y0, 1,0,
            x1,y1, 1,1,
        ], dtype="f4")
        self.gizmo_ui_vbo.write(quad.tobytes())

    # ------------------------------------------------------------
    # Input: mouse
    # ------------------------------------------------------------

    def on_mouse_press_event(self, x, y, button):
        LEFT   = self.wnd.mouse.left
        MIDDLE = self.wnd.mouse.middle

        if button == MIDDLE:
            self._drag_pan = True
            self._last_mouse = (x, y)
            return

        if button != LEFT:
            return

        # If click is inside gizmo: prioritize gizmo UI buttons -> cube faces -> orbit drag
        if self._in_gizmo(x, y):
            payload = self._hit_gizmo_ui_button(x, y)
            if payload is not None:
                kind, val = payload
                if kind == 'axis':
                    self._snap_plane_to_axis(val)
                elif kind == 'face':
                    self._snap_plane_to_face(val)
                self._drag_gizmo = False
                self._drag_plane = False
                self._last_mouse = None
                return

            face = self._pick_gizmo_face(x, y)
            if face is not None:
                self._snap_plane_to_face(face)
                self._drag_gizmo = False
                self._drag_plane = False
                self._last_mouse = None
                return

            # otherwise orbit gizmo camera
            self._drag_gizmo = True
            self._drag_plane = False
            self._last_mouse = (x, y)
            return

        # main view rotates plane
        self._drag_plane = True
        self._drag_gizmo = False
        self._last_mouse = (x, y)

    def on_mouse_position_event(self, x, y, dx, dy):
        """
        Optional: keep last mouse coords updated even when not dragging.
        Doesn't rotate by itself; drag happens in mouse_drag_event.
        """
        self._mouse_px_x = x
        self._mouse_px_y = y

    def on_mouse_release_event(self, x, y, button):
        LEFT   = self.wnd.mouse.left
        MIDDLE = self.wnd.mouse.middle

        if button == LEFT:
            self._drag_plane = False
            self._drag_gizmo = False
        if button == MIDDLE:
            self._drag_pan = False

        self._last_mouse = None

    def on_mouse_drag_event(self, x, y, dx, dy):
        if not (self._drag_plane or self._drag_gizmo or self._drag_pan):
            return

        mdx = dx
        mdy = dy

        # rotate plane in main view
        if self._drag_plane:
            self.yaw   += mdx * 0.005
            self.pitch += -mdy * 0.005
            self.pitch = float(np.clip(self.pitch, -1.55, 1.55))
            self._update_plane_axes()
            self._push_slice_uniforms()
            self._update_gizmo_geometry()

        # orbit gizmo camera
        if self._drag_gizmo:
            self.gizmo_yaw   += mdx * 0.01
            self.gizmo_pitch += -mdy * 0.01
            self.gizmo_pitch = float(np.clip(self.gizmo_pitch, -1.2, 1.2))

        # pan plane center
        if self._drag_pan:
            W, H = max(self.wnd.width, 1), max(self.wnd.height, 1)
            aspect = W / max(H, 1)

            du = (mdx / W) * (2.0 * self.scale) * aspect
            dv = (-mdy / H) * (2.0 * self.scale)

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
    # Input: keys (continuous while holding)
    # ------------------------------------------------------------

    def on_key_event(self, key, action, modifiers):
        k = self.wnd.keys

        # one-shot keys
        if action == k.ACTION_PRESS and key == k.ESCAPE:
            self.wnd.close()
            return

        if action == k.ACTION_PRESS and key == k.T:
            self.show_help = not self.show_help
            if self.show_help:
                self._ensure_help_tex()
            return

        if action == k.ACTION_PRESS and key == k.R:
            self.yaw = 0.0
            self.pitch = 0.0
            self.center[:] = (0.5, 0.5, 0.5)
            self.scale = 0.55
            self.gizmo_yaw = 0.8
            self.gizmo_pitch = 0.5
            self.gizmo_radius = 2.4
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
        if self.show_help:
            self._ensure_help_tex()
        self._ensure_gizmo_ui_tex()

    # ------------------------------------------------------------
    # Render
    # ------------------------------------------------------------

    def on_render(self, time: float, frame_time: float):
        W, H = self.wnd.width, self.wnd.height

        # continuous movement while held:
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

        # plane slab (3D)
        self.gizmo_prog["u_color"].value = (1.0, 0.15, 0.15, 0.80)
        self.plane_vao.render(mode=moderngl.TRIANGLES)

        # normal arrow
        self.gizmo_prog["u_color"].value = (1.0, 0.90, 0.25, 1.0)
        self.n_vao.render(mode=moderngl.LINES)

        self.ctx.disable(moderngl.DEPTH_TEST)

        # ---- gizmo UI overlay (buttons)
        self._ensure_gizmo_ui_tex()

        self.ctx.viewport = (0, 0, W, H)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.gizmo_ui_tex.use(location=5)
        self.hud_prog["u_tex"].value = 5
        self.gizmo_ui_vao.render(mode=moderngl.TRIANGLES)

        # ---- help tooltip overlay (top-left)
        if self.show_help:
            self._ensure_help_tex()
            self.help_tex.use(location=3)
            self.hud_prog["u_tex"].value = 3
            self.help_vao.render(mode=moderngl.TRIANGLES)

        self.ctx.disable(moderngl.BLEND)


if __name__ == "__main__":
    mglw.run_window_config(MPRPlaneUI)