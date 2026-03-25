import ctypes
import os
from pathlib import Path
import math

import cv2
import numpy as np
import moderngl
import moderngl_window as mglw
from skimage import measure

import json
from pathlib import Path

PRESET_PATH = Path("plane_preset.json")
MAX_POLY_POINTS = 16
# ============================================================
# Best-effort: prefer NVIDIA GPU on Windows
# ============================================================
try:
    ctypes.windll.ntdll.NtSetInformationProcess(
        ctypes.windll.kernel32.GetCurrentProcess(),
        0x27,
        ctypes.byref(ctypes.c_ulong(1)),
        ctypes.sizeof(ctypes.c_ulong),
    )
except Exception:
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"


# ============================================================
# Config
# ============================================================
VOL_PATH = Path("threshold_images") / "volume_uint8.npy"
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

MC_DOWNSAMPLE = 2

# 3D Physarum grid is downsampled to keep memory reasonable
PHYS3D_DOWNSAMPLE = 6
PHYS3D_AGENT_COUNT = 5000


# ============================================================
# Slice shaders
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

# =========================
# 2) REPLACE SLICE_FRAG WITH THIS
# =========================
SLICE_FRAG = r"""
#version 330

uniform sampler2DArray tex_array;
uniform int   u_num_layers;

uniform sampler2DArray trail_tex_array;
uniform int   u_trail_num_layers;
uniform int   u_phys3d_enable;
uniform float u_phys3d_gain;

uniform vec2  u_slice_px;

uniform vec3  u_center;
uniform vec3  u_axis_u;
uniform vec3  u_axis_v;
uniform vec3  u_axis_n;
uniform float u_scale;

uniform vec3  u_box_min;
uniform vec3  u_box_size;
uniform float u_box_diag;

uniform int   u_heap_enable;
uniform vec2  u_mouse;
uniform float u_radius;
uniform float u_softness;
uniform float u_layer_stretch;
uniform float u_heap_depth;
uniform float u_heap_dir;

uniform int   u_flip_y;
uniform int   u_bgr_input;

uniform int   u_threshold_enable;
uniform float u_threshold_value;
uniform int   u_threshold_invert;

// -------- plane preset uniforms --------
uniform int   u_plane_mode;   // 0 regular, 1 bezier, 2 noise, 3 polyline
uniform float u_plane_width;

uniform vec3  u_bezier_p0;
uniform vec3  u_bezier_p1;
uniform vec3  u_bezier_p2;
uniform vec3  u_bezier_p3;

uniform int   u_poly_count;
uniform vec3  u_poly_points[16];

uniform float u_noise_amp;
uniform float u_noise_freq;
uniform float u_noise_speed;
uniform float u_time;
// --------------------------------------

in vec2 v_uv;
out vec4 fragColor;

vec3 world_to_norm(vec3 p_world) {
    return (p_world - u_box_min) / max(u_box_size, vec3(1e-8));
}

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float noise2(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);

    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);

    return mix(a, b, u.x)
         + (c - a) * u.y * (1.0 - u.x)
         + (d - b) * u.x * u.y;
}

vec4 sample_volume_norm(vec3 p) {
    float zf = clamp(p.z, 0.0, 1.0) * float(u_num_layers - 1);
    int   z0 = int(floor(zf));
    int   z1 = min(z0 + 1, u_num_layers - 1);
    float t  = fract(zf);

    vec2 uv = p.xy;
    if (u_flip_y != 0) uv.y = 1.0 - uv.y;

    vec4 a = texture(tex_array, vec3(uv, float(z0)));
    vec4 b = texture(tex_array, vec3(uv, float(z1)));
    vec4 c = mix(a, b, t);
    if (u_bgr_input != 0) c.rgb = c.bgr;
    return c;
}

float sample_trail_norm(vec3 p) {
    float zf = clamp(p.z, 0.0, 1.0) * float(u_trail_num_layers - 1);
    int   z0 = int(floor(zf));
    int   z1 = min(z0 + 1, u_trail_num_layers - 1);
    float t  = fract(zf);

    vec2 uv = p.xy;
    if (u_flip_y != 0) uv.y = 1.0 - uv.y;

    float a = texture(trail_tex_array, vec3(uv, float(z0))).r;
    float b = texture(trail_tex_array, vec3(uv, float(z1))).r;
    return mix(a, b, t);
}

float binary_mask_from_color(vec3 rgb) {
    float g = dot(rgb, vec3(0.299, 0.587, 0.114));
    float m = (g >= u_threshold_value) ? 1.0 : 0.0;
    if (u_threshold_invert != 0) m = 1.0 - m;
    return m;
}

vec3 bezier_cubic(vec3 p0, vec3 p1, vec3 p2, vec3 p3, float t) {
    float u = 1.0 - t;
    return
        (u*u*u) * p0 +
        3.0 * (u*u) * t * p1 +
        3.0 * u * (t*t) * p2 +
        (t*t*t) * p3;
}

vec3 bezier_tangent(vec3 p0, vec3 p1, vec3 p2, vec3 p3, float t) {
    float u = 1.0 - t;
    return
        3.0 * (u*u) * (p1 - p0) +
        6.0 * u * t * (p2 - p1) +
        3.0 * (t*t) * (p3 - p2);
}

vec3 safe_side(vec3 tangent) {
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 side = cross(up, tangent);
    if (length(side) < 1e-5) {
        side = cross(vec3(1.0, 0.0, 0.0), tangent);
    }
    return normalize(side);
}

vec3 eval_polyline_center(float t) {
    if (u_poly_count <= 1) return u_center;

    float lens[16];
    float total = 0.0;
    for (int i = 0; i < 15; ++i) {
        if (i < u_poly_count - 1) {
            lens[i] = length(u_poly_points[i + 1] - u_poly_points[i]);
            total += lens[i];
        } else {
            lens[i] = 0.0;
        }
    }

    if (total < 1e-6) return u_poly_points[0];

    float target = clamp(t, 0.0, 1.0) * total;
    float accum = 0.0;

    for (int i = 0; i < 15; ++i) {
        if (i >= u_poly_count - 1) break;
        float seg = lens[i];
        if (target <= accum + seg || i == u_poly_count - 2) {
            float lt = (target - accum) / max(seg, 1e-6);
            return mix(u_poly_points[i], u_poly_points[i + 1], clamp(lt, 0.0, 1.0));
        }
        accum += seg;
    }

    return u_poly_points[u_poly_count - 1];
}

vec3 eval_polyline_tangent(float t) {
    if (u_poly_count <= 1) return normalize(u_axis_u);

    float eps = 0.01;
    vec3 a = eval_polyline_center(clamp(t - eps, 0.0, 1.0));
    vec3 b = eval_polyline_center(clamp(t + eps, 0.0, 1.0));
    return normalize(b - a);
}

vec3 eval_plane_world(vec2 uv_in) {
    vec2 s = (uv_in * 2.0 - 1.0);
    float aspect = u_slice_px.x / max(u_slice_px.y, 1.0);
    s.x *= aspect;

    if (u_plane_mode == 1) {
        float t = clamp(uv_in.x, 0.0, 1.0);
        float lateral = (uv_in.y * 2.0 - 1.0) * u_plane_width * 0.5;
        vec3 c = bezier_cubic(u_bezier_p0, u_bezier_p1, u_bezier_p2, u_bezier_p3, t);
        vec3 tan = normalize(bezier_tangent(u_bezier_p0, u_bezier_p1, u_bezier_p2, u_bezier_p3, t));
        vec3 side = safe_side(tan);
        return c + side * lateral;
    }

    if (u_plane_mode == 3) {
        float t = clamp(uv_in.x, 0.0, 1.0);
        float lateral = (uv_in.y * 2.0 - 1.0) * u_plane_width * 0.5;
        vec3 c = eval_polyline_center(t);
        vec3 tan = eval_polyline_tangent(t);
        vec3 side = safe_side(tan);
        return c + side * lateral;
    }

    vec3 p = u_center + (u_axis_u * (s.x * u_scale)) + (u_axis_v * (s.y * u_scale));

    if (u_plane_mode == 2) {
        float h = noise2(p.xz * u_noise_freq + vec2(u_time * u_noise_speed, u_time * 0.27 * u_noise_speed));
        h = (h - 0.5) * 2.0;
        p += u_axis_n * (h * u_noise_amp);
    }

    return p;
}

vec4 eval_slice(vec2 uv_in) {
    vec3 p0_world = eval_plane_world(uv_in);
    vec3 p_world = p0_world;

    vec3 p0 = world_to_norm(p0_world);

    if (any(lessThan(p0, vec3(0.0))) || any(greaterThan(p0, vec3(1.0)))) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    if (u_heap_enable != 0) {
        float d = distance(uv_in, u_mouse);

        float r = max(u_radius, 1e-6);
        float t = clamp(d / r, 0.0, 1.0);
        float shaped_t = pow(t, max(u_layer_stretch, 1e-6));
        float depth_world = (1.0 - shaped_t) * u_heap_depth * u_box_diag;

        p_world = p0_world + (u_axis_n * (depth_world * u_heap_dir));

        vec3 p_test = world_to_norm(p_world);
        if (any(lessThan(p_test, vec3(0.0))) || any(greaterThan(p_test, vec3(1.0)))) {
            p_world = p0_world;
        }

        vec3 pn_out = world_to_norm(p0_world);
        vec3 pn_in  = world_to_norm(p_world);

        vec4 outside = sample_volume_norm(pn_out);
        vec4 inside  = sample_volume_norm(pn_in);

        float rim0 = 1.0 - (u_softness / r);
        rim0 = clamp(rim0, 0.0, 1.0);
        float edge_fade = 1.0 - smoothstep(rim0, 1.0, t);

        vec4 c = mix(outside, inside, edge_fade);

        if (u_phys3d_enable != 0) {
            float trail = sample_trail_norm(pn_in) * u_phys3d_gain;
            c.rgb = mix(c.rgb, vec3(0.1, 1.0, 0.35), clamp(trail, 0.0, 1.0));
        }
        return c;
    }

    vec4 c = sample_volume_norm(p0);
    if (u_phys3d_enable != 0) {
        float trail = sample_trail_norm(p0) * u_phys3d_gain;
        c.rgb = mix(c.rgb, vec3(0.1, 1.0, 0.35), clamp(trail, 0.0, 1.0));
    }
    return c;
}

void main() {
    vec4 c = eval_slice(v_uv);

    if (u_threshold_enable != 0) {
        float m = binary_mask_from_color(c.rgb);

        vec2 px = 1.0 / max(u_slice_px, vec2(1.0));
        vec4 cx = eval_slice(v_uv + vec2(px.x, 0.0));
        vec4 cy = eval_slice(v_uv + vec2(0.0, px.y));

        float mx = binary_mask_from_color(cx.rgb);
        float my = binary_mask_from_color(cy.rgb);

        float edge = max(abs(m - mx), abs(m - my));

        vec3 fillColor = mix(vec3(0.0), vec3(1.0), m);
        vec3 edgeColor = vec3(1.0, 1.0, 0.0);
        vec3 outColor = mix(fillColor, edgeColor, edge);

        fragColor = vec4(outColor, 1.0);
        return;
    }

    fragColor = vec4(c.rgb, 1.0);
}
"""


# =========================
# 3) ADD THESE METHODS INSIDE MPRPlaneUI
# =========================
def _load_plane_preset(self):
    self.plane_mode = 0   # 0 regular, 1 bezier, 2 noise, 3 polyline
    self.plane_width = 0.06

    self.bezier_p0 = np.array([-0.8, 0.0, -0.2], dtype=np.float32)
    self.bezier_p1 = np.array([-0.2, 0.0,  0.1], dtype=np.float32)
    self.bezier_p2 = np.array([ 0.2, 0.0, -0.1], dtype=np.float32)
    self.bezier_p3 = np.array([ 0.8, 0.0,  0.2], dtype=np.float32)

    self.poly_points = np.zeros((MAX_POLY_POINTS, 3), dtype=np.float32)
    self.poly_count = 0

    self.noise_amp = 0.12
    self.noise_freq = 2.5
    self.noise_speed = 0.8
    self.noise_plane_w = 1.2
    self.noise_plane_h = 0.8
    self.noise_grid_u = 60
    self.noise_grid_v = 30

    if not PRESET_PATH.exists():
        print("[plane preset] no plane_preset.json found, using regular plane")
        return

    try:
        data = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[plane preset] failed to parse json: {e}")
        return

    mode = data.get("mode", "regular")
    self.plane_width = float(data.get("width", 0.06))

    if mode == "bezier":
        self.plane_mode = 1
        pts = data.get("points", [])
        if len(pts) >= 4:
            self.bezier_p0 = np.array(pts[0], dtype=np.float32)
            self.bezier_p1 = np.array(pts[1], dtype=np.float32)
            self.bezier_p2 = np.array(pts[2], dtype=np.float32)
            self.bezier_p3 = np.array(pts[3], dtype=np.float32)

    elif mode == "noise":
        self.plane_mode = 2
        self.noise_amp = float(data.get("noise_amp", 0.12))
        self.noise_freq = float(data.get("noise_freq", 2.5))
        self.noise_speed = float(data.get("noise_speed", 0.8))
        self.noise_plane_w = float(data.get("plane_w", 1.2))
        self.noise_plane_h = float(data.get("plane_h", 0.8))
        self.noise_grid_u = int(data.get("grid_u", 60))
        self.noise_grid_v = int(data.get("grid_v", 30))
        self.scale = 0.5 * max(self.noise_plane_w, self.noise_plane_h)

    elif mode == "polyline":
        self.plane_mode = 3
        pts = data.get("points", [])
        self.poly_count = min(len(pts), MAX_POLY_POINTS)
        for i in range(self.poly_count):
            self.poly_points[i] = np.array(pts[i], dtype=np.float32)

    else:
        self.plane_mode = 0

    print(f"[plane preset] loaded mode={mode}")


def _noise2_cpu(self, x, z, t):
    return np.sin(x * self.noise_freq + t * self.noise_speed) * np.cos(z * self.noise_freq + t * 0.27 * self.noise_speed)


def _bezier_cubic_cpu(self, p0, p1, p2, p3, t):
    u = 1.0 - t
    return ((u**3) * p0 +
            3.0 * (u**2) * t * p1 +
            3.0 * u * (t**2) * p2 +
            (t**3) * p3)


def _bezier_tangent_cpu(self, p0, p1, p2, p3, t):
    u = 1.0 - t
    return (3.0 * (u**2) * (p1 - p0) +
            6.0 * u * t * (p2 - p1) +
            3.0 * (t**2) * (p3 - p2))


def _safe_side_cpu(self, tangent):
    tangent = normalize(np.array(tangent, dtype=np.float32))
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    side = np.cross(up, tangent)
    if np.linalg.norm(side) < 1e-6:
        side = np.cross(np.array([1.0, 0.0, 0.0], dtype=np.float32), tangent)
    return normalize(side)


def _eval_polyline_center_cpu(self, t):
    if self.poly_count <= 1:
        return self.center.copy()

    pts = self.poly_points[:self.poly_count]
    lens = []
    total = 0.0
    for i in range(self.poly_count - 1):
        seg = float(np.linalg.norm(pts[i + 1] - pts[i]))
        lens.append(seg)
        total += seg

    if total < 1e-6:
        return pts[0].copy()

    target = np.clip(t, 0.0, 1.0) * total
    accum = 0.0
    for i in range(self.poly_count - 1):
        seg = lens[i]
        if target <= accum + seg or i == self.poly_count - 2:
            lt = (target - accum) / max(seg, 1e-6)
            return (1.0 - lt) * pts[i] + lt * pts[i + 1]
        accum += seg

    return pts[-1].copy()


def _eval_polyline_tangent_cpu(self, t):
    if self.poly_count <= 1:
        return normalize(self.u.copy())
    eps = 0.01
    a = self._eval_polyline_center_cpu(max(0.0, t - eps))
    b = self._eval_polyline_center_cpu(min(1.0, t + eps))
    return normalize(b - a)

# ============================================================
# Mesh shader
# ============================================================
MESH_VERT = r"""
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
out vec3 v_pos_world;
void main() {
    v_pos_world = in_pos;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

MESH_FRAG = r"""
#version 330
in vec3 v_pos_world;
out vec4 fragColor;

uniform vec4 u_color;
uniform vec3 u_plane_center;
uniform vec3 u_plane_normal;
uniform float u_plane_band;

void main() {
    float d = dot(v_pos_world - u_plane_center, normalize(u_plane_normal));
    float band = 1.0 - smoothstep(0.0, u_plane_band, abs(d));

    vec3 base = u_color.rgb;
    vec3 hitColor = base * 0.25 + vec3(0.20, 0.03, 0.03);
    vec3 finalColor = mix(base, hitColor, band);

    fragColor = vec4(finalColor, u_color.a);
}
"""


# ============================================================
# Simple color shader
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
void main() {
    fragColor = u_color;
}
"""


# ============================================================
# Helpers
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


def contours_to_svg(contours, width, height, out_path: Path):
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<g fill="none" stroke="black" stroke-width="1">'
    ]
    for cnt in contours:
        if len(cnt) < 2:
            continue
        pts = cnt[:, 0, :]
        d = [f"M {pts[0,0]} {pts[0,1]}"]
        for p in pts[1:]:
            d.append(f"L {p[0]} {p[1]}")
        d.append("Z")
        lines.append(f'<path d="{" ".join(d)}" />')
    lines.append("</g></svg>")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def blur3d_separable(vol: np.ndarray) -> np.ndarray:
    out_xy = np.empty_like(vol)
    for z in range(vol.shape[0]):
        out_xy[z] = cv2.GaussianBlur(vol[z], (0, 0), sigmaX=1.0)
    out_z = (np.roll(out_xy, 1, axis=0) + out_xy + np.roll(out_xy, -1, axis=0)) / 3.0
    out_z[0] = (out_xy[0] + out_xy[1]) * 0.5 if out_xy.shape[0] > 1 else out_xy[0]
    out_z[-1] = (out_xy[-2] + out_xy[-1]) * 0.5 if out_xy.shape[0] > 1 else out_xy[-1]
    return out_z



class MPRPlaneUI(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "MPR Plane UI — 3D Physarum trail volume + live threshold + transparent mesh"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if not VOL_PATH.exists():
            raise FileNotFoundError(f"Missing {VOL_PATH}")

        V = np.load(VOL_PATH, mmap_mode="r")
        if V.dtype != np.uint8 or V.ndim != 4 or V.shape[3] != 3:
            raise ValueError(f"Unexpected volume: shape={V.shape}, dtype={V.dtype}")

        self.Z, self.H, self.W, _ = V.shape
        self.V = V
        print(f"[volume] Z={self.Z} H={self.H} W={self.W} dtype=uint8")

        self.slice_prog = self.ctx.program(vertex_shader=SLICE_VERT, fragment_shader=SLICE_FRAG)
        self.mesh_prog = self.ctx.program(vertex_shader=MESH_VERT, fragment_shader=MESH_FRAG)
        self.gizmo_prog = self.ctx.program(vertex_shader=GIZMO_VERT, fragment_shader=GIZMO_FRAG)

        fsq = np.array([
            -1, -1,
             1, -1,
            -1,  1,
            -1,  1,
             1, -1,
             1,  1
        ], dtype="f4")
        self.slice_vbo = self.ctx.buffer(fsq.tobytes())
        self.slice_vao = self.ctx.simple_vertex_array(self.slice_prog, self.slice_vbo, "in_pos")

        # anatomical volume texture
        self.tex = self.ctx.texture_array((self.W, self.H, self.Z), components=4, dtype="f1")
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        alpha = np.full((self.H, self.W, 1), 255, dtype=np.uint8)
        for z in range(self.Z):
            slab = np.ascontiguousarray(self.V[z])
            slab_rgba = np.concatenate([slab, alpha], axis=2)
            self.tex.write(slab_rgba.tobytes(), viewport=(0, 0, z, self.W, self.H, 1))
            if z % 100 == 0 or z == self.Z - 1:
                print(f"  uploaded anatomical {z+1}/{self.Z}")

        self.tex.use(location=0)
        self.slice_prog["tex_array"].value = 0
        self.slice_prog["u_num_layers"].value = int(self.Z)

        # world box
        self.box_min = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        self.box_max = np.array([ 0.5,  0.5,  0.5], dtype=np.float32)
        self.box_size = self.box_max - self.box_min
        self.box_center = 0.5 * (self.box_min + self.box_max)
        self.box_diag = float(np.linalg.norm(self.box_size))

        # plane state
        self.yaw = 0.0
        self.pitch = 0.0
        self.center = self.box_center.copy()
        self.scale = 0.18 * self.box_diag

        # heap brush
        self.heap_enable = True
        self.mouse_uv = (0.5, 0.5)
        self.heap_radius = 0.18
        self.heap_softness = 0.06
        self.heap_depth = 0.08
        self.heap_stretch = 1.0
        self.heap_dir = -1.0

        self.flip_y = 1
        self.bgr_input = 1

        # threshold / live MC
        self.threshold_enabled = False
        self.threshold_value = 128
        self.threshold_invert = False
        self._mesh_dirty = False

        # held keys
        self._drag_plane = False
        self._drag_pan = False
        self._held_keys = set()
        self._held_threshold_keys = set()
        self._threshold_repeat_rate = 120.0
        self._threshold_accum = 0.0

        # gizmo camera
        self.gizmo_yaw = 0.8
        self.gizmo_pitch = 0.5
        self.gizmo_radius = 2.4
        self.mesh_alpha = 0.28

        # extracted mesh
        self.mesh_vao = None
        self.mesh_vbo = None

        # -------- 3D Physarum state --------
        self.phys3d_enabled = False
        self.phys3d_gain = 1.25

        self.phys3d_ds = PHYS3D_DOWNSAMPLE
        self.pZ = max(2, (self.Z + self.phys3d_ds - 1) // self.phys3d_ds)
        self.pH = max(2, (self.H + self.phys3d_ds - 1) // self.phys3d_ds)
        self.pW = max(2, (self.W + self.phys3d_ds - 1) // self.phys3d_ds)

        self.phys3d_trail = np.zeros((self.pZ, self.pH, self.pW), dtype=np.float32)
        self.phys3d_mask = np.zeros((self.pZ, self.pH, self.pW), dtype=np.uint8)
        self.phys3d_agents = None

        self.phys3d_step_size = 0.9
        self.phys3d_sensor_dist = 3.0
        self.phys3d_turn_angle = 0.35
        self.phys3d_deposit = 1.0
        self.phys3d_decay = 0.985
        self.phys3d_diffuse = 0.18

        self._phys3d_tex_dirty = True
        self._phys3d_upload_stride = 2
        self._frame_counter = 0

        # trail texture array
        self.trail_tex = self.ctx.texture_array((self.pW, self.pH, self.pZ), components=1, dtype="f4")
        self.trail_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.trail_tex.use(location=1)
        self.slice_prog["trail_tex_array"].value = 1
        self.slice_prog["u_trail_num_layers"].value = int(self.pZ)

        self._rebuild_phys3d_mask()
        self._seed_phys3d_agents(PHYS3D_AGENT_COUNT)

        # build GL objects
        self._build_plane_buffers()
        self._build_box()
        self._update_plane_axes()
        self._load_obj_mesh()
        self._update_gizmo_geometry()
        self._push_slice_uniforms()
        self._upload_phys3d_trail_texture(force=True)

        print("Ready.")
        print("LMB drag: rotate plane")
        print("MMB drag: pan plane")
        print("Wheel: zoom plane")
        print("WASDQE: move plane")
        print("H: toggle heap brush")
        print("I/K: heap depth +/-")
        print("J/L: heap radius +/-")
        print("[ ] hold: threshold -/+ continuously")
        print("T: toggle threshold overlay")
        print("V: invert threshold")
        print("P: export current thresholded slice to SVG")
        print("B: toggle 3D Physarum")
        print("M: reseed 3D Physarum from current threshold")
        print("R: reset")
        print("ESC: quit")

    # ------------------------------------------------------------
    # Marching cubes mesh from anatomy threshold
    # ------------------------------------------------------------
    def _load_obj_mesh(self):
        self.mesh_vao = None
        self.mesh_vbo = None

        iso_threshold = int(self.threshold_value)
        downsample = int(max(1, MC_DOWNSAMPLE))

        vol_bgr = np.asarray(self.V[::downsample, ::downsample, ::downsample], dtype=np.float32)
        Zs, Hs, Ws, _ = vol_bgr.shape

        gray = (
            0.114 * vol_bgr[..., 0] +
            0.587 * vol_bgr[..., 1] +
            0.299 * vol_bgr[..., 2]
        ).astype(np.float32)

        occ = gray >= float(iso_threshold)
        if self.threshold_invert:
            occ = ~occ

        filled = int(np.count_nonzero(occ))
        total = int(occ.size)
        print(f"[mc] downsample={downsample}, threshold={iso_threshold}, filled={filled}/{total}")

        if filled == 0 or filled == total:
            print("[mc] skipping mesh extraction")
            return

        field = occ.astype(np.float32)

        try:
            verts_zyx, faces, normals, values = measure.marching_cubes(
                field,
                level=0.5,
                spacing=(1.0, 1.0, 1.0),
            )
        except Exception as e:
            print(f"[mc] marching cubes failed: {e}")
            return

        if len(verts_zyx) == 0 or len(faces) == 0:
            print("[mc] empty mesh after marching cubes")
            return

        verts_xyz = np.empty_like(verts_zyx, dtype=np.float32)
        verts_xyz[:, 0] = verts_zyx[:, 2] / max(Ws - 1, 1)
        verts_xyz[:, 1] = verts_zyx[:, 1] / max(Hs - 1, 1)
        verts_xyz[:, 2] = verts_zyx[:, 0] / max(Zs - 1, 1)

        verts_world = self.box_min[None, :] + verts_xyz * self.box_size[None, :]
        tri_verts = verts_world[faces.reshape(-1)].astype("f4")

        self.mesh_vbo = self.ctx.buffer(tri_verts.tobytes())
        self.mesh_vao = self.ctx.simple_vertex_array(self.mesh_prog, self.mesh_vbo, "in_pos")
        print(f"[mc] verts={len(verts_xyz)} faces={len(faces)} uploaded")

    # ------------------------------------------------------------
    # 3D Physarum
    # ------------------------------------------------------------
    def _rebuild_phys3d_mask(self):
        ds = self.phys3d_ds
        vol_bgr = np.asarray(self.V[::ds, ::ds, ::ds], dtype=np.float32)
        gray = (
            0.114 * vol_bgr[..., 0] +
            0.587 * vol_bgr[..., 1] +
            0.299 * vol_bgr[..., 2]
        ).astype(np.float32)
        mask = gray >= float(self.threshold_value)
        if self.threshold_invert:
            mask = ~mask
        self.phys3d_mask = mask.astype(np.uint8)

    def _seed_phys3d_agents(self, count=PHYS3D_AGENT_COUNT):
        idx = np.argwhere(self.phys3d_mask > 0)
        if len(idx) == 0:
            idx = np.argwhere(np.ones_like(self.phys3d_mask, dtype=np.uint8))

        choice = np.random.choice(len(idx), size=min(count, len(idx)), replace=True)
        pts = idx[choice].astype(np.float32)  # z,y,x

        dirs = np.random.normal(size=(len(pts), 3)).astype(np.float32)
        norms = np.linalg.norm(dirs, axis=1, keepdims=True)
        dirs /= np.maximum(norms, 1e-6)

        # agents stored as x,y,z,dx,dy,dz in trail-grid coordinates
        self.phys3d_agents = np.zeros((len(pts), 6), dtype=np.float32)
        self.phys3d_agents[:, 0] = pts[:, 2]
        self.phys3d_agents[:, 1] = pts[:, 1]
        self.phys3d_agents[:, 2] = pts[:, 0]
        self.phys3d_agents[:, 3:6] = dirs

        self.phys3d_trail.fill(0.0)
        self._phys3d_tex_dirty = True

    def _sample_phys3d_trail_point(self, x, y, z):
        xi = int(np.clip(x, 0, self.pW - 1))
        yi = int(np.clip(y, 0, self.pH - 1))
        zi = int(np.clip(z, 0, self.pZ - 1))
        return self.phys3d_trail[zi, yi, xi]

    def _mask_at(self, x, y, z):
        xi = int(np.clip(x, 0, self.pW - 1))
        yi = int(np.clip(y, 0, self.pH - 1))
        zi = int(np.clip(z, 0, self.pZ - 1))
        return self.phys3d_mask[zi, yi, xi] > 0

    def _basis_from_dir(self, d):
        d = normalize(d.astype(np.float32))
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if abs(float(np.dot(d, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        u = normalize(np.cross(ref, d))
        v = normalize(np.cross(d, u))
        return u, v

    def _step_phys3d(self):
        if not self.phys3d_enabled or self.phys3d_agents is None or len(self.phys3d_agents) == 0:
            return

        A = self.phys3d_agents

        for i in range(len(A)):
            x, y, z, dx, dy, dz = A[i]
            d = normalize(np.array([dx, dy, dz], dtype=np.float32))
            u, v = self._basis_from_dir(d)

            def sensor(offset_vec):
                p = np.array([x, y, z], dtype=np.float32) + offset_vec * self.phys3d_sensor_dist
                return self._sample_phys3d_trail_point(p[0], p[1], p[2])

            forward = sensor(d)
            left = sensor(normalize(d + self.phys3d_turn_angle * u))
            right = sensor(normalize(d - self.phys3d_turn_angle * u))
            up = sensor(normalize(d + self.phys3d_turn_angle * v))
            down = sensor(normalize(d - self.phys3d_turn_angle * v))

            vals = np.array([forward, left, right, up, down], dtype=np.float32)
            dirs = [
                d,
                normalize(d + self.phys3d_turn_angle * u),
                normalize(d - self.phys3d_turn_angle * u),
                normalize(d + self.phys3d_turn_angle * v),
                normalize(d - self.phys3d_turn_angle * v),
            ]

            best_idx = int(np.argmax(vals))
            nd = dirs[best_idx]

            if best_idx == 0 and np.random.rand() < 0.08:
                nd = normalize(d + 0.25 * np.random.normal(size=3).astype(np.float32))

            np_pos = np.array([x, y, z], dtype=np.float32) + nd * self.phys3d_step_size

            if (
                np_pos[0] < 0 or np_pos[0] >= self.pW or
                np_pos[1] < 0 or np_pos[1] >= self.pH or
                np_pos[2] < 0 or np_pos[2] >= self.pZ or
                not self._mask_at(np_pos[0], np_pos[1], np_pos[2])
            ):
                # respawn inside valid mask
                idx = np.argwhere(self.phys3d_mask > 0)
                if len(idx) > 0:
                    rz, ry, rx = idx[np.random.randint(len(idx))]
                    np_pos = np.array([rx, ry, rz], dtype=np.float32)
                nd = normalize(np.random.normal(size=3).astype(np.float32))

            xi = int(np.clip(np_pos[0], 0, self.pW - 1))
            yi = int(np.clip(np_pos[1], 0, self.pH - 1))
            zi = int(np.clip(np_pos[2], 0, self.pZ - 1))
            self.phys3d_trail[zi, yi, xi] += self.phys3d_deposit

            A[i, 0:3] = np_pos
            A[i, 3:6] = nd

        self.phys3d_trail *= self.phys3d_decay
        blurred = blur3d_separable(self.phys3d_trail)
        self.phys3d_trail = (
            (1.0 - self.phys3d_diffuse) * self.phys3d_trail
            + self.phys3d_diffuse * blurred
        )
        self._phys3d_tex_dirty = True

    def _upload_phys3d_trail_texture(self, force=False):
        if not force and not self._phys3d_tex_dirty:
            return

        trail = self.phys3d_trail
        maxv = max(float(trail.max()), 1e-6)
        trail_norm = np.clip(trail / maxv, 0.0, 1.0).astype(np.float32)

        for z in range(self.pZ):
            slab = np.ascontiguousarray(trail_norm[z])
            self.trail_tex.write(slab.tobytes(), viewport=(0, 0, z, self.pW, self.pH, 1))

        self._phys3d_tex_dirty = False

    # ------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------
    def _build_plane_buffers(self):
        self.plane_vbo = self.ctx.buffer(reserve=36 * 3 * 4)
        self.plane_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.plane_vbo, "in_pos")

        self.n_vbo = self.ctx.buffer(reserve=2 * 3 * 4)
        self.n_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.n_vbo, "in_pos")

    def _build_box(self):
        xmin, ymin, zmin = self.box_min
        xmax, ymax, zmax = self.box_max

        corners = [
            np.array([xmin, ymin, zmin], np.float32),
            np.array([xmax, ymin, zmin], np.float32),
            np.array([xmin, ymax, zmin], np.float32),
            np.array([xmax, ymax, zmin], np.float32),
            np.array([xmin, ymin, zmax], np.float32),
            np.array([xmax, ymin, zmax], np.float32),
            np.array([xmin, ymax, zmax], np.float32),
            np.array([xmax, ymax, zmax], np.float32),
        ]

        edges = [
            (0,1),(0,2),(1,3),(2,3),
            (4,5),(4,6),(5,7),(6,7),
            (0,4),(1,5),(2,6),(3,7),
        ]

        box_lines = []
        for a, b in edges:
            box_lines.append(corners[a])
            box_lines.append(corners[b])

        box_lines = np.array(box_lines, dtype="f4")
        self.box_vbo = self.ctx.buffer(box_lines.tobytes())
        self.box_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.box_vbo, "in_pos")

    def _update_plane_axes(self):
        self.n = yaw_pitch_to_normal(self.yaw, self.pitch)
        self.u, self.v = orthonormal_basis_from_normal(self.n)

    def _push_slice_uniforms(self):
        self.slice_prog["u_center"].value = tuple(map(float, self.center))
        self.slice_prog["u_axis_u"].value = tuple(map(float, self.u))
        self.slice_prog["u_axis_v"].value = tuple(map(float, self.v))
        self.slice_prog["u_axis_n"].value = tuple(map(float, self.n))
        self.slice_prog["u_scale"].value = float(self.scale)

        self.slice_prog["u_box_min"].value = tuple(map(float, self.box_min))
        self.slice_prog["u_box_size"].value = tuple(map(float, self.box_size))
        self.slice_prog["u_box_diag"].value = float(self.box_diag)
        self.slice_prog["u_slice_px"].value = (float(self.wnd.width), float(self.wnd.height))

        self.slice_prog["u_heap_enable"].value = int(self.heap_enable)
        self.slice_prog["u_mouse"].value = tuple(map(float, self.mouse_uv))
        self.slice_prog["u_radius"].value = float(self.heap_radius)
        self.slice_prog["u_softness"].value = float(self.heap_softness)
        self.slice_prog["u_layer_stretch"].value = float(self.heap_stretch)
        self.slice_prog["u_heap_depth"].value = float(self.heap_depth)
        self.slice_prog["u_heap_dir"].value = float(self.heap_dir)

        self.slice_prog["u_flip_y"].value = int(self.flip_y)
        self.slice_prog["u_bgr_input"].value = int(self.bgr_input)

        self.slice_prog["u_threshold_enable"].value = int(self.threshold_enabled)
        self.slice_prog["u_threshold_value"].value = float(self.threshold_value / 255.0)
        self.slice_prog["u_threshold_invert"].value = int(self.threshold_invert)

        self.slice_prog["u_phys3d_enable"].value = int(self.phys3d_enabled)
        self.slice_prog["u_phys3d_gain"].value = float(self.phys3d_gain)

    def _update_gizmo_geometry(self):
        c = self.center
        u = self.u
        v = self.v
        n = self.n
        s = self.scale

        p00 = c - u*s - v*s
        p10 = c + u*s - v*s
        p01 = c - u*s + v*s
        p11 = c + u*s + v*s

        t = 0.01 * self.box_diag
        off = n * (t * 0.5)

        a00, a10, a01, a11 = p00 + off, p10 + off, p01 + off, p11 + off
        b00, b10, b01, b11 = p00 - off, p10 - off, p01 - off, p11 - off

        tris = []
        tris += [a00, a10, a01, a01, a10, a11]
        tris += [b00, b01, b10, b01, b11, b10]
        tris += [a00, b00, a10, a10, b00, b10]
        tris += [a10, b10, a11, a11, b10, b11]
        tris += [a11, b11, a01, a01, b11, b01]
        tris += [a01, b01, a00, a00, b01, b00]

        plane = np.array(tris, dtype=np.float32)
        self.plane_vbo.orphan(size=plane.nbytes)
        self.plane_vbo.write(plane.tobytes())

        arrow_len = 0.18 * self.box_diag
        arrow = np.array([c, c + self.n * arrow_len], dtype=np.float32)
        self.n_vbo.write(arrow.tobytes())

    # ------------------------------------------------------------
    # CPU slice sampling / SVG export
    # ------------------------------------------------------------
    def world_to_volume_norm(self, p_world):
        return (p_world - self.box_min) / np.maximum(self.box_size, 1e-8)

    def world_to_phys3d_norm(self, p_world):
        return (p_world - self.box_min) / np.maximum(self.box_size, 1e-8)

    def _sample_volume_cpu(self, p):
        p = np.clip(p, 0.0, 1.0)

        zf = p[..., 2] * (self.Z - 1)
        z0 = np.floor(zf).astype(np.int32)
        z1 = np.clip(z0 + 1, 0, self.Z - 1)
        tz = (zf - z0)[..., None]

        uvx = p[..., 0]
        uvy = 1.0 - p[..., 1] if self.flip_y else p[..., 1]

        xf = uvx * (self.W - 1)
        yf = uvy * (self.H - 1)

        x0 = np.floor(xf).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, self.W - 1)
        y0 = np.floor(yf).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, self.H - 1)

        tx = (xf - x0)[..., None]
        ty = (yf - y0)[..., None]

        c000 = self.V[z0, y0, x0].astype(np.float32)
        c100 = self.V[z0, y0, x1].astype(np.float32)
        c010 = self.V[z0, y1, x0].astype(np.float32)
        c110 = self.V[z0, y1, x1].astype(np.float32)

        c001 = self.V[z1, y0, x0].astype(np.float32)
        c101 = self.V[z1, y0, x1].astype(np.float32)
        c011 = self.V[z1, y1, x0].astype(np.float32)
        c111 = self.V[z1, y1, x1].astype(np.float32)

        c00 = c000 * (1 - tx) + c100 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c11 = c011 * (1 - tx) + c111 * tx

        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        c = c0 * (1 - tz) + c1 * tz

        if self.bgr_input:
            c = c[..., ::-1]

        return np.clip(c, 0, 255).astype(np.uint8)

    def _sample_phys3d_cpu(self, p):
        p = np.clip(p, 0.0, 1.0)

        zf = p[..., 2] * (self.pZ - 1)
        z0 = np.floor(zf).astype(np.int32)
        z1 = np.clip(z0 + 1, 0, self.pZ - 1)
        tz = (zf - z0)

        uvx = p[..., 0]
        uvy = 1.0 - p[..., 1] if self.flip_y else p[..., 1]

        xf = uvx * (self.pW - 1)
        yf = uvy * (self.pH - 1)

        x0 = np.floor(xf).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, self.pW - 1)
        y0 = np.floor(yf).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, self.pH - 1)

        tx = (xf - x0)
        ty = (yf - y0)

        c000 = self.phys3d_trail[z0, y0, x0]
        c100 = self.phys3d_trail[z0, y0, x1]
        c010 = self.phys3d_trail[z0, y1, x0]
        c110 = self.phys3d_trail[z0, y1, x1]

        c001 = self.phys3d_trail[z1, y0, x0]
        c101 = self.phys3d_trail[z1, y0, x1]
        c011 = self.phys3d_trail[z1, y1, x0]
        c111 = self.phys3d_trail[z1, y1, x1]

        c00 = c000 * (1 - tx) + c100 * tx
        c01 = c001 * (1 - tx) + c101 * tx
        c10 = c010 * (1 - tx) + c110 * tx
        c11 = c011 * (1 - tx) + c111 * tx

        c0 = c00 * (1 - ty) + c10 * ty
        c1 = c01 * (1 - ty) + c11 * ty
        c = c0 * (1 - tz) + c1 * tz

        maxv = max(float(self.phys3d_trail.max()), 1e-6)
        return np.clip(c / maxv, 0.0, 1.0)

    def render_current_slice_to_image(self, out_w=1024, out_h=1024, include_phys3d=True):
        xs = np.linspace(0.0, 1.0, out_w, dtype=np.float32)
        ys = np.linspace(0.0, 1.0, out_h, dtype=np.float32)
        uu, vv = np.meshgrid(xs, ys)

        s = np.stack([uu * 2.0 - 1.0, vv * 2.0 - 1.0], axis=-1)
        aspect = out_w / max(out_h, 1)
        s[..., 0] *= aspect

        p0_world = (
            self.center
            + self.u * (s[..., 0:1] * self.scale)
            + self.v * (s[..., 1:2] * self.scale)
        )
        p_world = np.array(p0_world, copy=True)

        if self.heap_enable:
            d = np.sqrt((uu - self.mouse_uv[0]) ** 2 + (vv - self.mouse_uv[1]) ** 2)
            r = max(self.heap_radius, 1e-6)
            t = np.clip(d / r, 0.0, 1.0)
            shaped_t = np.power(t, max(self.heap_stretch, 1e-6))
            depth_world = (1.0 - shaped_t) * self.heap_depth * self.box_diag
            p_world = p0_world + self.n * (depth_world[..., None] * self.heap_dir)

        p0 = self.world_to_volume_norm(p0_world)
        p = self.world_to_volume_norm(p_world)

        outside = np.any((p < 0.0) | (p > 1.0), axis=-1)
        p[outside] = p0[outside]

        outside_base = np.any((p0 < 0.0) | (p0 > 1.0), axis=-1)
        p = np.clip(p, 0.0, 1.0)

        img = self._sample_volume_cpu(p)
        img[outside_base] = 0

        if include_phys3d and self.phys3d_enabled:
            tp = self.world_to_phys3d_norm(p_world)
            trail = self._sample_phys3d_cpu(tp) * self.phys3d_gain
            imgf = img.astype(np.float32)
            imgf = imgf * (1.0 - np.clip(trail[..., None], 0.0, 1.0)) + np.array([25.0, 255.0, 90.0], dtype=np.float32) * np.clip(trail[..., None], 0.0, 1.0)
            img = np.clip(imgf, 0, 255).astype(np.uint8)

        return img

    def export_current_slice_svg(self):
        img = self.render_current_slice_to_image(1024, 1024, include_phys3d=False)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mode = cv2.THRESH_BINARY_INV if self.threshold_invert else cv2.THRESH_BINARY
        _, mask = cv2.threshold(gray, self.threshold_value, 255, mode)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        out_svg = EXPORT_DIR / "slice_export.svg"
        contours_to_svg(contours, mask.shape[1], mask.shape[0], out_svg)
        print(f"[export] wrote {out_svg}")

    # ------------------------------------------------------------
    # Gizmo camera
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
        giz_px = int(min(W, H) * 0.32)
        pad = 12
        gx0 = W - giz_px - pad
        gy0 = H - giz_px - pad
        return gx0, gy0, giz_px

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------
    def on_mouse_position_event(self, x, y, dx, dy):
        u = x / max(1, self.wnd.width)
        v = 1.0 - (y / max(1, self.wnd.height))
        self.mouse_uv = (float(u), float(v))
        self.slice_prog["u_mouse"].value = self.mouse_uv

    def on_mouse_press_event(self, x, y, button):
        if button == self.wnd.mouse.left:
            self._drag_plane = True
        if button == self.wnd.mouse.middle:
            self._drag_pan = True

    def on_mouse_release_event(self, x, y, button):
        if button == self.wnd.mouse.left:
            self._drag_plane = False
        if button == self.wnd.mouse.middle:
            self._drag_pan = False

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self._drag_plane:
            self.yaw += dx * 0.005
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
            self.center[:] = np.minimum(np.maximum(self.center, self.box_min), self.box_max)

            self._push_slice_uniforms()
            self._update_gizmo_geometry()

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.scale *= float(0.92 ** y_offset)
        self.scale = float(np.clip(self.scale, 0.02 * self.box_diag, 1.5 * self.box_diag))
        self._push_slice_uniforms()
        self._update_gizmo_geometry()

    def on_key_event(self, key, action, modifiers):
        k = self.wnd.keys

        if action == k.ACTION_PRESS and key == k.ESCAPE:
            self.wnd.close()
            return

        threshold_hold_keys = {k.LEFT_BRACKET, k.RIGHT_BRACKET}
        if key in threshold_hold_keys:
            if action == k.ACTION_PRESS:
                self._held_threshold_keys.add(key)
            elif action == k.ACTION_RELEASE:
                self._held_threshold_keys.discard(key)
            return

        if action == k.ACTION_PRESS:
            if key == k.P:
                self.export_current_slice_svg()
                return

            if key == k.T:
                self.threshold_enabled = not self.threshold_enabled
                self._push_slice_uniforms()
                print(f"threshold_enabled={self.threshold_enabled}")
                return

            if key == k.V:
                self.threshold_invert = not self.threshold_invert
                self._push_slice_uniforms()
                self._mesh_dirty = True
                self._rebuild_phys3d_mask()
                self._seed_phys3d_agents(PHYS3D_AGENT_COUNT)
                print(f"threshold_invert={self.threshold_invert}")
                return

            if key == k.B:
                self.phys3d_enabled = not self.phys3d_enabled
                self._push_slice_uniforms()
                print(f"phys3d_enabled={self.phys3d_enabled}")
                return

            if key == k.M:
                self._rebuild_phys3d_mask()
                self._seed_phys3d_agents(PHYS3D_AGENT_COUNT)
                print("phys3d reseeded from current threshold")
                return

            if key == k.H:
                self.heap_enable = not self.heap_enable
                self._push_slice_uniforms()
                print(f"heap_enable={self.heap_enable}")
                return

            if key == k.N:
                self.heap_dir *= -1.0
                self._push_slice_uniforms()
                print(f"heap_dir={self.heap_dir:+.0f}")
                return

            if key == k.Y:
                self.flip_y = 0 if self.flip_y else 1
                self._push_slice_uniforms()
                print(f"flip_y={self.flip_y}")
                return

            if key == k.J:
                self.heap_radius = max(0.01, self.heap_radius - 0.01)
                self._push_slice_uniforms()
                return

            if key == k.L:
                self.heap_radius = min(0.9, self.heap_radius + 0.01)
                self._push_slice_uniforms()
                return

            if key == k.I:
                self.heap_depth = min(1.0, self.heap_depth + 0.02)
                self._push_slice_uniforms()
                return

            if key == k.K:
                self.heap_depth = max(0.0, self.heap_depth - 0.02)
                self._push_slice_uniforms()
                return

            if key == k.COMMA:
                self.heap_stretch = max(0.1, self.heap_stretch - 0.1)
                self._push_slice_uniforms()
                return

            if key == k.PERIOD:
                self.heap_stretch = min(10.0, self.heap_stretch + 0.1)
                self._push_slice_uniforms()
                return

            if key == k.R:
                self.yaw = 0.0
                self.pitch = 0.0
                self.center = self.box_center.copy()
                self.scale = 0.18 * self.box_diag
                self.heap_enable = True
                self.heap_radius = 0.18
                self.heap_softness = 0.06
                self.heap_depth = 0.08
                self.heap_stretch = 1.0
                self.heap_dir = -1.0
                self.flip_y = 1
                self.threshold_enabled = False
                self.threshold_value = 128
                self.threshold_invert = False
                self._mesh_dirty = True
                self._rebuild_phys3d_mask()
                self._seed_phys3d_agents(PHYS3D_AGENT_COUNT)
                self._update_plane_axes()
                self._update_gizmo_geometry()
                self._push_slice_uniforms()
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
        base = 0.22 * self.box_diag

        for key, is_shift in list(self._held_keys):
            step = base * dt * (3.0 if is_shift else 1.0)
            if key == k.W:
                self.center += self.n * step
            if key == k.S:
                self.center -= self.n * step
            if key == k.A:
                self.center -= self.u * step
            if key == k.D:
                self.center += self.u * step
            if key == k.Q:
                self.center -= self.v * step
            if key == k.E:
                self.center += self.v * step

        self.center[:] = np.minimum(np.maximum(self.center, self.box_min), self.box_max)
        self._push_slice_uniforms()
        self._update_gizmo_geometry()

    def _apply_held_threshold_keys(self, dt: float):
        if not self._held_threshold_keys:
            self._threshold_accum = 0.0
            return

        delta = 0.0
        if self.wnd.keys.LEFT_BRACKET in self._held_threshold_keys:
            delta -= self._threshold_repeat_rate * dt
        if self.wnd.keys.RIGHT_BRACKET in self._held_threshold_keys:
            delta += self._threshold_repeat_rate * dt

        self._threshold_accum += delta
        step = int(self._threshold_accum)

        if step != 0:
            self.threshold_value = int(np.clip(self.threshold_value + step, 0, 255))
            self._threshold_accum -= step
            self._push_slice_uniforms()
            self._mesh_dirty = True
            self._rebuild_phys3d_mask()

    def resize(self, width, height):
        self._push_slice_uniforms()

    # ------------------------------------------------------------
    # Render
    # ------------------------------------------------------------
    def on_render(self, time: float, frame_time: float):
        self._apply_held_keys(frame_time)
        self._apply_held_threshold_keys(frame_time)

        if self.phys3d_enabled:
            self._step_phys3d()

        if self._mesh_dirty:
            self._load_obj_mesh()
            self._seed_phys3d_agents(PHYS3D_AGENT_COUNT)
            self._mesh_dirty = False

        self._frame_counter += 1
        if self._phys3d_tex_dirty and (self._frame_counter % self._phys3d_upload_stride == 0):
            self._upload_phys3d_trail_texture()

        W, H = self.wnd.width, self.wnd.height

        # main slice
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.tex.use(location=0)
        self.trail_tex.use(location=1)
        self.slice_vao.render()

        # gizmo
        gx0, gy0, giz_px = self._gizmo_viewport()
        self.ctx.viewport = (gx0, gy0, giz_px, giz_px)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)

        P = perspective(45.0, 1.0, 0.05, 20.0)
        V = look_at(self._gizmo_eye(), [0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
        MVP = (P @ V).astype(np.float32)

        self.gizmo_prog["u_mvp"].write(MVP.tobytes())
        self.gizmo_prog["u_color"].value = (0.85, 0.90, 0.98, 1.0)
        self.box_vao.render(mode=moderngl.LINES)

        self.gizmo_prog["u_color"].value = (1.0, 0.15, 0.15, 0.88)
        self.plane_vao.render(mode=moderngl.TRIANGLES)

        if self.mesh_vao is not None:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (
                moderngl.SRC_ALPHA,
                moderngl.ONE_MINUS_SRC_ALPHA,
                moderngl.ONE,
                moderngl.ONE_MINUS_SRC_ALPHA,
            )
            self.ctx.depth_mask = False

            self.mesh_prog["u_mvp"].write(MVP.tobytes())
            self.mesh_prog["u_color"].value = (0.72, 0.78, 0.86, self.mesh_alpha)
            self.mesh_prog["u_plane_center"].value = tuple(map(float, self.center))
            self.mesh_prog["u_plane_normal"].value = tuple(map(float, self.n))
            self.mesh_prog["u_plane_band"].value = 0.02 * self.box_diag
            self.mesh_vao.render(mode=moderngl.TRIANGLES)

            self.ctx.depth_mask = True
            self.ctx.disable(moderngl.BLEND)

        self.gizmo_prog["u_color"].value = (1.0, 0.90, 0.25, 1.0)
        self.n_vao.render(mode=moderngl.LINES)

        self.ctx.disable(moderngl.DEPTH_TEST)


if __name__ == "__main__":
    mglw.run_window_config(MPRPlaneUI)