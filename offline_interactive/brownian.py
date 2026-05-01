import ctypes, math, os, time
from pathlib import Path

import numpy as np
import moderngl
import moderngl_window as mglw
from PIL import Image, ImageDraw, ImageFont

try:
    import pyspacemouse
except Exception:
    pyspacemouse = None

try:
    _user32 = ctypes.windll.user32
except Exception:
    _user32 = None

def _get_cursor_pos():
    if _user32 is None:
        return None
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]
    pt = POINT()
    if _user32.GetCursorPos(ctypes.byref(pt)):
        return int(pt.x), int(pt.y)
    return None

def _set_cursor_pos(x: int, y: int):
    if _user32 is not None:
        _user32.SetCursorPos(int(x), int(y))


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
# Auto motion + SpaceMouse helpers
# ============================================================

class AutoMotionController:
    def __init__(self):
        self.enabled = True
        self.modulate_heap = True
        self.center_vel = np.zeros(3, dtype=np.float32)
        self.yaw_vel = 0.0
        self.pitch_vel = 0.0
        self.seed = time.perf_counter()

        # Ornstein-Uhlenbeck style drift parameters
        self.center_sigma = 0.085
        self.center_damping = 1.75
        self.angular_sigma = 0.55
        self.angular_damping = 1.4
        self.center_speed_cap = 0.22
        self.angular_speed_cap = 0.75

        self.heap_depth_base = 0.22
        self.heap_radius_base = 0.18
        self.heap_softness_base = 0.06
        self.heap_depth_amp = 0.08
        self.heap_radius_amp = 0.05
        self.heap_softness_amp = 0.015
        self.heap_freq_1 = 0.21
        self.heap_freq_2 = 0.13
        self.heap_freq_3 = 0.31
        self.tilt_from_motion = True
        self.tilt_coupling = 2.2
        self.tilt_damping = 4.0
        self.roll_phase = 0.0

    def reset(self, center, yaw, pitch, heap_depth, heap_radius, heap_softness):
        self.center_vel[:] = 0.0
        self.yaw_vel = 0.0
        self.pitch_vel = 0.0
        self.heap_depth_base = float(heap_depth)
        self.heap_radius_base = float(heap_radius)
        self.heap_softness_base = float(heap_softness)

    def step(self, app, dt, now):
        if not self.enabled:
            return

        dt = float(np.clip(dt, 1.0 / 240.0, 0.05))

        self.center_vel += np.random.normal(
            loc=0.0,
            scale=self.center_sigma * math.sqrt(dt),
            size=3,
        ).astype(np.float32)
        self.center_vel -= self.center_vel * (self.center_damping * dt)
        speed = float(np.linalg.norm(self.center_vel))
        if speed > self.center_speed_cap:
            self.center_vel *= self.center_speed_cap / max(speed, 1e-8)

        self.yaw_vel += float(np.random.normal(0.0, self.angular_sigma * math.sqrt(dt)))
        self.pitch_vel += float(np.random.normal(0.0, self.angular_sigma * math.sqrt(dt)))
        self.yaw_vel -= self.yaw_vel * (self.angular_damping * dt)
        self.pitch_vel -= self.pitch_vel * (self.angular_damping * dt)
        self.yaw_vel = float(np.clip(self.yaw_vel, -self.angular_speed_cap, self.angular_speed_cap))
        self.pitch_vel = float(np.clip(self.pitch_vel, -self.angular_speed_cap, self.angular_speed_cap))

        app.center += self.center_vel * dt
        for axis in range(3):
            if app.center[axis] < 0.03:
                app.center[axis] = 0.03
                self.center_vel[axis] = abs(self.center_vel[axis]) * 0.8
            elif app.center[axis] > 0.97:
                app.center[axis] = 0.97
                self.center_vel[axis] = -abs(self.center_vel[axis]) * 0.8

        motion = self.center_vel.copy()
        lateral = float(np.dot(motion, app.u)) if hasattr(app, 'u') else float(motion[0])
        vertical = float(np.dot(motion, app.v)) if hasattr(app, 'v') else float(motion[2])

        if self.tilt_from_motion:
            target_yaw_vel = lateral * self.tilt_coupling
            target_pitch_vel = -vertical * self.tilt_coupling
            self.yaw_vel += (target_yaw_vel - self.yaw_vel) * min(1.0, self.tilt_damping * dt)
            self.pitch_vel += (target_pitch_vel - self.pitch_vel) * min(1.0, self.tilt_damping * dt)

        app.yaw += self.yaw_vel * dt
        app.pitch += self.pitch_vel * dt
        app.pitch = float(np.clip(app.pitch, -1.45, 1.45))
        app._update_plane_axes()

        if self.modulate_heap:
            phase = now - self.seed
            app.heap_depth = float(np.clip(
                self.heap_depth_base
                + self.heap_depth_amp * math.sin(phase * self.heap_freq_1 * math.tau)
                + 0.025 * math.sin(phase * 0.47 * math.tau + 0.7),
                0.02,
                0.95,
            ))
            app.heap_radius = float(np.clip(
                self.heap_radius_base
                + self.heap_radius_amp * math.sin(phase * self.heap_freq_2 * math.tau + 1.2),
                0.03,
                0.90,
            ))
            app.heap_softness = float(np.clip(
                self.heap_softness_base
                + self.heap_softness_amp * math.sin(phase * self.heap_freq_3 * math.tau + 2.1),
                0.0,
                min(app.heap_radius * 0.9, 0.45),
            ))


class SpaceMouseController:
    def __init__(self):
        self.available = pyspacemouse is not None
        self.enabled = self.available
        self._ctx = None
        self.device = None
        self.last_buttons = []
        self.cursor_control = True
        self.cursor_speed = 1400.0
        self.cursor_deadzone = 0.03
        self.rotation_enabled = True

    def connect(self):
        if not self.available or self.device is not None:
            return
        try:
            ctx = pyspacemouse.open()
            if hasattr(ctx, "__enter__"):
                self._ctx = ctx
                self.device = ctx.__enter__()
            else:
                self.device = ctx
            self.enabled = self.device is not None
            if self.enabled:
                print("[3Dconnexion] connected")
        except Exception as exc:
            self.enabled = False
            self.device = None
            print(f"[3Dconnexion] unavailable: {exc}")

    def close(self):
        if self._ctx is not None:
            try:
                self._ctx.__exit__(None, None, None)
            except Exception:
                pass
        self._ctx = None
        self.device = None

    def apply(self, app, dt):
        if not self.enabled:
            return False
        if self.device is None:
            self.connect()
            if self.device is None:
                return False

        try:
            state = self.device.read()
        except Exception as exc:
            print(f"[3Dconnexion] read failed: {exc}")
            self.close()
            self.enabled = False
            return False

        if state is None:
            return False

        move_speed = 0.34
        rot_speed = 1.65
        scale_speed = 0.60
        heap_speed = 0.45

        tx = float(getattr(state, "x", 0.0))
        ty = float(getattr(state, "y", 0.0))
        tz = float(getattr(state, "z", 0.0))
        rr = float(getattr(state, "roll", 0.0))
        rp = float(getattr(state, "pitch", 0.0))
        ry = float(getattr(state, "yaw", 0.0))
        buttons = list(getattr(state, "buttons", []) or [])

        dead = 0.05
        def dz(v, cutoff=dead):
            return 0.0 if abs(v) < cutoff else v

        tx, ty, tz, rr, rp, ry = map(dz, (tx, ty, tz, rr, rp, ry))

        changed = False

        app.center += app.u * (tx * move_speed * dt)
        app.center += app.v * (-ty * move_speed * dt)
        app.center += app.n * (-tz * move_speed * dt)
        app.center[:] = np.clip(app.center, 0.0, 1.0)
        changed = changed or any(abs(v) > 0.0 for v in (tx, ty, tz))

        if self.rotation_enabled:
            app.yaw += rr * rot_speed * dt
            app.pitch += rp * rot_speed * dt
            app.pitch = float(np.clip(app.pitch, -1.55, 1.55))
            app.scale = float(np.clip(app.scale * (1.0 - ry * scale_speed * dt), 0.05, 2.0))
            changed = changed or any(abs(v) > 0.0 for v in (rr, rp, ry))

        # Extra control: press button 0 / 1 to dig or pull heap depth
        if len(buttons) > 0 and buttons[0]:
            app.heap_depth = float(np.clip(app.heap_depth + heap_speed * dt, 0.0, 1.0))
            changed = True
        if len(buttons) > 1 and buttons[1]:
            app.heap_depth = float(np.clip(app.heap_depth - heap_speed * dt, 0.0, 1.0))
            changed = True

        # Move the in-app mouse / heap cursor from the SpaceMouse cap.
        # On Windows, also mirror this onto the OS cursor when possible.
        cx = dz(tx, self.cursor_deadzone)
        cy = dz(ty, self.cursor_deadzone)
        if self.cursor_control and (cx != 0.0 or cy != 0.0):
            du = cx * self.cursor_speed * dt / max(app.wnd.width, 1)
            dv = cy * self.cursor_speed * dt / max(app.wnd.height, 1)
            mouse_u = float(np.clip(app.mouse_uv[0] + du, 0.0, 1.0))
            mouse_v = float(np.clip(app.mouse_uv[1] + dv, 0.0, 1.0))
            app.mouse_uv = (mouse_u, mouse_v)
            app.slice_prog["u_mouse"].value = app.mouse_uv
            changed = True

            # Best-effort OS cursor sync for Windows desktops.
            cur = _get_cursor_pos()
            if cur is not None:
                x, y = cur
                dx_px = int(round(cx * self.cursor_speed * dt))
                dy_px = int(round(-cy * self.cursor_speed * dt))
                if dx_px != 0 or dy_px != 0:
                    _set_cursor_pos(x + dx_px, y + dy_px)

        return changed

# ============================================================
# Main app
# ============================================================

class MPRPlaneUI(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "MPR Plane UI — Brownian auto motion + 3Dconnexion SpaceMouse"
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

        # ---------- auto motion / SpaceMouse ----------
        self.auto_motion = AutoMotionController()
        self.spacemouse = SpaceMouseController()
        self.auto_motion.reset(
            center=self.center,
            yaw=self.yaw,
            pitch=self.pitch,
            heap_depth=0.22,
            heap_radius=0.18,
            heap_softness=0.06,
        )

        # ---------- heap brush state ----------
        self.heap_enable = True
        self.mouse_uv = (0.5, 0.5)     # bottom-left UV
        self.heap_radius = 0.18
        self.heap_softness = 0.06
        self.heap_depth = 0.22         # normalized volume units
        self.heap_stretch = 1.0
        self.heap_dir = -1.0           # -1 digs along -N; +1 digs along +N
        self.auto_motion.reset(
            center=self.center,
            yaw=self.yaw,
            pitch=self.pitch,
            heap_depth=self.heap_depth,
            heap_radius=self.heap_radius,
            heap_softness=self.heap_softness,
        )
        self.spacemouse.connect()

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
        print("  B = toggle Brownian auto motion | M = toggle heap modulation | X = toggle 3Dconnexion input")
        print("  Brownian tilt now follows motion direction; SpaceMouse also drives the heap cursor / mouse")
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

            if key == k.B:
                self.auto_motion.enabled = not self.auto_motion.enabled
                print(f"brownian_auto_motion={self.auto_motion.enabled}")
                return

            if key == k.M:
                self.auto_motion.modulate_heap = not self.auto_motion.modulate_heap
                print(f"heap_modulation={self.auto_motion.modulate_heap}")
                return

            if key == k.X:
                self.spacemouse.enabled = not self.spacemouse.enabled
                if self.spacemouse.enabled and self.spacemouse.device is None:
                    self.spacemouse.connect()
                print(f"spacemouse_enabled={self.spacemouse.enabled}")
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
                self.auto_motion.reset(
                    center=self.center,
                    yaw=self.yaw,
                    pitch=self.pitch,
                    heap_depth=self.heap_depth,
                    heap_radius=self.heap_radius,
                    heap_softness=self.heap_softness,
                )
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


    def close(self):
        self.spacemouse.close()

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

        changed = False
        self.auto_motion.step(self, frame_time, time)
        changed = True if self.auto_motion.enabled else changed
        if self.spacemouse.apply(self, frame_time):
            self._update_plane_axes()
            changed = True

        if changed:
            self._push_slice_uniforms()
            self._update_gizmo_geometry()

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

        # plane slab
        self.gizmo_prog["u_color"].value = (1.0, 0.15, 0.15, 0.80)
        self.plane_vao.render(mode=moderngl.TRIANGLES)

        # normal arrow
        self.gizmo_prog["u_color"].value = (1.0, 0.90, 0.25, 1.0)
        self.n_vao.render(mode=moderngl.LINES)

        self.ctx.disable(moderngl.DEPTH_TEST)


if __name__ == "__main__":
    mglw.run_window_config(MPRPlaneUI)