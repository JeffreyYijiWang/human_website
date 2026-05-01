import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import moderngl
import moderngl_window as mglw


PRESET_PATH = Path("plane_preset.json")


# ============================================================
# Data model
# ============================================================
@dataclass
class PlanePreset:
    mode: str = "bezier"   # bezier | noise | polyline

    # shared
    width: float = 0.12

    # bezier / polyline control points
    points: list = None

    # noise plane
    grid_u: int = 60
    grid_v: int = 30
    plane_w: float = 1.2
    plane_h: float = 0.8
    noise_amp: float = 0.12
    noise_freq: float = 2.5
    noise_speed: float = 0.8

    def __post_init__(self):
        if self.points is None:
            self.points = [
                [-0.8, 0.0, -0.2],
                [-0.2, 0.0,  0.1],
                [ 0.2, 0.0, -0.1],
                [ 0.8, 0.0,  0.2],
            ]


# ============================================================
# Shaders
# ============================================================
PREVIEW_VERT = r"""
#version 330
uniform mat4 u_mvp;
uniform float u_time;
uniform int u_mode;      // 0=static, 1=noise
uniform float u_amp;
uniform float u_freq;
uniform float u_speed;

in vec3 in_pos;

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

void main() {
    vec3 p = in_pos;

    if (u_mode == 1) {
        float h = noise2(p.xz * u_freq + vec2(u_time * u_speed, u_time * 0.27 * u_speed));
        h = (h - 0.5) * 2.0;
        p.y += h * u_amp;
    }

    gl_Position = u_mvp * vec4(p, 1.0);
}
"""

PREVIEW_FRAG = r"""
#version 330
uniform vec4 u_color;
out vec4 fragColor;
void main() {
    fragColor = u_color;
}
"""

LINE_VERT = r"""
#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

LINE_FRAG = r"""
#version 330
uniform vec4 u_color;
out vec4 fragColor;
void main() {
    fragColor = u_color;
}
"""


# ============================================================
# Math
# ============================================================
def normalize(v):
    n = float(np.linalg.norm(v))
    return v if n < 1e-8 else (v / n)

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
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) / 2.0)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / max(aspect, 1e-8)
    M[1, 1] = f
    M[2, 2] = (zfar + znear) / (znear - zfar)
    M[2, 3] = (2 * zfar * znear) / (znear - zfar)
    M[3, 2] = -1.0
    return M

def bezier_cubic(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (
        (u**3) * p0
        + 3.0 * (u**2) * t * p1
        + 3.0 * u * (t**2) * p2
        + (t**3) * p3
    )

def bezier_tangent(p0, p1, p2, p3, t):
    u = 1.0 - t
    return (
        3.0 * (u**2) * (p1 - p0)
        + 6.0 * u * t * (p2 - p1)
        + 3.0 * (t**2) * (p3 - p2)
    )

def make_ribbon_from_curve(points, width, steps=80):
    p0, p1, p2, p3 = [np.array(p, dtype=np.float32) for p in points]
    verts = []
    idx = []

    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    for i in range(steps + 1):
        t = i / steps
        c = bezier_cubic(p0, p1, p2, p3, t)
        tan = normalize(bezier_tangent(p0, p1, p2, p3, t))

        side = np.cross(up, tan)
        if np.linalg.norm(side) < 1e-6:
            side = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        side = normalize(side)

        l = c - side * width * 0.5
        r = c + side * width * 0.5

        verts.append(l)
        verts.append(r)

    for i in range(steps):
        a = i * 2
        b = a + 1
        c = a + 2
        d = a + 3
        idx.extend([a, b, c, c, b, d])

    verts = np.array(verts, dtype=np.float32)
    idx = np.array(idx, dtype=np.uint32)
    return verts, idx

def make_noise_plane(grid_u, grid_v, w, h):
    verts = []
    idx = []

    for j in range(grid_v + 1):
        v = j / grid_v
        z = (v - 0.5) * h
        for i in range(grid_u + 1):
            u = i / grid_u
            x = (u - 0.5) * w
            verts.append([x, 0.0, z])

    for j in range(grid_v):
        for i in range(grid_u):
            a = j * (grid_u + 1) + i
            b = a + 1
            c = a + (grid_u + 1)
            d = c + 1
            idx.extend([a, c, b, b, c, d])

    return (
        np.array(verts, dtype=np.float32),
        np.array(idx, dtype=np.uint32),
    )

def make_polyline(points):
    return np.array(points, dtype=np.float32)

def make_control_points(points):
    return np.array(points, dtype=np.float32)


# ============================================================
# Pre-launch editor
# ============================================================
class PlaneDesigner(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Plane Designer — 1:Bezier  2:Noise  3:Polyline  Enter:Accept"
    window_size = (1400, 900)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.preset = PlanePreset()
        self.selected_cp = 0

        self.cam_yaw = 0.6
        self.cam_pitch = 0.45
        self.cam_dist = 2.4

        self.drag_rotate = False
        self.held = set()

        self.preview_prog = self.ctx.program(vertex_shader=PREVIEW_VERT, fragment_shader=PREVIEW_FRAG)
        self.line_prog = self.ctx.program(vertex_shader=LINE_VERT, fragment_shader=LINE_FRAG)

        self.mesh_vbo = None
        self.ibo = None
        self.mesh_vao = None

        self.line_vbo = None
        self.line_vao = None

        self.cp_vbo = None
        self.cp_vao = None

        self._rebuild_geometry()

    def _eye(self):
        cy, sy = np.cos(self.cam_yaw), np.sin(self.cam_yaw)
        cp, sp = np.cos(self.cam_pitch), np.sin(self.cam_pitch)
        return np.array([
            self.cam_dist * sy * cp,
            self.cam_dist * sp,
            self.cam_dist * cy * cp,
        ], dtype=np.float32)

    def _mvp(self):
        P = perspective(45.0, self.wnd.width / max(self.wnd.height, 1), 0.01, 20.0)
        V = look_at(self._eye(), [0, 0, 0], [0, 1, 0])
        return (P @ V).astype(np.float32)

    def _rebuild_geometry(self):
        self.mesh_vao = None
        self.line_vao = None
        self.cp_vao = None

        if self.preset.mode == "bezier":
            verts, idx = make_ribbon_from_curve(self.preset.points, self.preset.width, steps=80)
            self.mesh_vbo = self.ctx.buffer(verts.tobytes())
            self.ibo = self.ctx.buffer(idx.tobytes())
            self.mesh_vao = self.ctx.vertex_array(
                self.preview_prog,
                [(self.mesh_vbo, "3f", "in_pos")],
                index_buffer=self.ibo,
            )

            line_pts = make_polyline(self.preset.points)
            self.line_vbo = self.ctx.buffer(line_pts.tobytes())
            self.line_vao = self.ctx.simple_vertex_array(self.line_prog, self.line_vbo, "in_pos")

            cp_pts = make_control_points(self.preset.points)
            self.cp_vbo = self.ctx.buffer(cp_pts.tobytes())
            self.cp_vao = self.ctx.simple_vertex_array(self.line_prog, self.cp_vbo, "in_pos")

        elif self.preset.mode == "noise":
            verts, idx = make_noise_plane(
                self.preset.grid_u,
                self.preset.grid_v,
                self.preset.plane_w,
                self.preset.plane_h,
            )
            self.mesh_vbo = self.ctx.buffer(verts.tobytes())
            self.ibo = self.ctx.buffer(idx.tobytes())
            self.mesh_vao = self.ctx.vertex_array(
                self.preview_prog,
                [(self.mesh_vbo, "3f", "in_pos")],
                index_buffer=self.ibo,
            )

        elif self.preset.mode == "polyline":
            pts = make_polyline(self.preset.points)
            self.line_vbo = self.ctx.buffer(pts.tobytes())
            self.line_vao = self.ctx.simple_vertex_array(self.line_prog, self.line_vbo, "in_pos")

            cp_pts = make_control_points(self.preset.points)
            self.cp_vbo = self.ctx.buffer(cp_pts.tobytes())
            self.cp_vao = self.ctx.simple_vertex_array(self.line_prog, self.cp_vbo, "in_pos")

    def _move_selected_point(self, delta):
        if self.preset.mode not in ("bezier", "polyline"):
            return
        p = np.array(self.preset.points[self.selected_cp], dtype=np.float32)
        p += np.array(delta, dtype=np.float32)
        self.preset.points[self.selected_cp] = p.tolist()
        self._rebuild_geometry()

    def _reset_mode(self):
        if self.preset.mode == "bezier":
            self.preset.points = [
                [-0.8, 0.0, -0.2],
                [-0.2, 0.0,  0.1],
                [ 0.2, 0.0, -0.1],
                [ 0.8, 0.0,  0.2],
            ]
            self.preset.width = 0.12
        elif self.preset.mode == "noise":
            self.preset.grid_u = 60
            self.preset.grid_v = 30
            self.preset.plane_w = 1.2
            self.preset.plane_h = 0.8
            self.preset.noise_amp = 0.12
            self.preset.noise_freq = 2.5
            self.preset.noise_speed = 0.8
        elif self.preset.mode == "polyline":
            self.preset.points = [
                [-0.9, 0.0, -0.25],
                [-0.3, 0.08, 0.1],
                [0.15, -0.05, -0.05],
                [0.55, 0.10, 0.18],
                [0.95, 0.0, -0.15],
            ]
            self.preset.width = 0.06
        self._rebuild_geometry()

    def _save_and_exit(self):
        PRESET_PATH.write_text(json.dumps(asdict(self.preset), indent=2), encoding="utf-8")
        print(f"saved preset to {PRESET_PATH}")
        self.wnd.close()

    def on_mouse_press_event(self, x, y, button):
        if button == self.wnd.mouse.left:
            self.drag_rotate = True

    def on_mouse_release_event(self, x, y, button):
        if button == self.wnd.mouse.left:
            self.drag_rotate = False

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self.drag_rotate:
            self.cam_yaw += dx * 0.005
            self.cam_pitch += -dy * 0.005
            self.cam_pitch = float(np.clip(self.cam_pitch, -1.4, 1.4))

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.cam_dist *= 0.92 ** y_offset
        self.cam_dist = float(np.clip(self.cam_dist, 0.5, 8.0))

    def on_key_event(self, key, action, modifiers):
        k = self.wnd.keys

        if action == k.ACTION_PRESS:
            if key == k.ESCAPE:
                self.wnd.close()
                return
            if key == k.ENTER:
                self._save_and_exit()
                return
            if key == k.NUMBER_1:
                self.preset.mode = "bezier"
                self._reset_mode()
                return
            if key == k.NUMBER_2:
                self.preset.mode = "noise"
                self._reset_mode()
                return
            if key == k.NUMBER_3:
                self.preset.mode = "polyline"
                self._reset_mode()
                return
            if key == k.TAB and self.preset.mode in ("bezier", "polyline"):
                self.selected_cp = (self.selected_cp + 1) % len(self.preset.points)
                return
            if key == k.R:
                self._reset_mode()
                return

        held_keys = {
            k.W, k.A, k.S, k.D, k.Q, k.E,
            k.LEFT_BRACKET, k.RIGHT_BRACKET,
            k.COMMA, k.PERIOD,
        }
        if key in held_keys:
            if action == k.ACTION_PRESS:
                self.held.add(key)
            elif action == k.ACTION_RELEASE:
                self.held.discard(key)

    def _step_controls(self, dt):
        step = 0.5 * dt

        if self.preset.mode in ("bezier", "polyline"):
            delta = np.zeros(3, dtype=np.float32)
            if self.wnd.keys.W in self.held: delta[2] -= step
            if self.wnd.keys.S in self.held: delta[2] += step
            if self.wnd.keys.A in self.held: delta[0] -= step
            if self.wnd.keys.D in self.held: delta[0] += step
            if self.wnd.keys.Q in self.held: delta[1] -= step
            if self.wnd.keys.E in self.held: delta[1] += step
            if np.linalg.norm(delta) > 0:
                self._move_selected_point(delta)

        if self.wnd.keys.LEFT_BRACKET in self.held:
            self.preset.width = max(0.01, self.preset.width - 0.25 * dt)
            if self.preset.mode in ("bezier", "polyline"):
                self._rebuild_geometry()

        if self.wnd.keys.RIGHT_BRACKET in self.held:
            self.preset.width = min(0.6, self.preset.width + 0.25 * dt)
            if self.preset.mode in ("bezier", "polyline"):
                self._rebuild_geometry()

        if self.preset.mode == "noise":
            changed = False
            if self.wnd.keys.COMMA in self.held:
                self.preset.grid_u = max(4, self.preset.grid_u - 1)
                self.preset.grid_v = max(2, self.preset.grid_v - 1)
                changed = True
            if self.wnd.keys.PERIOD in self.held:
                self.preset.grid_u = min(200, self.preset.grid_u + 1)
                self.preset.grid_v = min(200, self.preset.grid_v + 1)
                changed = True
            if changed:
                self._rebuild_geometry()

    def on_render(self, time, frame_time):
        self._step_controls(frame_time)

        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.02, 0.02, 0.025, 1.0)

        mvp = self._mvp()

        if self.mesh_vao is not None:
            self.preview_prog["u_mvp"].write(mvp.tobytes())
            self.preview_prog["u_time"].value = float(time)
            self.preview_prog["u_mode"].value = 1 if self.preset.mode == "noise" else 0
            self.preview_prog["u_amp"].value = float(self.preset.noise_amp)
            self.preview_prog["u_freq"].value = float(self.preset.noise_freq)
            self.preview_prog["u_speed"].value = float(self.preset.noise_speed)
            self.preview_prog["u_color"].value = (0.85, 0.18, 0.18, 1.0)
            self.mesh_vao.render(mode=moderngl.TRIANGLES)

        if self.line_vao is not None:
            self.line_prog["u_mvp"].write(mvp.tobytes())
            self.line_prog["u_color"].value = (0.95, 0.95, 1.0, 1.0)
            self.line_vao.render(mode=moderngl.LINE_STRIP)

        if self.cp_vao is not None:
            self.ctx.point_size = 10.0
            self.line_prog["u_mvp"].write(mvp.tobytes())
            self.line_prog["u_color"].value = (0.2, 1.0, 0.3, 1.0)
            self.cp_vao.render(mode=moderngl.POINTS)


if __name__ == "__main__":
    mglw.run_window_config(PlaneDesigner)