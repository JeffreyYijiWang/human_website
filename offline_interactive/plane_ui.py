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
    vec2 uv = p.xy;

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
# Gizmo shaders (3D box + plane rectangle)
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
# Tooltip shaders (2D textured quad)
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
    if abs(n[2]) < 0.9:
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
    M[0, 0] = f / aspect
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
    title = "MPR Plane UI — LMB rotate | MMB pan | wheel zoom | T help | WASDQE move | gizmo rotates"
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
        self.hud_prog   = self.ctx.program(vertex_shader=HUD_TEX_VERT, fragment_shader=HUD_TEX_FRAG)

        # ---------- fullscreen quad ----------
        fsq = np.array([-1,-1,  1,-1,  -1, 1,
                        -1, 1,  1,-1,   1, 1], dtype="f4")
        self.slice_vbo = self.ctx.buffer(fsq.tobytes())
        self.slice_vao = self.ctx.simple_vertex_array(self.slice_prog, self.slice_vbo, "in_pos")

        # ---------- upload texture array (RGBA normalized) ----------
        # This avoids the “black screen” from using integer textures with sampler2DArray.
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
        self.center = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # [0,1]
        self.scale  = 0.55

        self._update_plane_axes()
        self._push_slice_uniforms()

        # ---------- input state ----------
        self._drag_plane = False
        self._drag_gizmo = False
        self._drag_pan   = False
        self._last_mouse = None  # last mouse pos for manual dx/dy

        self._held_keys = set()  # smooth movement while holding

        # ---------- gizmo camera state ----------
        self.gizmo_yaw = 0.8
        self.gizmo_pitch = 0.5
        self.gizmo_radius = 2.4

        # ---------- gizmo geometry ----------
        # map [0,1]^3 -> [-0.5,0.5]^3
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

        self.plane_vbo = self.ctx.buffer(reserve=6 * 3 * 4)
        self.plane_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.plane_vbo, "in_pos")

        self.n_vbo = self.ctx.buffer(reserve=2 * 3 * 4)
        self.n_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.n_vbo, "in_pos")

        self._update_gizmo_geometry()

        # ---------- help tooltip overlay ----------
        self.show_help = False
        self.hud_vbo = self.ctx.buffer(reserve=6 * 4 * 4)  # (x,y,u,v)*6
        self.hud_vao = self.ctx.simple_vertex_array(self.hud_prog, self.hud_vbo, "in_pos", "in_uv")
        self.hud_tex = None
        self._hud_px = (0, 0)

        print("Ready.")
        print("  LMB drag (main): rotate plane")
        print("  MMB drag: pan plane (U/V)")
        print("  Wheel: zoom plane size")
        print("  LMB drag (gizmo): orbit gizmo camera")
        print("  WASDQE: move plane (hold to move continuously)")
        print("  T: toggle on-screen help")
        print("  R: reset")
        print("  ESC: quit")

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

    def _update_gizmo_geometry(self):
        # Plane quad in volume coords (center +/- u*scale +/- v*scale), map to gizmo space
        c = self.center
        u = self.u
        v = self.v
        s = self.scale

        p00 = (c - u*s - v*s) - 0.5
        p10 = (c + u*s - v*s) - 0.5
        p01 = (c - u*s + v*s) - 0.5
        p11 = (c + u*s + v*s) - 0.5

        plane = np.array([p00, p10, p01,  p01, p10, p11], dtype=np.float32)
        self.plane_vbo.write(plane.tobytes())

        # Normal arrow
        start = (c - 0.5)
        end   = (c + self.n * 0.35 - 0.5)
        arrow = np.array([start, end], dtype=np.float32)
        self.n_vbo.write(arrow.tobytes())

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

    # ------------------------------------------------------------
    # Help tooltip overlay helpers
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
            "LMB drag (main): rotate plane",
            "MMB drag: pan plane (U/V)",
            "Wheel: zoom plane size",
            "LMB drag (gizmo): orbit gizmo camera",
            "",
            "Hold keys (smooth):",
            "W/S: move along normal",
            "A/D: move along U",
            "Q/E: move along V",
            "",
            "R: reset   T: toggle this help",
        ]

        f_title = self._font(max(12, h//10))
        f_line  = self._font(max(10, h//16))

        y = 10
        d.text((12, y), lines[0], font=f_title, fill=(220, 235, 255, 255))
        y += f_title.size + 8

        for s in lines[1:]:
            d.text((12, y), s, font=f_line, fill=(200, 220, 255, 235))
            y += f_line.size + 6

        return np.ascontiguousarray(np.array(img, dtype=np.uint8))

    def _ensure_help_tex(self):
        W, H = max(self.wnd.width, 1), max(self.wnd.height, 1)
        pw = max(340, int(W * 0.34))
        ph = max(200, int(H * 0.30))

        if self.hud_tex is None or self._hud_px != (pw, ph):
            if self.hud_tex:
                self.hud_tex.release()
            self.hud_tex = self.ctx.texture((pw, ph), 4)
            self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._hud_px = (pw, ph)

            rgba = self._build_help_image(pw, ph)
            self.hud_tex.write(rgba.tobytes())

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
            self.hud_vbo.write(quad.tobytes())

    # ------------------------------------------------------------
    # Input: mouse (continuous while holding)
    # ------------------------------------------------------------

    def on_mouse_drag_event(self, x, y, dx, dy):

        mdx = dx
        mdy = dy

        if not (self._drag_plane or self._drag_gizmo or self._drag_pan):
            return

        # --- rotate plane in main view ---
        if self._drag_plane:
            self.yaw   += mdx * 0.005
            self.pitch += -mdy * 0.005
            self.pitch = float(np.clip(self.pitch, -1.55, 1.55))

            self._update_plane_axes()
            self._push_slice_uniforms()
            self._update_gizmo_geometry()

        # --- orbit gizmo camera ---
        if self._drag_gizmo:
            self.gizmo_yaw   += mdx * 0.01
            self.gizmo_pitch += -mdy * 0.01
            self.gizmo_pitch = float(np.clip(self.gizmo_pitch, -1.2, 1.2))

        # --- pan plane center (MMB drag) ---
        if self._drag_pan:
            W, H = max(self.wnd.width, 1), max(self.wnd.height, 1)
            aspect = W / max(H, 1)

            du = (mdx / W) * (2.0 * self.scale) * aspect
            dv = (-mdy / H) * (2.0 * self.scale)

            self.center += self.u * du + self.v * dv
            self.center[:] = np.clip(self.center, 0.0, 1.0)

            self._push_slice_uniforms()
            self._update_gizmo_geometry()


    def mouse_position_event(self, x, y, dx, dy):
        """
        Optional: keep last mouse coords updated even when not dragging.
        Doesn't rotate by itself; drag happens in mouse_drag_event.
        """
        self._mouse_px_x = x
        self._mouse_px_y = y

    def on_mouse_press_event(self, x, y, button):
        LEFT   = self.wnd.mouse.left
        MIDDLE = self.wnd.mouse.middle  # = 3 in your log

        if button == LEFT:
            self._drag_gizmo = self._in_gizmo(x, y)
            self._drag_plane = not self._drag_gizmo
            self._last_mouse = (x, y)
            return

        if button == MIDDLE:
            self._drag_pan = True
            self._last_mouse = (x, y)
            return
    def on_mouse_release_event(self, x, y, button):
        LEFT   = self.wnd.mouse.left
        MIDDLE = self.wnd.mouse.middle

        if button == LEFT:
            self._drag_plane = False
            self._drag_gizmo = False
        if button == MIDDLE:
            self._drag_pan = False

        self._last_mouse = None

   

    def on_mouse_position_event(self, x, y, dx, dy):
        if self._last_mouse is None:
            self._last_mouse = (x, y)
            return

        lx, ly = self._last_mouse
        mdx = x - lx
        mdy = y - ly
        self._last_mouse = (x, y)

        if not (self._drag_plane or self._drag_gizmo or self._drag_pan):
            return

        if self._drag_plane:
            self.yaw   += mdx * 0.005
            self.pitch += -mdy * 0.005
            self.pitch = float(np.clip(self.pitch, -1.55, 1.55))
            self._update_plane_axes()
            self._push_slice_uniforms()
            self._update_gizmo_geometry()

        if self._drag_gizmo:
            self.gizmo_yaw   += mdx * 0.01
            self.gizmo_pitch += -mdy * 0.01
            self.gizmo_pitch = float(np.clip(self.gizmo_pitch, -1.2, 1.2))

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

        base = 0.22  # normalized units per second (tune)
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

    def on_resize(self, width, height):
        self._push_slice_uniforms()
        if self.show_help:
            self._ensure_help_tex()

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

        # No scissor clear (compat). We draw an opaque background quad by just clearing the whole screen once.
        # If you want it always opaque regardless of what's under it, we could add a background quad later.

        self.ctx.enable(moderngl.DEPTH_TEST)

        P = perspective(45.0, 1.0, 0.05, 10.0)
        V = look_at(eye=self._gizmo_eye(), target=[0.0, 0.0, 0.0], up=[0.0, 0.0, 1.0])
        MVP = (P @ V).astype(np.float32)

        self.gizmo_prog["u_mvp"].write(MVP.tobytes())

        # box wireframe
        self.gizmo_prog["u_color"].value = (0.85, 0.90, 0.98, 1.0)
        self.box_vao.render(mode=moderngl.LINES)

        # plane rectangle
        self.gizmo_prog["u_color"].value = (1.0, 0.15, 0.15, 1.0)
        self.plane_vao.render(mode=moderngl.TRIANGLES)

        # normal arrow
        self.gizmo_prog["u_color"].value = (1.0, 0.90, 0.25, 1.0)
        self.n_vao.render(mode=moderngl.LINES)

        self.ctx.disable(moderngl.DEPTH_TEST)

        # ---- help tooltip overlay (top-left)
        if self.show_help:
            self._ensure_help_tex()

            self.ctx.viewport = (0, 0, W, H)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

            self.hud_tex.use(location=3)
            self.hud_prog["u_tex"].value = 3
            self.hud_vao.render(mode=moderngl.TRIANGLES)

            self.ctx.disable(moderngl.BLEND)


if __name__ == "__main__":
    mglw.run_window_config(MPRPlaneUI)