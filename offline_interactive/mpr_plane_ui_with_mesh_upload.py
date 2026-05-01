import ctypes
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# third-party runtime deps used by the app itself
import moderngl
import moderngl_window as mglw
import trimesh

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None


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
# Slice shaders (full-screen view) + HEAP BRUSH
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
uniform int   u_num_layers;
uniform vec2  u_slice_px;

uniform vec3  u_center;
uniform vec3  u_axis_u;
uniform vec3  u_axis_v;
uniform vec3  u_axis_n;
uniform float u_scale;
uniform int   u_surface_mode;
uniform float u_curve_amp;
uniform float u_curve_freq;

uniform int   u_heap_enable;
uniform vec2  u_mouse;
uniform float u_radius;
uniform float u_softness;
uniform float u_layer_stretch;
uniform float u_heap_depth;
uniform float u_heap_dir;

uniform int   u_flip_y;
uniform int   u_bgr_input;

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

    vec3 p_surface = p0;
    if (u_surface_mode == 1) {
        float wave = sin((s.x + s.y) * u_curve_freq) * u_curve_amp;
        p_surface = p0 + u_axis_n * wave;
    }

    if (any(lessThan(p_surface, vec3(0.0))) || any(greaterThan(p_surface, vec3(1.0)))) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 p = p_surface;

    if (u_heap_enable != 0) {
        float d = distance(v_uv, u_mouse);
        float r = max(u_radius, 1e-6);
        float t = clamp(d / r, 0.0, 1.0);
        float shaped_t = pow(t, max(u_layer_stretch, 1e-6));
        float depth = (1.0 - shaped_t) * u_heap_depth;

        p = p0 + (u_axis_n * (depth * u_heap_dir));

        if (any(lessThan(p, vec3(0.0))) || any(greaterThan(p, vec3(1.0)))) {
            p = p_surface;
        }

        vec4 outside = sample_volume(p_surface);
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
# Gizmo shaders
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
# Mesh shaders
# ============================================================

MESH_VERT = r"""
#version 330
uniform mat4 u_mvp;
uniform mat4 u_model;
in vec3 in_pos;
in vec3 in_texcoord3d;
in vec3 in_normal;
out vec3 v_texcoord3d;
out vec3 v_world_n;
void main() {
    v_texcoord3d = in_texcoord3d;
    v_world_n = mat3(u_model) * in_normal;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

MESH_FRAG = r"""
#version 330
uniform sampler2DArray tex_array;
uniform int u_num_layers;
uniform int u_flip_y;
uniform int u_bgr_input;
uniform vec3 u_light_dir;
in vec3 v_texcoord3d;
in vec3 v_world_n;
out vec4 fragColor;

vec4 sample_volume(vec3 p) {
    vec3 q = clamp(p, 0.0, 1.0);
    float zf = q.z * float(u_num_layers - 1);
    int z0 = int(floor(zf));
    int z1 = min(z0 + 1, u_num_layers - 1);
    float t = fract(zf);

    vec2 uv = q.xy;
    if (u_flip_y != 0) uv.y = 1.0 - uv.y;

    vec4 a = texture(tex_array, vec3(uv, float(z0)));
    vec4 b = texture(tex_array, vec3(uv, float(z1)));
    return mix(a, b, t);
}

void main() {
    vec4 c = sample_volume(v_texcoord3d);
    if (u_bgr_input != 0) c.rgb = c.bgr;

    vec3 N = normalize(v_world_n);
    vec3 L = normalize(u_light_dir);
    float ndl = max(dot(N, L), 0.0);
    float ambient = 0.30;
    float lit = ambient + (1.0 - ambient) * ndl;

    fragColor = vec4(c.rgb * lit, 1.0);
}
"""


# ============================================================
# HUD textured quad shaders
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


def translation_matrix(offset):
    M = np.eye(4, dtype=np.float32)
    M[:3, 3] = np.array(offset, dtype=np.float32)
    return M


# ============================================================
# Main app
# ============================================================

class MPRPlaneUI(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "MPR Plane UI + Mesh Upload (U)"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        vol_path = Path("threshold_images") / "volume_uint8.npy"
        if not vol_path.exists():
            raise FileNotFoundError(f"Missing {vol_path}. Put your prebuilt volume there.")

        V = np.load(vol_path, mmap_mode="r")
        if V.dtype != np.uint8 or V.ndim != 4 or V.shape[3] != 3:
            raise ValueError(f"Unexpected volume: shape={V.shape} dtype={V.dtype} (expected (Z,H,W,3) uint8)")

        self.Z, self.H, self.W, _ = V.shape
        self.V = V

        print(f"[volume] Z={self.Z} H={self.H} W={self.W} dtype=uint8 BGR")

        self.slice_prog = self.ctx.program(vertex_shader=SLICE_VERT, fragment_shader=SLICE_FRAG)
        self.gizmo_prog = self.ctx.program(vertex_shader=GIZMO_VERT, fragment_shader=GIZMO_FRAG)
        self.mesh_prog = self.ctx.program(vertex_shader=MESH_VERT, fragment_shader=MESH_FRAG)
        self.hud_prog = self.ctx.program(vertex_shader=HUD_TEX_VERT, fragment_shader=HUD_TEX_FRAG)

        fsq = np.array(
            [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1],
            dtype="f4",
        )
        self.slice_vbo = self.ctx.buffer(fsq.tobytes())
        self.slice_vao = self.ctx.simple_vertex_array(self.slice_prog, self.slice_vbo, "in_pos")

        hud_quad = np.array(
            [
                -1, -1, 0, 0,
                1, -1, 1, 0,
                -1, 1, 0, 1,
                -1, 1, 0, 1,
                1, -1, 1, 0,
                1, 1, 1, 1,
            ],
            dtype="f4",
        )
        self.hud_vbo = self.ctx.buffer(hud_quad.tobytes())
        self.hud_vao = self.ctx.vertex_array(
            self.hud_prog,
            [(self.hud_vbo, "2f 2f", "in_pos", "in_uv")],
        )

        self.tex = self.ctx.texture_array((self.W, self.H, self.Z), components=4, dtype="f1")
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        alpha = np.full((self.H, self.W, 1), 255, dtype=np.uint8)
        for z in range(self.Z):
            slab = np.ascontiguousarray(self.V[z])
            slab_rgba = np.concatenate([slab, alpha], axis=2)
            self.tex.write(slab_rgba.tobytes(), viewport=(0, 0, z, self.W, self.H, 1))
            if z % 25 == 0 or z == self.Z - 1:
                print(f"  uploaded {z + 1}/{self.Z}")

        self.tex.use(location=0)
        self.slice_prog["tex_array"].value = 0
        self.slice_prog["u_num_layers"].value = int(self.Z)
        self.mesh_prog["tex_array"].value = 0
        self.mesh_prog["u_num_layers"].value = int(self.Z)
        self.mesh_prog["u_light_dir"].value = (0.6, 0.5, 1.0)

        self.yaw = 0.0
        self.pitch = 0.0
        self.center = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.scale = 0.55
        self._update_plane_axes()

        self.heap_enable = True
        self.mouse_uv = (0.5, 0.5)
        self.heap_radius = 0.18
        self.heap_softness = 0.06
        self.heap_depth = 0.22
        self.heap_stretch = 1.0
        self.heap_dir = -1.0
        self.flip_y = 1
        self.bgr_input = 1
        self.surface_mode = 0
        self.curve_amp = 0.08
        self.curve_freq = 8.0
        self._push_slice_uniforms()

        self._drag_plane = False
        self._drag_pan = False
        self._drag_mesh_orbit = False
        self._drag_mesh_pan = False
        self._held_keys = set()

        self.gizmo_yaw = 0.8
        self.gizmo_pitch = 0.5
        self.gizmo_radius = 2.4

        self.curve_edit_mode = False
        self.curve_view = "perspective"
        self.curve_type = "cubic"
        self.curve_points = [
            np.array([-0.35, -0.20, -0.20], dtype=np.float32),
            np.array([-0.10, 0.25, -0.05], dtype=np.float32),
            np.array([0.18, -0.25, 0.12], dtype=np.float32),
            np.array([0.38, 0.15, 0.28], dtype=np.float32),
        ]
        self.curve_selected_idx = 0
        self.curve_drag_axis = None
        self.curve_drag_point = False


        self.mesh_rot_yaw = 0.8
        self.mesh_rot_pitch = 0.45
        self.mesh_pan = np.array([0.0, 0.0], dtype=np.float32)
        self.mesh_zoom = 1.0

        self.view_mode = "slice"
        self.surface_mode = 0
        self.curve_amp = 0.08
        self.curve_freq = 8.0
        self.mesh_path = None
        self.mesh_loaded = False
        self.mesh_vbo = None
        self.mesh_vao = None
        self.mesh_vertex_count = 0
        self.mesh_model = np.eye(4, dtype=np.float32)
        self.mesh_normals_ok = False

        self._init_gizmo_geometry()
        self._build_hud_texture()

        # frame history compositor (last 100 frames)
        self.history_max = 100
        self.history_enabled = False
        self.history_tex = self.ctx.texture((self.wnd.width, self.wnd.height), 4, dtype="f1")
        self.history_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.history_fbo = self.ctx.framebuffer(color_attachments=[self.history_tex])
        self._history_frames = []

        print("Ready.")
        print("  Slice mode: LMB rotate plane | MMB pan plane | wheel zoom")
        print("  U: upload mesh / toggle mesh mode")
        print("  Shift+U: reopen mesh picker")
        print("  Mesh mode: rotatable plane camera (LMB rotate mesh on plane, MMB pan, wheel zoom)")
        print("  T: cycle surface mode (plane -> curved -> uploaded mesh UV)")

    # ------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------

    def _init_gizmo_geometry(self):
        def to_gizmo(p01):
            return np.array(p01, np.float32) - 0.5

        corners = [
            to_gizmo([0, 0, 0]), to_gizmo([1, 0, 0]),
            to_gizmo([0, 1, 0]), to_gizmo([1, 1, 0]),
            to_gizmo([0, 0, 1]), to_gizmo([1, 0, 1]),
            to_gizmo([0, 1, 1]), to_gizmo([1, 1, 1]),
        ]
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (5, 7), (6, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        box_lines = []
        for a, b in edges:
            box_lines.append(corners[a])
            box_lines.append(corners[b])
        box_lines = np.array(box_lines, dtype="f4")

        self.box_vbo = self.ctx.buffer(box_lines.tobytes())
        self.box_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.box_vbo, "in_pos")

        self.plane_vbo = self.ctx.buffer(reserve=36 * 3 * 4)
        self.plane_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.plane_vbo, "in_pos")

        self.n_vbo = self.ctx.buffer(reserve=2 * 3 * 4)
        self.n_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.n_vbo, "in_pos")

        self.curve_vbo = self.ctx.buffer(reserve=512 * 3 * 4)
        self.curve_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.curve_vbo, "in_pos")
        self.curve_pts_vbo = self.ctx.buffer(reserve=128 * 3 * 4)
        self.curve_pts_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.curve_pts_vbo, "in_pos")
        self.curve_gizmo_vbo = self.ctx.buffer(reserve=6 * 3 * 4)
        self.curve_gizmo_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.curve_gizmo_vbo, "in_pos")

        self._update_gizmo_geometry()
        self._update_curve_geometry()

    def _build_hud_texture(self):
        img = Image.new("RGBA", (900, 180), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 22)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        draw.rounded_rectangle((0, 0, 899, 179), radius=18, fill=(15, 18, 25, 180))
        draw.text((18, 16), "U: upload / toggle mesh   |   Shift+U: choose another mesh", fill=(255, 255, 255, 255), font=font)
        draw.text((18, 56), "slice mode: rotate plane / heap brush    |    mesh mode: orbit uploaded mesh with volume texture", fill=(210, 220, 235, 255), font=font_small)
        draw.text((18, 90), "supported by trimesh: OBJ / PLY / STL / GLB / GLTF (triangle meshes recommended)", fill=(190, 205, 225, 255), font=font_small)
        draw.text((18, 124), "the uploaded mesh is normalized into the volume bounds and textured by sampling the 3D scan", fill=(190, 205, 225, 255), font=font_small)

        self.hud_tex = self.ctx.texture(img.size, 4, img.tobytes())
        self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hud_prog["u_tex"].value = 0

    # ------------------------------------------------------------
    # Plane state
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
        self.slice_prog["u_scale"].value = float(self.scale)
        self.slice_prog["u_surface_mode"].value = int(self.surface_mode)
        self.slice_prog["u_curve_amp"].value = float(self.curve_amp)
        self.slice_prog["u_curve_freq"].value = float(self.curve_freq)
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

        self.mesh_prog["u_flip_y"].value = int(self.flip_y)
        self.mesh_prog["u_bgr_input"].value = int(self.bgr_input)

    # ------------------------------------------------------------
    # Mesh loading
    # ------------------------------------------------------------

    def _open_mesh_picker(self):
        if filedialog is None or tk is None:
            print("[mesh] tkinter file dialog is unavailable on this Python build.")
            return None
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(
            title="Choose a 3D mesh",
            filetypes=[
                ("3D Mesh", "*.obj *.ply *.stl *.glb *.gltf"),
                ("OBJ", "*.obj"),
                ("PLY", "*.ply"),
                ("STL", "*.stl"),
                ("GLB/GLTF", "*.glb *.gltf"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return path or None

    def _load_mesh_from_path(self, path: str):
        loaded = trimesh.load(path, force="mesh")
        if isinstance(loaded, trimesh.Scene):
            geometries = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not geometries:
                raise ValueError("No mesh geometry found in scene")
            mesh = trimesh.util.concatenate(geometries)
        else:
            mesh = loaded

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Loaded asset is not a triangle mesh")

        mesh = mesh.copy()
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            raise ValueError("Mesh has no vertices or faces")

        if mesh.vertex_normals is None or len(mesh.vertex_normals) != len(mesh.vertices):
            mesh.rezero()
            mesh.vertex_normals

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        norms = np.asarray(mesh.vertex_normals, dtype=np.float32)

        bmin = verts.min(axis=0)
        bmax = verts.max(axis=0)
        extent = np.maximum(bmax - bmin, 1e-6)
        center = 0.5 * (bmin + bmax)
        max_extent = float(np.max(extent))

        verts01 = (verts - bmin) / extent
        verts_draw = (verts - center) / max_extent

        tri_pos = verts_draw[faces].reshape(-1, 3)
        tri_tex = verts01[faces].reshape(-1, 3)
        tri_nrm = norms[faces].reshape(-1, 3)

        interleaved = np.hstack([tri_pos, tri_tex, tri_nrm]).astype("f4", copy=False)

        if self.mesh_vbo is not None:
            self.mesh_vbo.release()
            self.mesh_vao.release()

        self.mesh_vbo = self.ctx.buffer(interleaved.tobytes())
        self.mesh_vao = self.ctx.vertex_array(
            self.mesh_prog,
            [(self.mesh_vbo, "3f 3f 3f", "in_pos", "in_texcoord3d", "in_normal")],
        )
        self.mesh_vertex_count = interleaved.shape[0]
        self.mesh_model = np.eye(4, dtype=np.float32)
        self.mesh_loaded = True
        self.mesh_path = path
        self.view_mode = "mesh"

        print(f"[mesh] loaded: {Path(path).name} | verts={len(verts)} faces={len(faces)}")

    def _handle_u_press(self, modifiers):
        shift_down = bool(modifiers.shift)
        if (not self.mesh_loaded) or shift_down:
            path = self._open_mesh_picker()
            if path:
                try:
                    self._load_mesh_from_path(path)
                except Exception as exc:
                    print(f"[mesh] failed to load '{path}': {exc}")
            return

        self.view_mode = "mesh" if self.view_mode == "slice" else "slice"
        print(f"[view] mode={self.view_mode}")

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

    def _mesh_eye(self):
        return np.array([0.0, 0.0, 2.2], dtype=np.float32)

    def _gizmo_viewport(self):
        W, H = self.wnd.width, self.wnd.height
        giz_px = int(min(W, H) * 0.28)
        pad = 12
        gx0 = W - giz_px - pad
        gy0 = H - giz_px - pad
        return gx0, gy0, giz_px

    def _update_gizmo_geometry(self):
        c = self.center
        u = self.u
        v = self.v
        n = self.n
        s = self.scale

        p00 = (c - u * s - v * s) - 0.5
        p10 = (c + u * s - v * s) - 0.5
        p01 = (c - u * s + v * s) - 0.5
        p11 = (c + u * s + v * s) - 0.5

        t = 0.02
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

        start = c - 0.5
        end = c + self.n * 0.35 - 0.5
        arrow = np.array([start, end], dtype=np.float32)
        self.n_vbo.write(arrow.tobytes())


    def _curve_segments(self):
        if len(self.curve_points) < 2:
            return []
        if self.curve_type == "poly":
            return [self.curve_points]
        seg_size = 3 if self.curve_type == "quadratic" else 4
        stride = 2 if self.curve_type == "quadratic" else 3
        segs = []
        i = 0
        while i + seg_size <= len(self.curve_points):
            segs.append(self.curve_points[i:i+seg_size])
            i += stride
        return segs if segs else [self.curve_points]

    def _bezier_point(self, seg, t):
        work = [p.copy() for p in seg]
        while len(work) > 1:
            work = [work[i] * (1.0 - t) + work[i + 1] * t for i in range(len(work)-1)]
        return work[0]

    def _curve_view_matrix(self):
        if self.curve_view == "front":
            return look_at([0, -2.0, 0], [0,0,0], [0,0,1])
        if self.curve_view == "side":
            return look_at([2.0, 0, 0], [0,0,0], [0,0,1])
        if self.curve_view == "top":
            return look_at([0, 0, 2.0], [0,0,0], [0,1,0])
        return look_at(eye=self._gizmo_eye(), target=[0,0,0], up=[0,0,1])

    def _update_curve_geometry(self):
        samples = []
        for seg in self._curve_segments():
            for i in range(40):
                samples.append(self._bezier_point(seg, i / 39.0))
        if not samples:
            samples = self.curve_points
        curve = np.array(samples, dtype=np.float32)
        self.curve_vbo.orphan(size=max(curve.nbytes, 12))
        self.curve_vbo.write(curve.tobytes())
        self._curve_count = len(curve)

        pts = np.array(self.curve_points, dtype=np.float32)
        self.curve_pts_vbo.orphan(size=max(pts.nbytes, 12))
        self.curve_pts_vbo.write(pts.tobytes())
        self._curve_pts_count = len(pts)

        p = self.curve_points[self.curve_selected_idx]
        g = np.array([p, p + np.array([0.18,0,0],np.float32), p, p + np.array([0,0.18,0],np.float32), p, p + np.array([0,0,0.18],np.float32)], dtype=np.float32)
        self.curve_gizmo_vbo.write(g.tobytes())

    # ------------------------------------------------------------
    # Input
    # ------------------------------------------------------------

    def on_mouse_position_event(self, x, y, dx, dy):
        u = x / max(1, self.wnd.width)
        v = 1.0 - (y / max(1, self.wnd.height))
        self.mouse_uv = (float(u), float(v))
        self.slice_prog["u_mouse"].value = self.mouse_uv

    def on_mouse_press_event(self, x, y, button):
        LEFT = self.wnd.mouse.left
        MIDDLE = self.wnd.mouse.middle

        if self.curve_edit_mode:
            self._curve_drag_start = (x, y)
            self.curve_drag_point = True
            return

        if self.view_mode == "slice":
            if button == LEFT:
                self._drag_plane = True
            if button == MIDDLE:
                self._drag_pan = True
        else:
            if button == LEFT:
                self._drag_mesh_orbit = True
            if button == MIDDLE:
                self._drag_mesh_pan = True

    def on_mouse_release_event(self, x, y, button):
        LEFT = self.wnd.mouse.left
        MIDDLE = self.wnd.mouse.middle

        if button == LEFT:
            self._drag_plane = False
            self._drag_mesh_orbit = False
            self.curve_drag_point = False
        if button == MIDDLE:
            self._drag_pan = False
            self._drag_mesh_pan = False

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self.curve_edit_mode and self.curve_drag_point:
            p = self.curve_points[self.curve_selected_idx]
            p[0] += dx * 0.0015
            p[1] += -dy * 0.0015
            p[:] = np.clip(p, -0.5, 0.5)
            self.curve_points[self.curve_selected_idx] = p
            self._update_curve_geometry()
            return

        if self.view_mode == "slice":
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
                self.center[:] = np.clip(self.center, 0.0, 1.0)
                self._push_slice_uniforms()
                self._update_gizmo_geometry()
            return

        if self._drag_mesh_orbit:
            self.mesh_rot_yaw += dx * 0.005
            self.mesh_rot_pitch += -dy * 0.005
            self.mesh_rot_pitch = float(np.clip(self.mesh_rot_pitch, -1.55, 1.55))

        if self._drag_mesh_pan:
            self.mesh_pan += np.array([dx, -dy], dtype=np.float32) * 0.002

    def on_mouse_scroll_event(self, x_offset, y_offset):
        if self.view_mode == "slice":
            self.scale *= float(0.92 ** y_offset)
            self.scale = float(np.clip(self.scale, 0.05, 2.0))
            self._push_slice_uniforms()
            self._update_gizmo_geometry()
        else:
            self.mesh_zoom *= float(0.92 ** y_offset)
            self.mesh_zoom = float(np.clip(self.mesh_zoom, 0.2, 4.0))

    def on_key_event(self, key, action, modifiers):
        k = self.wnd.keys

        if action == k.ACTION_PRESS and key == k.ESCAPE:
            self.wnd.close()
            return

        if action == k.ACTION_PRESS:
            if key == k.U:
                self._handle_u_press(modifiers)
                return


            if key == k.T:
                self.surface_mode = (self.surface_mode + 1) % 3
                self.slice_prog["u_surface_mode"].value = int(0 if self.surface_mode == 2 else self.surface_mode)
                modes = {0: "plane", 1: "curve", 2: "uploaded-mesh"}
                print(f"surface_mode={modes[self.surface_mode]}")
                return

            if key == k.TAB:
                self.history_enabled = not self.history_enabled
                print(f"history_enabled={self.history_enabled}")
                return

            if key == k.C:
                self.curve_edit_mode = not self.curve_edit_mode
                print(f"curve_edit_mode={self.curve_edit_mode}")
                return
            if key == k.V:
                views = ["perspective", "front", "side", "top"]
                self.curve_view = views[(views.index(self.curve_view) + 1) % len(views)]
                print(f"curve_view={self.curve_view}")
                return
            if key == k.B:
                types = ["cubic", "quadratic", "poly"]
                self.curve_type = types[(types.index(self.curve_type) + 1) % len(types)]
                self._update_curve_geometry()
                print(f"curve_type={self.curve_type}")
                return
            if key == k.P:
                last = self.curve_points[-1]
                self.curve_points.append((last + np.array([0.1, 0.0, 0.05], dtype=np.float32)).astype(np.float32))
                self.curve_selected_idx = len(self.curve_points)-1
                self._update_curve_geometry()
                return

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
                self.mesh_prog["u_flip_y"].value = int(self.flip_y)
                print(f"flip_y={self.flip_y}")
                return

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
                self.mesh_rot_yaw = 0.8
                self.mesh_rot_pitch = 0.45
                self.mesh_pan[:] = 0.0
                self.mesh_zoom = 1.0
                self.surface_mode = 0
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

        if self.view_mode == "slice":
            base = 0.22
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
            self.center[:] = np.clip(self.center, 0.0, 1.0)
            self._push_slice_uniforms()
            self._update_gizmo_geometry()
            return

        for key, is_shift in list(self._held_keys):
            step = dt * (120.0 if is_shift else 60.0)
            if key == k.W:
                self.mesh_pan[1] += step * 0.0005
            if key == k.S:
                self.mesh_pan[1] -= step * 0.0005
            if key == k.A:
                self.mesh_pan[0] -= step * 0.0005
            if key == k.D:
                self.mesh_pan[0] += step * 0.0005

    # ------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------

    def resize(self, width, height):
        self._push_slice_uniforms()
        if hasattr(self, "history_tex"):
            self.history_tex.release()
            self.history_fbo.release()
            self.history_tex = self.ctx.texture((max(1,width), max(1,height)), 4, dtype="f1")
            self.history_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.history_fbo = self.ctx.framebuffer(color_attachments=[self.history_tex])
            self._history_frames.clear()

    # ------------------------------------------------------------
    # Render paths
    # ------------------------------------------------------------

    def _render_slice_view(self):
        W, H = self.wnd.width, self.wnd.height
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        self.tex.use(location=0)
        self.slice_vao.render()

    def _render_mesh_view(self):
        W, H = self.wnd.width, self.wnd.height
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.02, 0.02, 0.03, 1.0)

        if not self.mesh_loaded or self.mesh_vao is None:
            self.ctx.disable(moderngl.DEPTH_TEST)
            return

        P = perspective(50.0, W / max(H, 1), 0.01, 50.0)
        V = look_at(self._mesh_eye(), [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        cy, sy = np.cos(self.mesh_rot_yaw), np.sin(self.mesh_rot_yaw)
        cp, sp = np.cos(self.mesh_rot_pitch), np.sin(self.mesh_rot_pitch)
        Ry = np.array([[cy,0,sy,0],[0,1,0,0],[-sy,0,cy,0],[0,0,0,1]], dtype=np.float32)
        Rx = np.array([[1,0,0,0],[0,cp,-sp,0],[0,sp,cp,0],[0,0,0,1]], dtype=np.float32)
        S = np.diag([self.mesh_zoom,self.mesh_zoom,self.mesh_zoom,1.0]).astype(np.float32)
        T = translation_matrix([float(self.mesh_pan[0]), float(self.mesh_pan[1]), 0.0])
        M = T @ Ry @ Rx @ S
        MVP = (P @ V @ M).astype(np.float32)

        self.mesh_prog["u_mvp"].write(MVP.tobytes())
        self.mesh_prog["u_model"].write(M.astype(np.float32).tobytes())

        self.tex.use(location=0)
        self.mesh_vao.render(mode=moderngl.TRIANGLES)
        self.ctx.disable(moderngl.DEPTH_TEST)

    def _render_gizmo(self):
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

        if self.curve_edit_mode:
            CV = self._curve_view_matrix()
            MVPc = (P @ CV).astype(np.float32)
            self.gizmo_prog["u_mvp"].write(MVPc.tobytes())
            self.gizmo_prog["u_color"].value = (0.15, 0.85, 1.0, 1.0)
            self.curve_vao.render(mode=moderngl.LINE_STRIP, vertices=self._curve_count)
            self.gizmo_prog["u_color"].value = (0.95, 0.95, 0.95, 1.0)
            self.curve_pts_vao.render(mode=moderngl.POINTS, vertices=self._curve_pts_count)
            self.gizmo_prog["u_color"].value = (1.0, 0.3, 0.3, 1.0)
            self.curve_gizmo_vao.render(mode=moderngl.LINES)

        self.ctx.disable(moderngl.DEPTH_TEST)

    def _render_hud(self):
        W, H = self.wnd.width, self.wnd.height
        hud_w = min(900, W - 24)
        hud_h = 180
        self.ctx.viewport = (12, max(12, H - hud_h - 12), hud_w, hud_h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.hud_tex.use(location=0)
        self.hud_vao.render(mode=moderngl.TRIANGLES)


    def _capture_frame(self):
        W, H = self.wnd.width, self.wnd.height
        data = self.ctx.screen.read(viewport=(0, 0, W, H), components=4, dtype="f1")
        arr = np.frombuffer(data, dtype=np.uint8).reshape(H, W, 4).copy()
        self._history_frames.append(arr)
        if len(self._history_frames) > self.history_max:
            self._history_frames = self._history_frames[-self.history_max:]

    def _composite_history(self):
        if not self._history_frames:
            return
        stack = np.stack(self._history_frames, axis=0).astype(np.float32)
        blended = np.mean(stack, axis=0).astype(np.uint8)
        self.history_tex.write(blended.tobytes())
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, self.wnd.width, self.wnd.height)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.hud_prog["u_tex"].value = 0
        self.history_tex.use(location=0)
        self.hud_vao.render(mode=moderngl.TRIANGLES)

    def on_render(self, time: float, frame_time: float):
        self._apply_held_keys(frame_time)

        if self.surface_mode == 2 and self.mesh_loaded:
            self._render_mesh_view()
        elif self.view_mode == "mesh" and self.mesh_loaded:
            self._render_mesh_view()
        else:
            self._render_slice_view()

        self._render_gizmo()
        self._render_hud()
        self._capture_frame()
        if self.history_enabled:
            self._composite_history()


if __name__ == "__main__":
    mglw.run_window_config(MPRPlaneUI)
