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
in vec3 in_normal;
in vec3 in_color;
out vec3 v_world_n;
out vec3 v_color;
void main() {
    v_world_n = mat3(u_model) * in_normal;
    v_color = in_color;
    gl_Position = u_mvp * vec4(in_pos, 1.0);
}
"""

MESH_FRAG = r"""
#version 330
uniform vec3 u_light_dir;
in vec3 v_world_n;
in vec3 v_color;
out vec4 fragColor;
void main() {
    vec3 N = normalize(v_world_n);
    vec3 L = normalize(u_light_dir);
    float ndl = max(dot(N, L), 0.0);
    float ambient = 0.28;
    float lit = ambient + (1.0 - ambient) * ndl;
    fragColor = vec4(v_color * lit, 1.0);
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
        volume_dims = np.array([self.W, self.H, self.Z], dtype=np.float32)
        self.volume_scale = volume_dims / float(np.max(volume_dims))

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
        self._push_slice_uniforms()

        self._drag_plane = False
        self._drag_pan = False
        self._drag_mesh_orbit = False
        self._drag_mesh_pan = False
        self._held_keys = set()

        self.gizmo_yaw = 0.8
        self.gizmo_pitch = 0.5
        self.gizmo_radius = 2.4

        self.mesh_view_yaw = 0.8
        self.mesh_view_pitch = 0.45
        self.mesh_view_radius = 1.8
        self.mesh_object_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.mesh_view_target = self.mesh_object_center.copy()

        self.view_mode = "slice"
        self.mesh_path = None
        self.mesh_loaded = False
        self.mesh_vbo = None
        self.mesh_vao = None
        self.mesh_vertex_count = 0
        self.mesh_model = np.eye(4, dtype=np.float32)
        self.mesh_normals_ok = False

        self._init_gizmo_geometry()
        self._build_hud_texture()

        print("Ready.")
        print("  Slice mode: LMB rotate plane | MMB pan plane | wheel zoom")
        print("  U: upload mesh / toggle mesh mode")
        print("  Shift+U: reopen mesh picker")
        print("  Mesh mode: LMB orbit mesh | MMB pan | wheel zoom")

    # ------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------

    def _make_volume_box_lines(self):
        sx, sy, sz = 0.5 * self.volume_scale
        corners = [
            np.array([-sx, -sy, -sz], np.float32), np.array([ sx, -sy, -sz], np.float32),
            np.array([-sx,  sy, -sz], np.float32), np.array([ sx,  sy, -sz], np.float32),
            np.array([-sx, -sy,  sz], np.float32), np.array([ sx, -sy,  sz], np.float32),
            np.array([-sx,  sy,  sz], np.float32), np.array([ sx,  sy,  sz], np.float32),
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
        return np.array(box_lines, dtype="f4")

    def _init_gizmo_geometry(self):
        box_lines = self._make_volume_box_lines()

        self.box_vbo = self.ctx.buffer(box_lines.tobytes())
        self.box_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.box_vbo, "in_pos")

        self.plane_vbo = self.ctx.buffer(reserve=36 * 3 * 4)
        self.plane_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.plane_vbo, "in_pos")

        self.n_vbo = self.ctx.buffer(reserve=2 * 3 * 4)
        self.n_vao = self.ctx.simple_vertex_array(self.gizmo_prog, self.n_vbo, "in_pos")

        self._update_gizmo_geometry()

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
        draw.text((18, 56), "slice mode: plane + heap brush    |    mesh mode: actual uploaded 3D mesh with perspective camera", fill=(210, 220, 235, 255), font=font_small)
        draw.text((18, 90), "supported by trimesh: OBJ / PLY / STL / GLB / GLTF (triangle meshes recommended)", fill=(190, 205, 225, 255), font=font_small)
        draw.text((18, 124), "mesh mode renders solid triangle geometry; the volume box stays separate and no plane is used", fill=(190, 205, 225, 255), font=font_small)

        img = img.transpose(Image.FLIP_TOP_BOTTOM)
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
        loaded = trimesh.load(path, process=True)
        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                raise ValueError("No mesh geometry found in scene")
            mesh = loaded.dump(concatenate=True)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise ValueError("Loaded asset is not a triangle mesh or scene")

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Loaded asset is not a triangle mesh")

        mesh = mesh.copy()
        if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
            raise ValueError("Mesh has no vertices or faces")

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int32)
        normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
        if normals.shape[0] != verts.shape[0]:
            mesh.vertex_normals
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

        bmin = verts.min(axis=0)
        bmax = verts.max(axis=0)
        center = 0.5 * (bmin + bmax)
        extent = np.maximum(bmax - bmin, 1e-6)
        max_extent = float(np.max(extent))

        # keep the uploaded mesh as real 3D geometry in object space,
        # only normalize uniformly so it fits inside the scene.
        verts_draw = (verts - center) / max_extent

        colors = None
        try:
            vc = np.asarray(mesh.visual.vertex_colors, dtype=np.float32)
            if vc.ndim == 2 and vc.shape[0] == verts.shape[0] and vc.shape[1] >= 3:
                colors = vc[:, :3] / 255.0
        except Exception:
            colors = None
        if colors is None:
            colors = np.tile(np.array([[0.78, 0.80, 0.86]], dtype=np.float32), (verts.shape[0], 1))

        tri_pos = verts_draw[faces].reshape(-1, 3)
        tri_nrm = normals[faces].reshape(-1, 3)
        tri_col = colors[faces].reshape(-1, 3)
        interleaved = np.hstack([tri_pos, tri_nrm, tri_col]).astype("f4", copy=False)

        if self.mesh_vbo is not None:
            self.mesh_vbo.release()
            self.mesh_vao.release()

        self.mesh_vbo = self.ctx.buffer(interleaved.tobytes())
        self.mesh_vao = self.ctx.vertex_array(
            self.mesh_prog,
            [(self.mesh_vbo, "3f 3f 3f", "in_pos", "in_normal", "in_color")],
        )
        self.mesh_vertex_count = interleaved.shape[0]
        self.mesh_model = np.eye(4, dtype=np.float32)
        self.mesh_object_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.mesh_view_target = self.mesh_object_center.copy()
        mesh_radius = float(np.max(np.linalg.norm(verts_draw, axis=1)))
        self.mesh_view_radius = max(1.2, mesh_radius * 3.0)
        self.mesh_loaded = True
        self.mesh_path = path
        self.view_mode = "mesh"

        print(f"[mesh] loaded actual 3D mesh: {Path(path).name} | verts={len(verts)} faces={len(faces)}")

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
        cy, sy = np.cos(self.mesh_view_yaw), np.sin(self.mesh_view_yaw)
        cp, sp = np.cos(self.mesh_view_pitch), np.sin(self.mesh_view_pitch)
        x = self.mesh_view_radius * sy * cp
        y = self.mesh_view_radius * cy * cp
        z = self.mesh_view_radius * sp
        return self.mesh_view_target + np.array([x, y, z], dtype=np.float32)

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

        vol_scale = self.volume_scale
        p00 = ((c - u * s - v * s) - 0.5) * vol_scale
        p10 = ((c + u * s - v * s) - 0.5) * vol_scale
        p01 = ((c - u * s + v * s) - 0.5) * vol_scale
        p11 = ((c + u * s + v * s) - 0.5) * vol_scale

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

        start = (c - 0.5) * vol_scale
        end = (c + self.n * 0.35 - 0.5) * vol_scale
        arrow = np.array([start, end], dtype=np.float32)
        self.n_vbo.write(arrow.tobytes())

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
        if button == MIDDLE:
            self._drag_pan = False
            self._drag_mesh_pan = False

    def on_mouse_drag_event(self, x, y, dx, dy):
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
            self.mesh_view_yaw += dx * 0.005
            self.mesh_view_pitch += -dy * 0.005
            self.mesh_view_pitch = float(np.clip(self.mesh_view_pitch, -1.55, 1.55))

        if self._drag_mesh_pan:
            eye = self._mesh_eye()
            forward = normalize(self.mesh_view_target - eye)
            right = normalize(np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32)))
            up = normalize(np.cross(right, forward))
            pan_scale = 0.002 * self.mesh_view_radius
            self.mesh_view_target += (-dx * pan_scale) * right + (dy * pan_scale) * up

    def on_mouse_scroll_event(self, x_offset, y_offset):
        if self.view_mode == "slice":
            self.scale *= float(0.92 ** y_offset)
            self.scale = float(np.clip(self.scale, 0.05, 2.0))
            self._push_slice_uniforms()
            self._update_gizmo_geometry()
        else:
            self.mesh_view_radius *= float(0.92 ** y_offset)
            self.mesh_view_radius = float(np.clip(self.mesh_view_radius, 0.25, 10.0))

    def on_key_event(self, key, action, modifiers):
        k = self.wnd.keys

        if action == k.ACTION_PRESS and key == k.ESCAPE:
            self.wnd.close()
            return

        if action == k.ACTION_PRESS:
            if key == k.U:
                self._handle_u_press(modifiers)
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
                self.mesh_view_yaw = 0.8
                self.mesh_view_pitch = 0.45
                self.mesh_view_radius = 1.8
                self.mesh_view_target = self.mesh_object_center.copy()
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

        base = 0.8
        forward = normalize(self.mesh_view_target - self._mesh_eye())
        right = normalize(np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32)))
        up = normalize(np.cross(right, forward))
        for key, is_shift in list(self._held_keys):
            step = base * dt * (3.0 if is_shift else 1.0)
            if key == k.W:
                self.mesh_view_target += forward * step
            if key == k.S:
                self.mesh_view_target -= forward * step
            if key == k.A:
                self.mesh_view_target -= right * step
            if key == k.D:
                self.mesh_view_target += right * step
            if key == k.Q:
                self.mesh_view_target -= up * step
            if key == k.E:
                self.mesh_view_target += up * step

    # ------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------

    def resize(self, width, height):
        self._push_slice_uniforms()

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
        V = look_at(self._mesh_eye(), self.mesh_view_target, [0.0, 0.0, 1.0])
        M = self.mesh_model
        MVP = (P @ V @ M).astype(np.float32)

        self.mesh_prog["u_mvp"].write(MVP.tobytes())
        self.mesh_prog["u_model"].write(M.astype(np.float32).tobytes())

        self.ctx.disable(moderngl.CULL_FACE)
        self.mesh_vao.render(mode=moderngl.TRIANGLES)

        # draw the full volume as a rectangular box in mesh mode; no slice plane participates here
        self.gizmo_prog["u_mvp"].write((P @ V).astype(np.float32).tobytes())
        self.gizmo_prog["u_color"].value = (0.45, 0.80, 1.00, 1.0)
        self.box_vao.render(mode=moderngl.LINES)

        self.ctx.disable(moderngl.DEPTH_TEST)

    def _render_gizmo(self):
        gx0, gy0, giz_px = self._gizmo_viewport()
        self.ctx.viewport = (gx0, gy0, giz_px, giz_px)
        self.ctx.enable(moderngl.DEPTH_TEST)

        target = self.mesh_object_center if (self.view_mode == "mesh" and self.mesh_loaded) else np.array([0.0, 0.0, 0.0], dtype=np.float32)
        P = perspective(45.0, 1.0, 0.05, 10.0)
        V = look_at(eye=self._gizmo_eye(), target=target, up=[0.0, 0.0, 1.0])
        MVP = (P @ V).astype(np.float32)
        self.gizmo_prog["u_mvp"].write(MVP.tobytes())

        self.gizmo_prog["u_color"].value = (0.85, 0.90, 0.98, 1.0)
        self.box_vao.render(mode=moderngl.LINES)

        if self.view_mode == "slice":
            self.gizmo_prog["u_color"].value = (1.0, 0.15, 0.15, 0.80)
            self.plane_vao.render(mode=moderngl.TRIANGLES)

            self.gizmo_prog["u_color"].value = (1.0, 0.90, 0.25, 1.0)
            self.n_vao.render(mode=moderngl.LINES)

        self.ctx.disable(moderngl.DEPTH_TEST)

    def _render_hud(self):
        W, H = self.wnd.width, self.wnd.height
        hud_w = min(900, W - 24)
        hud_h = 180
        self.ctx.viewport = (12, max(12, H - hud_h - 12), hud_w, hud_h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.hud_tex.use(location=0)
        self.hud_vao.render(mode=moderngl.TRIANGLES)

    def on_render(self, time: float, frame_time: float):
        self._apply_held_keys(frame_time)

        if self.view_mode == "mesh" and self.mesh_loaded:
            self._render_mesh_view()
        else:
            self._render_slice_view()

        self._render_gizmo()
        self._render_hud()


if __name__ == "__main__":
    mglw.run_window_config(MPRPlaneUI)
