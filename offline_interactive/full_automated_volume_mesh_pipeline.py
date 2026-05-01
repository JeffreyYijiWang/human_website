import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import trimesh


def parse_args():
    ap = argparse.ArgumentParser(description="Fully automated volume -> mesh -> Blender headless pipeline")
    ap.add_argument("--volume", required=True, help="Path to volume_uint8.npy with shape (Z,H,W,3)")
    ap.add_argument("--mesh", required=True, help="Input mesh path (.obj/.ply/.stl/.glb/.gltf)")
    ap.add_argument("--baked-mesh", required=True, help="Output baked mesh path, preferably .ply")
    ap.add_argument("--flip-y", type=int, default=1, help="1 to flip Y when sampling volume, 0 otherwise")
    ap.add_argument("--bgr-input", type=int, default=1, help="1 if volume channels are BGR, 0 if RGB")
    ap.add_argument("--blender-exe", default=None, help="Optional blender executable for headless stage")
    ap.add_argument("--save-blend", default=None, help="Optional output .blend path")
    ap.add_argument("--export-obj", default=None, help="Optional output OBJ path from Blender")
    ap.add_argument("--skin", type=int, default=1, help="1 to add Blender Skin modifier, 0 to skip")
    ap.add_argument("--subsurf", type=int, default=1, help="1 to add Subsurf after Skin, 0 to skip")
    return ap.parse_args()


# ------------------------------------------------------------
# Volume sampling
# ------------------------------------------------------------

def load_volume(path: str) -> np.ndarray:
    vol = np.load(path)
    if vol.ndim != 4 or vol.shape[3] != 3 or vol.dtype != np.uint8:
        raise ValueError(f"Expected (Z,H,W,3) uint8 volume, got shape={vol.shape} dtype={vol.dtype}")
    return vol


def load_mesh(path: str) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=True)
    if isinstance(loaded, trimesh.Scene):
        geoms = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("No mesh geometry found in scene")
        mesh = trimesh.util.concatenate(geoms)
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded asset is not a triangle mesh")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("Mesh has no vertices or faces")
    return mesh.copy()


def vertices_to_volume_uvw(vertices: np.ndarray) -> np.ndarray:
    bmin = vertices.min(axis=0)
    bmax = vertices.max(axis=0)
    extent = np.maximum(bmax - bmin, 1e-6)
    uvw = (vertices - bmin) / extent
    return np.clip(uvw, 0.0, 1.0)


def trilinear_sample_volume(volume: np.ndarray, uvw: np.ndarray, flip_y: bool = True, bgr_input: bool = True) -> np.ndarray:
    zdim, ydim, xdim, _ = volume.shape

    q = np.clip(uvw, 0.0, 1.0).astype(np.float32)
    x = q[:, 0] * (xdim - 1)
    y = q[:, 1]
    if flip_y:
        y = 1.0 - y
    y = y * (ydim - 1)
    z = q[:, 2] * (zdim - 1)

    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    z0 = np.floor(z).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, xdim - 1)
    y1 = np.clip(y0 + 1, 0, ydim - 1)
    z1 = np.clip(z0 + 1, 0, zdim - 1)

    tx = (x - x0).reshape(-1, 1)
    ty = (y - y0).reshape(-1, 1)
    tz = (z - z0).reshape(-1, 1)

    c000 = volume[z0, y0, x0].astype(np.float32)
    c100 = volume[z0, y0, x1].astype(np.float32)
    c010 = volume[z0, y1, x0].astype(np.float32)
    c110 = volume[z0, y1, x1].astype(np.float32)
    c001 = volume[z1, y0, x0].astype(np.float32)
    c101 = volume[z1, y0, x1].astype(np.float32)
    c011 = volume[z1, y1, x0].astype(np.float32)
    c111 = volume[z1, y1, x1].astype(np.float32)

    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx

    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    c = c0 * (1.0 - tz) + c1 * tz

    if bgr_input:
        c = c[:, ::-1]

    return np.clip(c, 0, 255).astype(np.uint8)


# ------------------------------------------------------------
# Baking + export
# ------------------------------------------------------------

def bake_vertex_colors(mesh: trimesh.Trimesh, volume: np.ndarray, flip_y: bool, bgr_input: bool) -> trimesh.Trimesh:
    uvw = vertices_to_volume_uvw(mesh.vertices)
    rgb = trilinear_sample_volume(volume, uvw, flip_y=flip_y, bgr_input=bgr_input)
    rgba = np.concatenate([rgb, np.full((len(rgb), 1), 255, dtype=np.uint8)], axis=1)
    mesh.visual.vertex_colors = rgba
    return mesh


def export_baked_mesh(mesh: trimesh.Trimesh, output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    print(f"[export] baked mesh saved to: {out}")


# ------------------------------------------------------------
# Blender headless stage
# ------------------------------------------------------------

def run_blender_headless(blender_exe: str, baked_mesh: str, save_blend: str | None, export_obj: str | None, skin: int, subsurf: int):
    script_path = Path(__file__).with_name("blender_batch_headless_import_skin.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Missing companion Blender script: {script_path}")

    cmd = [
        blender_exe,
        "--background",
        "--factory-startup",
        "--python",
        str(script_path),
        "--",
        "--mesh",
        baked_mesh,
        "--skin",
        str(int(skin)),
        "--subsurf",
        str(int(subsurf)),
    ]

    if save_blend:
        cmd += ["--save-blend", save_blend]
    if export_obj:
        cmd += ["--export-obj", export_obj]

    print("[blender] running headless Blender...")
    print(" ".join(f'"{c}"' if ' ' in c else c for c in cmd))
    subprocess.run(cmd, check=True)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    args = parse_args()

    volume = load_volume(args.volume)
    mesh = load_mesh(args.mesh)

    print(f"[volume] shape={volume.shape} dtype={volume.dtype}")
    print(f"[mesh] verts={len(mesh.vertices)} faces={len(mesh.faces)}")

    baked = bake_vertex_colors(
        mesh,
        volume,
        flip_y=bool(args.flip_y),
        bgr_input=bool(args.bgr_input),
    )
    export_baked_mesh(baked, args.baked_mesh)

    if args.blender_exe:
        run_blender_headless(
            blender_exe=args.blender_exe,
            baked_mesh=args.baked_mesh,
            save_blend=args.save_blend,
            export_obj=args.export_obj,
            skin=args.skin,
            subsurf=args.subsurf,
        )
    else:
        print("[done] baked mesh created. Blender stage skipped because --blender-exe was not provided.")


if __name__ == "__main__":
    main()
