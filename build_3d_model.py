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

# build_volume_and_reslice_color.py
from pathlib import Path
import re
import numpy as np
import cv2

# -----------------------------
# Config
# -----------------------------
INPUT_DIR = Path("abdomen_png")
OUT_ROOT  = Path("threshold_images")

WRITE_VOLUME   = True
WRITE_RESLICES = True

# Spacing (mm)
XY_SPACING_MM = 0.33
Z_SPACING_MM  = 1.0   # <- you said 1mm intervals

# Sorting pattern (e.g., a_vm1455.png)
_vm_re = re.compile(r"_vm(\d+)\.png$", re.IGNORECASE)

def sorted_pngs(folder: Path):
    paths = list(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNGs found in {folder}")
    def key(p: Path):
        m = _vm_re.search(p.name)
        return int(m.group(1)) if m else 10**12
    return sorted(paths, key=key)

def ensure_dirs():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "coronal_plane").mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "sagittal_plane").mkdir(parents=True, exist_ok=True)

def rescale_z_axis(img, z_to_xy_scale: float):
    # img is (Z, W, 3) or (Z, H, 3) where Z is the "vertical" axis.
    if z_to_xy_scale == 1.0:
        return img
    new_h = max(1, int(round(img.shape[0] * z_to_xy_scale)))
    return cv2.resize(img, (img.shape[1], new_h), interpolation=cv2.INTER_LINEAR)

def main():
    ensure_dirs()
    paths = sorted_pngs(INPUT_DIR)
    Z = len(paths)
    print(f"Found {Z} axial PNGs")

    # Read first image to set size
    first = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"Failed to read {paths[0]}")
    if first.ndim == 3 and first.shape[2] == 4:
        first = first[:, :, :3]
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Expected 24-bit color images (3 channels).")

    H, W = first.shape[:2]  # OpenCV uses (H,W)
    print(f"Slice size: {W}x{H} (color)")

    # Allocate volume: V[z,y,x,3] uint8 (BGR)
    V = np.empty((Z, H, W, 3), dtype=np.uint8)
    V[0] = first

    # Load remaining slices
    for i in range(1, Z):
        img = cv2.imread(str(paths[i]), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read {paths[i]}")
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if img.shape[:2] != (H, W):
            raise ValueError(f"Size mismatch at {paths[i].name}: got {img.shape[:2]}, expected {(H,W)}")
        V[i] = img
        if i % 25 == 0 or i == Z - 1:
            print(f"  loaded {i+1}/{Z}: {paths[i].name}")

    # Save single volume (memory-mappable)
    if WRITE_VOLUME:
        vol_path = OUT_ROOT / "volume_uint8.npy"
        meta_path = OUT_ROOT / "volume_meta.npz"
        np.save(vol_path, V)
        np.savez(
            meta_path,
            width_px=W,
            height_px=H,
            num_slices=Z,
            xy_spacing_mm=XY_SPACING_MM,
            z_spacing_mm=Z_SPACING_MM,
            channels=3,
            channel_order="BGR",
            dtype="uint8",
            layout="Z,Y,X,C",
        )
        print(f"\nWrote volume: {vol_path}")
        print(f"Wrote meta:   {meta_path}")
        print("Load fast with: V = np.load('threshold_images/volume_uint8.npy', mmap_mode='r')")

    if not WRITE_RESLICES:
        print("\nDone (no reslices).")
        return

    # Aspect correction factor for Z vs XY
    z_to_xy_scale = float(Z_SPACING_MM / XY_SPACING_MM)  # ~3.03

    out_cor = OUT_ROOT / "coronal_plane"
    out_sag = OUT_ROOT / "sagittal_plane"

    # CORONAL (frontal): fix y => V[:, y, :, :] => (Z, W, 3)
    print("\nWriting coronal (frontal) plane slices...")
    for y in range(H):
        cor = V[:, y, :, :]                 # (Z, W, 3)
        cor = rescale_z_axis(cor, z_to_xy_scale)
        cv2.imwrite(str(out_cor / f"coronal_y{y:04d}.png"), cor)
        if y % 60 == 0 or y == H - 1:
            print(f"  {y+1}/{H}")

    # SAGITTAL: fix x => V[:, :, x, :] => (Z, H, 3)
    print("\nWriting sagittal plane slices...")
    for x in range(W):
        sag = V[:, :, x, :]                 # (Z, H, 3)
        sag = rescale_z_axis(sag, z_to_xy_scale)
        cv2.imwrite(str(out_sag / f"sagittal_x{x:04d}.png"), sag)
        if x % 128 == 0 or x == W - 1:
            print(f"  {x+1}/{W}")

    print(f"\nReslices written to:\n  {out_cor}\n  {out_sag}")
    print("Done.")

if __name__ == "__main__":
    main()