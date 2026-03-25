import ctypes
import os
from pathlib import Path
import re
import numpy as np
import cv2

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
INPUT_ROOT = Path("images/male")
OUT_ROOT = Path("threshold_images")

WRITE_VOLUME = True

# Spacing (mm)
XY_SPACING_MM = 0.33
Z_SPACING_MM = 1.0

# Optional numeric extractor if filenames contain numbers.
# Example matches:
#   slice_001.png
#   a_vm1455.png
#   img12.png
_number_re = re.compile(r"(\d+)")

def ensure_dirs():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

def collect_pngs_recursive(folder: Path):
    paths = [p for p in folder.rglob("*.png") if p.is_file()]
    if not paths:
        raise FileNotFoundError(f"No PNGs found under {folder}")
    return paths

def filename_sort_key(p: Path):
    """
    Sort primarily by filename, with numeric awareness.
    Examples:
      img2.png < img10.png
      a_vm9.png < a_vm12.png
    If two files have the same name in different subfolders,
    the relative path is used as a tiebreaker.
    """
    stem = p.stem.lower()
    parts = re.split(r"(\d+)", stem)
    key = []
    for part in parts:
        if part.isdigit():
            key.append(int(part))
        else:
            key.append(part)
    # Use relative path as tiebreaker so ordering is stable
    return (key, str(p.relative_to(INPUT_ROOT)).lower())

def sorted_pngs(folder: Path):
    paths = collect_pngs_recursive(folder)
    return sorted(paths, key=filename_sort_key)

def main():
    ensure_dirs()

    paths = sorted_pngs(INPUT_ROOT)
    Z = len(paths)
    print(f"Found {Z} PNGs under {INPUT_ROOT}")

    # Show a few examples of the sorted order
    print("First few files in sorted order:")
    for p in paths[:10]:
        print(f"  {p}")

    # Read first image to determine volume size
    first = cv2.imread(str(paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"Failed to read {paths[0]}")

    # Drop alpha if present
    if first.ndim == 3 and first.shape[2] == 4:
        first = first[:, :, :3]

    # Ensure color image
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError(f"Expected 3-channel color PNG, got shape {first.shape} for {paths[0]}")

    H, W = first.shape[:2]
    print(f"Slice size: {W} x {H} (color)")

    # Allocate volume as (Z, H, W, 3), uint8, BGR
    V = np.empty((Z, H, W, 3), dtype=np.uint8)
    V[0] = first

    # Load remaining slices
    for i in range(1, Z):
        img = cv2.imread(str(paths[i]), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read {paths[i]}")

        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Expected 3-channel color PNG, got shape {img.shape} for {paths[i]}")

        if img.shape[:2] != (H, W):
            raise ValueError(
                f"Size mismatch at {paths[i]}: got {img.shape[:2]}, expected {(H, W)}"
            )

        V[i] = img

        if i % 25 == 0 or i == Z - 1:
            print(f"  loaded {i+1}/{Z}: {paths[i].name}")

    # Save only the volume + metadata
    if WRITE_VOLUME:
        vol_path = OUT_ROOT / "male_uint8.npy"
        meta_path = OUT_ROOT / "male_meta.npz"

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
            source_root=str(INPUT_ROOT),
        )

        print(f"\nWrote volume: {vol_path}")
        print(f"Wrote meta:   {meta_path}")
        print("Load with:")
        print("  V = np.load('threshold_images/volume_uint8.npy', mmap_mode='r')")

    print("\nDone.")

if __name__ == "__main__":
    main()