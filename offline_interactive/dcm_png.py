"""
Convert all .dcm files found recursively under the VHP-F folder
into numbered PNGs inside  mri_photos/

Usage:
    python dcm_to_png.py

Output files are named  mri_photos/mri_0000.png, mri_0001.png, ...
sorted by their DICOM InstanceNumber (slice position) when available,
otherwise by filename.
"""

import numpy as np
from pathlib import Path
from PIL import Image

try:
    import pydicom
except ImportError:
    raise SystemExit("Run:  pip install pydicom pillow numpy")


# ── config ─────────────────────────────────────────────────────────────────────

SRC  = Path(r"images\nlm_visible_human_project\VHP-F\1.3.6.1.4.1.5962.1.2.2716.1672334394.26545")
DST  = Path("mri_photos")

# Abdomen MRI soft-tissue window  (adjust if you want bone / other contrast)
# Set WINDOW_CENTER / WINDOW_WIDTH to None to use the full pixel range instead.
WINDOW_CENTER = None   # e.g. 50  for T1 soft tissue, or None = auto
WINDOW_WIDTH  = None   # e.g. 350

# ── helpers ────────────────────────────────────────────────────────────────────

def apply_window(arr: np.ndarray, center, width) -> np.ndarray:
    lo = center - width / 2
    hi = center + width / 2
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def auto_normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)


def dcm_sort_key(ds):
    """Sort key: prefer InstanceNumber, fall back to filename."""
    try:
        return int(ds.InstanceNumber)
    except AttributeError:
        return 0


def load_pixel_array(ds) -> np.ndarray:
    """Return rescaled float32 pixel array (handles RescaleSlope/Intercept)."""
    arr = ds.pixel_array.astype(np.float32)
    slope     = float(getattr(ds, "RescaleSlope",     1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return arr * slope + intercept


def to_uint8(arr: np.ndarray) -> np.ndarray:
    if WINDOW_CENTER is not None and WINDOW_WIDTH is not None:
        return apply_window(arr, WINDOW_CENTER, WINDOW_WIDTH)

    # Try DICOM-embedded window values as fallback
    # (many MR files store WindowCenter / WindowWidth)
    try:
        wc = float(arr.mean())   # placeholder; overwritten below
        raise AttributeError    # force auto path for now
    except AttributeError:
        return auto_normalize(arr)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    DST.mkdir(parents=True, exist_ok=True)

    dcm_files = sorted(SRC.rglob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found under {SRC}")

    print(f"Found {len(dcm_files)} DICOM file(s) under:\n  {SRC}\n")

    # Load all headers first so we can sort by slice position
    datasets = []
    for path in dcm_files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=False)
            ds._source_path = path          # stash path for later
            datasets.append(ds)
        except Exception as e:
            print(f"  SKIP {path.name}: {e}")

    # Sort by InstanceNumber (slice order)
    datasets.sort(key=dcm_sort_key)

    # Convert
    pad = len(str(len(datasets)))
    ok  = 0
    for idx, ds in enumerate(datasets):
        try:
            arr  = load_pixel_array(ds)
            u8   = to_uint8(arr)

            # Handle multi-frame (some MR files pack all slices in one DCM)
            if u8.ndim == 3 and u8.shape[0] > 1:
                for frame_idx, frame in enumerate(u8):
                    out = DST / f"mri_{ok:0{pad}d}.png"
                    Image.fromarray(frame).convert("RGB").save(out)
                    ok += 1
            else:
                if u8.ndim == 3:
                    u8 = u8[0]          # single-frame stored as (1,H,W)
                out = DST / f"mri_{ok:0{pad}d}.png"
                Image.fromarray(u8).convert("RGB").save(out)
                ok += 1

            print(f"  [{idx+1}/{len(datasets)}] {ds._source_path.name} "
                  f"→ {out.name}  shape={arr.shape}")

        except Exception as e:
            print(f"  ERROR {ds._source_path.name}: {e}")

    print(f"\nDone — {ok} PNG(s) saved to  {DST.resolve()}")


if __name__ == "__main__":
    main()