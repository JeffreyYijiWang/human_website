import argparse
import re
from pathlib import Path

import numpy as np
from PIL import Image

_vm_re = re.compile(r"_vm(\d+)\.png$", re.IGNORECASE)

def sorted_pngs(folder: Path):
    paths = list(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNGs found in {folder}")
    def key(p: Path):
        m = _vm_re.search(p.name)
        return int(m.group(1)) if m else 10**12
    return sorted(paths, key=key)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_file", required=True)
    ap.add_argument("--max_layers", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = sorted_pngs(in_dir)
    if args.max_layers and args.max_layers > 0:
        paths = paths[:args.max_layers]

    first = Image.open(paths[0]).convert("RGBA")
    w, h = first.size
    n = len(paths)

    # file layout = raw uint8 array of shape (n, h, w, 4)
    mm = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(n, h, w, 4))

    for i, p in enumerate(paths):
        im = Image.open(p).convert("RGBA")
        if im.size != (w, h):
            im = im.resize((w, h), Image.BILINEAR)
        arr = np.array(im, dtype=np.uint8, copy=True)
        arr[..., 3] = 255
        mm[i] = arr
        if (i % 25) == 0 or i == n - 1:
            print(f"[pack] {i+1}/{n} {p.name}")

    mm.flush()
    print(f"[pack] wrote {out_path}  shape=({n},{h},{w},4)  bytes={out_path.stat().st_size}")

if __name__ == "__main__":
    main()