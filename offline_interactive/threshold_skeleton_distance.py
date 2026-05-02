"""Utilities to preprocess threshold PNG slices for offline volume workflows.

Pipeline:
1) Convert to grayscale and threshold to binary (non-background => white).
2) Skeletonize the white mask.
3) Build a normalized distance gradient where center pixels are brighter.

Supports one image or all PNG images in a directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def threshold_binary(gray: np.ndarray, threshold: int, invert: bool = False) -> np.ndarray:
    if invert:
        return gray <= threshold
    return gray > threshold


def zhang_suen_thinning(binary: np.ndarray, max_iters: int = 256) -> np.ndarray:
    img = binary.astype(np.uint8).copy()
    h, w = img.shape
    for _ in range(max_iters):
        changed = False
        for step in (0, 1):
            to_remove = []
            for y in range(1, h - 1):
                for x in range(1, w - 1):
                    p1 = img[y, x]
                    if p1 != 1:
                        continue
                    p2, p3, p4 = img[y - 1, x], img[y - 1, x + 1], img[y, x + 1]
                    p5, p6, p7 = img[y + 1, x + 1], img[y + 1, x], img[y + 1, x - 1]
                    p8, p9 = img[y, x - 1], img[y - 1, x - 1]
                    nb = [p2, p3, p4, p5, p6, p7, p8, p9]
                    B = sum(nb)
                    if B < 2 or B > 6:
                        continue
                    A = sum((nb[i] == 0 and nb[(i + 1) % 8] == 1) for i in range(8))
                    if A != 1:
                        continue
                    if step == 0:
                        c1 = p2 * p4 * p6
                        c2 = p4 * p6 * p8
                    else:
                        c1 = p2 * p4 * p8
                        c2 = p2 * p6 * p8
                    if c1 == 0 and c2 == 0:
                        to_remove.append((y, x))
            if to_remove:
                changed = True
                for y, x in to_remove:
                    img[y, x] = 0
        if not changed:
            break
    return img.astype(bool)


def distance_transform_chamfer(binary: np.ndarray) -> np.ndarray:
    inf = 10**9
    dist = np.where(binary, inf, 0).astype(np.int32)
    h, w = dist.shape
    for y in range(h):
        for x in range(w):
            if dist[y, x] == 0:
                continue
            best = dist[y, x]
            if y > 0:
                best = min(best, dist[y - 1, x] + 3)
                if x > 0:
                    best = min(best, dist[y - 1, x - 1] + 4)
                if x + 1 < w:
                    best = min(best, dist[y - 1, x + 1] + 4)
            if x > 0:
                best = min(best, dist[y, x - 1] + 3)
            dist[y, x] = best
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if dist[y, x] == 0:
                continue
            best = dist[y, x]
            if y + 1 < h:
                best = min(best, dist[y + 1, x] + 3)
                if x > 0:
                    best = min(best, dist[y + 1, x - 1] + 4)
                if x + 1 < w:
                    best = min(best, dist[y + 1, x + 1] + 4)
            if x + 1 < w:
                best = min(best, dist[y, x + 1] + 3)
            dist[y, x] = best
    return dist.astype(np.float32)


def make_gradient(binary: np.ndarray) -> np.ndarray:
    d = distance_transform_chamfer(binary)
    d *= binary.astype(np.float32)
    maxv = float(d.max())
    if maxv <= 0:
        return np.zeros_like(d, dtype=np.uint8)
    return np.clip((d / maxv) * 255.0, 0, 255).astype(np.uint8)


def process_image(src: Path, out_dir: Path, threshold: int, invert: bool) -> None:
    gray = load_gray(src)
    binary = threshold_binary(gray, threshold=threshold, invert=invert)
    skeleton = zhang_suen_thinning(binary)
    gradient = make_gradient(binary)

    stem = src.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray((binary.astype(np.uint8) * 255), mode="L").save(out_dir / f"{stem}_binary.png")
    Image.fromarray((skeleton.astype(np.uint8) * 255), mode="L").save(out_dir / f"{stem}_skeleton.png")
    Image.fromarray(gradient, mode="L").save(out_dir / f"{stem}_distance_gradient.png")


def iter_pngs(path: Path) -> Iterable[Path]:
    yield from sorted([p for p in path.iterdir() if p.suffix.lower() == ".png"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Threshold + skeleton + distance-gradient for PNG slices.")
    ap.add_argument("--input", type=Path, required=True, help="Input PNG file or directory of PNG files")
    ap.add_argument("--output", type=Path, required=True, help="Output directory to store generated images")
    ap.add_argument("--threshold", type=int, default=10)
    ap.add_argument("--invert", action="store_true", help="Set when background is white")
    ap.add_argument("--all", action="store_true", help="Process all PNG files in input directory")
    args = ap.parse_args()

    if args.input.is_file():
        process_image(args.input, args.output, args.threshold, args.invert)
        return

    if args.input.is_dir() and args.all:
        for img in iter_pngs(args.input):
            process_image(img, args.output, args.threshold, args.invert)
        return

    raise SystemExit("Use --input <file.png> for one image, or --input <dir> --all for a directory.")


if __name__ == "__main__":
    main()
