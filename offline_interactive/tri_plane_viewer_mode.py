"""Offline tri-plane viewer mode for frontal/sagittal/transverse visualization.

This module adds a split-screen renderer (3 panels) and mouse-linked crosshair
for coordinate relationships across orthogonal planes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


class TriPlaneViewer:
    def __init__(self, volume: np.ndarray):
        if volume.ndim == 4 and volume.shape[-1] == 3:
            self.volume = volume.mean(axis=-1).astype(np.uint8)
        elif volume.ndim == 3:
            self.volume = volume.astype(np.uint8)
        else:
            raise ValueError("Volume must be (Z,H,W) or (Z,H,W,3)")
        self.z, self.h, self.w = self.volume.shape

    def _extract(self, z: int, y: int, x: int):
        frontal = self.volume[np.clip(z, 0, self.z - 1)]
        sagittal = self.volume[:, :, np.clip(x, 0, self.w - 1)]
        transverse = self.volume[:, np.clip(y, 0, self.h - 1), :]
        return frontal, sagittal, transverse

    def render_triptych(self, z: int, y: int, x: int, out_path: Path, scale: int = 1) -> None:
        frontal, sagittal, transverse = self._extract(z, y, x)

        panels = [frontal, sagittal, transverse]
        pil = []
        for p in panels:
            im = Image.fromarray(p, mode="L").convert("RGB")
            if scale > 1:
                im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
            pil.append(im)

        canvas_w = sum(im.width for im in pil)
        canvas_h = max(im.height for im in pil)
        canvas = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))

        xoff = 0
        labels = ["Frontal", "Sagittal", "Transverse"]
        for im, label in zip(pil, labels):
            canvas.paste(im, (xoff, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((xoff + 8, 8), label, fill=(255, 64, 64))
            xoff += im.width

        # mouse-linked guide lines (simple crosshair projection)
        draw = ImageDraw.Draw(canvas)
        fw = pil[0].width
        sw = pil[1].width

        sx = int(np.clip(x * scale, 0, pil[0].width - 1))
        sy = int(np.clip(y * scale, 0, pil[0].height - 1))
        sz = int(np.clip(z * scale, 0, max(pil[1].height - 1, 1)))

        draw.line([(sx, 0), (sx, pil[0].height)], fill=(0, 255, 0))
        draw.line([(0, sy), (pil[0].width, sy)], fill=(0, 255, 0))

        # sagittal panel: axes are z (vertical) and y (horizontal)
        sag_x0 = fw
        draw.line([(sag_x0 + sy, 0), (sag_x0 + sy, pil[1].height)], fill=(0, 255, 255))
        draw.line([(sag_x0, sz), (sag_x0 + pil[1].width, sz)], fill=(0, 255, 255))

        # transverse panel: axes are x (horizontal) and z (vertical)
        tr_x0 = fw + sw
        draw.line([(tr_x0 + sx, 0), (tr_x0 + sx, pil[2].height)], fill=(255, 255, 0))
        draw.line([(tr_x0, sz), (tr_x0 + pil[2].width, sz)], fill=(255, 255, 0))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a 3-up orthogonal plane view from a volume.")
    ap.add_argument("--volume", type=Path, required=True, help="Path to .npy volume (Z,H,W or Z,H,W,3)")
    ap.add_argument("--z", type=int, default=0)
    ap.add_argument("--y", type=int, default=0)
    ap.add_argument("--x", type=int, default=0)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--output", type=Path, default=Path("out/triplane.png"))
    args = ap.parse_args()

    vol = np.load(args.volume)
    viewer = TriPlaneViewer(vol)
    viewer.render_triptych(args.z, args.y, args.x, args.output, scale=args.scale)


if __name__ == "__main__":
    main()
