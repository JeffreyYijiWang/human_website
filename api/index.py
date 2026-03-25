import json
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image, ImageEnhance

from slider_heap_image import (
    VolumeConfigError,
    VolumeLoadError,
    get_slice,
    list_models,
    load_bundle,
)

app = FastAPI(title="Volume Visualizer API")


def _parse_polyline(polyline: str | None) -> list[tuple[float, float]]:
    if not polyline:
        return []
    try:
        data = json.loads(polyline)
        points = []
        for p in data:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                x = float(p[0])
                y = float(p[1])
                points.append((x, y))
        return points[:500]
    except (ValueError, TypeError):
        return []


def _distance_to_segment(px: np.ndarray, py: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> np.ndarray:
    ax, ay = a
    bx, by = b
    abx, aby = bx - ax, by - ay
    denom = (abx * abx + aby * aby) + 1e-12
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = np.clip(t, 0.0, 1.0)
    cx = ax + t * abx
    cy = ay + t * aby
    return np.hypot(px - cx, py - cy)


def _apply_curve_heap(image_arr: np.ndarray, points: list[tuple[float, float]], heap_depth: float, curve_mode: str) -> np.ndarray:
    if len(points) < 2:
        return image_arr

    h, w = image_arr.shape
    px, py = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dist = np.full((h, w), np.inf, dtype=np.float32)

    for i in range(1, len(points)):
        x0, y0 = points[i - 1]
        x1, y1 = points[i]
        a = (x0 * (w - 1), y0 * (h - 1))
        b = (x1 * (w - 1), y1 * (h - 1))
        d = _distance_to_segment(px, py, a, b)
        dist = np.minimum(dist, d)

    sigma = max(2.0, min(w, h) * 0.04)
    line_mask = np.exp(-(dist * dist) / (2.0 * sigma * sigma))

    arr = image_arr.astype(np.float32)
    if curve_mode == "surface":
        # Closed polyline approximates a freehand surface region.
        poly = np.array([(x * (w - 1), y * (h - 1)) for x, y in points], dtype=np.float32)
        inside = np.zeros((h, w), dtype=np.float32)
        # even-odd rule ray cast
        for y in range(h):
            yy = y + 0.5
            inter = []
            for i in range(len(poly)):
                x1, y1 = poly[i - 1]
                x2, y2 = poly[i]
                if (y1 > yy) != (y2 > yy):
                    x = x1 + (yy - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
                    inter.append(x)
            inter.sort()
            for j in range(0, len(inter), 2):
                if j + 1 < len(inter):
                    x0, x1 = int(max(0, inter[j])), int(min(w - 1, inter[j + 1]))
                    inside[y, x0 : x1 + 1] = 1.0
        mask = np.clip(inside + line_mask * 0.6, 0.0, 1.0)
    else:
        mask = line_mask

    gain = 50.0 + 130.0 * heap_depth
    arr = np.clip(arr + (mask * gain), 0.0, 255.0)
    return arr.astype(np.uint8)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/models")
def models() -> JSONResponse:
    try:
        names = list_models()
    except VolumeConfigError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return JSONResponse({"models": names})


@app.get("/api/volume-meta")
def volume_meta(model: str = Query("default")) -> JSONResponse:
    try:
        bundle = load_bundle(model=model)
    except (VolumeConfigError, VolumeLoadError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    shape = list(bundle.volume.shape)
    return JSONResponse({"shape": shape, "meta": bundle.meta})


@app.get("/api/slice")
def volume_slice(
    axis: str = Query("z", pattern="^(x|y|z)$"),
    index: int = Query(0, ge=0),
    model: str = Query("default"),
    yaw: float = Query(0.0, ge=-180.0, le=180.0),
    pitch: float = Query(0.0, ge=-90.0, le=90.0),
    heap_depth: float = Query(0.5, ge=0.0, le=1.0),
    curve_mode: str = Query("curve", pattern="^(curve|surface)$"),
    polyline: str | None = Query(None),
) -> Response:
    try:
        slice_2d = get_slice(axis=axis, index=index, model=model)
    except (VolumeConfigError, VolumeLoadError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    points = _parse_polyline(polyline)
    if points:
        slice_2d = _apply_curve_heap(slice_2d, points, heap_depth, curve_mode)

    image = Image.fromarray(slice_2d, mode="L")
    image = image.rotate(yaw, resample=Image.Resampling.BILINEAR)
    if abs(pitch) > 1e-6:
        h_shift = int((pitch / 90.0) * max(1, image.height // 8))
        image = image.transform(
            image.size,
            Image.Transform.AFFINE,
            (1, 0, 0, 0, 1, h_shift),
            resample=Image.Resampling.BILINEAR,
        )

    contrast = 0.75 + (heap_depth * 1.25)
    image = ImageEnhance.Contrast(image).enhance(contrast)

    buf = BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


public_dir = Path(__file__).resolve().parent.parent / "public"
if public_dir.exists():
    app.mount("/", StaticFiles(directory=str(public_dir), html=True), name="public")
