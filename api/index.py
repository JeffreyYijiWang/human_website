from io import BytesIO

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageEnhance

from slider_heap_image import (
    VolumeConfigError,
    VolumeLoadError,
    get_slice,
    load_bundle,
)

app = FastAPI(title="Volume Visualizer API")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/volume-meta")
def volume_meta() -> JSONResponse:
    try:
        bundle = load_bundle()
    except (VolumeConfigError, VolumeLoadError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    shape = list(bundle.volume.shape)
    return JSONResponse({"shape": shape, "meta": bundle.meta})


@app.get("/api/slice")
def volume_slice(
    axis: str = Query("z", pattern="^(x|y|z)$"),
    index: int = Query(0, ge=0),
    yaw: float = Query(0.0, ge=-180.0, le=180.0),
    pitch: float = Query(0.0, ge=-90.0, le=90.0),
    heap_depth: float = Query(0.5, ge=0.0, le=1.0),
) -> Response:
    try:
        slice_2d = get_slice(axis=axis, index=index)
    except (VolumeConfigError, VolumeLoadError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image = Image.fromarray(slice_2d, mode="L")

    # lightweight visual transforms to support motion experiments from the UI
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
