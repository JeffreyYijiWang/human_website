from io import BytesIO

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from PIL import Image

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
) -> Response:
    try:
        slice_2d = get_slice(axis=axis, index=index)
    except (VolumeConfigError, VolumeLoadError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image = Image.fromarray(slice_2d, mode="L")
    buf = BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
