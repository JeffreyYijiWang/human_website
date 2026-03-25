# human_website

Vercel-ready volume visualizer that loads `volume_uint8.npy` and `volume_meta.npz` from remote object storage.

## Why object storage instead of GitHub for volume files?

For large binary arrays (`.npy`, `.npz`), use **object storage** (not a relational database):

- Vercel Blob (easy with Vercel)
- AWS S3
- Cloudflare R2
- Google Cloud Storage

These are better for large files than GitHub repo storage and also simpler than putting binary blobs in SQL databases.

## Deploy to Vercel

1. Push this repo to GitHub (code only, not giant volume binaries).
2. Upload these files to object storage:
   - `volume_uint8.npy`
   - `volume_meta.npz`
3. Copy their public/signed URLs.
4. In Vercel project settings → Environment Variables, add:
   - `VOLUME_UINT8_URL`
   - `VOLUME_META_URL`
5. Deploy.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.index:app --reload
```

Then open `http://127.0.0.1:8000` for the static UI and call:

- `GET /api/health`
- `GET /api/volume-meta`
- `GET /api/slice?axis=z&index=0`

## Notes

- `slider_heap_image.py` loads and caches the remote volume/meta data.
- The API converts slices to grayscale PNGs for browser display.
- If your object storage requires signed URLs, store those full URLs in Vercel env vars.


## SpaceMouse support (desktop app only)

Vercel serverless functions cannot read USB HID devices, so SpaceMouse input must run in your local desktop OpenGL viewer process.

This repo now includes `spacemouse_integration.py` with a reusable `SpaceMouseController` that maps SpaceMouse axes to viewer state (`yaw`, `pitch`, `center`, `heap_depth`, `scale`) using deadzones and gains.

Install locally:

```bash
pip install pyspacemouse
```

In your desktop render loop, call:

```python
self._apply_held_keys(frame_time)
self.spacemouse_controller.apply(frame_time)
```

This follows the mapping you requested:

- twist (yaw axis) -> plane yaw
- tilt (roll axis) -> plane pitch
- push/pull (z) -> normal traversal + heap depth
- slide x/y -> in-plane movement on `u`/`v`


## Brownian/Perlin-style autopilot + 10-minute recording

The web UI now includes:

- Brownian motion toggle that animates slice index (plane move), yaw, pitch, and heap depth.
- A pseudo-Perlin layer mixed with Brownian drift for smoother random motion.
- A **"Record 10 min video"** button that captures the visualization from the canvas using `MediaRecorder` and downloads a `.webm` file after 10 minutes.

The API endpoint `/api/slice` accepts optional parameters so motion controls are reflected server-side:

- `yaw` in `[-180, 180]`
- `pitch` in `[-90, 90]`
- `heap_depth` in `[0, 1]`


## Multi-model switching

You can now configure multiple volumes and switch between them in the UI.

Set one env var in Vercel:

```json
MODEL_CONFIG_JSON={
  "brain_a": {"volume_url": "https://.../volume_a.npy", "meta_url": "https://.../meta_a.npz"},
  "brain_b": {"volume_url": "https://.../volume_b.npy", "meta_url": "https://.../meta_b.npz"}
}
```

API endpoints:

- `GET /api/models`
- `GET /api/volume-meta?model=brain_a`
- `GET /api/slice?model=brain_b&axis=z&index=20`

## Polyline-driven curve/surface heap map

The canvas now supports drawing a freehand polyline.

- **Curve mode**: uses the polyline as a 3D-curve-like influence line on the slice.
- **Surface mode**: treats a closed polyline region as a surface mask and boosts that region.

The UI sends the polyline to `/api/slice` with `curve_mode=curve|surface` and `polyline=[[x0,y0],...]` in normalized coordinates.
