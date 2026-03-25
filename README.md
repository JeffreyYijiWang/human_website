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
