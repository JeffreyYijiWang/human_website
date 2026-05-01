from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from urllib.parse import urlparse
from io import BytesIO
from typing import Any

import numpy as np
import requests


class VolumeConfigError(RuntimeError):
    """Raised when environment configuration is invalid."""


class VolumeLoadError(RuntimeError):
    """Raised when remote volume/meta loading fails."""


@dataclass(frozen=True)
class VolumeBundle:
    shape: tuple[int, int, int]
    dtype: np.dtype
    fortran_order: bool
    data_offset: int
    volume_url: str
    meta: dict[str, Any]


def _get_env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if not value:
        raise VolumeConfigError(f"Missing required environment variable: {name}")
    return value


def _resolve_meta_url() -> str:
    explicit = os.getenv("VOLUME_META_URL")
    if explicit:
        return explicit

    blob_base = os.getenv("VERCEL_BLOB_PUBLIC_BASE")
    if blob_base:
        base = blob_base.rstrip("/")
        meta_name = os.getenv("VOLUME_META_FILENAME", "volume_meta.npz")
        return f"{base}/{meta_name}"

    raise VolumeConfigError(
        "Set VOLUME_META_URL, or set VERCEL_BLOB_PUBLIC_BASE (+ optional VOLUME_META_FILENAME)."
    )


def _parse_npy_header(url: str) -> tuple[tuple[int, ...], np.dtype, bool, int]:
    # First bytes include magic/version + header length. Fetch 64 KiB to reliably include header.
    res = requests.get(url, headers={"Range": "bytes=0-65535"}, timeout=60)
    if res.status_code not in (200, 206):
        raise VolumeLoadError(f"Failed reading NPY header: HTTP {res.status_code}")
    buf = BytesIO(res.content)
    try:
        version = np.lib.format.read_magic(buf)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(buf)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(buf)
        else:
            raise VolumeLoadError(f"Unsupported NPY version: {version}")
    except Exception as exc:  # noqa: BLE001
        raise VolumeLoadError(f"Could not parse NPY header: {exc}") from exc

    return shape, dtype, fortran_order, buf.tell()


def _load_meta(meta_url: str) -> dict[str, Any]:
    res = requests.get(meta_url, timeout=120)
    if res.status_code != 200:
        raise VolumeLoadError(f"Failed loading meta from {meta_url}: HTTP {res.status_code}")

    try:
        with np.load(BytesIO(res.content), allow_pickle=True) as data:
            # Normalize npz into JSON-serializable payload for UI.
            out: dict[str, Any] = {}
            for key in data.files:
                value = data[key]
                if isinstance(value, np.ndarray):
                    out[key] = value.tolist()
                else:
                    out[key] = value
            return out
    except Exception:
        # fallback for JSON payloads if user stores metadata as json
        try:
            return json.loads(res.text)
        except Exception as exc:  # noqa: BLE001
            raise VolumeLoadError(f"Could not parse metadata payload: {exc}") from exc


def _default_model_id_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = parsed.path.rsplit('/', 1)[-1] or 'default'
    return name.replace('.npy', '').replace('.', '_')


def list_models() -> list[dict[str, str]]:
    config_json = os.getenv("MODEL_CONFIG_JSON")
    if config_json:
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as exc:
            raise VolumeConfigError(f"MODEL_CONFIG_JSON is not valid JSON: {exc}") from exc
        if not isinstance(config, list):
            raise VolumeConfigError("MODEL_CONFIG_JSON must be a JSON list of model objects")
        models: list[dict[str, str]] = []
        for item in config:
            if not isinstance(item, dict) or not item.get("id") or not item.get("volume_url"):
                raise VolumeConfigError("Each model must include id and volume_url")
            models.append({"id": str(item["id"]), "label": str(item.get("label", item["id"]))})
        if not models:
            raise VolumeConfigError("MODEL_CONFIG_JSON cannot be empty")
        return models

    volume_url = _get_env(
        "VOLUME_UINT8_URL",
        "https://pub-541cfda29b1c457b8b1b8dc2e2cc2956.r2.dev/volume_uint8.npy",
    )
    model_id = os.getenv("VOLUME_MODEL_ID", _default_model_id_from_url(volume_url))
    model_label = os.getenv("VOLUME_MODEL_LABEL", model_id)
    return [{"id": model_id, "label": model_label}]


def _resolve_model(model_id: str | None) -> tuple[str, str]:
    config_json = os.getenv("MODEL_CONFIG_JSON")
    if config_json:
        config = json.loads(config_json)
        selected = config[0] if model_id is None else next((m for m in config if str(m.get("id")) == model_id), None)
        if not selected:
            raise VolumeConfigError(f"Unknown model id: {model_id}")
        volume_url = selected.get("volume_url")
        meta_url = selected.get("meta_url")
        if not volume_url or not meta_url:
            raise VolumeConfigError("Each model must include volume_url and meta_url")
        return volume_url, meta_url

    volume_url = _get_env(
        "VOLUME_UINT8_URL",
        "https://pub-541cfda29b1c457b8b1b8dc2e2cc2956.r2.dev/volume_uint8.npy",
    )
    return volume_url, _resolve_meta_url()


@lru_cache(maxsize=1)
def load_bundle(model_id: str | None = None) -> VolumeBundle:
    volume_url, meta_url = _resolve_model(model_id)
    shape, dtype, fortran_order, data_offset = _parse_npy_header(volume_url)
    if len(shape) != 3:
        raise VolumeLoadError(f"Expected 3D volume, got shape={shape}")
    if dtype != np.uint8:
        raise VolumeLoadError(f"Expected uint8 volume, got dtype={dtype}")

    meta = _load_meta(meta_url)

    return VolumeBundle(
        shape=tuple(int(s) for s in shape),
        dtype=dtype,
        fortran_order=fortran_order,
        data_offset=int(data_offset),
        volume_url=volume_url,
        meta=meta,
    )


def _range_get(url: str, start: int, end_inclusive: int) -> bytes:
    headers = {"Range": f"bytes={start}-{end_inclusive}"}
    res = requests.get(url, headers=headers, timeout=180)
    if res.status_code not in (200, 206):
        raise VolumeLoadError(f"Range request failed: HTTP {res.status_code}")
    return res.content


def get_slice(axis: str, index: int, model_id: str | None = None) -> np.ndarray:
    bundle = load_bundle(model_id)
    z, y, x = bundle.shape
    itemsize = int(bundle.dtype.itemsize)

    if axis == "z":
        if not (0 <= index < z):
            raise IndexError(f"index out of range for z axis: {index} not in [0, {z - 1}]")
        n = y * x
        start = bundle.data_offset + (index * n * itemsize)
        end = start + (n * itemsize) - 1
        raw = _range_get(bundle.volume_url, start, end)
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((y, x))
        return arr

    if axis == "y":
        if not (0 <= index < y):
            raise IndexError(f"index out of range for y axis: {index} not in [0, {y - 1}]")
        rows = []
        for z_idx in range(z):
            plane_base = bundle.data_offset + (z_idx * y * x * itemsize)
            row_start = plane_base + (index * x * itemsize)
            row_end = row_start + (x * itemsize) - 1
            rows.append(np.frombuffer(_range_get(bundle.volume_url, row_start, row_end), dtype=np.uint8))
        return np.vstack(rows)

    if axis == "x":
        if not (0 <= index < x):
            raise IndexError(f"index out of range for x axis: {index} not in [0, {x - 1}]")
        cols = []
        for z_idx in range(z):
            row_vals = np.empty(y, dtype=np.uint8)
            plane_base = bundle.data_offset + (z_idx * y * x * itemsize)
            for y_idx in range(y):
                off = plane_base + (y_idx * x + index) * itemsize
                row_vals[y_idx] = np.frombuffer(_range_get(bundle.volume_url, off, off), dtype=np.uint8)[0]
            cols.append(row_vals)
        return np.vstack(cols)

    raise ValueError("axis must be one of x,y,z")
