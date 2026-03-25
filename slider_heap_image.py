"""Helpers for loading and slicing 3D uint8 volumes for web visualization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO
import hashlib
import hmac
import json
import os
from typing import Any
from urllib.parse import quote

import numpy as np
import requests


@dataclass(frozen=True)
class VolumeBundle:
    volume: np.ndarray
    meta: dict[str, Any]


class VolumeConfigError(RuntimeError):
    """Raised when required volume configuration is missing or invalid."""


class VolumeLoadError(RuntimeError):
    """Raised when remote volume assets could not be downloaded/parsed."""


def _download_bytes(url: str, timeout: int = 30) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def _download_r2_bytes(bucket: str, key: str) -> bytes:
    endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
    access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()

    if not endpoint or not access_key or not secret_key:
        raise VolumeConfigError(
            "Missing R2 credentials. Set R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY."
        )

    encoded_key = quote(key, safe="/-_.~")
    object_url = f"{endpoint.rstrip('/')}/{bucket}/{encoded_key}"

    now = datetime.now(timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    payload_hash = hashlib.sha256(b"").hexdigest()

    canonical_headers = (
        f"host:{endpoint.removeprefix('https://').removeprefix('http://').rstrip('/')}\n"
        f"x-amz-content-sha256:{payload_hash}\n"
        f"x-amz-date:{amz_date}\n"
    )
    signed_headers = "host;x-amz-content-sha256;x-amz-date"
    canonical_request = (
        f"GET\n/{bucket}/{encoded_key}\n\n"
        f"{canonical_headers}\n{signed_headers}\n{payload_hash}"
    )
    algorithm = "AWS4-HMAC-SHA256"
    credential_scope = f"{date_stamp}/auto/s3/aws4_request"
    string_to_sign = (
        f"{algorithm}\n{amz_date}\n{credential_scope}\n"
        f"{hashlib.sha256(canonical_request.encode()).hexdigest()}"
    )

    def _sign(k: bytes, msg: str) -> bytes:
        return hmac.new(k, msg.encode(), hashlib.sha256).digest()

    k_date = _sign(("AWS4" + secret_key).encode(), date_stamp)
    k_region = _sign(k_date, "auto")
    k_service = _sign(k_region, "s3")
    signing_key = _sign(k_service, "aws4_request")
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()

    authorization = (
        f"{algorithm} Credential={access_key}/{credential_scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    headers = {
        "x-amz-date": amz_date,
        "x-amz-content-sha256": payload_hash,
        "Authorization": authorization,
    }
    try:
        response = requests.get(object_url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.content
    except requests.RequestException as exc:
        raise VolumeLoadError(f"Failed to download R2 object s3://{bucket}/{key}.") from exc


def _download_volume_bytes(location: str) -> bytes:
    if location.startswith("r2://"):
        payload = location[len("r2://") :]
        if "/" not in payload:
            raise VolumeConfigError("R2 volume URL must be in format r2://<bucket>/<key>.")
        bucket, key = payload.split("/", 1)
        if not bucket or not key:
            raise VolumeConfigError("R2 volume URL must include bucket and key.")
        return _download_r2_bytes(bucket, key)
    return _download_bytes(location)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _model_registry() -> dict[str, tuple[str, str]]:
    """Return model_name -> (volume_url, meta_url)."""
    registry: dict[str, tuple[str, str]] = {}

    config_raw = os.getenv("MODEL_CONFIG_JSON", "").strip()
    if config_raw:
        try:
            payload = json.loads(config_raw)
        except json.JSONDecodeError as exc:
            raise VolumeConfigError("MODEL_CONFIG_JSON is not valid JSON") from exc

        if not isinstance(payload, dict):
            raise VolumeConfigError("MODEL_CONFIG_JSON must be an object")

        for name, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            vurl = entry.get("volume_url")
            murl = entry.get("meta_url")
            if isinstance(vurl, str) and isinstance(murl, str) and vurl and murl:
                registry[str(name)] = (vurl, murl)

    # Backward-compatible single-model fallback.
    default_v = os.getenv("VOLUME_UINT8_URL")
    default_m = os.getenv("VOLUME_META_URL")
    if default_v and default_m:
        registry.setdefault("default", (default_v, default_m))

    # Opinionated fallback for this deployment:
    # - volume from Cloudflare R2 via authenticated r2:// URI
    # - meta from public Vercel Blob URLs
    registry.setdefault(
        "default",
        (
            "r2://humanvisualization/volume_uint8.npy",
            "https://11srgbl8ig1rod0l.public.blob.vercel-storage.com/threshold_images/volume_meta.npz",
        ),
    )
    registry.setdefault(
        "male",
        (
            "r2://humanvisualization/volume_uint8.npy",
            "https://11srgbl8ig1rod0l.public.blob.vercel-storage.com/threshold_images/male_meta.npz",
        ),
    )

    if not registry:
        raise VolumeConfigError(
            "No volume models configured. Set MODEL_CONFIG_JSON or VOLUME_UINT8_URL/VOLUME_META_URL."
        )

    return registry


def list_models() -> list[str]:
    return sorted(_model_registry().keys())


@lru_cache(maxsize=8)
def load_bundle(model: str = "default") -> VolumeBundle:
    """Load and cache volume + meta from remote object storage URLs."""
    registry = _model_registry()
    if model not in registry:
        model = "default" if "default" in registry else next(iter(registry))

    volume_url, meta_url = registry[model]

    try:
        volume_bytes = _download_volume_bytes(volume_url)
        meta_bytes = _download_bytes(meta_url)

        volume = np.load(BytesIO(volume_bytes), allow_pickle=False)
        if volume.ndim != 3 or volume.dtype != np.uint8:
            raise VolumeLoadError(
                f"Expected 3D uint8 volume, got shape={volume.shape} dtype={volume.dtype}."
            )

        with np.load(BytesIO(meta_bytes), allow_pickle=False) as npz:
            meta = {key: _to_jsonable(npz[key]) for key in npz.files}

        meta["model_name"] = model
        return VolumeBundle(volume=volume, meta=meta)
    except (requests.RequestException, ValueError, OSError) as exc:
        raise VolumeLoadError("Failed to load volume assets from object storage.") from exc


def get_slice(axis: str, index: int, model: str = "default") -> np.ndarray:
    """Return a 2D uint8 slice from the loaded 3D volume."""
    bundle = load_bundle(model)
    volume = bundle.volume

    axis = axis.lower()
    axes = {"z": 0, "y": 1, "x": 2}
    if axis not in axes:
        raise ValueError("axis must be one of: x, y, z")

    axis_id = axes[axis]
    max_index = volume.shape[axis_id] - 1
    if index < 0 or index > max_index:
        raise IndexError(f"index out of range for axis {axis}: 0..{max_index}")

    if axis == "z":
        return volume[index, :, :]
    if axis == "y":
        return volume[:, index, :]
    return volume[:, :, index]
