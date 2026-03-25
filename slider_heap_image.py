"""Helpers for loading and slicing 3D uint8 volumes for web visualization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import json
import os
from typing import Any

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
        volume_bytes = _download_bytes(volume_url)
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
