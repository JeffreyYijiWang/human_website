"""Helpers for loading and slicing a 3D uint8 volume for web visualization.

This module is used by the Vercel FastAPI app in api/index.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
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


@lru_cache(maxsize=1)
def load_bundle() -> VolumeBundle:
    """Load and cache volume + meta from remote object storage URLs."""
    volume_url = os.getenv("VOLUME_UINT8_URL")
    meta_url = os.getenv("VOLUME_META_URL")

    if not volume_url or not meta_url:
        raise VolumeConfigError(
            "Missing VOLUME_UINT8_URL or VOLUME_META_URL environment variable."
        )

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

        return VolumeBundle(volume=volume, meta=meta)
    except (requests.RequestException, ValueError, OSError) as exc:
        raise VolumeLoadError("Failed to load volume assets from object storage.") from exc


def get_slice(axis: str, index: int) -> np.ndarray:
    """Return a 2D uint8 slice from the loaded 3D volume."""
    bundle = load_bundle()
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
