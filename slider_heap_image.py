"""Helpers for loading and slicing a 3D uint8 volume for web visualization.

This module is used by the Vercel FastAPI app in api/index.py.
It also includes local-only helpers for Brownian/procedural motion experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import argparse
import math
import os
import random
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


@dataclass
class MotionConfig:
    """Local Brownian + procedural curve modulation controls."""

    speed: float = 1.0
    slice_volatility: float = 25.0
    yaw_volatility: float = 90.0
    pitch_volatility: float = 70.0
    heap_volatility: float = 0.55

    slice_damping: float = 0.92
    yaw_damping: float = 0.90
    pitch_damping: float = 0.90
    heap_damping: float = 0.88


@dataclass
class MotionState:
    """Mutable Brownian state for local procedural modulation."""

    noise_time: float = 0.0
    velocity_slice: float = 0.0
    velocity_yaw: float = 0.0
    velocity_pitch: float = 0.0
    velocity_heap: float = 0.0


@dataclass(frozen=True)
class Local3DConnexionSettings:
    """Local 3Dconnexion toggle/settings loaded from environment variables."""

    enabled: bool
    deadzone_t: float
    deadzone_r: float
    yaw_gain: float
    pitch_gain: float
    move_n_gain: float
    move_u_gain: float
    move_v_gain: float
    heap_gain: float
    scale_gain: float


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


def _pseudo_perlin(t: float, freqs: tuple[float, float, float], phases: tuple[float, float, float]) -> float:
    (f1, f2, f3) = freqs
    (p1, p2, p3) = phases
    return (
        math.sin(t * f1 + p1) * 0.55
        + math.sin(t * f2 + p2) * 0.30
        + math.sin(t * f3 + p3) * 0.15
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def apply_brownian_procedural_modulation(
    *,
    state: MotionState,
    dt: float,
    max_slice: int,
    slice_index: float,
    yaw: float,
    pitch: float,
    heap_depth: float,
    config: MotionConfig | None = None,
    rng: random.Random | None = None,
) -> dict[str, float]:
    """Advance one Brownian + procedural curve modulation step.

    This mirrors the browser autopilot logic so local Python experiments can run
    with the same motion model as the UI.
    """
    cfg = config or MotionConfig()
    prng = rng or random

    speed = max(0.01, cfg.speed)
    state.noise_time += dt * speed

    state.velocity_slice += (prng.random() - 0.5) * dt * cfg.slice_volatility * speed
    state.velocity_yaw += (prng.random() - 0.5) * dt * cfg.yaw_volatility * speed
    state.velocity_pitch += (prng.random() - 0.5) * dt * cfg.pitch_volatility * speed
    state.velocity_heap += (prng.random() - 0.5) * dt * cfg.heap_volatility * speed

    state.velocity_slice *= cfg.slice_damping
    state.velocity_yaw *= cfg.yaw_damping
    state.velocity_pitch *= cfg.pitch_damping
    state.velocity_heap *= cfg.heap_damping

    slice_noise = _pseudo_perlin(state.noise_time, (0.9, 1.5, 2.2), (0.2, 1.1, 2.0)) * 3.0
    yaw_noise = _pseudo_perlin(state.noise_time, (0.6, 1.1, 1.7), (0.9, 1.8, 3.2)) * 4.2
    pitch_noise = _pseudo_perlin(state.noise_time, (0.7, 1.4, 2.4), (0.5, 2.5, 4.1)) * 2.9
    heap_noise = _pseudo_perlin(state.noise_time, (0.5, 1.0, 2.1), (1.3, 0.7, 2.8)) * 0.03

    next_slice = _clamp(slice_index + state.velocity_slice + slice_noise, 0.0, float(max_slice))
    next_yaw = _clamp(yaw + state.velocity_yaw + yaw_noise, -180.0, 180.0)
    next_pitch = _clamp(pitch + state.velocity_pitch + pitch_noise, -90.0, 90.0)
    next_heap = _clamp(heap_depth + state.velocity_heap + heap_noise, 0.0, 1.0)

    return {
        "slice": float(round(next_slice)),
        "yaw": float(next_yaw),
        "pitch": float(next_pitch),
        "heap_depth": float(next_heap),
    }


def generate_procedural_curve(samples: int = 120, cycles: float = 1.0) -> list[tuple[float, float, float]]:
    """Return normalized (x, y, z) curve points for local procedural camera paths."""
    samples = max(2, int(samples))
    points: list[tuple[float, float, float]] = []

    for i in range(samples):
        t = (i / (samples - 1)) * (2.0 * math.pi * max(0.01, cycles))
        x = 0.5 + 0.4 * math.cos(t)
        y = 0.5 + 0.3 * math.sin(t * 2.0)
        z = 0.5 + 0.2 * math.sin(t * 3.0 + 0.5)
        points.append((_clamp(x, 0.0, 1.0), _clamp(y, 0.0, 1.0), _clamp(z, 0.0, 1.0)))

    return points


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def load_local_3dconnexion_settings() -> Local3DConnexionSettings:
    """Load local 3Dconnexion tuning from environment variables.

    This is intentionally local-only (desktop apps) and mirrors defaults from
    ``spacemouse_integration.SpaceMouseConfig``.
    """
    return Local3DConnexionSettings(
        enabled=_env_bool("LOCAL_3DCONNEXION_ENABLED", True),
        deadzone_t=_env_float("LOCAL_3DCONNEXION_DEADZONE_T", 0.08),
        deadzone_r=_env_float("LOCAL_3DCONNEXION_DEADZONE_R", 0.08),
        yaw_gain=_env_float("LOCAL_3DCONNEXION_YAW_GAIN", 1.8),
        pitch_gain=_env_float("LOCAL_3DCONNEXION_PITCH_GAIN", 1.6),
        move_n_gain=_env_float("LOCAL_3DCONNEXION_MOVE_N_GAIN", 0.45),
        move_u_gain=_env_float("LOCAL_3DCONNEXION_MOVE_U_GAIN", 0.35),
        move_v_gain=_env_float("LOCAL_3DCONNEXION_MOVE_V_GAIN", 0.35),
        heap_gain=_env_float("LOCAL_3DCONNEXION_HEAP_GAIN", 0.50),
        scale_gain=_env_float("LOCAL_3DCONNEXION_SCALE_GAIN", 0.85),
    )


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


def _run_local_motion_demo(frames: int, fps: int, max_slice: int) -> None:
    dt = 1.0 / max(1, fps)
    state = MotionState()
    motion = {"slice": max_slice / 2.0, "yaw": 0.0, "pitch": 0.0, "heap_depth": 0.5}

    print("Local 3Dconnexion settings:", load_local_3dconnexion_settings())
    print("frame,slice,yaw,pitch,heap_depth")

    for frame in range(frames):
        motion = apply_brownian_procedural_modulation(
            state=state,
            dt=dt,
            max_slice=max_slice,
            slice_index=motion["slice"],
            yaw=motion["yaw"],
            pitch=motion["pitch"],
            heap_depth=motion["heap_depth"],
        )
        print(
            f"{frame},{motion['slice']:.0f},{motion['yaw']:.3f},"
            f"{motion['pitch']:.3f},{motion['heap_depth']:.4f}"
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local helpers for slider_heap_image motion experiments")
    parser.add_argument("--demo-motion", action="store_true", help="Run local Brownian/procedural motion demo")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames for --demo-motion")
    parser.add_argument("--fps", type=int, default=20, help="Frame rate for --demo-motion")
    parser.add_argument("--max-slice", type=int, default=255, help="Maximum slice index for --demo-motion")
    parser.add_argument("--curve-samples", type=int, default=12, help="Samples to generate for procedural curve output")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.demo_motion:
        _run_local_motion_demo(frames=max(1, args.frames), fps=max(1, args.fps), max_slice=max(1, args.max_slice))
    else:
        points = generate_procedural_curve(samples=args.curve_samples)
        print("Procedural curve points (first 5):")
        for point in points[:5]:
            print(point)


if __name__ == "__main__":
    main()
