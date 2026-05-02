"""Offline waypoint recorder and timeline interpolation helper.

Provides three non-overlapping hotkeys:
- C: camera/plane waypoint
- B: brush waypoint
- V: combined camera+brush waypoint

Writes JSON on close and can load JSON to preview an interpolated timeline path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import numpy as np


@dataclass
class CameraState:
    t: float
    position: list[float]
    euler_deg: list[float]
    plane_normal: list[float]


@dataclass
class BrushState:
    t: float
    mouse_uv: list[float]
    strength: float


class WaypointRecorder:
    def __init__(self) -> None:
        self.camera_waypoints: List[CameraState] = []
        self.brush_waypoints: List[BrushState] = []
        self.combined_waypoints: List[Dict[str, Any]] = []

    def on_key_c(self, state: CameraState) -> None:
        self.camera_waypoints.append(state)

    def on_key_b(self, state: BrushState) -> None:
        self.brush_waypoints.append(state)

    def on_key_v(self, cam: CameraState, brush: BrushState) -> None:
        self.combined_waypoints.append({"camera": asdict(cam), "brush": asdict(brush)})

    def save(self, out_json: Path) -> None:
        payload = {
            "camera_waypoints": [asdict(x) for x in self.camera_waypoints],
            "brush_waypoints": [asdict(x) for x in self.brush_waypoints],
            "combined_waypoints": self.combined_waypoints,
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cubic_catmull_rom(points: np.ndarray, num: int) -> np.ndarray:
    if len(points) < 2:
        return points
    out = []
    for i in range(len(points) - 1):
        p0 = points[max(i - 1, 0)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(i + 2, len(points) - 1)]
        for t in np.linspace(0.0, 1.0, num=num, endpoint=False):
            t2, t3 = t * t, t * t * t
            pt = 0.5 * ((2 * p1) + (-p0 + p2) * t + (2*p0 - 5*p1 + 4*p2 - p3) * t2 + (-p0 + 3*p1 - 3*p2 + p3) * t3)
            out.append(pt)
    out.append(points[-1])
    return np.array(out)


def add_noise(path: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return path
    return path + np.random.normal(0.0, sigma, size=path.shape)


def build_timeline(input_json: Path, out_json: Path, samples_per_segment: int, noise_sigma: float) -> None:
    data = json.loads(input_json.read_text(encoding="utf-8"))
    cams = data.get("camera_waypoints", [])
    if len(cams) < 2:
        raise ValueError("Need at least two camera waypoints for timeline interpolation.")

    pos = np.array([c["position"] for c in cams], dtype=np.float32)
    rot = np.array([c["euler_deg"] for c in cams], dtype=np.float32)

    pos_curve = cubic_catmull_rom(pos, samples_per_segment)
    rot_curve = cubic_catmull_rom(rot, samples_per_segment)
    pos_curve = add_noise(pos_curve, noise_sigma)

    timeline = [{"frame": i, "position": p.tolist(), "euler_deg": r.tolist()} for i, (p, r) in enumerate(zip(pos_curve, rot_curve))]
    out_json.write_text(json.dumps({"timeline": timeline, "interpolation": "catmull_rom"}, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Waypoint recorder/timeline utility for offline camera paths.")
    sub = ap.add_subparsers(dest="mode", required=True)

    p_interp = sub.add_parser("interpolate", help="Load waypoint JSON and generate interpolated timeline JSON")
    p_interp.add_argument("--input", type=Path, required=True)
    p_interp.add_argument("--output", type=Path, required=True)
    p_interp.add_argument("--samples-per-segment", type=int, default=16)
    p_interp.add_argument("--noise-sigma", type=float, default=0.0)

    args = ap.parse_args()

    if args.mode == "interpolate":
        build_timeline(args.input, args.output, args.samples_per_segment, args.noise_sigma)


if __name__ == "__main__":
    main()
