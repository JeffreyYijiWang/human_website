"""SpaceMouse integration helpers for a desktop MPR/OpenGL viewer.

This module is intentionally optional and is not used by the Vercel deployment.
Use it in a local desktop app (e.g., an existing ``MPRPlaneUI`` class) where a
3Dconnexion device is available over HID.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import pyspacemouse
except ImportError:  # pragma: no cover - optional dependency
    pyspacemouse = None


@dataclass
class SpaceMouseConfig:
    deadzone_t: float = 0.08
    deadzone_r: float = 0.08

    yaw_gain: float = 1.8
    pitch_gain: float = 1.6

    move_n_gain: float = 0.45
    move_u_gain: float = 0.35
    move_v_gain: float = 0.35

    heap_gain: float = 0.50
    scale_gain: float = 0.85


class SpaceMouseController:
    """Attach this to an existing viewer state object.

    The target object should expose:
      yaw, pitch, center, n, u, v, heap_depth, scale
    and methods:
      _update_plane_axes(), _push_slice_uniforms(), _update_gizmo_geometry()
    """

    def __init__(self, target: Any, config: SpaceMouseConfig | None = None) -> None:
        self.target = target
        self.config = config or SpaceMouseConfig()
        self.enabled = False

        if pyspacemouse is not None:
            try:
                self.enabled = bool(pyspacemouse.open())
            except Exception:
                self.enabled = False

    @staticmethod
    def _deadzone(x: float, dz: float) -> float:
        if abs(x) < dz:
            return 0.0
        s = 1.0 if x >= 0.0 else -1.0
        y = (abs(x) - dz) / max(1e-6, (1.0 - dz))
        return s * y

    def _clamp_center_and_sync(self) -> None:
        self.target.center[:] = np.clip(self.target.center, 0.0, 1.0)
        self.target.heap_depth = float(np.clip(self.target.heap_depth, 0.0, 1.0))
        self.target.pitch = float(np.clip(self.target.pitch, -1.55, 1.55))
        self.target.scale = float(np.clip(self.target.scale, 0.05, 2.0))

        self.target._update_plane_axes()
        self.target._push_slice_uniforms()
        self.target._update_gizmo_geometry()

    def apply(self, dt: float) -> None:
        if not self.enabled or pyspacemouse is None:
            return

        try:
            state = pyspacemouse.read()
        except Exception:
            return

        if state is None:
            return

        def getv(name: str, default: float = 0.0) -> float:
            if isinstance(state, dict):
                return float(state.get(name, default))
            return float(getattr(state, name, default))

        tx = self._deadzone(getv("x", 0.0), self.config.deadzone_t)
        ty = self._deadzone(getv("y", 0.0), self.config.deadzone_t)
        tz = self._deadzone(getv("z", 0.0), self.config.deadzone_t)

        rx = self._deadzone(getv("roll", 0.0), self.config.deadzone_r)
        _ry = self._deadzone(getv("pitch", 0.0), self.config.deadzone_r)
        rz = self._deadzone(getv("yaw", 0.0), self.config.deadzone_r)

        self.target.yaw += rz * self.config.yaw_gain * dt
        self.target.pitch += (-rx) * self.config.pitch_gain * dt

        self.target.center += self.target.n * (tz * self.config.move_n_gain * dt)
        self.target.center += self.target.u * (tx * self.config.move_u_gain * dt)
        self.target.center += self.target.v * (ty * self.config.move_v_gain * dt)

        self.target.heap_depth += tz * self.config.heap_gain * dt

        self._clamp_center_and_sync()
