"""Integrated offline MPR viewer with tri-plane preview and waypoint timeline hooks.

Builds on MPRPlaneUI and wires in:
- tri-plane split mode (frontal + sagittal + transverse snapshots)
- waypoint capture channels (camera / brush / combined)
- timeline import/export helpers
"""

from __future__ import annotations

import json
from pathlib import Path

import moderngl_window as mglw
import numpy as np

from mpr_plane_ui_with_mesh_upload import MPRPlaneUI
from tri_plane_viewer_mode import TriPlaneViewer
from waypoint_timeline_recorder import (
    BrushState,
    CameraState,
    WaypointRecorder,
    build_timeline,
)


class MPRPlaneUIIntegrated(MPRPlaneUI):
    title = "MPR Plane UI + Mesh + TriPlane + Waypoint Timeline"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.triplane_enabled = False
        self.triplane_output_dir = Path("out/triplane_frames")
        self.triplane_output_dir.mkdir(parents=True, exist_ok=True)

        self.waypoint_output = Path("out/waypoints/session_waypoints.json")
        self.timeline_output = Path("out/waypoints/session_timeline.json")
        self.waypoints = WaypointRecorder()

        self._last_time = 0.0

        self._tri_volume = self.V
        self._tri_viewer = TriPlaneViewer(self._tri_volume)

        print("[integrated] F6 toggle tri-plane snapshots")
        print("[integrated] F7 capture camera waypoint")
        print("[integrated] F8 capture brush waypoint")
        print("[integrated] F9 capture combined waypoint")
        print("[integrated] F10 save waypoint json")
        print("[integrated] F11 build timeline from saved waypoints")

    def _current_indices(self) -> tuple[int, int, int]:
        z = int(np.clip(round(self.center[2] * (self.Z - 1)), 0, self.Z - 1))
        y = int(np.clip(round(self.center[1] * (self.H - 1)), 0, self.H - 1))
        x = int(np.clip(round(self.center[0] * (self.W - 1)), 0, self.W - 1))
        return z, y, x

    def _camera_state(self) -> CameraState:
        return CameraState(
            t=float(self._last_time),
            position=[float(x) for x in self.center.tolist()],
            euler_deg=[float(np.degrees(self.pitch)), float(np.degrees(self.yaw)), 0.0],
            plane_normal=[float(x) for x in self.n.tolist()],
        )

    def _brush_state(self) -> BrushState:
        return BrushState(
            t=float(self._last_time),
            mouse_uv=[float(self.mouse_uv[0]), float(self.mouse_uv[1])],
            strength=float(self.heap_depth),
        )

    def _save_waypoints(self) -> None:
        self.waypoints.save(self.waypoint_output)
        print(f"[waypoint] wrote {self.waypoint_output}")

    def _load_waypoints_and_build_timeline(self) -> None:
        if not self.waypoint_output.exists():
            print(f"[waypoint] missing file: {self.waypoint_output}")
            return
        build_timeline(self.waypoint_output, self.timeline_output, samples_per_segment=12, noise_sigma=0.0)
        print(f"[timeline] wrote {self.timeline_output}")

    def _render_triplane_snapshot(self) -> None:
        z, y, x = self._current_indices()
        frame_idx = int(self._last_time * 30.0)
        out_path = self.triplane_output_dir / f"triplane_{frame_idx:06d}.png"
        self._tri_viewer.render_triptych(z=z, y=y, x=x, out_path=out_path, scale=1)

    def on_key_event(self, key, action, modifiers):
        super().on_key_event(key, action, modifiers)

        if action != self.wnd.keys.ACTION_PRESS:
            return

        k = self.wnd.keys
        if key == k.F6:
            self.triplane_enabled = not self.triplane_enabled
            print(f"[integrated] triplane_enabled={self.triplane_enabled}")
        elif key == k.F7:
            self.waypoints.on_key_c(self._camera_state())
            print(f"[waypoint] camera count={len(self.waypoints.camera_waypoints)}")
        elif key == k.F8:
            self.waypoints.on_key_b(self._brush_state())
            print(f"[waypoint] brush count={len(self.waypoints.brush_waypoints)}")
        elif key == k.F9:
            self.waypoints.on_key_v(self._camera_state(), self._brush_state())
            print(f"[waypoint] combined count={len(self.waypoints.combined_waypoints)}")
        elif key == k.F10:
            self._save_waypoints()
        elif key == k.F11:
            self._save_waypoints()
            self._load_waypoints_and_build_timeline()
        elif key == k.F12:
            self._save_waypoints()
            self.wnd.close()

    def on_render(self, time: float, frame_time: float):
        self._last_time = float(time)
        super().on_render(time, frame_time)
        if self.triplane_enabled:
            self._render_triplane_snapshot()


if __name__ == "__main__":
    mglw.run_window_config(MPRPlaneUIIntegrated)
