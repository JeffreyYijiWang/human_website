# offline_render.py
# Headless/offscreen renderer for your MPR slice + heap brush shader pipeline.
#
# Requires:
#   pip install moderngl numpy pillow
#
# Expects volume:
#   threshold_images/volume_uint8.npy   (Z,H,W,3) uint8 BGR  (same as your app)  :contentReference[oaicite:2]{index=2}
#
# Example:
#   python offline_render.py --seconds 6 --fps 30 --width 1280 --height 720 --seed 7
#   python offline_render.py --seconds 3 --fps 24 --width 2048 --height 2048 --out out_hi

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import moderngl
from PIL import Image

# --- Shaders copied to match your current pipeline (same uniforms/behavior) :contentReference[oaicite:3]{index=3}
SLICE_VERT = r"""
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = (in_pos * 0.5) + 0.5;   // 0..1, bottom-left origin
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

SLICE_FRAG = r"""
#version 330

uniform sampler2DArray tex_array;
uniform int   u_num_layers;     // Z layers in texture array
uniform vec2  u_slice_px;       // viewport (w,h) in pixels

uniform vec3  u_center;         // plane center in [0,1]^3
uniform vec3  u_axis_u;         // plane axis U (unit)
uniform vec3  u_axis_v;         // plane axis V (unit)
uniform vec3  u_axis_n;         // plane normal N (unit)
uniform float u_scale;          // half-width in normalized volume units

// heap brush
uniform int   u_heap_enable;    // 0/1
uniform vec2  u_mouse;          // [0,1], bottom-left origin
uniform float u_radius;         // brush radius in UV
uniform float u_softness;       // feather size (UV)
uniform float u_layer_stretch;  // shaping in radius space
uniform float u_heap_depth;     // max offset along +/-N (normalized volume units)
uniform float u_heap_dir;       // +1 or -1 (direction along N)

// color controls
uniform int   u_flip_y;         // 1 if texture rows are top-left origin (most PNG stacks), else 0
uniform int   u_bgr_input;      // 1 if stored as BGR in RGB channels, else 0

in vec2 v_uv;
out vec4 fragColor;

vec4 sample_volume(vec3 p) {
    float zf = clamp(p.z, 0.0, 1.0) * float(u_num_layers - 1);
    int   z0 = int(floor(zf));
    int   z1 = min(z0 + 1, u_num_layers - 1);
    float t  = fract(zf);

    vec2 uv = p.xy;
    if (u_flip_y != 0) uv.y = 1.0 - uv.y;

    vec4 a = texture(tex_array, vec3(uv, float(z0)));
    vec4 b = texture(tex_array, vec3(uv, float(z1)));
    return mix(a, b, t);
}

void main() {
    vec2 s = (v_uv * 2.0 - 1.0);

    float aspect = u_slice_px.x / max(u_slice_px.y, 1.0);
    s.x *= aspect;

    vec3 p0 = u_center + (u_axis_u * (s.x * u_scale)) + (u_axis_v * (s.y * u_scale));

    if (any(lessThan(p0, vec3(0.0))) || any(greaterThan(p0, vec3(1.0)))) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 p = p0;

    if (u_heap_enable != 0) {
        float d = distance(v_uv, u_mouse);

        float r = max(u_radius, 1e-6);
        float t = clamp(d / r, 0.0, 1.0);
        float shaped_t = pow(t, max(u_layer_stretch, 1e-6));

        float depth = (1.0 - shaped_t) * u_heap_depth;

        p = p0 + (u_axis_n * (depth * u_heap_dir));

        if (any(lessThan(p, vec3(0.0))) || any(greaterThan(p, vec3(1.0)))) {
            p = p0;
        }

        vec4 outside = sample_volume(p0);
        vec4 inside  = sample_volume(p);

        float rim0 = 1.0 - (u_softness / r);
        rim0 = clamp(rim0, 0.0, 1.0);
        float edge_fade = 1.0 - smoothstep(rim0, 1.0, t);

        vec4 c = mix(outside, inside, edge_fade);

        if (u_bgr_input != 0) c.rgb = c.bgr;
        fragColor = vec4(c.rgb, 1.0);
        return;
    }

    vec4 c = sample_volume(p);
    if (u_bgr_input != 0) c.rgb = c.bgr;
    fragColor = vec4(c.rgb, 1.0);
}
"""

# ---------- math helpers ----------
def _norm(v):
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else (v / n)

def orthonormal_basis_from_normal(n):
    n = _norm(n)
    a = np.array([0.0, 0.0, 1.0], dtype=np.float32) if abs(float(n[2])) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = _norm(np.cross(a, n))
    v = _norm(np.cross(n, u))
    return u, v

def yaw_pitch_to_normal(yaw, pitch):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x = sy * cp
    y = cy * cp
    z = sp
    return _norm(np.array([x, y, z], dtype=np.float32))

# ---------- offline renderer ----------
class OfflineMPR:
    def __init__(self, width, height, volume_path: Path):
        self.w = int(width)
        self.h = int(height)

        # Headless context
        self.ctx = moderngl.create_standalone_context(require=330)

        # Load volume: (Z,H,W,3) uint8 BGR :contentReference[oaicite:4]{index=4}
        V = np.load(volume_path, mmap_mode="r")
        if V.dtype != np.uint8 or V.ndim != 4 or V.shape[3] != 3:
            raise ValueError(f"Unexpected volume: shape={V.shape} dtype={V.dtype} (expected (Z,H,W,3) uint8)")
        self.Z, self.H, self.W = int(V.shape[0]), int(V.shape[1]), int(V.shape[2])
        self.V = V

        # Program / fullscreen tri quad
        self.prog = self.ctx.program(vertex_shader=SLICE_VERT, fragment_shader=SLICE_FRAG)
        fsq = np.array([-1,-1,  1,-1,  -1, 1,
                        -1, 1,  1,-1,   1, 1], dtype="f4")
        self.vbo = self.ctx.buffer(fsq.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_pos")

        # Texture array upload (RGBA8)
        self.tex = self.ctx.texture_array((self.W, self.H, self.Z), components=4, dtype="f1")
        self.tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        alpha = np.full((self.H, self.W, 1), 255, dtype=np.uint8)
        for z in range(self.Z):
            slab = np.ascontiguousarray(self.V[z])            # (H,W,3) BGR
            slab_rgba = np.concatenate([slab, alpha], axis=2) # (H,W,4)
            self.tex.write(slab_rgba.tobytes(), viewport=(0, 0, z, self.W, self.H, 1))

        self.tex.use(location=0)
        self.prog["tex_array"].value = 0
        self.prog["u_num_layers"].value = self.Z

        # Offscreen render target
        self.color = self.ctx.texture((self.w, self.h), components=4, dtype="f1")
        self.fbo = self.ctx.framebuffer(color_attachments=[self.color])

    def render(self, params: dict) -> np.ndarray:
        """
        params keys:
          center (3), yaw, pitch, scale
          heap_enable, mouse_uv(2), radius, softness, stretch, depth, dir
          flip_y, bgr_input
        """
        # plane basis
        n = yaw_pitch_to_normal(params["yaw"], params["pitch"])
        u, v = orthonormal_basis_from_normal(n)

        self.fbo.use()
        self.ctx.viewport = (0, 0, self.w, self.h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        # uniforms
        self.prog["u_center"].value = tuple(float(x) for x in params["center"])
        self.prog["u_axis_u"].value = tuple(float(x) for x in u)
        self.prog["u_axis_v"].value = tuple(float(x) for x in v)
        self.prog["u_axis_n"].value = tuple(float(x) for x in n)
        self.prog["u_scale"].value = float(params["scale"])
        self.prog["u_slice_px"].value = (float(self.w), float(self.h))

        self.prog["u_heap_enable"].value = int(params["heap_enable"])
        self.prog["u_mouse"].value = (float(params["mouse_uv"][0]), float(params["mouse_uv"][1]))
        self.prog["u_radius"].value = float(params["radius"])
        self.prog["u_softness"].value = float(params["softness"])
        self.prog["u_layer_stretch"].value = float(params["stretch"])
        self.prog["u_heap_depth"].value = float(params["depth"])
        self.prog["u_heap_dir"].value = float(params["dir"])

        self.prog["u_flip_y"].value = int(params["flip_y"])
        self.prog["u_bgr_input"].value = int(params["bgr_input"])

        self.fbo.clear(0.0, 0.0, 0.0, 1.0)
        self.vao.render(mode=moderngl.TRIANGLES)

        # Read back RGBA8
        raw = self.fbo.read(components=4, alignment=1)
        img = np.frombuffer(raw, dtype=np.uint8).reshape((self.h, self.w, 4))
        # moderngl returns bottom-to-top; flip to conventional top-to-bottom PNG
        img = np.flipud(img)
        return img

# ---------- synthetic input generator ----------
def generate_timeline(seconds, fps, seed):
    """
    Very slow, mostly imperceptible Brownian-like motion suitable for long renders.
    Over long durations (e.g. 1 hour) the center performs a slow wander that can
    eventually traverse most of the volume while keeping frame-to-frame change tiny.
    """
    rng = np.random.default_rng(seed)
    frames = int(round(seconds * fps))
    dt = 1.0 / max(float(fps), 1.0)

    # Start state
    state = {
        "center": np.array([0.5, 0.5, 0.5], dtype=np.float32),
        "yaw": 0.0,
        "pitch": 0.0,
        "scale": 0.55,
        "heap_enable": 1,
        "mouse_uv": np.array([0.5, 0.5], dtype=np.float32),
        "radius": 0.18,
        "softness": 0.06,
        "stretch": 1.0,
        "depth": 0.22,
        "dir": -1.0,
        "flip_y": 1,
        "bgr_input": 1,
    }

    # Target traversal in normalized units for the full clip; scales by duration.
    travel_ratio = float(np.clip(seconds / 3600.0, 0.05, 1.0))
    max_center_speed = (0.55 * travel_ratio) / max(seconds, 1.0)
    max_mouse_speed = (0.40 * travel_ratio) / max(seconds, 1.0)

    # Brownian / OU velocity states (small per-frame deltas)
    v_center = np.zeros(3, dtype=np.float32)
    v_mouse = np.zeros(2, dtype=np.float32)
    v_yaw = 0.0
    v_pitch = 0.0
    v_depth = 0.0

    events = []
    per_frame = []

    def clamp01(x):
        return np.clip(x, 0.0, 1.0)

    for f in range(frames):
        t = f / max(frames - 1, 1)

        # Tiny, sparse toggles for subtle non-static behavior.
        if rng.random() < 0.00035:
            state["dir"] *= -1.0
            events.append({"frame": f, "type": "flip_dir", "value": state["dir"]})
        if rng.random() < 0.00025:
            state["heap_enable"] = 0 if state["heap_enable"] else 1
            events.append({"frame": f, "type": "toggle_heap", "value": state["heap_enable"]})

        # Slow center wander (Brownian + weak deterministic drift to cover volume).
        drift = np.array([
            math.sin(2.0 * math.pi * (0.07 * t + 0.11)),
            math.sin(2.0 * math.pi * (0.05 * t + 0.41)),
            math.sin(2.0 * math.pi * (0.09 * t + 0.73)),
        ], dtype=np.float32) * (max_center_speed * 0.22)
        v_center = v_center * 0.997 + rng.normal(0.0, max_center_speed * 0.22, size=3).astype(np.float32) + drift
        speed = float(np.linalg.norm(v_center))
        if speed > max_center_speed:
            v_center *= (max_center_speed / max(speed, 1e-8))
        state["center"] = clamp01(state["center"] + v_center)

        # Slow mouse drift in UV with Brownian motion.
        mouse_drift = np.array([
            math.cos(2.0 * math.pi * (0.11 * t + 0.19)),
            math.sin(2.0 * math.pi * (0.08 * t + 0.62)),
        ], dtype=np.float32) * (max_mouse_speed * 0.28)
        v_mouse = v_mouse * 0.996 + rng.normal(0.0, max_mouse_speed * 0.25, size=2).astype(np.float32) + mouse_drift
        ms = float(np.linalg.norm(v_mouse))
        if ms > max_mouse_speed:
            v_mouse *= (max_mouse_speed / max(ms, 1e-8))
        state["mouse_uv"] = clamp01(state["mouse_uv"] + v_mouse)

        # Very slow axis rotation (yaw/pitch) with Brownian perturbation.
        v_yaw = (v_yaw * 0.9985) + (0.030 * math.sin(2.0 * math.pi * (0.016 * t + 0.30)) + rng.normal(0.0, 0.0012)) * dt
        v_pitch = (v_pitch * 0.9985) + (0.024 * math.cos(2.0 * math.pi * (0.013 * t + 0.57)) + rng.normal(0.0, 0.0010)) * dt
        state["yaw"] += v_yaw
        state["pitch"] = float(np.clip(state["pitch"] + v_pitch, -1.45, 1.45))

        # Heap depth/radius/softness/stretch: ultra-slow modulation + noise.
        v_depth = (v_depth * 0.996) + (0.060 * math.sin(2.0 * math.pi * (0.022 * t + 0.17)) + rng.normal(0.0, 0.0015)) * dt
        state["depth"] = float(np.clip(state["depth"] + v_depth, 0.03, 0.94))

        state["radius"] = float(np.clip(
            0.16 + 0.05 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.019 * t + 0.12))) + rng.normal(0.0, 0.0008),
            0.03,
            0.90,
        ))
        state["softness"] = float(np.clip(
            0.030 + 0.018 * (0.5 + 0.5 * math.cos(2.0 * math.pi * (0.015 * t + 0.44))) + rng.normal(0.0, 0.0006),
            0.0,
            state["radius"] * 0.9,
        ))
        state["stretch"] = float(np.clip(
            0.9 + 0.7 * (0.5 + 0.5 * math.sin(2.0 * math.pi * (0.011 * t + 0.81))) + rng.normal(0.0, 0.004),
            0.1,
            10.0,
        ))

        # Tiny scale breathing for life without obvious pulses.
        state["scale"] = float(np.clip(
            0.55 + 0.018 * math.sin(2.0 * math.pi * (0.020 * t + 0.13)) + rng.normal(0.0, 0.0007),
            0.05,
            2.0,
        ))

        per_frame.append({
            "frame": f,
            "center": [float(x) for x in state["center"]],
            "yaw": float(state["yaw"]),
            "pitch": float(state["pitch"]),
            "scale": float(state["scale"]),
            "heap_enable": int(state["heap_enable"]),
            "mouse_uv": [float(x) for x in state["mouse_uv"]],
            "radius": float(state["radius"]),
            "softness": float(state["softness"]),
            "stretch": float(state["stretch"]),
            "depth": float(state["depth"]),
            "dir": float(state["dir"]),
            "flip_y": int(state["flip_y"]),
            "bgr_input": int(state["bgr_input"]),
        })

    return per_frame, events

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", default="threshold_images/male_uint8.npy", help="Path to volume_uint8.npy")
    ap.add_argument("--out", default="offline_out", help="Output folder")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--seed", type=int, default=None,
                    help="Seed value. With --deterministic, defaults to 0. With --random, auto-generated if omitted.")
    seed_mode = ap.add_mutually_exclusive_group()
    seed_mode.add_argument("--random", dest="random_seed", action="store_true",
                           help="Use a random seed (default mode).")
    seed_mode.add_argument("--deterministic", dest="random_seed", action="store_false",
                           help="Use deterministic seed behavior (default seed=0 unless --seed is set).")
    ap.set_defaults(random_seed=True)
    ap.add_argument("--every", type=int, default=1, help="Save every Nth frame (1 = all)")
    ap.add_argument("--video", action="store_true", help="Also encode saved frames into an MP4 using ffmpeg.")
    ap.add_argument("--video-name", default="preview.mp4", help="Output video filename (inside --out).")
    args = ap.parse_args()

    if args.seed is None:
        if args.random_seed:
            # Use OS entropy so runs differ by default.
            args.seed = int.from_bytes(os.urandom(8), "big") ^ time.time_ns()
        else:
            args.seed = 0

    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    per_frame, events = generate_timeline(args.seconds, args.fps, args.seed)

    # Save timeline metadata
    (out_dir / "events.json").write_text(json.dumps(events, indent=2))
    (out_dir / "points.json").write_text(json.dumps({
        "seconds": args.seconds,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "seed": args.seed,
        "per_frame": per_frame,
    }, indent=2))

    renderer = OfflineMPR(args.width, args.height, Path(args.volume))

    total = len(per_frame)
    for i, st in enumerate(per_frame):
        if (i % args.every) != 0:
            continue
        img = renderer.render({
            "center": st["center"],
            "yaw": st["yaw"],
            "pitch": st["pitch"],
            "scale": st["scale"],
            "heap_enable": st["heap_enable"],
            "mouse_uv": st["mouse_uv"],
            "radius": st["radius"],
            "softness": st["softness"],
            "stretch": st["stretch"],
            "depth": st["depth"],
            "dir": st["dir"],
            "flip_y": st["flip_y"],
            "bgr_input": st["bgr_input"],
        })
        Image.fromarray(img, mode="RGBA").save(frames_dir / f"frame_{i:06d}.png")
        if (i % max(1, total // 10)) == 0:
            print(f"[render] {i}/{total}")

    if args.video:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg or run without --video.")

        video_path = out_dir / args.video_name
        cmd = [
            ffmpeg,
            "-y",
            "-framerate", str(args.fps),
            "-i", str(frames_dir / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(video_path),
        ]
        print("[ffmpeg]", " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"Video:    {video_path}")

    print(f"Seed: {args.seed}")
    print(f"Done. Frames in: {frames_dir}")
    print(f"Timeline: {out_dir/'points.json'}")
    print(f"Events:   {out_dir/'events.json'}")

if __name__ == "__main__":
    main()