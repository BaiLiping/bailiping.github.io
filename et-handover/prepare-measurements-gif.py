#!/usr/bin/env python3
"""Plot only BS1's recorded mc_0088 measurements used by the simulation slide.

This is offline plotting only. It neither invokes a filter nor generates data.
Only BS1's position, the configured FoV radius, and its range/bearing pairs reach
the renderer. Target detections and clutter use the same unlabelled marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import numpy as np
from PIL import Image


TRIAL = "mc_0088"
FRAME_COUNT = 100
DURATION_MS = 100
BS_KEYS = [f"bs{i}" for i in range(7)]
SELECTED_BS = "bs1"
GREEN = "#087f68"
GRAY = "#596d80"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_measurements(repo: Path) -> tuple[np.ndarray, float, list, dict]:
    """Require all 100 original frames; return only BS1 and its observations."""
    if not (repo / "npz_data_io.py").is_file():
        raise FileNotFoundError(f"Expected EO_Target_Handover checkout: {repo}")
    sys.path.insert(0, str(repo))
    from npz_data_io import load_npz

    config_path = repo / "Config/config.json"
    config = json.loads(config_path.read_text())
    all_positions = np.asarray(config["sensor_positions"], dtype=float)[:, :2]
    radius = float(config["measurement_range"])
    if all_positions.shape != (7, 2) or not np.isfinite(all_positions).all():
        raise ValueError("Expected seven finite BS positions")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Expected a positive finite FoV radius")
    positions = all_positions[[BS_KEYS.index(SELECTED_BS)]]

    frames = []
    records = []
    for number in range(FRAME_COUNT):
        path = repo / "Data" / TRIAL / "frames_data" / f"frame_{number:04d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing measurement frame: {path}")
        frame = load_npz(path)
        if frame["metadata"] != {"version": "2.0.0", "frame_number": number}:
            raise ValueError(f"Unexpected frame schema or number: {path}")
        data = frame["data"]
        if set(data["bs"]) != set(BS_KEYS) or set(data["measurements"]) != set(BS_KEYS):
            raise ValueError(f"Expected all seven BS blocks: {path}")
        points = []
        counts = {}
        for key, position in zip([SELECTED_BS], positions):
            if not np.array_equal(data["bs"][key]["position"], position):
                raise ValueError(f"BS geometry changed: {path}, {key}")
            block = data["measurements"][key]
            # Combine all observations before plotting. Never use track_id,
            # source type, truth position, cluster membership, or estimates.
            observations = block["real"] + block["clutter"]
            counts[key] = len(observations)
            for observation in observations:
                distance = float(observation["range"])
                bearing = np.deg2rad(float(observation["bearing"]))
                points.append(position + distance * np.array([np.cos(bearing), np.sin(bearing)]))
        xy = np.asarray(points, dtype=float).reshape(-1, 2)
        if not np.isfinite(xy).all():
            raise ValueError(f"Non-finite measurement coordinates: {path}")
        frames.append(xy)
        records.append({
            "source": str(path.relative_to(repo)),
            "sha256": sha256(path),
            "frame_number": number,
            "detection_count": len(xy),
            "detection_count_by_bs": counts,
            "xy_float64_le_sha256": hashlib.sha256(xy.astype("<f8").tobytes()).hexdigest(),
        })
    provenance = {
        "source_repository": "https://github.com/BaiLiping/EO_Target_Handover",
        "source_revision": subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "config": {"source": "Config/config.json", "sha256": sha256(config_path)},
        "trial": TRIAL,
        "selected_bs": SELECTED_BS,
        "source_frame_range_inclusive": [0, 99],
        "frame_count": FRAME_COUNT,
        "frame_duration_ms": DURATION_MS,
        "loop": 0,
        "bs_positions_m": dict(zip([SELECTED_BS], positions.tolist())),
        "fov_radius_m": radius,
        "coordinate_conversion": "x = BS_x + range*cos(bearing*pi/180); y = BS_y + range*sin(bearing*pi/180)",
        "display": "Only BS1 and its FoV; only BS1 detections including clutter, one identical marker; only the current frame; no other BS measurements, truth, trajectories, associations, extents, or estimates.",
        "frames": records,
    }
    return positions, radius, frames, provenance


def render(positions: np.ndarray, radius: float, frames: list, folder: Path) -> dict:
    """Keep one fixed canvas and replace the scatter offsets at every frame."""
    lo = positions.min(axis=0) - radius
    hi = positions.max(axis=0) + radius
    center = (lo + hi) / 2
    half_span = (hi - lo).max() / 2 + 15
    limits = np.column_stack((center - half_span, center + half_span))
    for xy in frames:
        if np.any(xy < limits[:, 0]) or np.any(xy > limits[:, 1]):
            raise ValueError("A measurement would be clipped by the fixed plot limits")

    fig = Figure(figsize=(10, 10), dpi=120, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.09, 0.085, 0.875, 0.875))
    ax.set(aspect="equal", xlim=limits[0], ylim=limits[1], xlabel="x [m]", ylabel="y [m]")
    ax.tick_params(labelsize=12, colors=GRAY)
    ax.xaxis.label.set(size=14, color=GRAY)
    ax.yaxis.label.set(size=14, color=GRAY)
    ax.grid(color="#e5eaf0", linewidth=0.7)
    for spine in ax.spines.values():
        spine.set_color("#a9b7c5")
    for key, (x, y) in zip([SELECTED_BS], positions):
        ax.add_patch(Circle((x, y), radius, fill=False, edgecolor=GREEN,
                            linestyle="--", linewidth=1.3, alpha=0.55, zorder=2))
        ax.scatter([x], [y], color=GREEN, marker="p", s=120, zorder=4)
        ax.annotate(key.upper(), (x, y), xytext=(8, 5), textcoords="offset points",
                    fontsize=11, color=GREEN, fontweight="bold", zorder=5)
    observations = ax.scatter([], [], color=GRAY, marker="x", s=17,
                              linewidths=0.85, alpha=0.9, zorder=3)
    counter = fig.text(0.965, 0.977, "", ha="right", fontsize=15, color="#16273e")
    for index, xy in enumerate(frames):
        observations.set_offsets(xy)
        counter.set_text(f"Frame {index + 1:03d} / {FRAME_COUNT}")
        fig.savefig(folder / f"frame-{index:03d}.png", dpi=120)
    return {"x_m": limits[0].tolist(), "y_m": limits[1].tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path, help="EO_Target_Handover checkout containing Data/mc_0088")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "assets")
    args = parser.parse_args()
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        raise RuntimeError("FFmpeg is required to encode the GIF")
    positions, radius, frames, provenance = load_measurements(args.repo.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "measurements-only.gif"
    with tempfile.TemporaryDirectory(prefix="handover-measurements-") as temporary:
        folder = Path(temporary)
        provenance["plot_limits"] = render(positions, radius, frames, folder)
        subprocess.run([
            encoder, "-v", "error", "-y", "-framerate", "10", "-i", str(folder / "frame-%03d.png"),
            "-filter_complex", "[0:v]split[frames][colors];[colors]palettegen=reserve_transparent=0[palette];[frames][palette]paletteuse=dither=none",
            "-loop", "0", str(output),
        ], check=True)
    with Image.open(output) as gif:
        if gif.size != (1200, 1200) or gif.n_frames != FRAME_COUNT or gif.info["loop"] != 0:
            raise ValueError("Unexpected GIF dimensions, frame count, or looping")
        for index in range(gif.n_frames):
            gif.seek(index)
            if gif.info["duration"] != DURATION_MS:
                raise ValueError(f"Unexpected GIF timing at frame {index}")
    provenance["renderer_sha256"] = sha256(Path(__file__))
    provenance["output"] = {"file": output.name, "sha256": sha256(output), "bytes": output.stat().st_size, "size_px": [1200, 1200]}
    manifest = args.output_dir / "measurements-only-provenance.json"
    manifest.write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"output": str(output), "frames": FRAME_COUNT, "detections": sum(map(len, frames)), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
