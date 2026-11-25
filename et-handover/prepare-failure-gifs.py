#!/usr/bin/env python3
"""Extract the requested PPT animations and stabilize their plotting canvas.

Pillow/NumPy only inspect the frames. FFmpeg performs the lossless integer
translations, title crop, and GIF encoding; no temporal interpolation is used.
Requires ffmpeg, Pillow, and NumPy. Run with the original PowerPoint path.
"""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image


ASSETS = Path(__file__).resolve().parent / "assets"
SOURCES = [
    (11, "image14.gif", "failure-extent-expansion-1.gif"),
    (12, "image15.gif", "failure-extent-expansion-2.gif"),
    (13, "image16.gif", "failure-multiple-initiation.gif"),
]
TITLE_ROWS = 36


def inspect(source):
    frames = []
    with Image.open(source) as gif:
        assert gif.size == (960, 840), "Recheck the crop for a different source size"
        loop = gif.info.get("loop")
        for index in range(gif.n_frames):
            gif.seek(index)
            rgb = np.asarray(gif.convert("RGB"))
            dark_gray = (np.ptp(rgb, axis=2) < 25) & (rgb.mean(axis=2) < 100)
            vertical = dark_gray[60:755].sum(axis=0)
            left = 60 + int(np.argmax(vertical[60:200]))
            right = 750 + int(np.argmax(vertical[750:900]))
            assert min(vertical[left], vertical[right]) > 650, "Plot border not found"
            assert 725 <= right - left <= 726, "Unexpected plot scale change"
            content_x = np.where(np.any(rgb[TITLE_ROWS:] < 245, axis=2))[1]
            frames.append({
                "left": left, "right": right,
                "content_left": int(content_x.min()), "content_right": int(content_x.max()),
                "duration_ms": gif.info["duration"],
            })
    assert loop == 0 and all(frame["duration_ms"] == 100 for frame in frames)
    return frames


def prepare(source, output):
    frames = inspect(source)
    lefts = [frame["left"] for frame in frames]
    reference = max(lefts)
    shifts = [reference - left for left in lefts]
    # Retain all original annotations, plus a small consistent outer margin.
    start = min(frame["content_left"] + shift for frame, shift in zip(frames, shifts)) - 8
    end = max(frame["content_right"] + shift for frame, shift in zip(frames, shifts)) + 9
    padding = max(shifts) + 8
    crop_x = str(padding - reference + start) + "".join(
        f"+{left}*eq(n,{index})" for index, left in enumerate(lefts)
    )
    stabilize = (
        # The title's underscores reach row 39; preserve the top y-axis tick
        # at x < 150 while clearing the complete title above the plot border.
        f"format=rgb24,drawbox=x=150:y=0:w=iw-150:h=44:color=white:t=fill,"
        f"crop=iw:ih-{TITLE_ROWS}:0:{TITLE_ROWS}:exact=1,"
        f"pad=iw+{2 * padding}:ih:{padding}:0:white,"
        f"crop={end - start}:ih:x='{crop_x}':y=0:exact=1"
    )
    filters = (
        f"[0:v]{stabilize},split[frames][colors];"
        "[colors]palettegen=stats_mode=full:reserve_transparent=0[palette];"
        "[frames][palette]paletteuse=dither=none:new=0[out]"
    )
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-filter_complex_threads", "1", "-filter_complex", filters, "-map", "[out]",
        "-fps_mode", "passthrough", "-gifflags", "0", "-loop", "0",
        "-final_delay", "10", str(output),
    ], check=True)

    stabilized_lefts = []
    with Image.open(output) as gif:
        assert gif.n_frames == len(frames) and gif.info.get("loop") == 0
        size = list(gif.size)
        for index in range(gif.n_frames):
            gif.seek(index)
            assert gif.info["duration"] == frames[index]["duration_ms"]
            assert gif.info.get("transparency") is None, "Frames must be opaque"
            assert gif.tile[0][1] == (0, 0, *gif.size), "Frames must cover the full canvas"
            rgb = np.asarray(gif.convert("RGB"))
            dark_gray = (np.ptp(rgb, axis=2) < 25) & (rgb.mean(axis=2) < 100)
            vertical = dark_gray[60 - TITLE_ROWS:755 - TITLE_ROWS].sum(axis=0)
            stabilized_lefts.append(40 + int(np.argmax(vertical[40:180])))
    assert len(set(stabilized_lefts)) == 1, "The plot border must stay fixed"
    return {
        "output": output.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "frames": len(frames), "duration_ms": sum(frame["duration_ms"] for frame in frames),
        "size": size, "title_rows_removed": TITLE_ROWS,
        "title_mask": {"x": 150, "y": 0, "width": 810, "height": 44},
        "original_axis_left_range": [min(lefts), max(lefts)],
        "stabilized_axis_left": stabilized_lefts[0],
        "horizontal_translations_px": shifts,
        "encoding": "one global palette; opaque full frames; 100 ms per frame; infinite loop",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("powerpoint", type=Path)
    args = parser.parse_args()
    report = {"source": args.powerpoint.name, "animations": []}
    ASSETS.mkdir(exist_ok=True)
    with ZipFile(args.powerpoint) as archive, tempfile.TemporaryDirectory(prefix="handover-failure-") as temp:
        for slide, media, name in SOURCES:
            source = Path(temp) / media
            source.write_bytes(archive.read("ppt/media/" + media))
            result = prepare(source, ASSETS / name)
            result.update({"powerpoint_slide": slide, "powerpoint_media": "ppt/media/" + media})
            report["animations"].append(result)
            print(f"Page {slide}: {name}, {result['frames']} frames, axis shift {result['original_axis_left_range']} -> fixed")
    (ASSETS / "failure-analysis-provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
