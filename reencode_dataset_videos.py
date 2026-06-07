"""Re-encode every video in a local LeRobot dataset to a target codec, in place.

Needed before aggregating datasets recorded with different codecs: aggregate_datasets
requires identical features (which include ``video.codec``) and concatenates video files
without re-encoding, so mixing e.g. av1 + h264 either fails validation or corrupts video.

Usage:
    uv run python reencode_dataset_videos.py <dataset_name> [h264|av1]   # default h264

Transcodes each videos/**/*.mp4 1:1 (frame count + fps preserved) to a temp file, verifies
the frame count is unchanged, swaps them all in, then rewrites meta/info.json's video.codec.
A failure before the swap leaves the original dataset untouched.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import phone_camera  # noqa: F401 - import side effect loads .env + .env.local

from recording import resolve_hf_username

ENCODERS = {
    # libx264 crf 18 is visually lossless at VGA and fast; passthrough keeps frames 1:1.
    "h264": [
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
    ],
    "av1": ["-c:v", "libsvtav1", "-crf", "30", "-pix_fmt", "yuv420p"],
}


def packet_count(path: Path) -> int:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_packets",
            "-show_entries",
            "stream=nb_read_packets",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return int(out)


def main() -> None:
    name = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else "h264"
    if target not in ENCODERS:
        sys.exit(f"target codec must be one of {list(ENCODERS)}, got {target!r}")

    user = resolve_hf_username()
    base = Path(
        os.environ.get("HF_LEROBOT_HOME")
        or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "lerobot"
    ).expanduser()
    root = base / user / name
    if not (root / "meta" / "info.json").exists():
        sys.exit(f"no dataset at {root}")

    mp4s = sorted(root.glob("videos/**/*.mp4"))
    if not mp4s:
        sys.exit(f"no videos under {root}/videos")
    print(f"Re-encoding {len(mp4s)} video file(s) in {user}/{name} -> {target}")

    # Pass 1: transcode each to a temp file and verify the frame count is preserved.
    tmps: list[tuple[Path, Path]] = []
    try:
        for mp4 in mp4s:
            tmp = mp4.with_suffix(".reenc.mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(mp4),
                    *ENCODERS[target],
                    "-fps_mode",
                    "passthrough",
                    "-an",
                    str(tmp),
                ],
                check=True,
            )
            before, after = packet_count(mp4), packet_count(tmp)
            if before != after:
                raise RuntimeError(
                    f"frame count changed for {mp4.name}: {before} -> {after}"
                )
            tmps.append((mp4, tmp))
            print(f"  ok {mp4.relative_to(root)} ({after} frames)")
    except BaseException:
        for _, tmp in tmps:
            tmp.unlink(missing_ok=True)
        raise

    # Pass 2: swap in the re-encoded files, then update info.json's codec.
    for mp4, tmp in tmps:
        tmp.replace(mp4)

    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    for ft in info["features"].values():
        if ft.get("dtype") == "video":
            ft["info"]["video.codec"] = target
    info_path.write_text(json.dumps(info, indent=4))
    print(f"Done. {root} is now {target}.")


if __name__ == "__main__":
    main()
