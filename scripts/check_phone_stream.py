"""Verify that an Android phone's IP Webcam stream reaches this Mac via LeRobot.

Connects to the phone's MJPEG stream through LeRobot's OpenCVCamera, reads a burst
of frames, reports resolution / achieved FPS / a latency proxy, and saves the first
frame to ``stream_sample.png`` as visual proof.

Usage:
    uv run python scripts/check_phone_stream.py http://<phone-ip>:8080/video
    # or set PHONE_CAM_URL and run with no argument
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Final

import cv2

# Allow running the script directly from the repo without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phone_camera import open_phone_camera  # noqa: E402

N_FRAMES: Final[int] = 150
SNAPSHOT_PATH: Final[Path] = Path("stream_sample.png")


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else None

    try:
        camera = open_phone_camera(url)
    except ConnectionError as e:
        print(f"Could not connect to the phone stream: {e}")
        print(
            "Checklist:\n"
            "  - IP Webcam app is running and you tapped 'Start server'.\n"
            "  - Phone and Mac are on the same WiFi (no guest/client isolation).\n"
            "  - The IP matches the one shown in the app.\n"
            "  - Try the '/videofeed' path if '/video' fails."
        )
        return 1
    except ValueError as e:
        print(e)
        return 2

    try:
        print(f"Connected. Detected resolution: {camera.width}x{camera.height}")

        latencies_ms: list[float] = []
        first_frame = None
        start = time.perf_counter()
        for _ in range(N_FRAMES):
            frame = camera.async_read(timeout_ms=5000)
            if first_frame is None:
                first_frame = frame
            if camera.latest_timestamp is not None:
                latencies_ms.append(
                    (time.perf_counter() - camera.latest_timestamp) * 1e3
                )
        elapsed = time.perf_counter() - start

        achieved_fps = N_FRAMES / elapsed if elapsed > 0 else 0.0
        avg_latency = (
            sum(latencies_ms) / len(latencies_ms) if latencies_ms else float("nan")
        )
        print(f"Read {N_FRAMES} frames in {elapsed:.1f}s -> {achieved_fps:.1f} FPS")
        print(f"Frame-age latency proxy: {avg_latency:.0f} ms (avg)")

        if first_frame is not None:
            # async_read returns RGB; cv2.imwrite expects BGR.
            cv2.imwrite(
                str(SNAPSHOT_PATH), cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR)
            )
            print(f"Saved sample frame to {SNAPSHOT_PATH.resolve()}")
    finally:
        camera.disconnect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
