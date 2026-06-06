"""Read RGB frames from a Luxonis OAK-D (depthai).

The OAK-D is an XLink/depthai device, not a UVC webcam, so OpenCV can't open it by
device index. :func:`build_oak_rgb_pipeline` is the shared depthai setup used by both
the dashboard preview (:class:`OakCamera`, a ``cv2.VideoCapture``-like polling source)
and the LeRobot camera backend (``oak_lerobot_camera.OakDepthAICamera``).
"""

from __future__ import annotations

from typing import Any

import depthai as dai

DEFAULT_SOCKET = dai.CameraBoardSocket.CAM_A


def build_oak_rgb_pipeline(
    width: int = 640,
    height: int = 480,
    *,
    blocking: bool = False,
    max_size: int = 4,
    socket: dai.CameraBoardSocket = DEFAULT_SOCKET,
) -> tuple[dai.Pipeline, Any]:
    """Create a started depthai pipeline streaming one BGR output queue.

    Returns ``(pipeline, queue)``. The queue yields ``ImgFrame`` packets whose
    ``getCvFrame()`` is a BGR ndarray (OpenCV convention). Caller owns teardown via
    ``pipeline.stop()``.
    """
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.Camera).build(socket)
    out = cam.requestOutput((width, height), dai.ImgFrame.Type.BGR888i)
    queue = out.createOutputQueue(maxSize=max_size, blocking=blocking)
    pipeline.start()
    return pipeline, queue


class OakCamera:
    """Minimal ``cv2.VideoCapture``-like wrapper around an OAK RGB stream.

    ``read()`` returns ``(ok, bgr_frame)`` and is non-blocking (``ok`` is False when no
    frame is ready yet), matching how the dashboard generators poll cv2 cameras.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        socket: dai.CameraBoardSocket = DEFAULT_SOCKET,
    ) -> None:
        self._pipeline, self._queue = build_oak_rgb_pipeline(
            width, height, blocking=False, socket=socket
        )
        self._running = True

    def isOpened(self) -> bool:  # noqa: N802 - mirror cv2.VideoCapture
        return self._running

    def read(self) -> tuple[bool, Any]:
        pkt = self._queue.tryGet()
        if pkt is None:
            return False, None
        return True, pkt.getCvFrame()  # already BGR (ImgFrame at runtime)

    def release(self) -> None:
        if self._running:
            self._running = False
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass


if __name__ == "__main__":
    import time

    import cv2

    cam = OakCamera()
    print("opened:", cam.isOpened())
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < 4:
        ok, frame = cam.read()
        if ok:
            n += 1
            if n == 1:
                cv2.imwrite("/tmp/oak_class_test.jpg", frame)
    cam.release()
    print(f"read {n} frames in ~4s -> /tmp/oak_class_test.jpg")
