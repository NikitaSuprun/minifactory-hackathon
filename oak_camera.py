"""Read RGB frames from a Luxonis OAK-D (depthai) as a cv2.VideoCapture-like source.

The OAK-D is an XLink/depthai device, not a UVC webcam, so OpenCV can't open it by
device index. This wraps a minimal depthai RGB pipeline behind the small slice of the
``cv2.VideoCapture`` interface the dashboard uses (``isOpened``/``read``/``release``),
so it drops into the same preview/MJPEG code path as the other cameras.
"""

from __future__ import annotations

from typing import Any

import depthai as dai


class OakCamera:
    """Minimal VideoCapture-like wrapper around an OAK RGB stream.

    ``read()`` returns ``(ok, bgr_frame)`` and is non-blocking (``ok`` is False when no
    frame is ready yet), matching how the dashboard generators poll cv2 cameras.
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        socket: dai.CameraBoardSocket = dai.CameraBoardSocket.CAM_A,
    ) -> None:
        self._pipeline = dai.Pipeline()
        cam = self._pipeline.create(dai.node.Camera).build(socket)
        out = cam.requestOutput((width, height), dai.ImgFrame.Type.BGR888i)
        self._queue = out.createOutputQueue(maxSize=4, blocking=False)
        self._pipeline.start()
        self._running = True

    def isOpened(self) -> bool:  # noqa: N802 - mirror cv2.VideoCapture
        return self._running

    def read(self) -> tuple[bool, Any]:
        pkt = self._queue.tryGet()
        if pkt is None:
            return False, None
        # tryGet() is typed as the generic ADatatype; at runtime it's an ImgFrame.
        return True, pkt.getCvFrame()  # pyright: ignore[reportAttributeAccessIssue]  # already BGR

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
