"""A LeRobot camera backend for the Luxonis OAK-D (depthai).

LeRobot's ``make_cameras_from_configs`` has no branch for the OAK (it isn't a UVC
webcam, so it has no OpenCV index). Its ``else`` branch, however, falls back to
``make_device_from_device_class``, which resolves a config ``XxxConfig`` to a device
class ``Xxx`` in the same module. So registering ``OakDepthAICameraConfig`` (type
``"oak"``) here lets the robot build an :class:`OakDepthAICamera` with no patching —
just put ``OakDepthAICameraConfig(...)`` in the robot's ``cameras`` dict.

The camera mirrors ``OpenCVCamera``'s threaded model: a background thread pulls frames
from the OAK and publishes the latest one, and ``async_read`` returns the freshest
unconsumed frame (RGB by default, matching what LeRobot policies expect).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Lock, Thread
from typing import Any

import cv2
import depthai as dai
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera
from lerobot.cameras.configs import CameraConfig, ColorMode

from oak_camera import build_oak_rgb_pipeline


@CameraConfig.register_subclass("oak")
@dataclass
class OakDepthAICameraConfig(CameraConfig):
    """Config for a Luxonis OAK-D camera (depthai). ``type`` is ``"oak"``.

    Inherits ``fps``/``width``/``height`` from ``CameraConfig``. The robot config
    requires width/height/fps, so set them (e.g. 640x480@30); the OAK is asked for
    exactly that output size.
    """

    color_mode: ColorMode = ColorMode.RGB
    warmup_s: float = 1.0
    socket: str = "CAM_A"


class OakDepthAICamera(Camera):
    """OAK-D RGB camera exposing LeRobot's :class:`Camera` interface."""

    def __init__(self, config: OakDepthAICameraConfig):
        super().__init__(config)
        self.config = config
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.socket_name = config.socket
        # Default the declared profile so the robot always has width/height/fps.
        self.width = config.width or 640
        self.height = config.height or 480
        self.fps = config.fps or 30
        self.capture_width, self.capture_height = self.width, self.height

        self._pipeline: Any = None
        self._queue: Any = None
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.socket_name})"

    @property
    def is_connected(self) -> bool:
        return (
            self._pipeline is not None
            and self.thread is not None
            and self.thread.is_alive()
        )

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        cams: list[dict[str, Any]] = []
        for d in dai.Device.getAllAvailableDevices():
            cams.append(
                {
                    "name": getattr(d, "name", "OAK"),
                    "type": "oak",
                    "id": d.deviceId,
                    "state": d.state.name,
                    "protocol": d.protocol.name,
                }
            )
        return cams

    def connect(self, warmup: bool = True) -> None:
        if self.is_connected:
            raise RuntimeError(f"{self} is already connected.")
        socket = getattr(dai.CameraBoardSocket, self.socket_name)
        width, height = int(self.width or 640), int(self.height or 480)
        self._pipeline, self._queue = build_oak_rgb_pipeline(
            width, height, blocking=False, socket=socket
        )
        self._start_read_thread()

        if warmup and self.warmup_s > 0:
            start = time.time()
            while time.time() - start < self.warmup_s:
                try:
                    self.async_read(timeout_ms=self.warmup_s * 1000)
                except TimeoutError:
                    pass
                time.sleep(0.1)
            with self.frame_lock:
                if self.latest_frame is None:
                    raise ConnectionError(
                        f"{self} failed to capture frames during warmup."
                    )

    def _read_from_hardware(self) -> NDArray[Any]:
        # Non-blocking poll with a deadline so a stalled device surfaces as an error
        # (the read loop's failure counter then handles transient gaps vs a dead link).
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            pkt = self._queue.tryGet()
            if pkt is not None:
                return pkt.getCvFrame()  # BGR, like OpenCV
            time.sleep(0.002)
        raise RuntimeError(f"{self} received no frame from the OAK within 2s.")

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        h, w, c = image.shape
        if c != 3:
            raise RuntimeError(f"{self} frame channels={c} do not match expected 3.")
        # The server resizes to the policy's input shape, and the OAK may deliver a
        # slightly different size than requested, so adopt the actual size rather than
        # raising (mirrors the project's tolerate_camera_resolution_drift patch).
        if h != self.capture_height or w != self.capture_width:
            self.capture_width, self.capture_height = w, h
            self.width, self.height = w, h
        if self.color_mode == ColorMode.RGB:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def read(self) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        self.new_frame_event.clear()
        return self.async_read(timeout_ms=10000)

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )
        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()
        if frame is None:
            raise RuntimeError(f"Internal error: event set but no frame for {self}.")
        return frame

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        with self.frame_lock:
            frame, ts = self.latest_frame, self.latest_timestamp
        if frame is None or ts is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        age_ms = (time.perf_counter() - ts) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(f"{self} latest frame is too old: {age_ms:.1f} ms.")
        return frame

    def _read_loop(self) -> None:
        assert self.stop_event is not None
        failure_count = 0
        while not self.stop_event.is_set():
            try:
                frame = self._postprocess_image(self._read_from_hardware())
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_timestamp = time.perf_counter()
                self.new_frame_event.set()
                failure_count = 0
            except Exception as e:  # noqa: BLE001 - tolerate transient gaps, die on a dead link
                if failure_count <= 10:
                    failure_count += 1
                else:
                    raise RuntimeError(
                        f"{self} exceeded maximum consecutive read failures."
                    ) from e

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(
            target=self._read_loop, name=f"{self}_read_loop", daemon=True
        )
        self.thread.start()
        time.sleep(0.1)

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None
        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    def disconnect(self) -> None:
        self._stop_read_thread()
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._pipeline = None
            self._queue = None
