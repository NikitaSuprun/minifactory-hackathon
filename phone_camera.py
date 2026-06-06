"""Use an Android phone (IP Webcam app) as a LeRobot camera over WiFi.

The phone runs the *IP Webcam* app and serves an MJPEG stream at
``http://<phone-ip>:8080/video``. LeRobot's :class:`OpenCVCamera` opens that URL
directly via ``cv2.VideoCapture`` — no macOS driver or virtual camera needed.

Two details make this work reliably:

- The URL must be passed as a plain ``str``. ``OpenCVCameraConfig.index_or_path``
  is typed ``int | Path``, but a ``Path`` would collapse ``http://`` to ``http:/``
  and break the URL, so we keep it a string.
- ``fps``/``width``/``height`` are left as ``None`` so LeRobot auto-detects the
  stream's native profile. Setting them would make OpenCV try ``VideoCapture.set``
  on a network stream and raise. Pick resolution/FPS in the IP Webcam app instead.

Later, to use the phone in a LeRobot robot for recording/inference, drop the
config into the robot's ``cameras`` dict::

    from phone_camera import build_phone_camera_config

    robot_config.cameras = {
        "phone": build_phone_camera_config("http://192.168.1.42:8080/video"),
    }
"""

from __future__ import annotations

import os

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

DEFAULT_URL_ENV = "PHONE_CAM_URL"


def build_phone_camera_config(
    url: str,
    *,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION,
) -> OpenCVCameraConfig:
    """Build an OpenCVCameraConfig pointed at a phone's network stream URL.

    Leave fps/width/height as None to auto-detect the stream profile; set
    resolution/FPS in the IP Webcam app rather than here.
    """
    if not isinstance(url, str) or "://" not in url:
        raise ValueError(
            f"Expected a stream URL string like 'http://<phone-ip>:8080/video', got {url!r}."
        )
    return OpenCVCameraConfig(
        # Typed int | Path, but cv2.VideoCapture needs the raw URL string; a Path
        # would collapse "http://" to "http:/" and break the stream.
        index_or_path=url,  # pyright: ignore[reportArgumentType]
        fps=fps,
        width=width,
        height=height,
        rotation=rotation,
    )


def open_phone_camera(url: str | None = None, **kwargs) -> OpenCVCamera:
    """Build, connect, and return an OpenCVCamera for the phone stream.

    If ``url`` is None, falls back to the ``PHONE_CAM_URL`` environment variable.
    Caller is responsible for ``camera.disconnect()``.
    """
    url = url or os.environ.get(DEFAULT_URL_ENV)
    if not url:
        raise ValueError(
            f"No stream URL provided and ${DEFAULT_URL_ENV} is not set. "
            "Pass e.g. 'http://192.168.1.42:8080/video'."
        )
    camera = OpenCVCamera(build_phone_camera_config(url, **kwargs))
    camera.connect()
    return camera
