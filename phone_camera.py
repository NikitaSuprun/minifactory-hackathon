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
from pathlib import Path
from typing import Final
from urllib.parse import quote, urlsplit, urlunsplit

from dotenv import load_dotenv

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

# Load connection settings from the committed .env next to this module, then
# overlay gitignored local secrets (HF_TOKEN etc.) from .env.local if present.
_HERE: Final[Path] = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
load_dotenv(_HERE / ".env.local", override=True)

DEFAULT_URL_ENV: Final[str] = "PHONE_CAM_URL"
USER_ENV: Final[str] = "PHONE_CAM_USER"
PASS_ENV: Final[str] = "PHONE_CAM_PASS"
HOST_ENV: Final[str] = "PHONE_CAM_HOST"
PORT_ENV: Final[str] = "PHONE_CAM_PORT"
PATH_ENV: Final[str] = "PHONE_CAM_PATH"


def phone_url_from_env() -> str | None:
    """Resolve the stream URL from environment / .env.

    Prefers an explicit ``PHONE_CAM_URL``; otherwise assembles it from
    ``PHONE_CAM_HOST`` (+ optional ``PHONE_CAM_PORT`` and ``PHONE_CAM_PATH``).
    Returns None if no host/url is configured.
    """
    url = os.environ.get(DEFAULT_URL_ENV)
    if url:
        return url
    host = os.environ.get(HOST_ENV)
    if not host:
        return None
    port = os.environ.get(PORT_ENV, "8080")
    path = os.environ.get(PATH_ENV, "/video")
    if not path.startswith("/"):
        path = "/" + path
    return f"http://{host}:{port}{path}"


def with_credentials(url: str, user: str, password: str) -> str:
    """Embed HTTP Basic Auth credentials into a stream URL.

    IP Webcam's "Login/password" option protects the stream with Basic Auth;
    cv2/FFmpeg reads ``http://user:pass@host:port/path``. Credentials are
    percent-encoded so special characters survive.
    """
    parts = urlsplit(url)
    netloc = f"{quote(user, safe='')}:{quote(password, safe='')}@{parts.hostname}"
    if parts.port:
        netloc += f":{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


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


def resolve_phone_url(
    url: str | None = None,
    *,
    user: str | None = None,
    password: str | None = None,
) -> str:
    """Resolve the full stream URL (with credentials) from args or .env.

    ``url`` falls back to ``$PHONE_CAM_URL`` or the ``$PHONE_CAM_HOST``/``_PORT``/
    ``_PATH`` trio; ``user``/``password`` fall back to ``$PHONE_CAM_USER``/
    ``$PHONE_CAM_PASS``. Credentials are injected only when the URL has none.
    """
    url = url or phone_url_from_env()
    if not url:
        raise ValueError(
            f"No stream URL provided and neither ${DEFAULT_URL_ENV} nor ${HOST_ENV} "
            "is set (check .env). Pass e.g. 'http://192.168.1.42:8080/video'."
        )
    user = user or os.environ.get(USER_ENV)
    password = password or os.environ.get(PASS_ENV)
    if user and password and "@" not in urlsplit(url).netloc:
        url = with_credentials(url, user, password)
    return url


def open_phone_camera(
    url: str | None = None,
    *,
    user: str | None = None,
    password: str | None = None,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION,
) -> OpenCVCamera:
    """Build, connect, and return an OpenCVCamera for the phone stream.

    See :func:`resolve_phone_url` for how the URL/credentials are resolved.
    Caller is responsible for ``camera.disconnect()``.
    """
    resolved_url: str = resolve_phone_url(url, user=user, password=password)
    config: OpenCVCameraConfig = build_phone_camera_config(
        resolved_url, fps=fps, width=width, height=height, rotation=rotation
    )
    camera = OpenCVCamera(config)
    camera.connect()
    return camera
