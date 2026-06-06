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
import shutil
import subprocess
from pathlib import Path
from typing import Any, Final
from urllib.parse import quote, urlsplit, urlunsplit

from dotenv import load_dotenv
from numpy.typing import NDArray

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

# USB streaming: tunnel the IP Webcam port over the cable with `adb forward` and
# connect to localhost instead of the phone's WiFi IP.
USB_ENV: Final[str] = "PHONE_CAM_USB"
ADB_BIN_ENV: Final[str] = "PHONE_CAM_ADB"
ADB_SERIAL_ENV: Final[str] = "PHONE_CAM_ADB_SERIAL"
ADB_LOCAL_PORT_ENV: Final[str] = "PHONE_CAM_ADB_LOCAL_PORT"
ADB_REMOTE_PORT_ENV: Final[str] = "PHONE_CAM_ADB_REMOTE_PORT"


def _usb_enabled() -> bool:
    """True when USB streaming (adb port-forward) is requested via env."""
    return os.environ.get(USB_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def ensure_adb_forward(
    local_port: int | str,
    remote_port: int | str,
    *,
    adb: str = "adb",
    serial: str | None = None,
) -> None:
    """Tunnel the phone's IP Webcam port to localhost over USB via ``adb forward``.

    Idempotent: re-binding the same local port just replaces the existing rule,
    and the binding persists in the adb server, so opening the stream later (even
    from a child process) still reaches the phone. Raises with an actionable
    message if adb is missing or no authorized device is attached.
    """
    if shutil.which(adb) is None:
        raise FileNotFoundError(
            f"'{adb}' not found. Install Android platform-tools "
            "(`brew install android-platform-tools`) or set $PHONE_CAM_ADB."
        )
    base = [adb, *(("-s", serial) if serial else ())]
    devices = subprocess.run(
        [*base, "devices"], capture_output=True, text=True, check=False
    )
    # `adb devices` lists one device per line after the header as "<serial>\tdevice".
    if "\tdevice" not in devices.stdout:
        raise ConnectionError(
            "No authorized USB device for adb. Checklist:\n"
            "  - Tablet is plugged in and Developer options -> USB debugging is on.\n"
            "  - You accepted the 'Allow USB debugging?' prompt on the tablet.\n"
            "  - `adb devices` lists it as 'device' (not 'unauthorized'/'offline')."
        )
    forward = subprocess.run(
        [*base, "forward", f"tcp:{local_port}", f"tcp:{remote_port}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if forward.returncode != 0:
        raise ConnectionError(
            f"adb forward tcp:{local_port} tcp:{remote_port} failed: "
            f"{forward.stderr.strip() or forward.stdout.strip()}"
        )


def _ensure_usb_forward() -> None:
    """Establish the adb port-forward described by the PHONE_CAM_ADB_* env vars."""
    port = os.environ.get(PORT_ENV, "8080")
    ensure_adb_forward(
        os.environ.get(ADB_LOCAL_PORT_ENV, port),
        os.environ.get(ADB_REMOTE_PORT_ENV, port),
        adb=os.environ.get(ADB_BIN_ENV, "adb"),
        serial=os.environ.get(ADB_SERIAL_ENV) or None,
    )


def phone_url_from_env() -> str | None:
    """Resolve the stream URL from environment / .env.

    Prefers an explicit ``PHONE_CAM_URL``; otherwise assembles it from
    ``PHONE_CAM_HOST`` (+ optional ``PHONE_CAM_PORT`` and ``PHONE_CAM_PATH``).
    Returns None if no host/url is configured.
    """
    url = os.environ.get(DEFAULT_URL_ENV)
    if url:
        return url
    path = os.environ.get(PATH_ENV, "/video")
    if not path.startswith("/"):
        path = "/" + path
    # USB mode reaches the phone through an adb forward, so the stream lives on
    # localhost:<local_port> regardless of the phone's WiFi IP.
    if _usb_enabled():
        port = os.environ.get(PORT_ENV, "8080")
        local_port = os.environ.get(ADB_LOCAL_PORT_ENV, port)
        return f"http://localhost:{local_port}{path}"
    host = os.environ.get(HOST_ENV)
    if not host:
        return None
    port = os.environ.get(PORT_ENV, "8080")
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


def adopt_network_stream_profile() -> None:
    """Make OpenCVCamera auto-detect resolution/FPS for network (URL) streams.

    A LeRobot *robot* config requires width/height/fps on every camera, so when the
    phone is used in a robot (not just the dashboard preview) those must be set. But an
    MJPEG/IP stream is read-only: OpenCV can't change its resolution or FPS, so
    ``VideoCapture.set`` returns ``False`` and LeRobot's enforcement
    (``_validate_width_and_height`` / ``_validate_fps``) raises — even when the stream
    is already at the requested resolution. This patches ``_configure_capture_settings``
    so a URL-based camera adopts whatever profile the stream actually provides (LeRobot's
    own behavior when width/height/fps are ``None``) instead of enforcing the declared
    values. Local (device-index) cameras are untouched. Idempotent; call once before
    connecting the robot.
    """
    if getattr(
        OpenCVCamera._configure_capture_settings, "_network_stream_patched", False
    ):
        return

    original = OpenCVCamera._configure_capture_settings

    def _configure(self) -> None:
        index = self.index_or_path
        if isinstance(index, str) and "://" in index:
            # Read-only stream: drop the declared values so the original method adopts
            # the stream's actual resolution/FPS and skips the set()-based checks.
            self.width = self.height = self.fps = None
        original(self)

    _configure._network_stream_patched = True  # pyright: ignore[reportFunctionMemberAccess]
    OpenCVCamera._configure_capture_settings = _configure


def tolerate_camera_resolution_drift() -> None:
    """Let an OpenCVCamera accept frames whose size differs from its configured size.

    Some USB cameras report one resolution at connect (so validation passes) but stream a
    different one (e.g. configured 640x480 yet deliver 640x360), and a few switch modes
    mid-stream. LeRobot's background read thread rejects mismatched frames
    (``_postprocess_image`` raises), and after a handful of failures the thread dies — then
    ``get_observation`` raises and observations are silently dropped, starving the policy
    server. Since the server resizes every image to the policy's input shape anyway, the
    exact source size is irrelevant; this patches ``_postprocess_image`` to adopt the
    frame's actual size on mismatch and accept it instead of raising. Idempotent.
    """
    if getattr(OpenCVCamera._postprocess_image, "_resolution_drift_patched", False):
        return

    original = OpenCVCamera._postprocess_image

    def _postprocess(self, image: NDArray[Any]) -> NDArray[Any]:
        try:
            return original(self, image)
        except RuntimeError as e:
            if "do not match configured" not in str(e):
                raise
            # Adopt the actual frame size, then re-run (now the size check passes).
            h, w = image.shape[:2]
            self.capture_width, self.capture_height = w, h
            self.width, self.height = w, h
            return original(self, image)

    _postprocess._resolution_drift_patched = True  # pyright: ignore[reportFunctionMemberAccess]
    OpenCVCamera._postprocess_image = _postprocess


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
    # In USB mode, establish the adb forward before handing back a localhost URL
    # so whoever opens the stream next (this process or a child) can reach it.
    if _usb_enabled():
        _ensure_usb_forward()
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
