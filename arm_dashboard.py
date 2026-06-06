"""Operator dashboard for the SO-101 arm + remote VLA inference.

Run:
    uv run python arm_dashboard.py        # http://localhost:8041

Capabilities:
- Connect/disconnect the SO-101 follower + leader; start/stop teleoperation
  (leader drives follower) in a background thread.
- Run VLA inference **remotely** on the GPU box: clicking Run inference frees the
  arm + wrist camera and launches ``run_robot_client.py`` as a managed subprocess
  (device cuda, model ``lerobot/smolvla_base``, editable task). Stop kills it.
- Live phone + USB-wrist camera previews with per-camera FPS.
- Clean, state-aware UI: status pills, joint table, buttons that enable/disable by
  state, plus GPU-box (SSH-tailed) and client log panels.

Config from .env (ports, calibration, server address); HF token from .env.local.
"""

from __future__ import annotations

import json
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, cast, get_args

import cv2
import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import camera_lock

# Importing phone_camera also loads .env and .env.local (HF_TOKEN) on import.
from phone_camera import open_phone_camera

# --- Configuration (from .env / environment) --------------------------------
_HERE: Final[Path] = Path(__file__).resolve().parent
FOLLOWER_PORT: Final[str] = os.environ.get("FOLLOWER_PORT", "")
LEADER_PORT: Final[str] = os.environ.get("LEADER_PORT", "")
FOLLOWER_ID: Final[str] = os.environ.get("ROBOT_ID", "so101_follower")
LEADER_ID: Final[str] = os.environ.get("LEADER_ID", "so101_leader")
TELEOP_FPS: Final[int] = int(os.environ.get("TELEOP_FPS", "60"))
DASHBOARD_PORT: Final[int] = int(os.environ.get("DASHBOARD_PORT", "8041"))
DASHBOARD_USER: Final[str] = os.environ.get("DASHBOARD_USER", "admin")
DASHBOARD_PASS: Final[str] = os.environ.get("DASHBOARD_PASS", "")
POLICY_PATH: Final[str] = os.environ.get("POLICY_PATH", "lerobot/smolvla_base")
DEFAULT_POLICY_TASK: Final[str] = os.environ.get("POLICY_TASK", "Pick up the cube")
SERVER_POLICY_DEVICE: Final[str] = os.environ.get("SERVER_POLICY_DEVICE", "cuda")
POLICY_SERVER_ADDRESS: Final[str] = os.environ.get("POLICY_SERVER_ADDRESS", "")
GPU_SSH_HOST: Final[str] = os.environ.get("GPU_SSH_HOST", "")
CALIB_DIR: Final[Path] = Path(
    os.environ.get("CALIBRATION_DIR") or _HERE / "calibration"
)
ARM_CAM_INDEX: Final[str] = os.environ.get("ARM_CAM_INDEX", "")
ARM_CAM_WIDTH: Final[int] = int(os.environ.get("ARM_CAM_WIDTH", "640"))
ARM_CAM_HEIGHT: Final[int] = int(os.environ.get("ARM_CAM_HEIGHT", "480"))
CAM3_INDEX: Final[str] = os.environ.get("CAM3_INDEX", "")
CAM3_WIDTH: Final[int] = int(os.environ.get("CAM3_WIDTH", "640"))
CAM3_HEIGHT: Final[int] = int(os.environ.get("CAM3_HEIGHT", "480"))
# "opencv" (cv2 device index) or "oak" (Luxonis depthai device, e.g. OAK-D Lite).
CAM3_SOURCE: Final[str] = os.environ.get("CAM3_SOURCE", "opencv").strip().lower()
CLIENT_LOG: Final[Path] = _HERE / "logs" / "client.out"
# Dataset recording: the recorder subprocess's stdout, the dashboard->recorder command
# file, and the recorder->dashboard progress file (all under the gitignored logs/ dir).
RECORD_LOG: Final[Path] = _HERE / "logs" / "record.out"
RECORD_CTRL: Final[Path] = _HERE / "logs" / "record.ctrl.json"
RECORD_PROGRESS: Final[Path] = _HERE / "logs" / "record.progress.json"
# Local LeRobot dataset cache (where record_dataset.py / the Hub store datasets on disk).
LEROBOT_ROOT: Final[Path] = Path(
    os.environ.get("HF_LEROBOT_HOME")
    or Path(os.environ.get("HF_HOME", "~/.cache/huggingface")) / "lerobot"
).expanduser()
SERVER_LOG_REMOTE: Final[str] = "~/minifactory-hackathon/policy_server.out"
# Built Vite SPA (committed). Served when present; otherwise the inline HTML below.
DIST: Final[Path] = _HERE / "frontend" / "dist"


# --- Authentication ---------------------------------------------------------
_basic: Final[HTTPBasic] = HTTPBasic()


def require_auth(credentials: HTTPBasicCredentials = Depends(_basic)) -> str:
    user_ok: bool = secrets.compare_digest(credentials.username, DASHBOARD_USER)
    pass_ok: bool = secrets.compare_digest(credentials.password, DASHBOARD_PASS)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


AUTH_ENABLED: Final[bool] = bool(DASHBOARD_PASS)
_auth_deps: Final[list[Any]] = [Depends(require_auth)] if AUTH_ENABLED else []


@dataclass
class AppState:
    # lerobot device objects are kept as Any so the dashboard imports without the
    # (heavy) robot modules and runs on a machine with no arms attached.
    robot: Any = None
    teleop: Any = None
    processors: tuple[Any, Any, Any] | None = None
    camera: Any = None
    wrist_cam: Any = None
    cam3: Any = None
    teleop_thread: threading.Thread | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Serializes wrist-cam read() against release() so freeing it (on inference start)
    # can't race a concurrent read in the preview generator and segfault OpenCV.
    wrist_cam_lock: threading.Lock = field(default_factory=threading.Lock)
    cam3_lock: threading.Lock = field(default_factory=threading.Lock)
    phone_cam_lock: threading.Lock = field(default_factory=threading.Lock)
    infer_proc: subprocess.Popen[bytes] | None = None
    infer_task: str = DEFAULT_POLICY_TASK
    inference_status: str = "idle"  # idle | running | error
    # Dataset recording (managed subprocess, mirrors the inference fields above).
    record_proc: subprocess.Popen[bytes] | None = None
    record_status: RecordStatus = "idle"
    record_repo_id: str | None = None  # repo of the in-flight / just-finished run
    record_last_done_repo: str | None = None  # so the Datasets tab can auto-select it
    control_fps: float = 0.0
    last_action: dict[str, float] = field(default_factory=dict)
    phone_fps: float = 0.0
    phone_fps_at: float = 0.0
    wrist_fps: float = 0.0
    wrist_fps_at: float = 0.0
    cam3_fps: float = 0.0
    cam3_fps_at: float = 0.0
    error: str | None = None

    @property
    def connected(self) -> bool:
        return (
            self.robot is not None
            and self.teleop is not None
            and self.robot.is_connected
            and self.teleop.is_connected
        )

    @property
    def teleop_running(self) -> bool:
        return self.teleop_thread is not None and self.teleop_thread.is_alive()

    @property
    def inference_running(self) -> bool:
        return self.infer_proc is not None and self.infer_proc.poll() is None

    @property
    def recording_running(self) -> bool:
        return self.record_proc is not None and self.record_proc.poll() is None


state: Final[AppState] = AppState()
app: Final[FastAPI] = FastAPI(title="SO-101 Arm Dashboard", dependencies=_auth_deps)
if not AUTH_ENABLED:
    print("WARNING: DASHBOARD_PASS not set in .env -> dashboard is UNAUTHENTICATED.")


RecordStatus = Literal[
    "idle",
    "starting",
    "recording",
    "resetting",
    "finalizing",
    "pushing",
    "done",
    "error",
]
_RECORD_STATUSES: Final[frozenset[str]] = frozenset(get_args(RecordStatus))


class InferenceRequest(BaseModel):
    task: str = DEFAULT_POLICY_TASK


class RecordRequest(BaseModel):
    name: str
    task: str = DEFAULT_POLICY_TASK
    episodes: int = 5
    episode_time: int = 60
    reset_time: int = 15
    fps: int = 30


class RecordEventRequest(BaseModel):
    event: Literal["end_episode", "rerecord"]


# --- Arm control ------------------------------------------------------------
def _connect_no_prompt(dev: Any) -> None:
    """Connect a follower/leader without lerobot's interactive calibration prompt.

    lerobot's ``connect()`` re-runs ``calibrate()`` whenever the motors' stored
    calibration doesn't match the committed file (e.g. after a power-cycle). With
    a file present that path blocks on ``input()`` ("Press ENTER to use provided
    calibration file...") — invisible from the browser, so Connect appears to
    hang. We pass ``calibrate=False`` to skip the prompt, then replicate the
    ENTER path ourselves: write the committed calibration straight to the motors.
    ``configure()`` (run inside ``connect``) only touches operating-mode/PID
    registers, so writing calibration afterwards is equivalent and safe.
    """
    dev.connect(calibrate=False)
    if not dev.is_calibrated and dev.calibration:
        dev.bus.write_calibration(dev.calibration)


def _connect_arms() -> None:
    # Imported lazily so the dashboard/camera still run on a machine with no arms.
    from lerobot.processor import make_default_processors
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    if not FOLLOWER_PORT or not LEADER_PORT:
        raise RuntimeError(
            "FOLLOWER_PORT / LEADER_PORT are not set in .env. "
            "Run `uv run lerobot-find-port` to discover them."
        )

    robot = SO101Follower(
        SO101FollowerConfig(
            port=FOLLOWER_PORT, id=FOLLOWER_ID, calibration_dir=CALIB_DIR
        )
    )
    teleop = SO101Leader(
        SO101LeaderConfig(port=LEADER_PORT, id=LEADER_ID, calibration_dir=CALIB_DIR)
    )
    _connect_no_prompt(robot)
    _connect_no_prompt(teleop)

    state.robot = robot
    state.teleop = teleop
    state.processors = make_default_processors()
    state.error = None


def _stop_teleop_thread() -> None:
    state.stop_event.set()
    if state.teleop_thread is not None and state.teleop_thread.is_alive():
        state.teleop_thread.join(timeout=3.0)
    state.teleop_thread = None


def _release_wrist_cam() -> None:
    # Hold wrist_cam_lock so we never release while the preview generator is mid-read().
    with state.wrist_cam_lock:
        if state.wrist_cam is not None:
            try:
                state.wrist_cam.release()
            except Exception:  # noqa: BLE001 - best-effort
                pass
            state.wrist_cam = None


def _release_cam3() -> None:
    with state.cam3_lock:
        if state.cam3 is not None:
            try:
                state.cam3.release()
            except Exception:  # noqa: BLE001 - best-effort
                pass
            state.cam3 = None


def _release_camera() -> None:
    # Phone camera (lerobot OpenCVCamera); disconnect stops its background read thread.
    with state.phone_cam_lock:
        if state.camera is not None:
            try:
                state.camera.disconnect()
            except Exception:  # noqa: BLE001 - best-effort
                pass
            state.camera = None


def _release_all_cameras() -> None:
    _release_camera()
    _release_wrist_cam()
    _release_cam3()


def inference_active() -> bool:
    """True when the model owns the cameras: our own inference subprocess is running,
    or another process (e.g. an external ``run_robot_client.py``) holds the camera lock."""
    return state.inference_running or camera_lock.active()


def _disconnect_arms() -> None:
    _stop_teleop_thread()
    for dev in (state.teleop, state.robot):
        try:
            if dev is not None and dev.is_connected:
                dev.disconnect()
        except Exception as e:  # noqa: BLE001 - best-effort cleanup
            state.error = f"disconnect: {e}"
    state.robot = state.teleop = state.processors = None


def _teleop_worker() -> None:
    assert state.processors is not None
    teleop_action_processor, robot_action_processor, _ = state.processors
    while not state.stop_event.is_set():
        loop_start = time.perf_counter()
        try:
            obs = state.robot.get_observation()
            raw_action = state.teleop.get_action()
            action = teleop_action_processor((raw_action, obs))
            to_send = robot_action_processor((action, obs))
            state.robot.send_action(to_send)
            state.last_action = {k: round(float(v), 2) for k, v in to_send.items()}
        except Exception as e:  # noqa: BLE001 - surface to the UI, stop the loop
            state.error = f"teleop: {e}"
            break
        dt = time.perf_counter() - loop_start
        time.sleep(max(1.0 / TELEOP_FPS - dt, 0.0))
        state.control_fps = round(1.0 / max(time.perf_counter() - loop_start, 1e-6), 1)
    state.control_fps = 0.0


# --- Remote inference (managed subprocess) ----------------------------------
def _start_inference(task: str) -> None:
    # Free the hardware so run_robot_client.py can own the arm + all cameras.
    _disconnect_arms()
    _release_all_cameras()
    CLIENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    state.infer_task = task
    state.error = None
    logf = CLIENT_LOG.open("wb")
    state.infer_proc = subprocess.Popen(
        [sys.executable, str(_HERE / "run_robot_client.py")],
        cwd=str(_HERE),
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ, "POLICY_TASK": task},
    )
    state.inference_status = "running"


def _stop_inference() -> None:
    proc = state.infer_proc
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
    state.infer_proc = None
    state.inference_status = "idle"


def _refresh_inference_state() -> None:
    """Detect a finished/crashed inference subprocess and update status."""
    proc = state.infer_proc
    if proc is not None and proc.poll() is not None:
        if state.inference_status == "running":
            clean = proc.returncode in (0, -signal.SIGTERM)
            state.inference_status = "idle" if clean else "error"
            if not clean:
                state.error = f"inference client exited (code {proc.returncode})"
        state.infer_proc = None


# --- Dataset recording (managed subprocess) ---------------------------------
def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _start_recording(
    name: str, task: str, episodes: int, episode_time: int, reset_time: int, fps: int
) -> None:
    # Free the hardware so record_dataset.py can own the arm + leader + all cameras.
    _disconnect_arms()
    _release_all_cameras()
    RECORD_LOG.parent.mkdir(parents=True, exist_ok=True)
    # Clear stale control/progress files so a new run starts clean.
    for p in (RECORD_CTRL, RECORD_PROGRESS):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    state.error = None
    state.record_status = "starting"
    state.record_repo_id = None
    logf = RECORD_LOG.open("wb")
    state.record_proc = subprocess.Popen(
        [
            sys.executable,
            str(_HERE / "record_dataset.py"),
            "--name",
            name,
            "--task",
            task,
            "--episodes",
            str(episodes),
            "--episode-time",
            str(episode_time),
            "--reset-time",
            str(reset_time),
            "--fps",
            str(fps),
            "--no-display",
            "--control-file",
            str(RECORD_CTRL),
            "--progress-file",
            str(RECORD_PROGRESS),
        ],
        cwd=str(_HERE),
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={**os.environ},
    )


def _write_record_ctrl(cmd: str) -> None:
    """Send a command (stop | end_episode | rerecord) to the recorder subprocess."""
    try:
        RECORD_CTRL.write_text(json.dumps({"cmd": cmd, "at": time.time()}))
    except OSError:
        pass


def _stop_recording() -> None:
    """Request a *graceful* stop: the recorder finalizes + pushes, then exits.

    Unlike inference, we must NOT SIGKILL on a short deadline — finalize()/push_to_hub()
    can take tens of seconds. We write the stop command, nudge with SIGTERM (the recorder
    treats it as a graceful-stop request in control-file mode), and return immediately;
    _refresh_record_state() reaps the process once it finishes in the background.
    """
    proc = state.record_proc
    if proc is not None and proc.poll() is None:
        _write_record_ctrl("stop")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:  # noqa: BLE001 - best-effort backup nudge
            pass
        if state.record_status in ("starting", "recording", "resetting"):
            state.record_status = "finalizing"


def _record_event(event: str) -> None:
    if event in ("end_episode", "rerecord"):
        _write_record_ctrl(event)


def _refresh_record_state() -> None:
    """Merge the recorder's progress file into state and reap a finished subprocess."""
    prog = _read_json(RECORD_PROGRESS)
    if prog:
        phase = prog.get("phase")
        if phase in _RECORD_STATUSES:
            state.record_status = cast(RecordStatus, phase)
        state.record_repo_id = prog.get("repo_id") or state.record_repo_id
    proc = state.record_proc
    if proc is not None and proc.poll() is not None:
        clean = proc.returncode in (0, -signal.SIGTERM)
        if prog.get("phase") == "done" or (clean and state.record_status != "error"):
            state.record_status = "done"
            state.record_last_done_repo = state.record_repo_id
        else:
            state.record_status = "error"
            if prog.get("phase") != "error":
                state.error = f"recorder exited (code {proc.returncode})"
        state.record_proc = None


# --- Logs -------------------------------------------------------------------
_srv_log_text: str = ""
_srv_log_at: float = 0.0


def _server_logs() -> str:
    """SSH-tail the policy server log on the GPU box (cached ~3s)."""
    global _srv_log_text, _srv_log_at
    if not GPU_SSH_HOST:
        return "(GPU_SSH_HOST not set in .env)"
    now = time.perf_counter()
    if _srv_log_text and now - _srv_log_at < 3.0:
        return _srv_log_text
    try:
        out = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                GPU_SSH_HOST,
                f"tail -n 80 {SERVER_LOG_REMOTE}",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        _srv_log_text = out.stdout.strip() or out.stderr.strip() or "(no output)"
    except Exception as e:  # noqa: BLE001
        _srv_log_text = f"(server log unavailable: {e})"
    _srv_log_at = now
    return _srv_log_text


def _client_logs() -> str:
    if not CLIENT_LOG.is_file():
        return "(no client log yet)"
    return CLIENT_LOG.read_bytes()[-8000:].decode("utf-8", "replace")


_reach: bool = False
_reach_at: float = 0.0


def _server_reachable() -> bool:
    """Cheap cached probe of the policy server's gRPC port."""
    global _reach, _reach_at
    now = time.perf_counter()
    if now - _reach_at < 5.0:
        return _reach
    _reach_at = now
    host, _, port = POLICY_SERVER_ADDRESS.rpartition(":")
    if not host or not port.isdigit():
        _reach = False
        return _reach
    try:
        with socket.create_connection((host, int(port)), timeout=1.5):
            _reach = True
    except Exception:  # noqa: BLE001
        _reach = False
    return _reach


# --- Camera previews --------------------------------------------------------
def _tick_fps(times: list[float]) -> float:
    """Append now to a rolling 1s window; return frames in the last second (~FPS)."""
    now = time.perf_counter()
    times.append(now)
    cutoff = now - 1.0
    while times and times[0] < cutoff:
        times.pop(0)
    return float(len(times))


_PLACEHOLDER_CACHE: dict[str, bytes] = {}


def _placeholder_jpeg(label: str) -> bytes:
    """Cached gray JPEG shown in a tile while the model owns that camera."""
    if label not in _PLACEHOLDER_CACHE:
        img = np.full((480, 640, 3), 38, dtype=np.uint8)
        cv2.putText(
            img,
            "camera in use",
            (160, 225),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            f"{label}: held by inference or recording",
            (60, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
        ok, jpg = cv2.imencode(".jpg", img)
        _PLACEHOLDER_CACHE[label] = jpg.tobytes() if ok else b""
    return _PLACEHOLDER_CACHE[label]


def _mjpeg_chunk(jpg: bytes) -> bytes:
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"


def _get_camera() -> Any:
    if inference_active():
        raise RuntimeError("cameras are owned by the running inference")
    if state.camera is None:
        with state.lock:
            if state.camera is None:
                state.camera = open_phone_camera()
    return state.camera


def _get_wrist_cam() -> Any:
    if inference_active():
        raise RuntimeError("cameras are owned by the running inference")
    if not ARM_CAM_INDEX:
        raise RuntimeError("ARM_CAM_INDEX is not set in .env")
    if state.wrist_cam is None:
        with state.lock:
            if state.wrist_cam is None:
                cap = cv2.VideoCapture(int(ARM_CAM_INDEX))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, ARM_CAM_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ARM_CAM_HEIGHT)
                if not cap.isOpened():
                    cap.release()
                    raise RuntimeError(
                        f"failed to open wrist camera index {ARM_CAM_INDEX}"
                    )
                state.wrist_cam = cap
    return state.wrist_cam


def _open_cam3() -> Any:
    # OAK-D (depthai) is an XLink device with no OpenCV index, so it needs its own
    # VideoCapture-like source; otherwise fall back to a plain cv2 device index.
    if CAM3_SOURCE == "oak":
        from oak_camera import OakCamera

        cap = OakCamera(width=CAM3_WIDTH, height=CAM3_HEIGHT)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError("failed to open OAK camera (cam3)")
        return cap
    if not CAM3_INDEX:
        raise RuntimeError("CAM3_INDEX is not set in .env")
    cap = cv2.VideoCapture(int(CAM3_INDEX))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM3_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM3_HEIGHT)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"failed to open cam3 index {CAM3_INDEX}")
    return cap


def _get_cam3() -> Any:
    if inference_active():
        raise RuntimeError("cameras are owned by the running inference")
    if state.cam3 is None:
        with state.lock:
            if state.cam3 is None:
                state.cam3 = _open_cam3()
    return state.cam3


def _mjpeg_generator() -> Iterator[bytes]:
    times: list[float] = []
    while True:
        # While the model owns the cameras, don't open/read this device — show a
        # placeholder. The stream stays open so it resumes live when inference ends.
        if inference_active():
            yield _mjpeg_chunk(_placeholder_jpeg("phone"))
            time.sleep(0.3)
            continue
        try:
            cam = _get_camera()
            frame_rgb = cam.async_read(timeout_ms=2000)
        except Exception:  # noqa: BLE001 - skip a dropped frame, keep streaming
            time.sleep(0.05)
            continue
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            continue
        state.phone_fps, state.phone_fps_at = _tick_fps(times), time.perf_counter()
        yield _mjpeg_chunk(jpg.tobytes())


def _wrist_mjpeg_generator() -> Iterator[bytes]:
    times: list[float] = []
    while True:
        if inference_active():
            yield _mjpeg_chunk(_placeholder_jpeg("wrist"))
            time.sleep(0.3)
            continue
        # Read under the lock and re-fetch each iteration so a release (by inference or
        # the watcher) can't race a concurrent read of freed memory.
        try:
            _get_wrist_cam()
        except Exception:  # noqa: BLE001 - camera unavailable; keep the stream alive
            time.sleep(0.3)
            continue
        with state.wrist_cam_lock:
            cap = state.wrist_cam
            if cap is None:  # released between _get_wrist_cam and here; retry
                continue
            ok, frame = cap.read()  # already BGR
        if not ok:
            time.sleep(0.05)
            continue
        ok2, jpg = cv2.imencode(".jpg", frame)
        if not ok2:
            continue
        state.wrist_fps, state.wrist_fps_at = _tick_fps(times), time.perf_counter()
        yield _mjpeg_chunk(jpg.tobytes())


def _cam3_mjpeg_generator() -> Iterator[bytes]:
    times: list[float] = []
    while True:
        if inference_active():
            yield _mjpeg_chunk(_placeholder_jpeg("camera3"))
            time.sleep(0.3)
            continue
        # Read under the lock and re-fetch each iteration so a release (by inference or
        # the watcher) can't race a concurrent read of freed memory.
        try:
            _get_cam3()
        except Exception:  # noqa: BLE001 - camera unavailable; keep the stream alive
            time.sleep(0.3)
            continue
        with state.cam3_lock:
            cap = state.cam3
            if cap is None:  # released between _get_cam3 and here; retry
                continue
            ok, frame = cap.read()  # already BGR
        if not ok:
            time.sleep(0.05)
            continue
        ok2, jpg = cv2.imencode(".jpg", frame)
        if not ok2:
            continue
        state.cam3_fps, state.cam3_fps_at = _tick_fps(times), time.perf_counter()
        yield _mjpeg_chunk(jpg.tobytes())


# --- Routes -----------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> Any:
    # Serve the built Vite SPA when present (auth-gated here so the browser caches
    # Basic-Auth creds for the same-origin asset/API requests); else inline HTML.
    spa = DIST / "index.html"
    if spa.is_file():
        return FileResponse(spa)
    return HTMLResponse(INDEX_HTML)


@app.get("/status")
def status() -> JSONResponse:
    _refresh_inference_state()
    _refresh_record_state()
    now = time.perf_counter()
    phone = state.phone_fps if now - state.phone_fps_at < 2.0 else 0.0
    wrist = state.wrist_fps if now - state.wrist_fps_at < 2.0 else 0.0
    cam3 = state.cam3_fps if now - state.cam3_fps_at < 2.0 else 0.0
    return JSONResponse(
        {
            "connected": state.connected,
            "teleop_running": state.teleop_running,
            "inference_running": state.inference_running,
            "inference_status": state.inference_status,
            "control_fps": state.control_fps,
            "camera_fps": {
                "phone": round(phone, 1),
                "wrist": round(wrist, 1),
                "camera3": round(cam3, 1),
            },
            "joints": state.last_action,
            "error": state.error,
            "device": SERVER_POLICY_DEVICE,
            "policy": POLICY_PATH,
            "task": state.infer_task,
            "cameras_locked": inference_active(),
            "server_reachable": _server_reachable(),
            "follower_port": FOLLOWER_PORT or None,
            "leader_port": LEADER_PORT or None,
            "recording_running": state.recording_running,
            "record_status": state.record_status,
            "record_repo_id": state.record_repo_id,
            "record_last_done_repo": state.record_last_done_repo,
            "record_progress": _read_json(RECORD_PROGRESS),
        }
    )


@app.get("/logs/server")
def logs_server() -> dict[str, str]:
    return {"text": _server_logs()}


@app.get("/logs/client")
def logs_client() -> dict[str, str]:
    return {"text": _client_logs()}


@app.post("/connect")
def connect() -> dict[str, str]:
    if state.inference_running:
        raise HTTPException(status_code=409, detail="Stop inference first.")
    if state.recording_running:
        raise HTTPException(status_code=409, detail="Stop recording first.")
    with state.lock:
        if state.connected:
            return {"status": "already connected"}
        try:
            _connect_arms()
        except Exception as e:  # noqa: BLE001 - report cleanly to the browser
            state.error = str(e)
            raise HTTPException(status_code=400, detail=str(e)) from e
    return {"status": "connected"}


@app.post("/disconnect")
def disconnect() -> dict[str, str]:
    with state.lock:
        _disconnect_arms()
    return {"status": "disconnected"}


@app.post("/teleop/start")
def teleop_start() -> dict[str, str]:
    if not state.connected:
        raise HTTPException(status_code=400, detail="Connect the arms first.")
    if state.inference_running:
        raise HTTPException(status_code=409, detail="Stop inference first.")
    if state.recording_running:
        raise HTTPException(status_code=409, detail="Stop recording first.")
    if state.teleop_running:
        return {"status": "already running"}
    state.stop_event.clear()
    state.teleop_thread = threading.Thread(target=_teleop_worker, daemon=True)
    state.teleop_thread.start()
    return {"status": "teleop started"}


@app.post("/teleop/stop")
def teleop_stop() -> dict[str, str]:
    _stop_teleop_thread()
    return {"status": "teleop stopped"}


@app.post("/inference/start")
def inference_start(req: InferenceRequest) -> dict[str, str]:
    if state.inference_running:
        return {"status": "already running"}
    if state.recording_running:
        raise HTTPException(status_code=409, detail="Stop recording first.")
    if not POLICY_SERVER_ADDRESS:
        raise HTTPException(
            status_code=400, detail="POLICY_SERVER_ADDRESS not set in .env."
        )
    with state.lock:
        _start_inference(req.task)
    return {"status": "inference starting", "task": req.task}


@app.post("/inference/stop")
def inference_stop() -> dict[str, str]:
    _stop_inference()
    return {"status": "inference stopped"}


@app.get("/logs/record")
def logs_record() -> dict[str, str]:
    if not RECORD_LOG.is_file():
        return {"text": "(no recorder log yet)"}
    return {"text": RECORD_LOG.read_bytes()[-8000:].decode("utf-8", "replace")}


@app.post("/record/start")
def record_start(req: RecordRequest) -> dict[str, str]:
    if state.recording_running:
        return {"status": "already recording"}
    if state.inference_running:
        raise HTTPException(status_code=409, detail="Stop inference first.")
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Dataset name is required.")
    if not os.environ.get("HF_TOKEN"):
        raise HTTPException(
            status_code=400, detail="HF_TOKEN not set in .env.local (needed to push)."
        )
    with state.lock:
        _start_recording(
            req.name.strip(),
            req.task,
            req.episodes,
            req.episode_time,
            req.reset_time,
            req.fps,
        )
    return {"status": "recording starting", "name": req.name.strip()}


@app.post("/record/stop")
def record_stop() -> dict[str, str]:
    _stop_recording()
    return {"status": "stopping"}


@app.post("/record/event")
def record_event(req: RecordEventRequest) -> dict[str, str]:
    if not state.recording_running:
        raise HTTPException(status_code=409, detail="Not recording.")
    _record_event(req.event)
    return {"status": "ok", "event": req.event}


# --- Datasets (record output: list, replay viewer, upload verification) ------
def _scan_datasets() -> list[dict[str, Any]]:
    """List local LeRobot datasets under the cache root ({user}/{name}/meta/info.json)."""
    out: list[dict[str, Any]] = []
    if not LEROBOT_ROOT.is_dir():
        return out
    for info_path in LEROBOT_ROOT.glob("*/*/meta/info.json"):
        ds_dir = info_path.parent.parent
        repo_id = str(ds_dir.relative_to(LEROBOT_ROOT))
        info = _read_json(info_path)
        if not info:
            continue
        cams = [
            k.split(".")[-1]
            for k, v in info.get("features", {}).items()
            if isinstance(v, dict) and v.get("dtype") == "video"
        ]
        out.append(
            {
                "repo_id": repo_id,
                "total_episodes": info.get("total_episodes", 0),
                "total_frames": info.get("total_frames", 0),
                "fps": info.get("fps", 0),
                "cameras": cams,
            }
        )
    out.sort(key=lambda d: d["repo_id"])
    return out


@app.get("/datasets")
def datasets() -> dict[str, list[dict[str, Any]]]:
    return {"datasets": _scan_datasets()}


@app.get("/datasets/{repo_id:path}/viewer", response_class=HTMLResponse)
def dataset_viewer(repo_id: str, episode: int = 0) -> Any:
    from make_viewer import build_viewer_html

    try:
        html = build_viewer_html(
            repo_id,
            episode,
            root=LEROBOT_ROOT,
            video_url_prefix=f"/datasets/{repo_id}/file",
        )
    except Exception as e:  # noqa: BLE001 - missing dataset/episode -> 404
        raise HTTPException(
            status_code=404, detail=f"dataset/episode not found: {e}"
        ) from e
    return HTMLResponse(html)


@app.get("/datasets/{repo_id:path}/file/{relpath:path}")
def dataset_file(repo_id: str, relpath: str) -> FileResponse:
    base = (LEROBOT_ROOT / repo_id).resolve()
    target = (base / relpath).resolve()
    # Path-traversal guard: target must stay inside the dataset dir.
    if base != target and base not in target.parents:
        raise HTTPException(status_code=403, detail="forbidden")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(
        target
    )  # FileResponse handles Range requests for <video> seeking


@app.get("/datasets/{repo_id:path}/verify")
def dataset_verify(repo_id: str) -> dict[str, Any]:
    """Compare the local dataset against its Hugging Face Hub repo to confirm the upload."""
    ds_dir = LEROBOT_ROOT / repo_id
    info = _read_json(ds_dir / "meta" / "info.json")
    if not info:
        raise HTTPException(status_code=404, detail="dataset not found")
    local = {
        "total_episodes": info.get("total_episodes", 0),
        "total_frames": info.get("total_frames", 0),
        "video_files": sum(1 for _ in (ds_dir / "videos").rglob("*.mp4")),
    }
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        files = list(api.list_repo_files(repo_id, repo_type="dataset"))
        hub = {
            "exists": True,
            "video_files": sum(1 for f in files if f.endswith(".mp4")),
            "has_info": any(f.endswith("meta/info.json") for f in files),
        }
        match = bool(hub["has_info"] and hub["video_files"] == local["video_files"])
    except Exception as e:  # noqa: BLE001 - repo missing / offline / no token
        return {
            "local": local,
            "hub": {"exists": False, "error": str(e)},
            "match": False,
        }
    return {"local": local, "hub": hub, "match": match}


# During inference the model owns the cameras: skip the pre-open probe so the stream
# starts and shows the placeholder instead of a 503. When not locked, keep the probe so
# a genuinely missing/broken camera still surfaces as 503.
@app.get("/camera.mjpeg")
def camera() -> StreamingResponse:
    if not inference_active():
        try:
            _get_camera()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"camera: {e}") from e
    return StreamingResponse(
        _mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/wrist.mjpeg")
def wrist_camera() -> StreamingResponse:
    if not inference_active():
        try:
            _get_wrist_cam()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"wrist camera: {e}") from e
    return StreamingResponse(
        _wrist_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/camera3.mjpeg")
def camera3_camera() -> StreamingResponse:
    if not inference_active():
        try:
            _get_cam3()
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=503, detail=f"cam3: {e}") from e
    return StreamingResponse(
        _cam3_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


INDEX_HTML: Final[str] = """<!doctype html>
<html><head><meta charset="utf-8"><title>SO-101 Dashboard</title>
<style>
  :root{color-scheme:dark}
  body{font-family:system-ui,sans-serif;margin:20px;background:#0f1115;color:#e6e6e6}
  h1{font-size:20px;margin:0 0 12px} h3{margin:0 0 8px;font-size:14px;color:#9aa4b2}
  .card{background:#141821;padding:14px 16px;border-radius:10px;margin-bottom:14px}
  .row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
  button{font-size:14px;padding:8px 14px;margin:4px;border:0;border-radius:6px;
    background:#2d6cdf;color:#fff;cursor:pointer}
  button.stop{background:#c0392b} button:disabled{opacity:.35;cursor:not-allowed}
  input{font-size:14px;padding:7px;border-radius:6px;border:1px solid #333;
    background:#0f1115;color:#e6e6e6;width:320px}
  .pill{display:inline-block;padding:3px 10px;border-radius:999px;font-size:13px;
    margin:3px 6px 3px 0;background:#2a2f3a}
  .on{background:#1e7e34}.off{background:#444}.warn{background:#b8860b}.err{background:#c0392b}
  table{border-collapse:collapse;font-size:13px}
  td{padding:2px 12px 2px 0} td.v{font-variant-numeric:tabular-nums;text-align:right}
  img{max-width:420px;border-radius:8px;background:#000;display:block}
  pre{background:#0b0d12;padding:10px;border-radius:8px;max-height:240px;overflow:auto;
    font-size:12px;white-space:pre-wrap;margin:0}
  .mono{font-variant-numeric:tabular-nums}
  .banner{background:#c0392b;padding:8px 12px;border-radius:8px;margin-bottom:12px;display:none}
  label{font-size:13px;color:#9aa4b2}
</style></head><body>
<h1>SO-101 Arm Dashboard</h1>
<div id="banner" class="banner"></div>

<div class="card">
  <h3>STATUS</h3>
  <div id="pills"></div>
  <div class="row" style="margin-top:8px">
    <table><tbody id="joints"></tbody></table>
  </div>
</div>

<div class="card">
  <h3>CONTROL</h3>
  <button id="b_connect" onclick="post('/connect')">Connect arms</button>
  <button id="b_disconnect" class="stop" onclick="post('/disconnect')">Disconnect</button>
  <button id="b_teleop_start" onclick="post('/teleop/start')">Start teleop</button>
  <button id="b_teleop_stop" class="stop" onclick="post('/teleop/stop')">Stop teleop</button>
</div>

<div class="card">
  <h3>INFERENCE (remote &middot; GPU box)</h3>
  <div><label>model</label> <span class="pill mono" id="model">…</span>
       <label>device</label> <span class="pill mono" id="device">…</span></div>
  <div style="margin:8px 0"><label>task</label><br><input id="task" value="Pick up the cube"></div>
  <button id="b_infer_start" onclick="startInfer()">Run inference</button>
  <button id="b_infer_stop" class="stop" onclick="post('/inference/stop')">Stop inference</button>
</div>

<div class="row">
  <div class="card"><h3>PHONE CAM <span id="phone_fps" class="mono"></span></h3>
    <img src="/camera.mjpeg" onerror="this.alt='phone camera unavailable'"></div>
  <div class="card"><h3>WRIST CAM <span id="wrist_fps" class="mono"></span></h3>
    <img src="/wrist.mjpeg" onerror="this.alt='wrist cam unavailable (used by client during inference)'"></div>
  <div class="card"><h3>CAMERA 3 <span id="cam3_fps" class="mono"></span></h3>
    <img src="/camera3.mjpeg" onerror="this.alt='cam3 unavailable (used by client during inference)'"></div>
</div>

<div class="row">
  <div class="card" style="flex:1;min-width:380px"><h3>GPU-BOX SERVER LOG</h3>
    <pre id="srvlog">…</pre></div>
  <div class="card" style="flex:1;min-width:380px"><h3>CLIENT LOG</h3>
    <pre id="clilog">…</pre></div>
</div>

<script>
function pill(label, cls){return `<span class="pill ${cls}">${label}</span>`}
async function post(p){
  try{const r=await fetch(p,{method:'POST'});
    if(!r.ok){alert((await r.json()).detail||r.statusText);}}
  catch(e){alert(e);} refresh();
}
async function startInfer(){
  if(!confirm('Run inference will release the arm to the remote client and move it. Continue?'))return;
  try{const r=await fetch('/inference/start',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({task:document.getElementById('task').value})});
    if(!r.ok){alert((await r.json()).detail||r.statusText);}}
  catch(e){alert(e);} refresh();
}
function setDisabled(id,v){document.getElementById(id).disabled=v}
async function refresh(){
  let s; try{s=await (await fetch('/status')).json();}catch(e){return}
  const inf=s.inference_running, con=s.connected, tel=s.teleop_running;
  document.getElementById('pills').innerHTML =
    pill(con?'arms: connected':'arms: disconnected', con?'on':'off') +
    pill(tel?`teleop: on (${s.control_fps} Hz)`:'teleop: off', tel?'on':'off') +
    pill(`inference: ${s.inference_status}`,
         s.inference_status==='running'?'on':s.inference_status==='error'?'err':'off') +
    pill(s.server_reachable?'server: reachable':'server: down', s.server_reachable?'on':'err');
  // joints
  const j=s.joints||{}; let rows='';
  for(const k of Object.keys(j)) rows+=`<tr><td>${k}</td><td class="v">${j[k]}</td></tr>`;
  document.getElementById('joints').innerHTML = rows || '<tr><td>(no joint data)</td></tr>';
  // error banner
  const b=document.getElementById('banner');
  if(s.error){b.style.display='block';b.textContent='⚠ '+s.error}else{b.style.display='none'}
  // inference info
  document.getElementById('model').textContent=s.policy;
  document.getElementById('device').textContent=s.device;
  // camera fps
  document.getElementById('phone_fps').textContent = s.camera_fps.phone? `${s.camera_fps.phone} fps`:'';
  document.getElementById('wrist_fps').textContent = s.camera_fps.wrist? `${s.camera_fps.wrist} fps`:'';
  document.getElementById('cam3_fps').textContent = s.camera_fps.camera3? `${s.camera_fps.camera3} fps`:'';
  // buttons
  setDisabled('b_connect', con||inf);
  setDisabled('b_disconnect', !con||inf);
  setDisabled('b_teleop_start', !con||tel||inf);
  setDisabled('b_teleop_stop', !tel);
  setDisabled('b_infer_start', inf);
  setDisabled('b_infer_stop', !inf);
}
async function refreshLogs(){
  try{document.getElementById('srvlog').textContent=(await (await fetch('/logs/server')).json()).text;}catch(e){}
  try{document.getElementById('clilog').textContent=(await (await fetch('/logs/client')).json()).text;}catch(e){}
}
setInterval(refresh,1000); setInterval(refreshLogs,4000); refresh(); refreshLogs();
</script></body></html>
"""


# Serve the SPA's hashed JS/CSS bundles (non-sensitive) when the build exists.
if (DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


def _camera_lock_watcher() -> None:
    """Release all cameras the moment the model takes the lock, so it always wins —
    even tiles no browser is viewing (whose cameras would otherwise stay cached-open)."""
    was_active = False
    while True:
        active = inference_active()
        if active and not was_active:
            _release_all_cameras()
        was_active = active
        time.sleep(0.25)


if __name__ == "__main__":
    threading.Thread(
        target=_camera_lock_watcher, name="camera_lock_watcher", daemon=True
    ).start()
    uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT)
