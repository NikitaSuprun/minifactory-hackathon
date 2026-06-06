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
from typing import Any, Final

import cv2
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
CLIENT_LOG: Final[Path] = _HERE / "logs" / "client.out"
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
    infer_proc: subprocess.Popen[bytes] | None = None
    infer_task: str = DEFAULT_POLICY_TASK
    inference_status: str = "idle"  # idle | running | error
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


state: Final[AppState] = AppState()
app: Final[FastAPI] = FastAPI(title="SO-101 Arm Dashboard", dependencies=_auth_deps)
if not AUTH_ENABLED:
    print("WARNING: DASHBOARD_PASS not set in .env -> dashboard is UNAUTHENTICATED.")


class InferenceRequest(BaseModel):
    task: str = DEFAULT_POLICY_TASK


# --- Arm control ------------------------------------------------------------
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
    robot.connect()
    teleop.connect()

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
    # Free the hardware so run_robot_client.py can own the arm + wrist camera.
    _disconnect_arms()
    _release_wrist_cam()
    _release_cam3()
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


def _get_camera() -> Any:
    if state.camera is None:
        with state.lock:
            if state.camera is None:
                state.camera = open_phone_camera()
    return state.camera


def _get_wrist_cam() -> Any:
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


def _get_cam3() -> Any:
    if not CAM3_INDEX:
        raise RuntimeError("CAM3_INDEX is not set in .env")
    if state.cam3 is None:
        with state.lock:
            if state.cam3 is None:
                cap = cv2.VideoCapture(int(CAM3_INDEX))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM3_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM3_HEIGHT)
                if not cap.isOpened():
                    cap.release()
                    raise RuntimeError(f"failed to open cam3 index {CAM3_INDEX}")
                state.cam3 = cap
    return state.cam3


def _mjpeg_generator() -> Iterator[bytes]:
    cam = _get_camera()
    times: list[float] = []
    while True:
        try:
            frame_rgb = cam.async_read(timeout_ms=2000)
        except Exception:  # noqa: BLE001 - skip a dropped frame, keep streaming
            time.sleep(0.05)
            continue
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            continue
        state.phone_fps, state.phone_fps_at = _tick_fps(times), time.perf_counter()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


def _wrist_mjpeg_generator() -> Iterator[bytes]:
    _get_wrist_cam()  # open it (if needed) before streaming
    times: list[float] = []
    while True:
        # Read under the lock and re-fetch each iteration: if inference released the
        # cam, state.wrist_cam is None and we stop instead of reading freed memory.
        with state.wrist_cam_lock:
            cap = state.wrist_cam
            if cap is None:
                return
            ok, frame = cap.read()  # already BGR
        if not ok:
            time.sleep(0.05)
            continue
        ok2, jpg = cv2.imencode(".jpg", frame)
        if not ok2:
            continue
        state.wrist_fps, state.wrist_fps_at = _tick_fps(times), time.perf_counter()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


def _cam3_mjpeg_generator() -> Iterator[bytes]:
    _get_cam3()  # open it (if needed) before streaming
    times: list[float] = []
    while True:
        # Read under the lock and re-fetch each iteration: if inference released the
        # cam, state.cam3 is None and we stop instead of reading freed memory.
        with state.cam3_lock:
            cap = state.cam3
            if cap is None:
                return
            ok, frame = cap.read()  # already BGR
        if not ok:
            time.sleep(0.05)
            continue
        ok2, jpg = cv2.imencode(".jpg", frame)
        if not ok2:
            continue
        state.cam3_fps, state.cam3_fps_at = _tick_fps(times), time.perf_counter()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


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
            "server_reachable": _server_reachable(),
            "follower_port": FOLLOWER_PORT or None,
            "leader_port": LEADER_PORT or None,
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


@app.get("/camera.mjpeg")
def camera() -> StreamingResponse:
    try:
        _get_camera()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"camera: {e}") from e
    return StreamingResponse(
        _mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/wrist.mjpeg")
def wrist_camera() -> StreamingResponse:
    try:
        _get_wrist_cam()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"wrist camera: {e}") from e
    return StreamingResponse(
        _wrist_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/camera3.mjpeg")
def camera3_camera() -> StreamingResponse:
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT)
