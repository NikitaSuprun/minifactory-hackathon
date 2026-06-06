"""Minimal web dashboard to control an SO-101 leader/follower arm pair.

Run:
    uv run python arm_dashboard.py
    # then open http://localhost:8041

What it does:
- Connect / disconnect the SO-101 follower (robot) + SO-101 leader (teleoperator).
- Start / stop teleoperation (leader drives follower) in a background thread,
  using LeRobot's canonical loop (get_observation -> get_action -> processors ->
  send_action) from ``lerobot_teleoperate``.
- Start / stop VLA policy inference: load a Hugging Face policy (pi0, SmolVLA, …),
  prompt it with a task string, and let it drive the follower (see policy_inference).
- Live phone-camera preview (the IP Webcam stream from phone_camera.py) as MJPEG.
- Status polling: connection state, loop FPS, latest joint commands.

Teleop and inference are mutually exclusive (both drive the follower).
Configuration comes from .env (ports via ``uv run lerobot-find-port``); the HF
token for gated policies comes from gitignored .env.local. Both arms plug into
this computer over USB.
"""

from __future__ import annotations

import os
import secrets
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import cv2
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel

# Importing phone_camera also loads .env and .env.local (HF_TOKEN) on import.
from phone_camera import open_phone_camera

# --- Configuration (from .env / environment) --------------------------------
FOLLOWER_PORT: Final[str] = os.environ.get("FOLLOWER_PORT", "")
LEADER_PORT: Final[str] = os.environ.get("LEADER_PORT", "")
FOLLOWER_ID: Final[str] = os.environ.get("ROBOT_ID", "so101_follower")
LEADER_ID: Final[str] = os.environ.get("LEADER_ID", "so101_leader")
TELEOP_FPS: Final[int] = int(os.environ.get("TELEOP_FPS", "60"))
INFERENCE_FPS: Final[int] = int(os.environ.get("INFERENCE_FPS", "30"))
DASHBOARD_PORT: Final[int] = int(os.environ.get("DASHBOARD_PORT", "8041"))
DASHBOARD_USER: Final[str] = os.environ.get("DASHBOARD_USER", "admin")
DASHBOARD_PASS: Final[str] = os.environ.get("DASHBOARD_PASS", "")
DEFAULT_POLICY_PATH: Final[str] = os.environ.get("POLICY_PATH", "lerobot/smolvla_base")
DEFAULT_POLICY_TASK: Final[str] = os.environ.get("POLICY_TASK", "Pick up the cube")
DEFAULT_POLICY_DEVICE: Final[str] = os.environ.get("POLICY_DEVICE", "")
# Repo-committed calibration (calibration/<id>.json), used for both arms.
CALIB_DIR: Final[Path] = Path(
    os.environ.get("CALIBRATION_DIR") or Path(__file__).resolve().parent / "calibration"
)


# --- Authentication ---------------------------------------------------------
# HTTP Basic Auth on every route so only people with the login/password (from
# .env) can open the dashboard or call the APIs over the network.
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
    # (heavy) robot/policy modules and runs on a machine with no arms attached.
    robot: Any = None
    teleop: Any = None
    processors: tuple[Any, Any, Any] | None = None
    policy_bundle: Any = None
    camera: Any = None
    teleop_thread: threading.Thread | None = None
    infer_thread: threading.Thread | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    infer_stop: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    fps: float = 0.0
    last_action: dict[str, float] = field(default_factory=dict)
    inference_status: str = "idle"  # idle | loading | running | error
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
        return self.infer_thread is not None and self.infer_thread.is_alive()


state: Final[AppState] = AppState()
app: Final[FastAPI] = FastAPI(title="SO-101 Arm Dashboard", dependencies=_auth_deps)
if not AUTH_ENABLED:
    print("WARNING: DASHBOARD_PASS not set in .env -> dashboard is UNAUTHENTICATED.")


class InferenceRequest(BaseModel):
    policy_path: str = DEFAULT_POLICY_PATH
    task: str = DEFAULT_POLICY_TASK
    device: str = DEFAULT_POLICY_DEVICE


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


def _disconnect_arms() -> None:
    _stop_thread(state.infer_stop, state.infer_thread)
    state.infer_thread = None
    state.inference_status = "idle"
    state.policy_bundle = None
    _stop_thread(state.stop_event, state.teleop_thread)
    state.teleop_thread = None
    for dev in (state.teleop, state.robot):
        try:
            if dev is not None and dev.is_connected:
                dev.disconnect()
        except Exception as e:  # noqa: BLE001 - best-effort cleanup
            state.error = f"disconnect: {e}"
    state.robot = state.teleop = state.processors = None


def _stop_thread(event: threading.Event, thread: threading.Thread | None) -> None:
    event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=3.0)


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
        state.fps = round(1.0 / max(time.perf_counter() - loop_start, 1e-6), 1)


def _inference_worker(policy_path: str, task: str, device: str) -> None:
    # Imported lazily: pulls in torch + the policy stack, which is heavy.
    from policy_inference import infer_action, load_policy

    try:
        state.inference_status = "loading"
        state.error = None
        bundle = load_policy(policy_path, state.robot, device=device or None)
        state.policy_bundle = bundle
        state.inference_status = "running"
        while not state.infer_stop.is_set():
            loop_start = time.perf_counter()
            to_send = infer_action(state.robot, bundle, task)
            state.robot.send_action(to_send)
            state.last_action = {k: round(float(v), 2) for k, v in to_send.items()}
            dt = time.perf_counter() - loop_start
            time.sleep(max(1.0 / INFERENCE_FPS - dt, 0.0))
            state.fps = round(1.0 / max(time.perf_counter() - loop_start, 1e-6), 1)
    except Exception as e:  # noqa: BLE001 - surface to the UI
        state.error = f"inference: {e}"
        state.inference_status = "error"
    else:
        state.inference_status = "idle"


# --- Camera preview ---------------------------------------------------------
def _get_camera() -> Any:
    if state.camera is None:
        with state.lock:
            if state.camera is None:
                state.camera = open_phone_camera()
    return state.camera


def _mjpeg_generator() -> Iterator[bytes]:
    cam = _get_camera()
    while True:
        try:
            frame_rgb = cam.async_read(timeout_ms=2000)
        except Exception:  # noqa: BLE001 - skip a dropped frame, keep streaming
            time.sleep(0.05)
            continue
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")


# --- Routes -----------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


@app.get("/status")
def status() -> JSONResponse:
    return JSONResponse(
        {
            "connected": state.connected,
            "teleop_running": state.teleop_running,
            "inference_running": state.inference_running,
            "inference_status": state.inference_status,
            "fps": state.fps,
            "last_action": state.last_action,
            "error": state.error,
            "follower_port": FOLLOWER_PORT or None,
            "leader_port": LEADER_PORT or None,
        }
    )


@app.post("/connect")
def connect() -> dict[str, str]:
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
    _stop_thread(state.stop_event, state.teleop_thread)
    state.teleop_thread = None
    return {"status": "teleop stopped"}


@app.post("/inference/start")
def inference_start(req: InferenceRequest) -> dict[str, str]:
    if not state.connected:
        raise HTTPException(status_code=400, detail="Connect the arms first.")
    if state.teleop_running:
        raise HTTPException(status_code=409, detail="Stop teleop first.")
    if state.inference_running:
        return {"status": "already running"}
    state.infer_stop.clear()
    state.infer_thread = threading.Thread(
        target=_inference_worker,
        args=(req.policy_path, req.task, req.device),
        daemon=True,
    )
    state.infer_thread.start()
    return {"status": "inference starting", "policy": req.policy_path, "task": req.task}


@app.post("/inference/stop")
def inference_stop() -> dict[str, str]:
    _stop_thread(state.infer_stop, state.infer_thread)
    state.infer_thread = None
    state.inference_status = "idle"
    return {"status": "inference stopped"}


# TODO(next): POST /record/start|stop -> wrap lerobot record (LeRobotDataset).


@app.get("/camera.mjpeg")
def camera() -> StreamingResponse:
    try:
        _get_camera()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"camera: {e}") from e
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


INDEX_HTML: Final[str] = """<!doctype html>
<html><head><meta charset="utf-8"><title>SO-101 Dashboard</title>
<style>
  body{font-family:system-ui,sans-serif;margin:24px;background:#0f1115;color:#e6e6e6}
  h1{font-size:20px} h3{margin-bottom:6px}
  button{font-size:15px;padding:8px 14px;margin:4px;border:0;border-radius:6px;
  background:#2d6cdf;color:#fff;cursor:pointer} button.stop{background:#c0392b}
  input{font-size:14px;padding:7px;margin:4px 0;width:340px;border-radius:6px;
  border:1px solid #333;background:#171a21;color:#e6e6e6}
  #cam{max-width:640px;border-radius:8px;background:#000}
  pre{background:#171a21;padding:12px;border-radius:8px;max-width:640px;overflow:auto}
  .row{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start}
  .card{background:#141821;padding:14px 16px;border-radius:10px;margin-bottom:14px}
</style></head><body>
<h1>SO-101 Arm Dashboard</h1>
<div class="card">
  <button onclick="post('/connect')">Connect arms</button>
  <button class="stop" onclick="post('/disconnect')">Disconnect</button>
  <button onclick="post('/teleop/start')">Start teleop</button>
  <button class="stop" onclick="post('/teleop/stop')">Stop teleop</button>
</div>
<div class="card">
  <h3>VLA policy inference</h3>
  <div>HF policy repo:<br><input id="policy" value="lerobot/smolvla_base"></div>
  <div>Task prompt:<br><input id="task" value="Pick up the cube"></div>
  <div>Device (blank = auto):<br><input id="device" placeholder="mps / cuda / cpu"></div>
  <button onclick="startInfer()">Run inference</button>
  <button class="stop" onclick="post('/inference/stop')">Stop inference</button>
</div>
<div class="row">
  <div><h3>Camera (phone)</h3><img id="cam" src="/camera.mjpeg"
       onerror="this.alt='camera unavailable'"></div>
  <div><h3>Status</h3><pre id="status">loading…</pre></div>
</div>
<script>
async function post(p){
  try{const r=await fetch(p,{method:'POST'});
    if(!r.ok){alert((await r.json()).detail||r.statusText);}}
  catch(e){alert(e);} refresh();
}
async function startInfer(){
  const body={policy_path:policy.value,task:task.value,device:device.value};
  try{const r=await fetch('/inference/start',
    {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!r.ok){alert((await r.json()).detail||r.statusText);}}
  catch(e){alert(e);} refresh();
}
async function refresh(){
  try{const r=await fetch('/status');
    document.getElementById('status').textContent=JSON.stringify(await r.json(),null,2);}
  catch(e){}
}
setInterval(refresh,1000); refresh();
</script></body></html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT)
