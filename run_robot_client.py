"""Launch LeRobot's async-inference RobotClient (run this on THIS Mac).

This owns the SO-101 follower + phone camera, streams observations to the remote
PolicyServer (run_policy_server.py), and executes the returned action chunks. It
assembles the long lerobot CLI from .env, including the phone camera URL.

    uv run python run_robot_client.py

This builds LeRobot's RobotClientConfig in-process rather than shelling out to the
`lerobot.async_inference.robot_client` CLI. The CLI routes `--robot.cameras` through
draccus, which decodes `index_or_path` (typed `int | Path`) by calling `Path(url)` and
collapses `http://` to `http:/`, corrupting the phone stream URL. Constructing the config
here keeps the URL a plain string (via `build_phone_camera_config`) and leaves the phone
camera's width/height/fps unset so LeRobot auto-detects the stream instead of trying
`VideoCapture.set` on a network stream (which raises). See phone_camera.py.
"""

from __future__ import annotations

import atexit
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Final

import camera_lock
import inference_handshake

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.transport import services_pb2  # type: ignore
from lerobot.utils.import_utils import register_third_party_plugins

# Importing recording pulls in phone_camera (which loads .env / .env.local) and the shared
# camera assembly. camera1 is required; camera2/camera3 are optional for inference.
from recording import build_cameras

_HERE: Final[Path] = Path(__file__).resolve().parent
CALIBRATION_DIR: Final[str] = os.environ.get("CALIBRATION_DIR") or str(
    _HERE / "calibration"
)

FOLLOWER_PORT: Final[str] = os.environ.get("FOLLOWER_PORT", "")
ROBOT_ID: Final[str] = os.environ.get("ROBOT_ID", "so101_follower")
SERVER_ADDRESS: Final[str] = os.environ.get("POLICY_SERVER_ADDRESS", "")
POLICY_TYPE: Final[str] = os.environ.get("POLICY_TYPE", "smolvla")
POLICY_PATH: Final[str] = os.environ.get("POLICY_PATH", "lerobot/smolvla_base")
POLICY_TASK: Final[str] = os.environ.get("POLICY_TASK", "Pick up the cube")
SERVER_POLICY_DEVICE: Final[str] = os.environ.get("SERVER_POLICY_DEVICE", "cuda")
CLIENT_DEVICE: Final[str] = os.environ.get("CLIENT_DEVICE", "cpu")
ACTIONS_PER_CHUNK: Final[str] = os.environ.get("ACTIONS_PER_CHUNK", "50")
CHUNK_SIZE_THRESHOLD: Final[str] = os.environ.get("CHUNK_SIZE_THRESHOLD", "0.5")
AGGREGATE_FN: Final[str] = os.environ.get("AGGREGATE_FN", "weighted_average")

# When set, do all the slow setup (cameras + arm + server model load + a CUDA warmup
# inference), then pause before the control loop until the dashboard writes the go flag.
# This lets "Run inference" start the arm moving almost instantly. See inference_handshake.
PREWARM: Final[bool] = os.environ.get("PREWARM", "0") == "1"
# How often to poll for the go flag while paused after prewarm.
GO_POLL_S: Final[float] = 0.05


# Seconds to wait after claiming the camera lock so the dashboard can release the
# devices (notably the OAK) before we open them.
CAMERA_LOCK_GRACE_S: Final[float] = 2.0


def _cleanup() -> None:
    camera_lock.release()
    inference_handshake.clear_ready()
    inference_handshake.clear_go()


def _release_and_exit(signum: int, frame: Any) -> None:
    # The dashboard stops inference with SIGTERM; drop the lock so it resumes promptly.
    _cleanup()
    sys.exit(0)


def main() -> None:
    if not FOLLOWER_PORT:
        sys.exit("FOLLOWER_PORT is not set in .env (run `uv run lerobot-find-port`).")
    if not SERVER_ADDRESS:
        sys.exit("POLICY_SERVER_ADDRESS is not set in .env (e.g. 192.168.1.50:8080).")

    # Claim the cameras for this run so the dashboard yields them (it watches this lock
    # and shows a placeholder instead of reading the same devices). Released on every
    # exit path; a hard kill is covered by the dashboard's stale-pid check. The grace
    # delay lets the dashboard free the devices before we open them.
    camera_lock.acquire()
    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, _release_and_exit)
    time.sleep(CAMERA_LOCK_GRACE_S)

    # Cameras sent to the policy: phone(camera1) + USB(camera2) + OAK-D/USB(camera3). All
    # three are required (same as recording) so the observation.images.<key> slots always
    # match the loaded policy (smolvla_base: camera1/camera2/camera3).
    try:
        cameras = build_cameras()
    except ValueError as e:
        sys.exit(str(e))

    robot_cfg = SO101FollowerConfig(
        port=FOLLOWER_PORT,
        id=ROBOT_ID,
        calibration_dir=Path(CALIBRATION_DIR),
        cameras=cameras,
        use_degrees=True,
    )
    client_cfg = RobotClientConfig(
        robot=robot_cfg,
        task=POLICY_TASK,
        server_address=SERVER_ADDRESS,
        policy_type=POLICY_TYPE,
        pretrained_name_or_path=POLICY_PATH,
        policy_device=SERVER_POLICY_DEVICE,
        client_device=CLIENT_DEVICE,
        actions_per_chunk=int(ACTIONS_PER_CHUNK),
        chunk_size_threshold=float(CHUNK_SIZE_THRESHOLD),
        aggregate_fn_name=AGGREGATE_FN,
    )

    # Mirrors lerobot.async_inference.robot_client's CLI entrypoint (async_client):
    # connect, run the action-receiver thread, drive the control loop, then tear down.
    register_third_party_plugins()
    client = RobotClient(client_cfg)
    # RobotClient.__init__ already connected the robot + opened the cameras (the slow part);
    # start() does the server handshake + SendPolicyInstructions, which makes the server load
    # the policy model. Neither moves the arm — that only happens in the control loop below.
    if not client.start():
        sys.exit("RobotClient failed to start (policy-server handshake failed).")

    task = client_cfg.task
    if PREWARM:
        task = _prewarm_and_wait(client, task)

    action_receiver = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver.start()
    try:
        client.control_loop(task=task)
    finally:
        client.stop()
        action_receiver.join()


def _prewarm_and_wait(client: RobotClient, task: str) -> str:
    """Warm the server (one discarded inference), mark ready, then block until the dashboard
    writes the go flag. Returns the task to run (the go flag may override it). The arm stays
    still throughout — we never call robot.send_action here."""
    # Best-effort CUDA/kernel warmup on the server: push one observation through and force a
    # single inference, discarding the result. control_loop_observation sends a must_go obs
    # (clearing the must_go flag), so we re-set it afterwards to leave the client in the same
    # state a fresh start would have for the real control loop's first observation.
    try:
        client.control_loop_observation(task)
        client.stub.GetActions(services_pb2.Empty())  # pyright: ignore[reportAttributeAccessIssue]
    except Exception as e:  # noqa: BLE001 - warmup is best-effort; model is already loaded
        print(f"[prewarm] warmup inference skipped: {e}", flush=True)
    client.must_go.set()

    inference_handshake.clear_go()  # drop any stale go from a previous run
    inference_handshake.mark_ready()
    print(
        "[prewarm] ready — cameras + arm + server model loaded; waiting for go",
        flush=True,
    )

    while client.running:
        go = inference_handshake.read_go()
        if go is not None:
            inference_handshake.clear_ready()
            inference_handshake.clear_go()
            return go or task
        time.sleep(GO_POLL_S)
    return task


if __name__ == "__main__":
    main()
