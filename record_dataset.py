"""Record a LeRobot dataset from our 3-camera SO-101 setup and push it to the Hub.

This is the recording counterpart to ``run_robot_client.py``: it owns the SO-101
follower + leader and all three cameras (phone -> camera1, USB workspace cam ->
camera2, Luxonis OAK-D -> camera3), teleoperates the follower with the leader, and
records episodes into a :class:`LeRobotDataset` which it pushes to Hugging Face.

    uv run python record_dataset.py

Why not the stock ``lerobot-record`` CLI? Same reason ``run_robot_client.py`` builds
its config in-process: draccus decodes ``--robot.cameras`` ``index_or_path`` via
``Path(url)``, collapsing the phone's ``http://`` stream URL to ``http:/``; and the
OAK-D has no OpenCV index, so it can't be expressed on the CLI at all. We assemble the
robot config here (reusing phone_camera.py / oak_lerobot_camera.py) and drive LeRobot's
``record_loop`` directly.

Recording controls (foreground terminal, via LeRobot's keyboard listener):
    ->  (right arrow)  end the current episode early
    <-  (left arrow)   re-record the current episode (discards it)
    Esc                stop recording and push to the Hub

NOTE: run this with the arms free — the web dashboard (arm_dashboard.py) must NOT be
connected to the follower/leader, or the serial ports will conflict. The camera lock
below makes the dashboard yield the cameras, but it does not cover the arms.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Final, TypeVar

import camera_lock

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.robots.so_follower import SO101Follower
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# Importing phone_camera also loads .env + .env.local (so HF_TOKEN is in os.environ).
from phone_camera import (
    adopt_network_stream_profile,
    build_phone_camera_config,
    resolve_phone_url,
    tolerate_camera_resolution_drift,
)

_HERE: Final[Path] = Path(__file__).resolve().parent
CALIBRATION_DIR: Final[str] = os.environ.get("CALIBRATION_DIR") or str(
    _HERE / "calibration"
)

FOLLOWER_PORT: Final[str] = os.environ.get("FOLLOWER_PORT", "")
LEADER_PORT: Final[str] = os.environ.get("LEADER_PORT", "")
ROBOT_ID: Final[str] = os.environ.get("ROBOT_ID", "so101_follower")
LEADER_ID: Final[str] = os.environ.get("LEADER_ID", "so101_leader")

# Camera names become the policy-facing slots (observation.images.<name>). Keep these
# aligned with run_robot_client.py / smolvla_base (camera1/camera2/camera3) so a dataset
# recorded here trains a policy that the inference path can serve unchanged.
PHONE_CAMERA_NAME: Final[str] = os.environ.get("ROBOT_CAMERA_NAME", "camera1")
PHONE_CAM_WIDTH: Final[str] = os.environ.get("PHONE_CAM_WIDTH", "640")
PHONE_CAM_HEIGHT: Final[str] = os.environ.get("PHONE_CAM_HEIGHT", "480")
PHONE_CAM_FPS: Final[str] = os.environ.get("PHONE_CAM_FPS", "30")
ARM_CAM_INDEX: Final[str] = os.environ.get("ARM_CAM_INDEX", "")
ARM_CAM_NAME: Final[str] = os.environ.get("ARM_CAM_NAME", "camera2")
ARM_CAM_WIDTH: Final[str] = os.environ.get("ARM_CAM_WIDTH", "640")
ARM_CAM_HEIGHT: Final[str] = os.environ.get("ARM_CAM_HEIGHT", "480")
ARM_CAM_FPS: Final[str] = os.environ.get("ARM_CAM_FPS", "30")
CAM3_INDEX: Final[str] = os.environ.get("CAM3_INDEX", "")
CAM3_NAME: Final[str] = os.environ.get("CAM3_NAME", "camera3")
CAM3_WIDTH: Final[str] = os.environ.get("CAM3_WIDTH", "640")
CAM3_HEIGHT: Final[str] = os.environ.get("CAM3_HEIGHT", "480")
CAM3_FPS: Final[str] = os.environ.get("CAM3_FPS", "30")
# "opencv" (cv2 device index) or "oak" (Luxonis OAK-D via depthai).
CAM3_SOURCE: Final[str] = os.environ.get("CAM3_SOURCE", "oak").strip().lower()

DEFAULT_FPS: Final[int] = int(os.environ.get("INFERENCE_FPS", "30"))
DEFAULT_TASK: Final[str] = os.environ.get("POLICY_TASK", "Pick up the cube")

# Seconds to wait after claiming the camera lock so the dashboard can release the
# devices (notably the OAK) before we open them. Mirrors run_robot_client.py.
CAMERA_LOCK_GRACE_S: Final[float] = 2.0

T = TypeVar("T")

# Set to the live ``events`` dict only in dashboard/control-file mode (see main). When set,
# SIGTERM requests a graceful stop (finalize + push) instead of a hard exit.
_events_ref: dict[str, bool] | None = None


def _release_and_exit(signum: int, frame: Any) -> None:
    # Dashboard mode: ask the record loop to stop cleanly so ``finally`` still
    # finalizes + pushes; the lock is released by the registered atexit handler.
    if _events_ref is not None:
        _events_ref["stop_recording"] = True
        _events_ref["exit_early"] = True
        return
    # Terminal mode: legacy behavior — drop the lock and exit immediately.
    camera_lock.release()
    sys.exit(0)


def _start_control_thread(path: Path, events: dict[str, bool]) -> None:
    """Poll a JSON control file and flip the shared ``events`` flags.

    Replaces lerobot's keyboard listener when launched headless from the dashboard:
    ``record_loop`` reads the very same ``events`` dict by reference, so mutating it
    here is equivalent to a key press. Mapping mirrors the keyboard exactly —
    ``stop`` = Esc, ``end_episode`` = ->, ``rerecord`` = <-.
    """

    def run() -> None:
        last_at = 0.0
        while not events["stop_recording"]:
            try:
                cmd = json.loads(path.read_text())
            except (FileNotFoundError, ValueError, OSError):
                cmd = None
            if cmd and float(cmd.get("at", 0)) > last_at:
                last_at = float(cmd["at"])
                c = cmd.get("cmd")
                if c == "stop":
                    events["stop_recording"] = True
                    events["exit_early"] = True
                elif c == "end_episode":
                    events["exit_early"] = True
                elif c == "rerecord":
                    events["rerecord_episode"] = True
                    events["exit_early"] = True
            time.sleep(0.2)

    threading.Thread(target=run, name="record_control", daemon=True).start()


def _prompt(label: str, default: T, cast: Callable[[str], T] = str) -> T:
    """Prompt with a default shown in brackets; empty input keeps the default."""
    suffix = f" [{default}]" if default != "" else ""
    while True:
        raw = input(f"{label}{suffix}: ").strip()
        if not raw:
            return default
        try:
            return cast(raw)
        except (ValueError, TypeError) as e:
            print(f"  invalid value ({e}); try again.")


def _resolve_hf_username() -> str:
    """Auto-detect the Hugging Face username from HF_TOKEN (in .env.local)."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit(
            "HF_TOKEN is not set. Put it in .env.local "
            "(get a write token at https://huggingface.co/settings/tokens)."
        )
    from huggingface_hub import whoami

    try:
        return whoami(token=token)["name"]
    except Exception as e:  # noqa: BLE001 - surface a clear, actionable message
        sys.exit(f"Could not resolve your Hugging Face username from HF_TOKEN: {e}")


def _build_cameras() -> dict[str, CameraConfig]:
    """Assemble all three cameras, aborting clearly if any is not configured.

    Mirrors run_robot_client.py's assembly but *requires* all three streams so every
    episode carries camera1/camera2/camera3.
    """
    adopt_network_stream_profile()
    tolerate_camera_resolution_drift()

    cameras: dict[str, CameraConfig] = {
        PHONE_CAMERA_NAME: build_phone_camera_config(
            resolve_phone_url(),
            width=int(PHONE_CAM_WIDTH),
            height=int(PHONE_CAM_HEIGHT),
            fps=int(PHONE_CAM_FPS),
        )
    }

    if ARM_CAM_INDEX == "":
        sys.exit(
            "ARM_CAM_INDEX is not set in .env, but all 3 cameras are required for "
            "recording (camera2 = workspace USB cam). Set it (e.g. ARM_CAM_INDEX=0)."
        )
    cameras[ARM_CAM_NAME] = OpenCVCameraConfig(
        index_or_path=int(ARM_CAM_INDEX),
        width=int(ARM_CAM_WIDTH),
        height=int(ARM_CAM_HEIGHT),
        fps=int(ARM_CAM_FPS),
    )

    if CAM3_SOURCE == "oak":
        # Registers the "oak" camera type so the robot can build it (see oak_lerobot_camera).
        from oak_lerobot_camera import OakDepthAICameraConfig

        cameras[CAM3_NAME] = OakDepthAICameraConfig(
            width=int(CAM3_WIDTH),
            height=int(CAM3_HEIGHT),
            fps=int(CAM3_FPS),
        )
    elif CAM3_INDEX != "":
        cameras[CAM3_NAME] = OpenCVCameraConfig(
            index_or_path=int(CAM3_INDEX),
            width=int(CAM3_WIDTH),
            height=int(CAM3_HEIGHT),
            fps=int(CAM3_FPS),
        )
    else:
        sys.exit(
            "camera3 is not configured, but all 3 cameras are required for recording. "
            "Set CAM3_SOURCE=oak (Luxonis OAK-D) or CAM3_INDEX=<n> (USB cam) in .env."
        )

    return cameras


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", help="Dataset name (skips the prompt).")
    parser.add_argument("--task", help="Task description (skips the prompt).")
    parser.add_argument("--episodes", type=int, help="Number of episodes.")
    parser.add_argument("--episode-time", type=int, help="Seconds per episode.")
    parser.add_argument(
        "--reset-time", type=int, help="Seconds to reset between episodes."
    )
    parser.add_argument(
        "--fps", type=int, help=f"Recording FPS (default {DEFAULT_FPS})."
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't open the rerun viewer (headless / faster).",
    )
    parser.add_argument(
        "--control-file",
        help="Headless dashboard mode: poll this JSON file for stop/end/rerecord "
        "commands instead of the keyboard listener.",
    )
    parser.add_argument(
        "--progress-file",
        help="Write phase/progress JSON here for the dashboard to display.",
    )
    args = parser.parse_args()

    control_path = Path(args.control_file) if args.control_file else None
    progress_path = Path(args.progress_file) if args.progress_file else None

    def progress(phase: str, **kw: Any) -> None:
        """Write the current recording phase for the dashboard (no-op without --progress-file)."""
        if progress_path is None:
            return
        payload = {
            "phase": phase,
            "repo_id": repo_id,
            "total_episodes": num_episodes,
            **kw,
        }
        try:
            progress_path.write_text(json.dumps(payload))
        except OSError:
            pass

    if not FOLLOWER_PORT:
        sys.exit("FOLLOWER_PORT is not set in .env (run `uv run lerobot-find-port`).")
    if not LEADER_PORT:
        sys.exit("LEADER_PORT is not set in .env (run `uv run lerobot-find-port`).")

    username = _resolve_hf_username()
    print(f"Hugging Face user: {username}\n")

    name = args.name or _prompt("Dataset name", "")
    while not name:
        print("  a dataset name is required.")
        name = _prompt("Dataset name", "")
    repo_id = f"{username}/{name}"

    task = args.task or _prompt("Task description", DEFAULT_TASK)
    num_episodes = args.episodes or _prompt("Number of episodes", 5, int)
    episode_time_s = args.episode_time or _prompt("Episode time (s)", 60, int)
    reset_time_s = (
        args.reset_time
        if args.reset_time is not None
        else _prompt("Reset time between episodes (s)", 15, int)
    )
    fps = args.fps or _prompt("FPS", DEFAULT_FPS, int)
    display = not args.no_display

    print(
        f"\nRecording {num_episodes} episode(s) -> {repo_id} (private)\n"
        f"  task: {task!r}\n"
        f"  {episode_time_s:.0f}s/episode, {reset_time_s:.0f}s reset, {fps} FPS\n"
        "  controls: ->  end episode  |  <-  re-record  |  Esc  stop & push\n"
    )

    register_third_party_plugins()

    # Claim the cameras so the dashboard yields them (notably the OAK, which can't be
    # shared); released on every exit path. Grace lets the dashboard free the devices.
    camera_lock.acquire()
    atexit.register(camera_lock.release)
    signal.signal(signal.SIGTERM, _release_and_exit)
    time.sleep(CAMERA_LOCK_GRACE_S)

    cameras = _build_cameras()

    robot = SO101Follower(
        SO101FollowerConfig(
            port=FOLLOWER_PORT,
            id=ROBOT_ID,
            calibration_dir=Path(CALIBRATION_DIR),
            cameras=cameras,
            use_degrees=True,
        )
    )
    teleop = SO101Leader(
        SO101LeaderConfig(
            port=LEADER_PORT,
            id=LEADER_ID,
            calibration_dir=Path(CALIBRATION_DIR),
            use_degrees=True,
        )
    )

    # Dataset features are derived from the robot's action/observation spaces, so the
    # three cameras above become observation.images.camera1/2/3 automatically.
    action_features = hw_to_dataset_features(robot.action_features, "action")  # pyright: ignore[reportArgumentType]
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_default_processors()
    )

    if control_path is not None:
        # Dashboard mode: no keyboard/terminal, no rerun window. Build the events dict
        # ourselves and drive it from the control file; SIGTERM stops gracefully.
        global _events_ref
        listener = None
        events = {
            "exit_early": False,
            "rerecord_episode": False,
            "stop_recording": False,
        }
        _events_ref = events
        _start_control_thread(control_path, events)
        display = False
    else:
        listener, events = init_keyboard_listener()
    if display:
        init_rerun(session_name="recording")

    robot.connect()
    teleop.connect()

    recorded_episodes = 0
    try:
        progress("starting")
        with VideoEncodingManager(dataset):
            while recorded_episodes < num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {recorded_episodes + 1} of {num_episodes}")
                progress(
                    "recording",
                    current_episode=recorded_episodes + 1,
                    episode_started_at=time.time(),
                    episode_time_s=episode_time_s,
                )
                record_loop(
                    robot=robot,
                    events=events,
                    fps=fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=episode_time_s,
                    single_task=task,
                    display_data=display,
                )

                # Unrecorded reset window to rearrange the scene; skipped after the last
                # episode (unless we're about to re-record this one).
                if not events["stop_recording"] and (
                    (recorded_episodes < num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment")
                    progress(
                        "resetting",
                        current_episode=recorded_episodes + 1,
                        episode_started_at=time.time(),
                        episode_time_s=reset_time_s,
                    )
                    record_loop(
                        robot=robot,
                        events=events,
                        fps=fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=reset_time_s,
                        single_task=task,
                        display_data=display,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode")
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
    except Exception as e:  # noqa: BLE001 - surface the failure to the dashboard
        progress("error", message=str(e))
        raise
    finally:
        log_say("Stop recording", blocking=True)
        progress("finalizing", current_episode=recorded_episodes)
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if listener is not None:
            listener.stop()

        url = f"https://huggingface.co/datasets/{repo_id}"
        log_say(f"Pushing {recorded_episodes} episode(s) to {repo_id}")
        progress("pushing", current_episode=recorded_episodes)
        dataset.push_to_hub(private=True)
        progress("done", current_episode=recorded_episodes, message=url)
        print(f"\nDone. Dataset at {url}")


if __name__ == "__main__":
    main()
