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
import os
import signal
import sys
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


def _release_and_exit(signum: int, frame: Any) -> None:
    camera_lock.release()
    sys.exit(0)


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
    args = parser.parse_args()

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

    listener, events = init_keyboard_listener()
    if display:
        init_rerun(session_name="recording")

    robot.connect()
    teleop.connect()

    recorded_episodes = 0
    try:
        with VideoEncodingManager(dataset):
            while recorded_episodes < num_episodes and not events["stop_recording"]:
                log_say(f"Recording episode {recorded_episodes + 1} of {num_episodes}")
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
    finally:
        log_say("Stop recording", blocking=True)
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if listener is not None:
            listener.stop()

        log_say(f"Pushing {recorded_episodes} episode(s) to {repo_id}")
        dataset.push_to_hub(private=True)
        print(f"\nDone. Dataset at https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
