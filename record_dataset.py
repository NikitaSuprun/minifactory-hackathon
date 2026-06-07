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
robot config here (reusing recording.build_cameras) and drive LeRobot's record loop.

Recording controls (foreground terminal, via LeRobot's keyboard listener):
    ->  (right arrow)  end the current episode early
    <-  (left arrow)   re-record the current episode (discards it)
    Esc                stop recording and push to the Hub

NOTE: run this with the arms free — the web dashboard (arm_dashboard.py) must NOT be
connected to the follower/leader, or the serial ports will conflict. The camera lock
below makes the dashboard yield the cameras, but it does not cover the arms. (The
dashboard can also record in-process from its Record tab, which avoids both conflicts.)
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

from lerobot.datasets.feature_utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.processor import make_default_processors
from lerobot.robots.so_follower import SO101Follower
from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

from recording import build_cameras, resolve_hf_username, run_record_session

_HERE: Final[Path] = Path(__file__).resolve().parent
CALIBRATION_DIR: Final[str] = os.environ.get("CALIBRATION_DIR") or str(
    _HERE / "calibration"
)

FOLLOWER_PORT: Final[str] = os.environ.get("FOLLOWER_PORT", "")
LEADER_PORT: Final[str] = os.environ.get("LEADER_PORT", "")
ROBOT_ID: Final[str] = os.environ.get("ROBOT_ID", "so101_follower")
LEADER_ID: Final[str] = os.environ.get("LEADER_ID", "so101_leader")

DEFAULT_FPS: Final[int] = int(os.environ.get("INFERENCE_FPS", "30"))
DEFAULT_TASK: Final[str] = os.environ.get("POLICY_TASK", "Pick up the cube")
# Video codec for recorded datasets. "auto" picks a hardware encoder so streaming encoding
# keeps up with the cameras in real time; "libsvtav1" is AV1 (software). See arm_dashboard.py.
RECORD_VCODEC: Final[str] = os.environ.get("RECORD_VCODEC", "auto")

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

    try:
        username = resolve_hf_username()
    except ValueError as e:
        sys.exit(str(e))
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

    try:
        cameras = build_cameras()
    except ValueError as e:
        sys.exit(str(e))

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
        # Encode video frames in background threads during capture so save_episode() is
        # near-instant and recording runs continuously (no per-episode encode wait).
        streaming_encoding=True,
        vcodec=RECORD_VCODEC,
    )

    processors = make_default_processors()

    listener, events = init_keyboard_listener()
    if display:
        init_rerun(session_name="recording")

    robot.connect()
    teleop.connect()

    recorded_episodes = 0
    try:
        recorded_episodes = run_record_session(
            robot=robot,
            teleop=teleop,
            processors=processors,
            dataset=dataset,
            episodes=num_episodes,
            episode_time_s=episode_time_s,
            reset_time_s=reset_time_s,
            task=task,
            fps=fps,
            events=events,
            display_data=display,
        )
    finally:
        log_say("Stop recording", blocking=True)
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if listener is not None:
            listener.stop()

        url = f"https://huggingface.co/datasets/{repo_id}"
        log_say(f"Pushing {recorded_episodes} episode(s) to {repo_id}")
        dataset.push_to_hub(private=True)
        print(f"\nDone. Dataset at {url}")


if __name__ == "__main__":
    main()
