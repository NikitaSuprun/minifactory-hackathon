"""Shared building blocks for recording a LeRobot dataset from our 3-camera SO-101 setup.

One source of truth for the pieces that were previously duplicated across
``record_dataset.py`` (the terminal CLI) and ``run_robot_client.py`` (inference), and now
also used by ``arm_dashboard.py`` to record in-process:

- :func:`build_cameras` assembles phone(camera1) + USB(camera2) + OAK-D(camera3) configs.
- :func:`resolve_hf_username` turns ``HF_TOKEN`` into the Hub username for the repo id.
- :func:`run_record_session` drives LeRobot's ``record_loop`` over N episodes with reset
  windows, honoring an externally-mutated ``events`` dict (keyboard in the CLI, HTTP
  endpoints in the dashboard) and reporting phases through an ``on_phase`` callback.

Camera names become the policy-facing slots (``observation.images.<name>``); they stay
aligned with smolvla_base (camera1/camera2/camera3) so a dataset recorded here trains a
policy the inference path serves unchanged.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Final

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.utils import log_say

# Importing phone_camera also loads .env + .env.local (so HF_TOKEN is in os.environ).
from phone_camera import (
    adopt_network_stream_profile,
    build_phone_camera_config,
    resolve_phone_url,
    tolerate_camera_resolution_drift,
)

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


def build_cameras() -> dict[str, CameraConfig]:
    """Assemble the 3 camera configs the robot owns: phone(camera1) + USB(camera2) + camera3.

    All three are required for both recording and inference so every episode / observation
    carries camera1/camera2/camera3 (the slots smolvla_base expects). A missing camera raises
    :class:`ValueError` (the CLI maps it to ``sys.exit``, the dashboard to HTTP 400).
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
        raise ValueError(
            "ARM_CAM_INDEX is not set in .env, but all 3 cameras are required "
            "(camera2 = workspace USB cam). Set it (e.g. ARM_CAM_INDEX=0)."
        )
    cameras[ARM_CAM_NAME] = OpenCVCameraConfig(
        index_or_path=int(ARM_CAM_INDEX),
        width=int(ARM_CAM_WIDTH),
        height=int(ARM_CAM_HEIGHT),
        fps=int(ARM_CAM_FPS),
    )

    if CAM3_SOURCE == "oak":
        # Importing registers the "oak" camera type so the robot can build it.
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
        raise ValueError(
            "camera3 is not configured, but all 3 cameras are required. "
            "Set CAM3_SOURCE=oak (Luxonis OAK-D) or CAM3_INDEX=<n> (USB cam) in .env."
        )

    return cameras


def resolve_hf_username() -> str:
    """Return the Hugging Face username for ``HF_TOKEN``; raise ``ValueError`` if it can't."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            "HF_TOKEN is not set. Put it in .env.local "
            "(get a write token at https://huggingface.co/settings/tokens)."
        )
    from huggingface_hub import whoami

    try:
        return whoami(token=token)["name"]
    except Exception as e:  # noqa: BLE001 - re-raise as a clear, actionable message
        raise ValueError(
            f"Could not resolve your Hugging Face username from HF_TOKEN: {e}"
        ) from e


def run_record_session(
    *,
    robot: Any,
    teleop: Any,
    processors: tuple[Any, Any, Any],
    dataset: Any,
    episodes: int,
    episode_time_s: int,
    reset_time_s: int,
    task: str,
    fps: int,
    events: dict[str, bool],
    display_data: bool = False,
    on_phase: Callable[..., None] = lambda *_a, **_k: None,
) -> int:
    """Record ``episodes`` episodes into ``dataset`` from a connected robot + teleop.

    Drives LeRobot's ``record_loop`` (which teleoperates *and* records), with an unrecorded
    reset window between episodes. ``events`` (``exit_early`` / ``rerecord_episode`` /
    ``stop_recording``) is mutated externally to end an episode early, re-record it, or stop.
    ``on_phase(phase, current_episode, total_episodes, phase_time_s)`` is called at the start
    of each recording / resetting window. Video encoding is finalized on exit via
    ``VideoEncodingManager``; the caller pushes to the Hub and disconnects.

    Returns the number of episodes actually saved.
    """
    teleop_ap, robot_ap, robot_op = processors
    recorded = 0
    with VideoEncodingManager(dataset):
        while recorded < episodes and not events["stop_recording"]:
            log_say(f"Recording episode {recorded + 1} of {episodes}")
            on_phase(
                "recording",
                current_episode=recorded + 1,
                total_episodes=episodes,
                phase_time_s=episode_time_s,
            )
            record_loop(
                robot=robot,
                events=events,
                fps=fps,
                teleop_action_processor=teleop_ap,
                robot_action_processor=robot_ap,
                robot_observation_processor=robot_op,
                teleop=teleop,
                dataset=dataset,
                control_time_s=episode_time_s,
                single_task=task,
                display_data=display_data,
            )

            # Unrecorded reset window to rearrange the scene; skipped after the last
            # episode (unless we're about to re-record this one).
            if not events["stop_recording"] and (
                (recorded < episodes - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                on_phase(
                    "resetting",
                    current_episode=recorded + 1,
                    total_episodes=episodes,
                    phase_time_s=reset_time_s,
                )
                record_loop(
                    robot=robot,
                    events=events,
                    fps=fps,
                    teleop_action_processor=teleop_ap,
                    robot_action_processor=robot_ap,
                    robot_observation_processor=robot_op,
                    teleop=teleop,
                    control_time_s=reset_time_s,
                    single_task=task,
                    display_data=display_data,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # Near-instant when the dataset was created with streaming_encoding=True: video
            # frames are encoded by background threads during capture, so this just finalizes
            # the episode instead of blocking on a full encode between episodes.
            dataset.save_episode()
            recorded += 1
    return recorded
