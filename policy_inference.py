"""Load a Hugging Face VLA/policy and run it on an SO-101 follower.

This mirrors the policy path of LeRobot's ``lerobot_record`` script, minus the
dataset writing: build the robot's dataset features, obtain metadata via a
throwaway :class:`LeRobotDataset`, load the pretrained policy + its pre/post
processors, then run single-step inference with :func:`predict_action`.

Language-conditioned VLAs (pi0, SmolVLA, …) are "prompted" via the ``task``
string passed to :func:`infer_action`.

Caveats (read before expecting motion):
- The robot must be connected, and the policy's expected inputs (camera names,
  state dimension) must match this robot's observation features, or loading /
  inference will raise. Use a checkpoint trained for an SO-101 with matching
  cameras, or expect to remap features.
- VLAs are large. pi0 is multi-billion-parameter; prefer ``lerobot/smolvla_base``
  on an Apple-Silicon Mac (mps). Gated repos need ``HF_TOKEN`` (loaded from
  ``.env.local`` via phone_camera).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import (
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import Robot
from lerobot.utils.constants import OBS_STR
from lerobot.utils.control_utils import predict_action
from lerobot.utils.device_utils import get_safe_torch_device

# Metadata-only FPS for the throwaway dataset; does not affect the control rate.
_SCRATCH_FPS: Final[int] = 30


def auto_device() -> str:
    """Pick the best available torch device string for this machine."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class PolicyBundle:
    """Everything needed to run one inference step against a robot."""

    policy: PreTrainedPolicy
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]]
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction]
    observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation]
    action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ]
    dataset_features: dict[str, Any]
    device: torch.device
    use_amp: bool


def load_policy(
    policy_path: str, robot: Robot, *, device: str | None = None
) -> PolicyBundle:
    """Load a pretrained policy and build the processors for ``robot``.

    Args:
        policy_path: Hugging Face repo id or local path of the policy checkpoint.
        robot: A connected robot whose observation/action features define the
            policy's input/output shapes.
        device: Torch device string ("cuda"/"mps"/"cpu"); auto-detected if None.
    """
    resolved_device: str = device or auto_device()

    teleop_action_processor, action_processor, observation_processor = (
        make_default_processors()
    )

    dataset_features: dict[str, Any] = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=observation_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=True,
        ),
    )

    # A throwaway dataset supplies the metadata (feature shapes/stats) make_policy
    # requires; no frames are ever written to it.
    scratch_root: Path = Path(tempfile.mkdtemp(prefix="lerobot_infer_"))
    scratch: LeRobotDataset = LeRobotDataset.create(
        repo_id="inference/scratch",
        fps=_SCRATCH_FPS,
        features=dataset_features,
        root=scratch_root,
        robot_type=robot.name,
        use_videos=True,
    )

    cfg: PreTrainedConfig = PreTrainedConfig.from_pretrained(policy_path)
    # The field is typed Path|None; a non-existent local Path is treated as a
    # Hugging Face repo id by from_pretrained downstream.
    cfg.pretrained_path = Path(policy_path)
    cfg.device = resolved_device

    policy: PreTrainedPolicy = make_policy(cfg, ds_meta=scratch.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        # Empty for a fresh scratch dataset; the pretrained processors carry the
        # checkpoint's own normalization. Cast bridges lerobot's numpy-typed stats.
        dataset_stats=cast(
            dict[str, dict[str, torch.Tensor]] | None, scratch.meta.stats
        ),
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )
    policy.reset()

    return PolicyBundle(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        observation_processor=observation_processor,
        action_processor=action_processor,
        dataset_features=scratch.features,
        device=get_safe_torch_device(cfg.device),
        use_amp=cfg.use_amp,
    )


def infer_action(robot: Robot, bundle: PolicyBundle, task: str) -> RobotAction:
    """Run one policy step and return the action ready for ``robot.send_action``."""
    obs: RobotObservation = robot.get_observation()
    obs_processed: RobotObservation = bundle.observation_processor(obs)
    observation_frame = build_dataset_frame(
        bundle.dataset_features, obs_processed, prefix=OBS_STR
    )
    action_values = predict_action(
        observation_frame,
        policy=bundle.policy,
        device=bundle.device,
        preprocessor=bundle.preprocessor,
        postprocessor=bundle.postprocessor,
        use_amp=bundle.use_amp,
        task=task,
        robot_type=robot.name,
    )
    policy_action: RobotAction = make_robot_action(
        action_values, bundle.dataset_features
    )
    return bundle.action_processor((policy_action, obs))
