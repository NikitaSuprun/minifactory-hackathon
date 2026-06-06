"""Launch LeRobot's async-inference RobotClient (run this on THIS Mac).

This owns the SO-101 follower + phone camera, streams observations to the remote
PolicyServer (run_policy_server.py), and executes the returned action chunks. It
assembles the long lerobot CLI from .env, including the phone camera URL.

    uv run python run_robot_client.py

Equivalent to:
    python -m lerobot.async_inference.robot_client \
        --robot.type=so101_follower --robot.port=... --robot.id=... \
        --robot.cameras="{phone: {type: opencv, index_or_path: <url>}}" \
        --task=... --server_address=HOST:PORT \
        --policy_type=... --pretrained_name_or_path=... \
        --policy_device=cuda --client_device=cpu
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Final

from phone_camera import resolve_phone_url

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
CAMERA_NAME: Final[str] = os.environ.get("ROBOT_CAMERA_NAME", "phone")


def main() -> None:
    if not FOLLOWER_PORT:
        sys.exit("FOLLOWER_PORT is not set in .env (run `uv run lerobot-find-port`).")
    if not SERVER_ADDRESS:
        sys.exit("POLICY_SERVER_ADDRESS is not set in .env (e.g. 192.168.1.50:8080).")

    cameras: dict[str, dict[str, str]] = {
        CAMERA_NAME: {"type": "opencv", "index_or_path": resolve_phone_url()}
    }

    argv: list[str] = [
        sys.executable,
        "-m",
        "lerobot.async_inference.robot_client",
        "--robot.type=so101_follower",
        f"--robot.port={FOLLOWER_PORT}",
        f"--robot.id={ROBOT_ID}",
        f"--robot.calibration_dir={CALIBRATION_DIR}",
        f"--robot.cameras={json.dumps(cameras)}",
        f"--task={POLICY_TASK}",
        f"--server_address={SERVER_ADDRESS}",
        f"--policy_type={POLICY_TYPE}",
        f"--pretrained_name_or_path={POLICY_PATH}",
        f"--policy_device={SERVER_POLICY_DEVICE}",
        f"--client_device={CLIENT_DEVICE}",
        f"--actions_per_chunk={ACTIONS_PER_CHUNK}",
        f"--chunk_size_threshold={CHUNK_SIZE_THRESHOLD}",
        f"--aggregate_fn_name={AGGREGATE_FN}",
    ]
    print("launching:", " ".join(argv))
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
