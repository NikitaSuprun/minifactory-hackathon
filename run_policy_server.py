"""Launch LeRobot's async-inference PolicyServer (run this on the REMOTE GPU box).

The server loads the VLA policy and does inference; the Mac runs run_robot_client.py
to stream observations here and execute the returned action chunks. The client tells
the server which policy to load, so this launcher only needs networking/timing.

    uv run python run_policy_server.py        # binds POLICY_SERVER_* from .env

This runs LeRobot's PolicyServer in-process (rather than `python -m
lerobot.async_inference.policy_server`) so we can lower the observation-similarity
threshold: LeRobot only re-runs inference when the new observation differs from the last
one by more than ``atol`` in joint space (default 1.0). That joint-only check skips too
aggressively when the arm is nearly still, so we patch it to a smaller ``atol``
(``OBS_SIMILARITY_ATOL``) before starting the gRPC server.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load .env (committed config) then .env.local (gitignored HF_TOKEN) so the
# policy can download gated checkpoints. This script does not import phone_camera.
_HERE: Final[Path] = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
load_dotenv(_HERE / ".env.local", override=True)

SERVER_HOST: Final[str] = os.environ.get("POLICY_SERVER_HOST", "0.0.0.0")
SERVER_PORT: Final[str] = os.environ.get("POLICY_SERVER_PORT", "8080")
SERVER_FPS: Final[str] = os.environ.get("INFERENCE_FPS", "30")
# Joint-space L2 tolerance (degrees) below which an observation is deemed "too similar"
# to the last one and skipped. LeRobot's default is 1.0; lower = more sensitive.
OBS_SIMILARITY_ATOL: Final[float] = float(os.environ.get("OBS_SIMILARITY_ATOL", "0.1"))


def _patch_similarity_threshold() -> None:
    """Lower the observation-similarity tolerance the server uses to skip inference.

    ``policy_server`` calls the module-global ``observations_similar`` without an ``atol``,
    so it uses the library default (1.0). Reassigning that global to a wrapper with a
    smaller ``atol`` makes the server re-infer on smaller joint changes.
    """
    import lerobot.async_inference.policy_server as policy_server
    from lerobot.async_inference.helpers import observations_similar

    def _similar(
        obs1, obs2, lerobot_features, atol: float = OBS_SIMILARITY_ATOL
    ) -> bool:
        return observations_similar(obs1, obs2, lerobot_features, atol=atol)

    policy_server.observations_similar = _similar


def main() -> None:
    _patch_similarity_threshold()

    import lerobot.async_inference.policy_server as policy_server

    # serve() is a draccus entrypoint that parses argv, so feed it the same flags the
    # `python -m ...` invocation used. Running in-process keeps our patch applied.
    sys.argv = [
        "policy_server",
        f"--host={SERVER_HOST}",
        f"--port={SERVER_PORT}",
        f"--fps={SERVER_FPS}",
    ]
    print(
        f"launching policy server on {SERVER_HOST}:{SERVER_PORT} (obs atol={OBS_SIMILARITY_ATOL})"
    )
    # serve() is draccus-wrapped: it parses cfg from argv at runtime, so the
    # declared `cfg` parameter is supplied there, not by this call.
    policy_server.serve()  # pyright: ignore[reportCallIssue]


if __name__ == "__main__":
    main()
