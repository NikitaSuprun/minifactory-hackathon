"""Launch LeRobot's async-inference PolicyServer (run this on the REMOTE GPU box).

The server loads the VLA policy and does inference; the Mac runs run_robot_client.py
to stream observations here and execute the returned action chunks. The client tells
the server which policy to load, so this launcher only needs networking/timing.

    uv run python run_policy_server.py        # binds POLICY_SERVER_* from .env

Equivalent to:
    python -m lerobot.async_inference.policy_server --host=... --port=... --fps=...
"""

from __future__ import annotations

import os
import sys
from typing import Final

SERVER_HOST: Final[str] = os.environ.get("POLICY_SERVER_HOST", "0.0.0.0")
SERVER_PORT: Final[str] = os.environ.get("POLICY_SERVER_PORT", "8080")
SERVER_FPS: Final[str] = os.environ.get("INFERENCE_FPS", "30")


def main() -> None:
    argv: list[str] = [
        sys.executable,
        "-m",
        "lerobot.async_inference.policy_server",
        f"--host={SERVER_HOST}",
        f"--port={SERVER_PORT}",
        f"--fps={SERVER_FPS}",
    ]
    print("launching:", " ".join(argv))
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
