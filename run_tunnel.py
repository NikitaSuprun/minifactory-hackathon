"""Open an SSH tunnel from this Mac to the remote GPU box's gRPC port.

When the policy server runs on a GPU box in the cloud / behind NAT, the Mac can't
reach its gRPC port directly. This forwards a local port to the server's port over
SSH, so you can set ``POLICY_SERVER_ADDRESS=localhost:<TUNNEL_LOCAL_PORT>``.

    uv run python run_tunnel.py     # keeps running; Ctrl-C to close

Equivalent to:
    ssh -N -L <local>:localhost:<remote> -p <ssh_port> <ssh_host>
"""

from __future__ import annotations

import os
import sys
from typing import Final

GPU_SSH_HOST: Final[str] = os.environ.get("GPU_SSH_HOST", "")
GPU_SSH_PORT: Final[str] = os.environ.get("GPU_SSH_PORT", "22")
LOCAL_PORT: Final[str] = os.environ.get("TUNNEL_LOCAL_PORT", "8080")
REMOTE_PORT: Final[str] = os.environ.get("POLICY_SERVER_PORT", "8080")


def main() -> None:
    if not GPU_SSH_HOST:
        sys.exit(
            "GPU_SSH_HOST is not set in .env (e.g. ubuntu@gpu-box or user@1.2.3.4)."
        )

    argv: list[str] = [
        "ssh",
        "-N",  # no remote command, just forward
        "-o",
        "ServerAliveInterval=30",  # keep the tunnel alive
        "-o",
        "ExitOnForwardFailure=yes",  # fail fast if the local port can't bind
        "-L",
        f"{LOCAL_PORT}:localhost:{REMOTE_PORT}",
        "-p",
        GPU_SSH_PORT,
        GPU_SSH_HOST,
    ]
    print("opening tunnel:", " ".join(argv))
    print(
        f"-> set POLICY_SERVER_ADDRESS=localhost:{LOCAL_PORT} for run_robot_client.py"
    )
    os.execvp(argv[0], argv)


if __name__ == "__main__":
    main()
