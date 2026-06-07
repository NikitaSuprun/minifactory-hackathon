"""Cross-process prewarm handshake between the dashboard and the inference client.

When the dashboard prewarms inference, it launches ``run_robot_client.py`` with
``PREWARM=1``. That subprocess does all the slow setup (opens the cameras, connects the
arm, makes the policy server load the model + warm CUDA), then **pauses** before the
control loop so the arm doesn't move. Two pid-free flag files under the gitignored
``logs/`` dir coordinate the pause:

- ``ready.flag``: written by the client once setup is done and it is paused waiting. The
  dashboard polls this (via its 1s status loop) to flip ``prewarming`` -> ``prewarmed``.
- ``go.flag``: written by the dashboard when the operator clicks Run inference. Its
  contents are the task string, so the task can be chosen at Run time rather than being
  committed at Prewarm time. The client polls this and, once present, enters the control
  loop (the arm starts moving). Writing it while the client is still prewarming is fine —
  the client picks it up as soon as it reaches the wait loop.

This is intentionally separate from ``camera_lock`` (camera-device ownership): the same
subprocess holds the camera lock for its whole run, but the prewarm pause is a distinct,
shorter-lived concern.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

_DIR: Final[Path] = Path(__file__).resolve().parent / "logs"
READY_PATH: Final[Path] = _DIR / "ready.flag"
GO_PATH: Final[Path] = _DIR / "go.flag"


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def mark_ready() -> None:
    """Signal that the client is prewarmed and paused (best-effort)."""
    READY_PATH.parent.mkdir(parents=True, exist_ok=True)
    READY_PATH.write_text("ready")


def clear_ready() -> None:
    _unlink(READY_PATH)


def is_ready() -> bool:
    return READY_PATH.exists()


def signal_go(task: str) -> None:
    """Release a paused/prewarming client into the control loop with ``task``."""
    GO_PATH.parent.mkdir(parents=True, exist_ok=True)
    GO_PATH.write_text(task or "")


def read_go() -> str | None:
    """Return the go-task string if the go flag is set, else None.

    An empty file means "go, but no task override" and returns "" (not None), so callers
    must distinguish ``is not None`` from truthiness.
    """
    try:
        return GO_PATH.read_text()
    except FileNotFoundError:
        return None
    except OSError:
        return None


def clear_go() -> None:
    _unlink(GO_PATH)
