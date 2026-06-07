"""Cross-process handshake + live control between the dashboard and the inference client.

The dashboard launches ``run_robot_client.py`` as a long-lived subprocess that owns the
cameras + arm and holds the policy-server connection open (so the model stays loaded). It
does the slow setup once, then runs a control loop that reads a live control file every
tick — so the dashboard can start/stop the arm following the policy and change the task
prompt **without** killing the subprocess or reloading the model. Files live under the
gitignored ``logs/`` dir:

- ``ready.flag``: written by the client once setup (cameras + arm + model load + CUDA
  warmup) is done and it has entered the control loop. The dashboard polls this (via its
  1s status loop) to flip ``prewarming`` -> warm.
- ``control.json`` (``{"following": bool, "task": str}``): the dashboard writes this; the
  client reads it every control-loop tick. ``following`` toggles whether the arm performs
  the policy's actions (false = arm holds position, model stays warm). ``task`` is the
  prompt sent with each observation; changing it takes effect on the next tick — no
  reload. Written atomically (temp + rename) so the client never reads a half-written file.

This is intentionally separate from ``camera_lock`` (camera-device ownership): the same
subprocess holds the camera lock for its whole run; follow/task control is a distinct
concern layered on top.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

_DIR: Final[Path] = Path(__file__).resolve().parent / "logs"
READY_PATH: Final[Path] = _DIR / "ready.flag"
CONTROL_PATH: Final[Path] = _DIR / "control.json"


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def mark_ready() -> None:
    """Signal that the client finished setup and is in its control loop (best-effort)."""
    READY_PATH.parent.mkdir(parents=True, exist_ok=True)
    READY_PATH.write_text("ready")


def clear_ready() -> None:
    _unlink(READY_PATH)


def is_ready() -> bool:
    return READY_PATH.exists()


def write_control(following: bool, task: str) -> None:
    """Set the live control state read by the client each control-loop tick.

    ``following``: whether the arm performs the policy's actions (false = hold position).
    ``task``: the prompt sent with each observation; takes effect next tick, no reload.
    Atomic (temp + rename) so the client never sees a partial write."""
    CONTROL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONTROL_PATH.with_name(CONTROL_PATH.name + ".tmp")
    tmp.write_text(json.dumps({"following": bool(following), "task": task or ""}))
    tmp.replace(CONTROL_PATH)


def read_control() -> tuple[bool, str] | None:
    """Return ``(following, task)`` from the control file, or None if absent/unreadable.

    Callers should keep their last known value on None (e.g. a transient read during the
    dashboard's atomic rewrite) rather than treating it as a state change."""
    try:
        d = json.loads(CONTROL_PATH.read_text())
        return bool(d.get("following", False)), str(d.get("task", ""))
    except (FileNotFoundError, ValueError, OSError):
        return None


def clear_control() -> None:
    _unlink(CONTROL_PATH)
