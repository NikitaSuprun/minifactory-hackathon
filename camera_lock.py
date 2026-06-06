"""Cross-process camera-ownership lock shared by the dashboard and the inference client.

`run_robot_client.py` holds this lock for its whole run so it owns the camera devices
exclusively; `arm_dashboard.py` watches it and stops reading the cameras while it's
held (showing a placeholder) so the two never contend for the same phone/USB/OAK
stream. The lock is a pid-stamped file under the gitignored ``logs/`` dir; a stale lock
left by a crashed client is detected via pid-liveness and ignored, so the dashboard
can never get stranded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

LOCK_PATH: Final[Path] = Path(__file__).resolve().parent / "logs" / "inference.lock"


def acquire() -> None:
    """Claim the cameras for this process by writing its pid to the lockfile."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(str(os.getpid()))


def release() -> None:
    """Drop the lock (best-effort; safe to call multiple times)."""
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by another user — still a live process.
        return True
    except OSError:
        return False
    return True


def active() -> bool:
    """True if a live process holds the lock.

    A lockfile whose pid is dead (e.g. after a ``kill -9``) is treated as stale: it is
    removed and reported inactive so the dashboard resumes instead of staying paused.
    """
    try:
        pid = int(LOCK_PATH.read_text().strip())
    except (FileNotFoundError, ValueError):
        return False
    if _pid_alive(pid):
        return True
    release()  # stale lock from a crashed holder
    return False
