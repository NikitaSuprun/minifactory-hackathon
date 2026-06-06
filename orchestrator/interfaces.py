"""Contracts between the Orchestrator and the three subsystems it coordinates.

These Protocols are the ONLY things the Orchestrator depends on. Both the real
hardware backends (carlink-backed Navigator/Sensors, LeRobot-backed Manipulator)
and the mocks in ``mocks.py`` satisfy them, so the state machine in ``core.py``
never changes when a backend is swapped in.

Keep these minimal — every method here is something the state machine actually
calls. Anything a real backend needs internally (calibration, dead-reckoning,
camera locks) stays inside that backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


@dataclass
class Pose:
    """A 2-D pose on the floor.

    ``x``/``y`` are a DEAD-RECKONED estimate (no encoders → they drift; see the
    design doc's measurement section). ``heading_deg`` comes from the car IMU and
    is the one component we trust. 0° = +x axis, CCW positive.
    """

    x: float
    y: float
    heading_deg: float = 0.0


class PickStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"        # the pick finished (button fired) — policy believes it's done
    TIMEOUT = "timeout"  # safety fallback fired before the button
    ERROR = "error"


@runtime_checkable
class Manipulator(Protocol):
    """The arm / VLA. "Prompting" a language-conditioned policy = a task string."""

    def pick(self, task: str) -> None:
        """Start one pick attempt (prompt the policy or open a teleop episode) and
        begin driving the follower. Non-blocking. The caller must have confirmed the
        car is stationary first."""

    @property
    def status(self) -> PickStatus: ...

    def to_neutral(self) -> None:
        """Drive the arm to a fixed rest pose clear of the car. BLOCKS until the arm
        has settled — the orchestrator must not release the car before this returns."""

    def stop(self) -> None:
        """Halt inference/recording immediately. Safe to call any time (e-stop)."""


@runtime_checkable
class Navigator(Protocol):
    """The car board: actuate the wheels + estimate pose from the IMU."""

    def move_to(self, x: float, y: float) -> None:
        """Begin driving toward ``(x, y)``: turn to the bearing (closed-loop on the
        IMU), then drive at the calibrated cruise speed. Non-blocking."""

    @property
    def pose(self) -> Pose:
        """Current dead-reckoned estimate (heading real, x/y drifts)."""
        ...

    @property
    def at_goal(self) -> bool:
        """The *estimate* says we've reached the goal. Approximate — used to slow for
        final approach, NOT as the authoritative stop (that's ``Sensors.car_present``)."""
        ...

    def reset_pose(self, p: Pose) -> None:
        """Re-zero the estimate to a known pose (called when docked, with the dock's
        known coordinates). This is what bounds dead-reckoning drift."""

    def stop(self) -> None:
        """Active brake. Safe to call any time (e-stop)."""


@runtime_checkable
class Sensors(Protocol):
    """The station board: arrival detection + the pick-done button. Read-only."""

    @property
    def car_distance_mm(self) -> float | None:
        """Range from the fixed station sensor to the approaching car (None until the
        first reading)."""

    @property
    def car_present(self) -> bool:
        """``car_distance_mm`` below the arrival threshold. AUTHORITATIVE arrival — a
        fixed sensor, so unlike the car's pose estimate it does not drift."""
        ...

    @property
    def button(self) -> bool:
        """Latched: True once the arm has pressed the station button, until
        ``clear_button()``. Edge-triggered so a held press doesn't re-fire."""
        ...

    def clear_button(self) -> None:
        """Reset the button latch for the next cycle."""
