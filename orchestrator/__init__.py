"""Orchestrator: coordinate the SO-101 arm and the car through a repeating
drive -> pick -> depart mission, gating arm inference to when the car is stationary.

See ``docs/orchestrator.md`` for the design. Phase 1 is the hardware-free skeleton:
the three interface Protocols, mock backends, and the state machine.
"""

from __future__ import annotations

from .core import Leg, Mission, Orchestrator, OrchState
from .interfaces import Manipulator, Navigator, PickStatus, Pose, Sensors

__all__ = [
    "Manipulator",
    "Navigator",
    "Sensors",
    "Pose",
    "PickStatus",
    "Orchestrator",
    "OrchState",
    "Mission",
    "Leg",
]
