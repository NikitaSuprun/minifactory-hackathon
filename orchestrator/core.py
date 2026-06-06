"""Orchestrator: the state machine that runs a drive -> pick -> depart mission.

It owns no hardware and no wire format — only the three Protocols in
``interfaces.py``. The interfaces are dumb; ALL sequencing and safety live here.
Mirrors the safe shape of carlink's PolicyRunner: a single tick loop, one state at
a time, and every exit path brakes the car and halts the arm.

Run the hardware-free demo:

    uv run python -m orchestrator
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .interfaces import Manipulator, Navigator, PickStatus, Pose, Sensors


class OrchState(Enum):
    IDLE = "idle"
    NAVIGATING = "navigating"  # driving toward the dock
    SETTLING = "settling"  # braked, confirming stationary before we pick
    PICKING = "picking"  # arm running; waiting for the button (or timeout)
    RETRACTING = "retracting"  # arm returning to neutral, clear of the car
    DEPARTING = "departing"  # driving away from the dock
    DONE = "done"
    ABORTED = "aborted"
    ERROR = "error"


_TERMINAL = {OrchState.DONE, OrchState.ABORTED, OrchState.ERROR}


@dataclass
class Leg:
    """One stop of the mission: drive to ``dock``, then prompt the arm with ``task``.
    ``dock`` doubles as the known coordinates we ``reset_pose`` to on arrival."""

    dock: Pose
    task: str


@dataclass
class Mission:
    legs: list[Leg]


@dataclass
class Orchestrator:
    nav: Navigator
    sensors: Sensors
    arm: Manipulator
    home: Pose = field(default_factory=lambda: Pose(0.0, 0.0, 0.0))
    settle_s: float = 1.0
    nav_timeout_s: float = 30.0
    pick_timeout_s: float = 25.0
    tick_hz: float = 10.0
    # Called on every transition (old, new, reason); defaults to a stdout log.
    on_transition: Callable[[OrchState, OrchState, str], None] | None = None

    _state: OrchState = field(default=OrchState.IDLE, init=False)
    _state_t: float = field(default=0.0, init=False)
    _leg_i: int = field(default=0, init=False)
    _mission: Mission | None = field(default=None, init=False)
    _abort: threading.Event = field(default_factory=threading.Event, init=False)
    history: list[OrchState] = field(default_factory=list, init=False)

    @property
    def state(self) -> OrchState:
        return self._state

    @property
    def _leg(self) -> Leg:
        assert self._mission is not None
        return self._mission.legs[self._leg_i]

    def abort(self) -> None:
        """Request an immediate stop from any thread. The loop brakes + halts."""
        self._abort.set()

    def run(self, mission: Mission) -> OrchState:
        """Run ``mission`` to a terminal state (blocking). Returns the final state.
        Brakes the car and halts the arm on every exit path (including exceptions)."""
        if not mission.legs:
            raise ValueError("mission has no legs")
        self._mission = mission
        self._leg_i = 0
        self._abort.clear()
        self._state = OrchState.IDLE
        self._state_t = time.perf_counter()
        self.history = [OrchState.IDLE]

        period = 1.0 / self.tick_hz if self.tick_hz > 0 else 0.0
        try:
            self._goto(OrchState.NAVIGATING, f"leg 1/{len(mission.legs)}")
            while self._state not in _TERMINAL:
                if self._abort.is_set():
                    self._goto(OrchState.ABORTED, "abort requested")
                    break
                t0 = time.perf_counter()
                self._tick()
                time.sleep(max(period - (time.perf_counter() - t0), 0.0))
        except Exception as e:  # noqa: BLE001 - surface, then brake in finally
            self._goto(OrchState.ERROR, f"unhandled: {e}")
        finally:
            self._safe_halt()
        return self._state

    # --- transitions ---
    def _goto(self, new: OrchState, reason: str) -> None:
        old = self._state
        self._state = new
        self._state_t = time.perf_counter()
        self.history.append(new)
        if self.on_transition is not None:
            self.on_transition(old, new, reason)
        else:
            print(f"[orch] {old.name:>11} -> {new.name:<11} ({reason})")
        self._on_enter(new)

    def _on_enter(self, s: OrchState) -> None:
        if s is OrchState.NAVIGATING:
            self.nav.move_to(self._leg.dock.x, self._leg.dock.y)
        elif s is OrchState.SETTLING:
            self.nav.stop()
            self.nav.reset_pose(self._leg.dock)  # re-zero drift at the known dock
        elif s is OrchState.PICKING:
            self.sensors.clear_button()
            self.arm.pick(self._leg.task)
        elif s is OrchState.RETRACTING:
            self.arm.to_neutral()  # blocks until the arm is clear of the car
        elif s is OrchState.DEPARTING:
            self.nav.move_to(self.home.x, self.home.y)
        # terminal states brake via _safe_halt() when run()'s loop exits

    def _tick(self) -> None:
        s = self._state
        elapsed = time.perf_counter() - self._state_t

        if s is OrchState.NAVIGATING:
            if self.sensors.car_present:
                self._goto(OrchState.SETTLING, "station reports car present")
            elif elapsed > self.nav_timeout_s:
                self._goto(OrchState.ERROR, f"nav timeout ({elapsed:.0f}s, no arrival)")

        elif s is OrchState.SETTLING:
            if elapsed >= self.settle_s:
                self._goto(OrchState.PICKING, "car settled")

        elif s is OrchState.PICKING:
            if self.sensors.button:
                self._goto(OrchState.RETRACTING, "button pressed (pick done)")
            elif self.arm.status is PickStatus.ERROR:
                self._goto(OrchState.ERROR, "arm reported error")
            elif elapsed > self.pick_timeout_s:
                # Don't hang: give up this pick, retract, and move on.
                self._goto(OrchState.RETRACTING, f"pick timeout ({elapsed:.0f}s)")

        elif s is OrchState.RETRACTING:
            # to_neutral() blocked on enter, so the arm is already clear here.
            if self.arm.status is not PickStatus.RUNNING:
                self._goto(OrchState.DEPARTING, "arm at neutral")

        elif s is OrchState.DEPARTING:
            if not self.sensors.car_present:  # cleared the dock
                self._advance_leg()

    def _advance_leg(self) -> None:
        assert self._mission is not None
        self._leg_i += 1
        n = len(self._mission.legs)
        if self._leg_i < n:
            self._goto(OrchState.NAVIGATING, f"leg {self._leg_i + 1}/{n}")
        else:
            self._goto(OrchState.DONE, "mission complete")

    def _safe_halt(self) -> None:
        """Brake the car and halt the arm. Never raises (safety path)."""
        for action in (self.nav.stop, self.arm.stop):
            try:
                action()
            except Exception as e:  # noqa: BLE001
                print(f"[orch] halt error: {e}")
