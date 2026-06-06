"""In-memory mocks of the three interfaces, for exercising the Orchestrator with
no hardware.

The causal chain mirrors the real rig: ``MockSensors`` is a passive "board" that
just reports what its sensors see; the car-arrival and the button-press are written
INTO it — by the navigator's travel timer and the arm's pick timer respectively —
exactly as the real station board would observe the car rolling in and the arm
pressing the button.

Delays use ``threading.Timer`` so the getters stay pure (no side effects on read)
and the demo reads like a real, slightly-asynchronous system.
"""

from __future__ import annotations

import math
import threading
import time

from .interfaces import PickStatus, Pose


class MockSensors:
    """Station board. Reads are the Sensors contract; the ``set_*``/``press_*``
    writes are how the 'world' (navigator/arm) pokes it — not part of the contract."""

    def __init__(self, present_distance_mm: float = 50.0, clear_distance_mm: float = 1000.0):
        self._car_present = False
        self._car_distance_mm: float | None = None
        self._button = False
        self._present_mm = present_distance_mm
        self._clear_mm = clear_distance_mm

    # --- Sensors contract (reads) ---
    @property
    def car_distance_mm(self) -> float | None:
        return self._car_distance_mm

    @property
    def car_present(self) -> bool:
        return self._car_present

    @property
    def button(self) -> bool:
        return self._button

    def clear_button(self) -> None:
        self._button = False

    # --- world writes (the car/arm cause these) ---
    def set_car_present(self, present: bool) -> None:
        self._car_present = present
        self._car_distance_mm = self._present_mm if present else self._clear_mm

    def press_button(self) -> None:
        self._button = True


class MockNavigator:
    """A car that takes ``travel_s`` to reach any goal, then tells the station it's
    present. Pose interpolates linearly toward the goal while moving."""

    def __init__(self, sensors: MockSensors, travel_s: float = 2.0):
        self.sensors = sensors
        self.travel_s = travel_s
        self._pose = Pose(0.0, 0.0, 0.0)
        self._origin = self._pose
        self._goal: tuple[float, float] | None = None
        self._start = 0.0
        self._arrived = True
        self._timer: threading.Timer | None = None

    def move_to(self, x: float, y: float) -> None:
        self._cancel()
        origin = self.pose  # snapshot before we start interpolating
        bearing = math.degrees(math.atan2(y - origin.y, x - origin.x))
        self._origin = Pose(origin.x, origin.y, bearing)
        self._goal = (x, y)
        self._start = time.perf_counter()
        self._arrived = False
        self.sensors.set_car_present(False)  # we've left / are moving
        self._timer = threading.Timer(self.travel_s, self._on_arrive)
        self._timer.daemon = True
        self._timer.start()

    def _on_arrive(self) -> None:
        if self._goal is None:
            return
        gx, gy = self._goal
        self._pose = Pose(gx, gy, self._origin.heading_deg)
        self._arrived = True
        self.sensors.set_car_present(True)  # the station now sees the car

    @property
    def pose(self) -> Pose:
        if self._goal is None or self._arrived:
            return self._pose
        frac = min(1.0, (time.perf_counter() - self._start) / max(self.travel_s, 1e-6))
        gx, gy = self._goal
        return Pose(
            self._origin.x + (gx - self._origin.x) * frac,
            self._origin.y + (gy - self._origin.y) * frac,
            self._origin.heading_deg,
        )

    @property
    def at_goal(self) -> bool:
        return self._arrived

    def reset_pose(self, p: Pose) -> None:
        self._pose = p
        self._goal = None

    def stop(self) -> None:
        self._cancel()
        self._goal = None

    def _cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


class MockManipulator:
    """An arm that 'picks' for ``pick_s`` and then presses the station button.
    ``to_neutral`` blocks for ``neutral_s`` to model the arm settling."""

    def __init__(self, sensors: MockSensors, pick_s: float = 2.0, neutral_s: float = 0.5):
        self.sensors = sensors
        self.pick_s = pick_s
        self.neutral_s = neutral_s
        self._status = PickStatus.IDLE
        self._timer: threading.Timer | None = None

    def pick(self, task: str) -> None:
        self._cancel()
        self._status = PickStatus.RUNNING
        self._timer = threading.Timer(self.pick_s, self.sensors.press_button)
        self._timer.daemon = True
        self._timer.start()

    @property
    def status(self) -> PickStatus:
        return self._status

    def to_neutral(self) -> None:
        time.sleep(self.neutral_s)  # blocks: the car must not move until the arm is clear
        self._status = PickStatus.IDLE

    def stop(self) -> None:
        self._cancel()
        self._status = PickStatus.IDLE

    def _cancel(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
