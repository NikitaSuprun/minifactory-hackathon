"""Driving policies + the runner that drives them — the DECISION layer.

A Policy reads car state and commands the car. It owns no connection and no wire
format (that's `atech.Board` + `Car`). PolicyRunner ticks a policy at a fixed rate
on a background thread and is the single place that enforces safety: any abort,
stop, or policy exception brakes the car.

    car = Car(connect_serial(), drive_action="motor_speed")
    runner = PolicyRunner(car, hz=20)
    runner.start(StraightUntilObstacle(speed=180, stop_distance_mm=300))
    ...
    runner.abort()   # brakes immediately
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Callable

from .car import Car


class Policy(ABC):
    name: str = "policy"

    def on_start(self, car: Car) -> None:
        """Optional setup when the runner starts this policy."""

    @abstractmethod
    def step(self, car: Car) -> None:
        """One control tick: read ``car`` state, command via ``car``."""

    def on_stop(self, car: Car) -> None:
        """Optional cleanup (the runner also brakes after)."""


class CallablePolicy(Policy):
    """Wrap a plain ``fn(car)`` as a Policy — handy for quick scripts."""

    def __init__(self, fn: Callable[[Car], None], name: str = "callable"):
        self.fn = fn
        self.name = name

    def step(self, car: Car) -> None:
        self.fn(car)


class StraightUntilObstacle(Policy):
    """Drive straight; brake when something is closer than ``stop_distance_mm``.

    A minimal sensor-reactive baseline to build a real driving policy on."""

    name = "straight_until_obstacle"

    def __init__(self, speed: int = 180, stop_distance_mm: float = 300.0):
        self.speed = speed
        self.stop_distance_mm = stop_distance_mm

    def step(self, car: Car) -> None:
        dist = car.distance_mm
        if dist is not None and dist < self.stop_distance_mm:
            car.stop()
        else:
            car.drive(self.speed)


class PolicyRunner:
    """Run a Policy at a fixed rate on a background thread, with safe abort.

    One policy at a time; starting a new one stops the previous."""

    def __init__(self, car: Car, hz: float = 20.0):
        self.car = car
        self.hz = hz
        self.policy: Policy | None = None
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def policy_name(self) -> str | None:
        return self.policy.name if self.policy is not None else None

    def start(self, policy: Policy) -> None:
        self.stop()  # one policy at a time
        self.policy = policy
        self.last_error = None
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="atech-policy", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the loop and brake. Safe to call when nothing is running."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            self.car.stop()  # brake
        except Exception as e:  # noqa: BLE001 - never raise on the safety path
            self.last_error = f"stop/brake: {e}"

    abort = stop  # the dashboard's e-stop button

    def _loop(self) -> None:
        assert self.policy is not None
        period = 1.0 / self.hz if self.hz > 0 else 0.0
        try:
            self.policy.on_start(self.car)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"on_start: {e}"
            return
        while not self._stop.is_set():
            t0 = time.perf_counter()
            try:
                self.policy.step(self.car)
            except Exception as e:  # noqa: BLE001 - brake and surface, then exit
                self.last_error = f"step: {e}"
                try:
                    self.car.stop()
                except Exception:  # noqa: BLE001
                    pass
                break
            time.sleep(max(period - (time.perf_counter() - t0), 0.0))
        try:
            self.policy.on_stop(self.car)
        except Exception as e:  # noqa: BLE001
            self.last_error = f"on_stop: {e}"
