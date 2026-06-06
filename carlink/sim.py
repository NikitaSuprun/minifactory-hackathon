"""In-memory car simulator for hardware-free development and demos.

Backs an `atech.Board` with `atech.MockTransport`, then runs a feeder thread that
reads the actions the host sends (via MockTransport.sent) and streams plausible
telemetry back — orientation that drifts with steering, depth that shrinks when
driving forward (with an obstacle stop), plus status/module-presence events.

    from carlink import Car
    from carlink.sim import SimCar
    sim = SimCar("car_a").start()
    car = Car(sim.board, name="car_a")
    car.forward(150); print(car.distance_mm)   # telemetry updates live
    sim.close()

So `car_dashboard.py --sim`, policy development, and UI work need no hardware.
"""

from __future__ import annotations

import json
import threading
import time

from atech import Board, MockTransport

MODULES = ("vl53l5cx", "imu", "fl", "fr", "rl", "rr")


def _wire(event_type: str, key: str, value, source: str | None = None) -> str:
    payload = {"event_type": event_type, "key": key, "value": value}
    if source:
        payload["source"] = source
    return json.dumps({"type": "event", "payload": payload})


class SimCar:
    """A simulated car. Use ``.board`` with `Car`; call ``.close()`` when done."""

    def __init__(self, name: str = "car", *, hz: float = 5.0):
        self.name = name
        self.mt = MockTransport()
        self.board = Board(self.mt)
        self._dt = 1.0 / hz
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name=f"sim-{name}", daemon=True)
        # simulated physical state
        self._heading = 0.0
        self._dist = 2000.0  # mm to the nearest obstacle ahead
        self._throttle = 0
        self._steer = 0
        self._seen = 0  # how many sent actions we've consumed
        self._last_action = time.time()  # for the deadman
        self._stale = False

    def start(self) -> "SimCar":
        # announce modules present + ready up front
        for m in MODULES:
            self.mt.push_wire(_wire("state", f"module.{m}", "ok"))
        self.mt.push_wire(_wire("state", "depth_sensor", "ok"))
        self.mt.push_wire(_wire("state", "link", "ok"))
        self.mt.push_wire(_wire("state", "status", "ready"))
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.board.close()

    def _consume_actions(self) -> None:
        sent = self.mt.sent
        if len(sent) > self._seen:
            self._last_action = time.time()  # any action (incl. ping) feeds the deadman
        for a in sent[self._seen :]:
            k, v = a.key, a.value
            try:
                n = int(float(v)) if v not in (None, "") else 0
            except (TypeError, ValueError):
                n = 0
            if k in ("stop", "motor_brake", "motor_stop", "disable"):
                self._throttle = self._steer = 0
            elif k == "forward":
                self._throttle, self._steer = n, 0
            elif k == "backward":
                self._throttle, self._steer = -n, 0
            elif k == "left":
                self._throttle, self._steer = 0, -n
            elif k == "right":
                self._throttle, self._steer = 0, n
            elif k == "motor_speed":
                self._throttle, self._steer = n, 0
        self._seen = len(sent)

    def _loop(self) -> None:
        last_obstacle = False
        while not self._stop.is_set():
            self._consume_actions()
            # deadman: brake + go stale if no action (incl. ping) for >700ms
            stale = (time.time() - self._last_action) > 0.7
            if stale != self._stale:
                self._stale = stale
                self.mt.push_wire(_wire("state", "link", "stale" if stale else "ok"))
            if stale:
                self._throttle = self._steer = 0
            # integrate a toy motion model
            self._heading = (self._heading + self._steer * self._dt * 0.5 + 180) % 360 - 180
            if self._throttle > 0:
                self._dist = max(80.0, self._dist - self._throttle * self._dt * 2.0)
            else:
                self._dist = min(2000.0, self._dist + 300.0 * self._dt)
            obstacle = self._dist < 300
            # stream telemetry
            self.mt.push_wire(
                _wire("sensor", "orientation", f"0.0,179.0,{self._heading:.1f}", "icm40608_imu")
            )
            self.mt.push_wire(_wire("sensor", "min_distance", int(self._dist), "vl53l5cx_distance"))
            self.mt.push_wire(_wire("state", "status", "ready"))
            if obstacle != last_obstacle:
                self.mt.push_wire(
                    _wire("state", "obstacle", "detected" if obstacle else "clear")
                )
                last_obstacle = obstacle
            time.sleep(self._dt)
