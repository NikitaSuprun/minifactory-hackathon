"""Car: a thin, configurable control surface over an `atech.Board`.

The official SDK gives us `board.send(key, value)` and `board.latest(key)`. A car
is just a naming convention on top of that, and the convention depends on the
firmware flashed:

- Per-wheel `dc_motor` modules (SDK catalog): actions are ``<instance>_speed`` /
  ``<instance>_brake`` / ``<instance>_stop`` — so set ``motors=("fl","fr","rl","rr")``.
- A single "car" program (what's on the board today): a global ``motor_speed`` /
  ``motor_brake`` — leave ``motors`` empty and set ``drive_action="motor_speed"``.

Either way, ``drive()`` / ``stop()`` do the right thing, and the policy/dashboard
stay firmware-agnostic. State is read by key via ``board.latest`` (kept fresh by
the transport's background reader).
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from atech import Board

from .util import as_bool, as_float

# Telemetry keys whose freshness the dashboard tracks.
_TELEMETRY_KEYS = ("orientation", "min_distance", "status", "obstacle")

# Module-health keys the firmware emits (state events module.<instance> = ok/missing).
_MODULE_KEYS = ("vl53l5cx", "imu", "fl", "rl", "fr", "rr")

# The firmware reads an action's value as a string (char*), and the official
# encoder maps None -> "". But atech.Action rejects None at construction, so for
# value-less actions (brake/stop/clear) we send "" rather than None.
_NULL = ""


@dataclass
class Car:
    board: Board
    name: str = "car"

    # Actuators — pick ONE motor style:
    motors: tuple[
        str, ...
    ] = ()  # per-wheel dc_motor instances, e.g. ("fl","fr","rl","rr")
    drive_action: str = "motor_speed"  # global action when motors is empty
    brake_action: str = "motor_brake"
    coast_action: str = "motor_stop"
    light: str | None = None  # neopixel instance name, if any

    # Sensors / state event keys (firmware-defined)
    running_key: str = "car_running"
    speed_key: str = "speed"
    distance_keys: tuple[str, ...] = ("min_distance", "distance", "distance_mm")

    # All outbound actions funnel through ONE writer thread + queue. Writing to a
    # serial port from multiple threads hangs on macOS (heartbeat + policy + route
    # threads), so every caller just enqueues and the single writer does os.write.
    _q: queue.Queue = field(default_factory=queue.Queue, repr=False)
    _stop_tx: threading.Event = field(default_factory=threading.Event, repr=False)
    _writer: Any = field(default=None, repr=False)
    last_send_error: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            self.name = "car"
        self._writer = threading.Thread(
            target=self._writer_loop, name=f"car-tx-{self.name}", daemon=True
        )
        self._writer.start()

    # -- outbound --
    def drive(self, speed: int) -> None:
        """Forward (+) / reverse (-), -255..255."""
        speed = max(-255, min(255, int(speed)))
        if self.motors:
            for m in self.motors:
                self._send(f"{m}_speed", speed)
        else:
            self._send(self.drive_action, speed)

    def stop(self) -> None:
        """Active brake — used for abort / e-stop. Jumps any queued commands."""
        self._drain()  # don't let a backed-up queue delay the brake
        if self.motors:
            for m in self.motors:
                self._send(f"{m}_brake", _NULL)
        else:
            self._send(self.brake_action, _NULL)

    def coast(self) -> None:
        if self.motors:
            for m in self.motors:
                self._send(f"{m}_stop", _NULL)
        else:
            self._send(self.coast_action, _NULL)

    # RC-car program actions (current firmware handleMessage):
    #   forward/backward/left/right take a 0..255 speed; stop brakes.
    def forward(self, speed: int = 150) -> None:
        self._send("forward", max(0, min(255, int(speed))))

    def backward(self, speed: int = 150) -> None:
        self._send("backward", max(0, min(255, int(speed))))

    def turn_left(self, speed: int = 150) -> None:
        self._send("left", max(0, min(255, int(speed))))

    def turn_right(self, speed: int = 150) -> None:
        self._send("right", max(0, min(255, int(speed))))

    def turn_to_heading(self, degrees: float) -> None:
        self._send("turn_to_heading", float(degrees))

    def tare_heading(self) -> None:
        self._send("tare_heading", _NULL)

    def enable(self) -> None:
        self._send("enable", _NULL)

    def disable(self) -> None:
        self._send("disable", _NULL)

    def set_light(self, r: int, g: int, b: int) -> None:
        if not self.light:
            return
        clamp = lambda c: max(0, min(255, int(c)))  # noqa: E731
        self._send(f"{self.light}_fill", {"r": clamp(r), "g": clamp(g), "b": clamp(b)})

    def send(self, key: str, value: Any = _NULL) -> None:
        """Escape hatch: send any action verbatim (None -> "" for value-less actions)."""
        self._send(key, _NULL if value is None else value)

    def ping(self) -> None:
        """Keepalive for the firmware deadman watchdog (resets its 500ms timer)."""
        self._send("ping", _NULL)

    def _send(self, key: str, value: Any) -> None:
        """Enqueue an action; the writer thread performs the actual serial write."""
        self._q.put((key, value))

    def _drain(self) -> None:
        """Discard any queued-but-unsent actions."""
        try:
            while True:
                self._q.get_nowait()
                self._q.task_done()
        except queue.Empty:
            pass

    def _writer_loop(self) -> None:
        while not self._stop_tx.is_set():
            try:
                key, value = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.board.send(key, value)
            except Exception as e:  # noqa: BLE001 - surface, keep the writer alive
                self.last_send_error = f"send {key}: {e}"
            finally:
                self._q.task_done()

    def close(self) -> None:
        """Stop the writer thread and close the underlying board."""
        self._stop_tx.set()
        if self._writer is not None:
            self._writer.join(timeout=1.5)
        self.board.close()

    # -- inbound / state --
    def value(self, key: str) -> Any:
        e = self.board.latest(key)
        return None if e is None else e.value

    def age_ms(self, key: str) -> int | None:
        """Milliseconds since the last event for ``key`` (None if never / no clock)."""
        e = self.board.latest(key)
        if e is None or e.received_at is None:
            return None
        return round((time.time() - e.received_at) * 1000)

    @property
    def is_running(self) -> bool | None:
        return as_bool(self.value(self.running_key))

    @property
    def speed(self) -> float | None:
        return as_float(self.value(self.speed_key))

    @property
    def distance_mm(self) -> float | None:
        for k in self.distance_keys:
            v = as_float(self.value(k))
            if v is not None:
                return v
        return None

    @property
    def orientation(self) -> tuple[float, float, float] | None:
        """(pitch, roll, heading) degrees, from the IMU 'orientation' event."""
        v = self.value("orientation")
        if not isinstance(v, str):
            return None
        try:
            pitch, roll, heading = (float(x) for x in v.split(","))
            return (pitch, roll, heading)
        except ValueError:
            return None

    @property
    def obstacle(self) -> bool | None:
        v = self.value("obstacle")
        return None if v is None else (v == "detected")

    @property
    def link_stale(self) -> bool | None:
        """Firmware deadman state ('link' = stale/ok). None until first report."""
        v = self.value("link")
        return None if v is None else (v == "stale")

    @property
    def depth_connected(self) -> bool | None:
        v = self.value("depth_sensor")
        return None if v is None else (v == "ok")

    def modules(self) -> dict[str, str | None]:
        """Per-module health from the firmware's module.<instance> state events."""
        return {m: self.value(f"module.{m}") for m in _MODULE_KEYS}

    def snapshot(self) -> dict[str, Any]:
        ages = {k: self.age_ms(k) for k in _TELEMETRY_KEYS}
        fresh = [a for a in ages.values() if a is not None]
        return {
            "name": self.name,
            "is_running": self.is_running,
            "speed": self.speed,
            "distance_mm": self.distance_mm,
            "orientation": self.orientation,
            "obstacle": self.obstacle,
            "status": self.value("status"),
            "link_stale": self.link_stale,
            "depth_connected": self.depth_connected,
            "modules": self.modules(),
            "motors": list(self.motors),
            "ages_ms": ages,
            "last_rx_age_ms": min(fresh) if fresh else None,
        }
