"""carlink: drive & monitor the hackathon cars on top of the official `atech` SDK.

The official `atech` package owns firmware (catalog/codegen/build/flash) and the
serial runtime (`atech.Board`). carlink is the thin host-side layer for the cars:

    gateway   GatewayTransport + connect_gateway  -> atech.Board over WiFi (the SDK is serial-only)
    car       Car: high-level drive/stop/light + state, over any atech.Board
    policy    Policy + PolicyRunner: decision loop with safe abort

Quick start (serial, current firmware):

    from carlink import connect_serial, Car, PolicyRunner, StraightUntilObstacle

    car = Car(connect_serial(), name="car_a")          # auto-discovers the USB board
    runner = PolicyRunner(car, hz=20)
    runner.start(StraightUntilObstacle(speed=180, stop_distance_mm=300))
    ...
    runner.abort()                                      # brakes
    car.board.close()

Quick start (WiFi gateway):

    from carlink import connect_gateway, Car
    car = Car(connect_gateway("my-project-id"))
"""

from atech import Board

from .car import Car
from .gateway import GatewayTransport, connect_gateway
from .policy import CallablePolicy, Policy, PolicyRunner, StraightUntilObstacle
from .util import as_bool, as_float


def connect_serial(port: str | None = None, baud: int = 115200) -> Board:
    """Open an atech.Board over USB serial (auto-discovers the port if None)."""
    return Board.connect(port, baud)


__all__ = [
    "Board",
    "connect_serial",
    "connect_gateway",
    "GatewayTransport",
    "Car",
    "Policy",
    "CallablePolicy",
    "PolicyRunner",
    "StraightUntilObstacle",
    "as_bool",
    "as_float",
]
