"""Hardware-free demo of the orchestrator state machine over the mock backends.

    uv run python -m orchestrator
"""

from __future__ import annotations

import threading
import time

from .core import Leg, Mission, Orchestrator
from .interfaces import Pose
from .mocks import MockManipulator, MockNavigator, MockSensors


def _happy_path() -> None:
    print("=== happy path: two legs ===")
    sensors = MockSensors()
    nav = MockNavigator(sensors, travel_s=1.5)
    arm = MockManipulator(sensors, pick_s=1.5, neutral_s=0.5)
    orch = Orchestrator(nav, sensors, arm, settle_s=0.5, tick_hz=20)
    mission = Mission(
        [
            Leg(Pose(1.0, 0.0), "pick up the red block"),
            Leg(Pose(0.0, 1.0), "pick up the blue block"),
        ]
    )
    final = orch.run(mission)
    print(f"--> final: {final.name}")
    print("--> trace:", " -> ".join(s.name for s in orch.history))


def _abort_mid_pick() -> None:
    print("\n=== abort mid-pick ===")
    sensors = MockSensors()
    nav = MockNavigator(sensors, travel_s=0.5)
    arm = MockManipulator(sensors, pick_s=5.0)  # long pick so we can interrupt it
    orch = Orchestrator(nav, sensors, arm, settle_s=0.3, tick_hz=20)
    runner = threading.Thread(
        target=orch.run, args=(Mission([Leg(Pose(1.0, 0.0), "pick")]),)
    )
    runner.start()
    time.sleep(1.5)  # let it reach PICKING
    orch.abort()
    runner.join()
    print(f"--> final: {orch.state.name}")
    print("--> trace:", " -> ".join(s.name for s in orch.history))


if __name__ == "__main__":
    _happy_path()
    _abort_mid_pick()
