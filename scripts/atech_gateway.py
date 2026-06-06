"""Control / monitor a car over the atech WiFi gateway (gateway.atech.dev).

Works with the online-generated car firmware (global actions: motor_speed,
turn_left/right, spin_left/right, motor_stop). The board must be in WiFi mode and
streaming to the gateway; auth is the project ID.

Usage:
    uv run python scripts/atech_gateway.py                 # monitor events ($ATECH_PROJECT_ID)
    uv run python scripts/atech_gateway.py <project-id>    # monitor a specific project
    uv run python scripts/atech_gateway.py --drive         # short drive demo, then stop
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from carlink import Car, connect_gateway  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    drive = "--drive" in sys.argv
    project_id = args[0] if args else os.environ.get("ATECH_PROJECT_ID")
    if not project_id:
        print("No project ID given and $ATECH_PROJECT_ID is unset.")
        return 2

    print(f"connecting to gateway for project {project_id} ...")
    try:
        board = connect_gateway(project_id)
    except Exception as e:  # noqa: BLE001
        print(f"Could not connect: {e}")
        return 1
    car = Car(board, name="car")

    try:
        if drive:
            print("DRIVE DEMO — watch the car (forward, turn, spin, stop)")
            steps = [
                ("forward 150", lambda: car.drive(150)),
                ("stop", car.stop),
            ]
            for label, fn in steps:
                print(f"  >>> {label}")
                fn()
                time.sleep(1.2)
            car.stop()
            print("stopped.")
        else:
            print("monitoring events (Ctrl-C to stop) ...")
            while True:
                time.sleep(1.0)
                print(
                    f"running={car.is_running} speed={car.speed} "
                    f"distance_mm={car.distance_mm}"
                )
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        car.stop()
        board.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
