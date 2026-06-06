# Orchestrator design

Coordinates an SO-101 arm and a wheeled car to run a repeating **drive → pick →
depart** cycle, while gating arm inference to moments when the car is stationary so
we can record clean training episodes even before the car is fully autonomous.

## The loop

```
car drives to the arm's station
  → car stops, settles (stationary)
  → orchestrator prompts the VLA to pick the object off the car
  → arm presses the station button when it believes it's done
  → arm returns to neutral
  → car drives away to fetch more
  → repeat
```

The invariant that makes the recorded data clean: **exactly one of {car moving, arm
moving} is ever active.** The arm is only ever prompted while the car is confirmed
stationary, so the recorded arm episodes are unpolluted by car motion.

## Hardware topology

Three independent units. The two atech units are **separate motherboards =
separate connections** (serial is single-owner; do not open a board twice).

| Unit | Connection | Carries | Backs interface |
|------|-----------|---------|-----------------|
| Car board | atech conn #1 (carlink `Car`) | IMU (orientation) + 4 DC motors (**no wheel encoders**) | `Navigator` |
| Station board | atech conn #2 (carlink, sensors only) | distance sensor (faces incoming car) + button (arm presses it) | `Sensors` |
| SO-101 arm | LeRobot (USB serial + cameras) | follower (+ leader for teleop), cameras → camera1/2/3 | `Manipulator` |

The station board is **fixed** at the arm's dock; its distance sensor watches the
car roll in (authoritative arrival), and its button is a fixed reach target the arm
presses. It has **no actuators** — read-only.

## Architecture

One process holds all three interfaces. No IPC: the button must end the arm's
recording episode *and* release the car, which is trivial in-process and painful
across a subprocess boundary.

```
                        ┌────────────────────────────┐
                        │        Orchestrator         │
                        │  (state machine + e-stop)   │
                        └──┬───────────┬───────────┬──┘
              pick()/status│  move_to()│   button  │
              to_neutral() │  pose     │  car_present
                           ▼  at_goal  ▼           ▼
                    ┌────────────┐ ┌──────────┐ ┌──────────┐
                    │ Manipulator│ │ Navigator│ │  Sensors │
                    │ (arm/VLA)  │ │ (car brd)│ │(stn brd) │
                    └─────┬──────┘ └────┬─────┘ └────┬─────┘
                       LeRobot       carlink       carlink
                    (cameras,        Car (IMU      Board (distance
                     record/infer)    + motors)     + button, RO)
```

## Interfaces

`Pose` is a dead-reckoned estimate, **not** ground truth (see Measurement). All
methods are non-blocking unless noted; the orchestrator polls the properties.

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


@dataclass
class Pose:
    x: float            # metres, in the floor frame defined by the dock layout
    y: float
    heading_deg: float  # 0 = +x axis, CCW positive; from the car IMU


class PickStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"        # button fired (policy believes it finished)
    TIMEOUT = "timeout"  # safety fallback fired first
    ERROR = "error"


class Manipulator(Protocol):
    """The arm. 'Prompting' a language-conditioned VLA = passing a task string."""

    def pick(self, task: str) -> None:
        """Start one pick attempt: load/prompt the policy (or open a teleop
        episode) and begin driving the follower. Non-blocking; the control loop
        runs on its own thread. Caller must have confirmed the car is stationary."""

    @property
    def status(self) -> PickStatus: ...

    def to_neutral(self) -> None:
        """Drive the arm to a fixed rest pose, clear of the car. BLOCKS until the
        arm has settled (the orchestrator must not release the car before this)."""

    def stop(self) -> None:
        """Halt inference/recording immediately. Safe to call any time (e-stop)."""


class Navigator(Protocol):
    """The car board: actuate the wheels + estimate pose from the IMU."""

    def move_to(self, x: float, y: float) -> None:
        """Begin driving toward (x, y): turn to the bearing (closed-loop on the
        IMU), then drive at the calibrated cruise speed. Non-blocking."""

    @property
    def pose(self) -> Pose:
        """Current dead-reckoned estimate. Heading is real (IMU); x/y drifts."""

    @property
    def at_goal(self) -> bool:
        """The *estimate* says we've reached the goal. Approximate — used to slow
        for final approach, NOT as the authoritative stop (that's Sensors)."""

    def reset_pose(self, p: Pose) -> None:
        """Re-zero the estimate to a known pose (called when docked, using the
        dock's known coordinates). This is what bounds dead-reckoning drift."""

    def stop(self) -> None:
        """Active brake. Jumps the queue (carlink Car.stop drains pending sends)."""


class Sensors(Protocol):
    """The station board: arrival detection + the pick-done button. Read-only."""

    @property
    def car_distance_mm(self) -> float | None:
        """Range from the fixed station sensor to the approaching car (None until
        the first reading)."""

    @property
    def car_present(self) -> bool:
        """car_distance_mm < arrival threshold. The AUTHORITATIVE arrival signal —
        a fixed sensor, so it does not drift like the car's pose estimate."""

    @property
    def button(self) -> bool:
        """Latched: True once the arm has pressed the station button, until
        clear_button(). Edge-triggered so a held press doesn't re-fire."""

    def clear_button(self) -> None:
        """Reset the button latch for the next cycle."""
```

### Backing implementations (carlink)

- `Navigator` wraps the car-board `Car`: `turn_to_heading()` / `drive()` /
  `stop()` for motion, `orientation` (`heading`) for the IMU, `tare_heading()` for
  re-zeroing. Owns the dead-reckoning integrator that produces `pose`.
- `Sensors` wraps the station-board `Board` (or a motorless `Car`): reads
  `value("min_distance")` and `value("button")` off the background reader. The
  button latch lives here.
- `Manipulator` wraps the in-process LeRobot loop. For **data collection** it drives
  `record_dataset.py`'s `record_loop`; `button → events["exit_early"]` ends the
  episode (the same seam the right-arrow key uses today). For **autonomous** rollout
  it drives `policy_inference.infer_action` / `run_robot_client` and terminates on
  the same button.

## State machine

Owned entirely by the orchestrator (`orchestrator/core.py`); the interfaces are
dumb. One state at a time. `OrchState` =
`IDLE · NAVIGATING · SETTLING · PICKING · RETRACTING · DEPARTING · DONE · ABORTED · ERROR`.

```
            ┌──────┐
            │ IDLE │
            └──┬───┘ run(mission)
               ▼
       ┌──────────────┐ car_present → SETTLING                       ┌───────┐
       │  NAVIGATING  │ nav_timeout (no arrival) ───────────────────▶│ ERROR │
       └──────┬───────┘                                              └───────┘
              │ stop(); reset_pose(dock)                                 ▲
              ▼                                                          │
       ┌──────────────┐ dwell ≥ settle_s → PICKING                      │
       │  SETTLING    │                                                 │
       └──────┬───────┘                                                 │
              ▼                                                          │
       ┌──────────────┐ clear_button(); pick(task)                      │
       │   PICKING    │ button → RETRACTING        arm status==ERROR ───┤
       └──────┬───────┘ pick_timeout → RETRACTING (give up, move on)    │
              ▼                                                          │
       ┌──────────────┐ to_neutral() (blocks until arm clear)           │
       │  RETRACTING  │                                                 │
       └──────┬───────┘ arm no longer RUNNING → DEPARTING               │
              ▼                                                          │
       ┌──────────────┐ move_to(home)                                   │
       │  DEPARTING   │ !car_present (cleared dock) → next leg          │
       └──────┬───────┘                                                 │
              ▼                                                          │
        more legs → NAVIGATING, else → DONE                             │
                                                                        │
   abort() from ANY state → ABORTED;  unhandled exception ─────────────▶┘
   every terminal state (DONE/ABORTED/ERROR): nav.stop() + arm.stop()
```

Transition notes:

- **Arrival fuses two boards.** `Navigator.at_goal` (the car's drifting estimate)
  is only for *slowing* on final approach inside the Navigator; the orchestrator's
  **stop trigger is `Sensors.car_present`** (fixed sensor, ground truth). On arrival
  it `reset_pose()`s to the dock's known coordinates (real backend also re-tares).
- **No car-side obstacle sensing**, so there's no "obstacle fault" — a leg that
  never arrives fails via **`nav_timeout_s` → ERROR** instead. (The Navigator
  Protocol therefore has no `fault` member; this replaced the earlier single-board
  guard idea.)
- **SETTLING is a real state.** The car must be confirmed stationary (brake + short
  dwell) before `pick()` — that's the whole point of gating recording.
- **PICKING never hangs.** It ends on the button (success), on `pick_timeout_s`
  (give up gracefully and still RETRACT → DEPART → continue), or on an arm error
  (→ ERROR).
- **RETRACTING blocks.** The car is **not** released until `to_neutral()` returns
  (arm no longer `RUNNING`), or it yanks the workpiece out from under a moving arm.
  Caveat: because `to_neutral()` blocks the tick loop, `abort()` is not serviced
  *during* the retract — acceptable while it's a short, safe motion.
- **Button is routed twice.** A press both ends the arm's episode
  (`events["exit_early"]`, in the real backend) and triggers `PICKING → RETRACTING`.
  `clear_button()` is called on entering PICKING. *(Phase 1: the orchestrator
  transitions on `Sensors.button`; binding the press to episode-save vs. -discard
  on success/timeout is a Phase 2 backend detail.)*
- **e-stop** brakes the car and halts the arm from any state. Only the car board has
  actuators; the station board is read-only, nothing to stop there.

## Navigation: open-loop dead-reckoning + closed-loop lock-in

No encoders, so position is open-loop. The design makes that acceptable by doing the
precision mechanically and with the fixed sensor, not with the control loop.

```
move_to(dock):
    bearing, dist = vector(pose → dock)
    turn_to_heading(bearing)         # CLOSED-loop on IMU heading — accurate
    drive(cruise) for ~dist / v_cal  # OPEN-loop — only needs to reach the funnel mouth
    creep slowly                      # until Sensors.car_present — CLOSED-loop on the
                                      #   fixed station sensor (final stop)
    stop(); reset_pose(dock)          # re-zero drift to known dock coords
    tare_heading()                    # free re-zero: true heading is known when docked
```

Why approximate is sufficient:

1. **`reset_pose()` at every dock bounds error to a single leg** — it never
   accumulates across the route.
2. **A wide funnel + backstop at the dock does the lateral/heading precision
   mechanically.** The nav only has to land in the funnel mouth (~±10–20 cm, ±15°),
   not on a coordinate.
3. **The final stop is closed-loop on the fixed station sensor** (creep until
   `car_present`), so the velocity calibration only has to get the car *into sensing
   range*, not to an exact point.

Calibration is just a **speed→velocity map**: drive a known cruise speed on the real
floor, tape-measure distance over a few seconds → cm/s. Turns need no calibration —
the IMU closes that loop.

## Measurement difficulties (read before trusting any number)

The sensing is the weakest part of the system. Each item below is *why* a naive
"drive to x/y" fails, and the mitigation we rely on.

- **No wheel encoders → no odometry.** Distance travelled is open-loop (calibrated
  speed × time). *Mitigation:* short legs; final stop closed-loop on the station
  sensor; `reset_pose()` at docks.
- **IMU gives orientation, not position.** You cannot integrate the accelerometer to
  get x/y — double-integrating noise/bias drifts metres within seconds. The IMU is
  trustworthy only for **heading**. *Mitigation:* use it for closed-loop turning;
  derive position from the (drifting) speed model, corrected at docks.
- **Wheel slip & surface variation** are stochastic — calibration removes the
  systematic scale error but not slip. Per-leg error depends on the floor.
  *Mitigation:* calibrate on the actual run surface; wide funnel; keep legs short.
- **Gyro heading drift.** If the IMU heading is gyro-only (no magnetometer fusion) it
  drifts over minutes. *Mitigation:* `tare_heading()` at each dock, where true
  heading is known. (Confirm whether the firmware fuses a magnetometer.)
- **`Car.value("speed")` is commanded, not measured.** Treat it as a setpoint, never
  as feedback.
- **Station distance sensor is noisy and rate-limited** (VL53-class, ~15 Hz, mm but
  jittery). At speed the car travels between samples. *Mitigation:* crawl on final
  approach; threshold with a small median/hysteresis; lead the brake for momentum.
- **Braking is not instantaneous** — the car coasts after `stop()`. *Mitigation:*
  trigger the stop slightly early and/or nose into a soft backstop; confirm
  stationary in SETTLING before picking.
- **The "success" button is self-reported.** A press means *the policy believes it
  finished*, not that the grasp is verified — the policy can press it after a failed
  pick. *Mitigation:* `Timeout` fallback so the loop never hangs; human spot-checks
  of the true success rate. (The OAK-D depth grasp-verifier we considered is **out** —
  no depth camera.)

None of these block the design — they are exactly why the docks (fixed sensor +
funnel + `reset_pose`) carry the precision and the control loop stays dumb.

## Integration notes / to-do

- **One process, in-process arm loop** so the button can reach
  `events["exit_early"]` directly.
- **Camera lock:** the orchestrator is a camera-owning process — `camera_lock.acquire()`
  and respect the dashboard's release grace (see `record_dataset.py`).
- **Two atech connections** (car board, station board) constructed once and shared;
  e-stop routes through the car `Car` (whose `stop()` already drains its send queue).
- **Firmware to-do:** the station board must *emit* the button as a state event
  (e.g. `button = pressed/released`), like `obstacle`/`orientation` today — see
  `firmware/PROMPT.md`. New input declaration, not new hardware plumbing.
- **Deps:** `atech` + `websocket-client` are now in `pyproject.toml`. Note `atech`
  pulls `platformio`, which caps `uvicorn <0.41`, so the project's uvicorn was
  relaxed from `>=0.49.0` to `>=0.16` (resolves to ~0.40.x).

## Status / layout

**Phase 1 (hardware-free skeleton) is built and passing** — interfaces, mock
backends, and the state machine. Run the demo (happy path + abort):

    uv run python -m orchestrator

```
orchestrator/
  interfaces.py   # Manipulator / Navigator / Sensors Protocols + Pose + PickStatus  ✅
  mocks.py        # MockSensors / MockNavigator / MockManipulator (threading.Timer)  ✅
  core.py         # Orchestrator + OrchState + Mission/Leg (the state machine)        ✅
  __main__.py     # the hardware-free demo                                           ✅
  navigator.py    # CarNavigator over carlink Car — move_to + dead-reckoning         ⏳ phase 2
  sensors.py      # StationSensors over the station Board — distance + button latch   ⏳ phase 2
  manipulator.py  # RecordManipulator (teleop) / InferManipulator (autonomous)       ⏳ phase 2
run_orchestrator.py  # entrypoint: build the three, camera_lock, run a Mission       ⏳ phase 3
```

The state machine is fully testable against the mocks (scripted scenarios assert the
emitted `Orchestrator.history` trace) — control logic is verified before any board is
plugged in.

## Open questions

- Funnel/backstop geometry — sets how sloppy the nav is allowed to be (highest
  leverage to pin down).
- Mission shape: single load↔arm ferry (two docks) or a multi-stop route? Decides
  whether the car must choose a *direction* at each dock.
- Phase 1 (teleop recording) vs phase 2 (autonomous rollout) first — same
  orchestrator, different `Manipulator` backend.
