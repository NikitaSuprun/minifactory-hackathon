# SO-101 arms — dashboard, teleop & calibration

A minimal browser dashboard to control the SO-101 leader/follower pair lives in
[`arm_dashboard.py`](../arm_dashboard.py). **Both arms plug into this computer over
USB.** It can connect/disconnect the arms, start/stop teleoperation (leader drives
follower), record datasets, run inference, shows the live phone + wrist cameras, and
polls status.

> For getting the cameras streaming first, see **[cameras.md](cameras.md)**.
> For recording datasets and running policies, see **[datasets.md](datasets.md)** and
> **[inference.md](inference.md)**.

## Setup

1. Find the USB serial ports and put them in [`.env`](../.env):
   ```bash
   uv run lerobot-find-port      # run once per arm, unplug to identify
   ```
   ```dotenv
   FOLLOWER_PORT=/dev/tty.usbmodemXXXX
   LEADER_PORT=/dev/tty.usbmodemYYYY
   ```
2. Set motor IDs — **only if the motors are fresh from the kit** (assigns IDs
   1–6; connect one motor at a time when prompted). Skip for pre-assembled arms:
   ```bash
   uv run lerobot-setup-motors --robot.type=so101_follower --robot.port=$FOLLOWER_PORT
   uv run lerobot-setup-motors --teleop.type=so101_leader  --teleop.port=$LEADER_PORT
   ```
3. Calibrate each arm once (interactive — needs a real terminal). Move the arm to
   the middle/rest pose, press Enter, then sweep every joint through its full range:
   ```bash
   uv run lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER_PORT --robot.id=so101_follower
   uv run lerobot-calibrate --teleop.type=so101_leader  --teleop.port=$LEADER_PORT  --teleop.id=so101_leader
   ```
   The `--id` **must match** `ROBOT_ID` / `LEADER_ID` in `.env` — calibrations are
   saved to `~/.cache/huggingface/lerobot/calibration/<type>/<id>.json` and loaded
   by id at connect time. Persistent; redo only if you swap motors or change `--id`.

   **This repo already ships calibration** under [`calibration/so101_follower.json`](../calibration/so101_follower.json)
   and [`calibration/so101_leader.json`](../calibration/so101_leader.json). The
   dashboard and `run_robot_client.py` point `calibration_dir` there automatically
   (override with `CALIBRATION_DIR` in `.env`), so connect won't re-prompt as long as
   the motors hold matching values. For bare `lerobot-teleoperate`/`-record`, add
   `--robot.calibration_dir=calibration`.
   Note: calibration is **per physical arm** — these files match *our* arms; recalibrate
   if you use different hardware.
4. Run the dashboard and open it:
   ```bash
   uv run python arm_dashboard.py    # http://localhost:8041
   ```
   The port is set by `DASHBOARD_PORT` in `.env` (default 8041).

## The dashboard UI

The dashboard is a **Vite + React + Tailwind** SPA (`frontend/`). The built bundle in
`frontend/dist/` is committed and served by FastAPI, so the command above works with no
Node step. It shows status pills, state-aware controls, the inference panel, phone +
wrist camera tiles with live FPS, and the GPU-box + client log panels. To work on the
UI:

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173 (HMR, proxies the API to :8041)
npm run build    # refresh frontend/dist (commit it)
```

If `frontend/dist` is absent, the dashboard falls back to a built-in inline HTML page.

## Login (protects the dashboard + APIs)

The dashboard binds to `0.0.0.0`, so it's reachable on the WiFi. Every route
(page, APIs, camera) is behind **HTTP Basic Auth** — the browser shows a login
prompt. Credentials come from `.env`:

```dotenv
DASHBOARD_USER=admin
DASHBOARD_PASS=123123     # leave empty to disable auth (prints a warning)
```

> Like the camera, this is Basic Auth over plain HTTP on the LAN — fine for a
> hackathon, but the password is base64 (not encrypted) on the wire. Keep it
> throwaway.
