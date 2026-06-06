# minifactory-hackathon

Stream an Android phone's camera to this Mac over WiFi and use it as a
[LeRobot](https://github.com/huggingface/lerobot) camera for model inference and
dataset recording.

The phone runs the **IP Webcam** app and serves an MJPEG stream; LeRobot's
`OpenCVCamera` opens that URL directly — no macOS driver or virtual camera needed.

## 1. Phone setup (IP Webcam)

1. Install **IP Webcam** (by Pavel Khlebovich) from the Play Store.
2. Open the app and, for low latency, set **Video preferences**:
   - Resolution: `640x480` (modest is fine; lower = lower latency)
   - Quality: ~50%
   - FPS limit: 30
   - Disable audio streaming
3. (Optional) Protect the stream: under **Connection → Login/password**, set a
   **Login** and **Password** (e.g. `admin` / `123123`). The stream then requires
   HTTP Basic Auth.
4. Scroll to the bottom and tap **Start server**.
5. The app shows a URL like `http://192.168.1.42:8080`. The MJPEG stream is that
   address + `/video`, e.g. `http://192.168.1.42:8080/video`.

## 2. Network check

Phone and Mac must be on the **same WiFi** (guest networks / "client isolation"
will block this). Confirm by opening `http://<phone-ip>:8080` in the Mac's browser —
you should see the live video and controls.

## 3. Configure `.env`

All connection settings live in the committed [`.env`](.env). Edit
`PHONE_CAM_HOST` to your phone's IP (and the login/password if you set them):

```dotenv
PHONE_CAM_HOST=192.168.1.42
PHONE_CAM_PORT=8080
PHONE_CAM_PATH=/video
PHONE_CAM_USER=admin
PHONE_CAM_PASS=123123
```

The URL is assembled from these as `http://USER:PASS@HOST:PORT/PATH`. Set
`PHONE_CAM_URL` instead if you want to override the whole thing.

> The `.env` is committed on purpose for this hackathon. That means the password
> is in git history — keep it a throwaway, LAN-only value, never a real secret.

## 4. Verify ingestion via LeRobot

```bash
uv run python scripts/check_phone_stream.py            # uses .env
# or pass an explicit URL (overrides .env):
uv run python scripts/check_phone_stream.py http://admin:123123@<phone-ip>:8080/video
```

Expected: it prints the detected resolution, achieved FPS, and a latency proxy,
then saves `stream_sample.png` showing the phone's camera view. That confirms
LeRobot's `OpenCVCamera` is ingesting the stream.

> `opencv-python-headless` (pulled in by LeRobot) has no GUI, so there's no live
> preview window — use the phone's browser page for live view; the script proves
> ingestion via stats + the saved snapshot.

## 5. Use the phone camera in LeRobot (later)

```python
from phone_camera import build_phone_camera_config

robot_config.cameras = {
    "phone": build_phone_camera_config("http://192.168.1.42:8080/video"),
}
```

`build_phone_camera_config` leaves `fps/width/height` unset so LeRobot auto-detects
the stream profile — set resolution/FPS in the IP Webcam app, not in code.

For a password-protected stream, embed credentials in the URL (optionally via the
`with_credentials` helper):

```python
from phone_camera import build_phone_camera_config, with_credentials

url = with_credentials("http://192.168.1.42:8080/video", "admin", "123123")
robot_config.cameras = {"phone": build_phone_camera_config(url)}
```

## 6. Web control dashboard (SO-101 teleop)

A minimal browser dashboard to control the SO-101 leader/follower pair lives in
[`arm_dashboard.py`](arm_dashboard.py). **Both arms plug into this computer over
USB.** It can connect/disconnect the arms, start/stop teleoperation (leader drives
follower), shows the live phone camera, and polls status. Record + inference are
scaffolded as TODO hooks (teleop first).

Setup:

1. Find the USB serial ports and put them in `.env`:
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
4. Run the dashboard and open it:
   ```bash
   uv run python arm_dashboard.py    # http://localhost:8041
   ```
   The port is set by `DASHBOARD_PORT` in `.env` (default 8041).

### Login (protects the dashboard + APIs)

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

### Local (single-machine) VLA inference

The dashboard's **Run inference** panel loads a Hugging Face policy *in this
process* (`policy_inference.py`) and drives the follower from a task prompt. Fine
for a quick local test on a Mac with a small policy (`lerobot/smolvla_base`);
heavy VLAs (pi0) want the remote setup below. Gated models need `HF_TOKEN` in
`.env.local` (gitignored — never commit it).

## 7. Remote (two-machine) VLA inference

For real VLAs, run inference on a GPU box and keep the arm on this Mac, using
LeRobot's built-in async inference:

```
Phone ──MJPEG──▶ Mac (run_robot_client.py: reads camera, owns the SO-101)
                   │
                   └─gRPC: observations (incl. images) ─▶ GPU box (run_policy_server.py)
                   ◀─gRPC: action chunks ────────────────┘
```

**The GPU box never connects to the phone** — the Mac reads the camera locally and
ships decoded frames over gRPC. So no reverse proxy is needed for the camera. The
only cross-network link is **Mac → `POLICY_SERVER_ADDRESS`** (the gRPC port). If
the GPU box is in the cloud / behind NAT, make that port reachable one of two ways.

**Option A — Tailscale (recommended).** A mesh VPN giving both machines stable
`100.x` IPs through NAT; no tunnel needed.

```bash
# This Mac:
brew install --cask tailscale            # then open the app, log in, Connect
/Applications/Tailscale.app/Contents/MacOS/Tailscale ip -4

# GPU box (Linux), same Tailscale account:
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4                          # -> use this as POLICY_SERVER_ADDRESS host
```

Then set `POLICY_SERVER_ADDRESS=<gpu-tailscale-ip>:8080` and run the server/client
directly — no `run_tunnel.py`.

**Option B — SSH tunnel.** If you can't use Tailscale, forward the port over SSH:

```bash
# helper: reads GPU_SSH_HOST / TUNNEL_LOCAL_PORT from .env and forwards the port
uv run python run_tunnel.py
# then set POLICY_SERVER_ADDRESS=localhost:8080
```

```dotenv
GPU_SSH_HOST=ubuntu@gpu-box   # or user@<public-ip>
GPU_SSH_PORT=22
TUNNEL_LOCAL_PORT=8080
```

Configure both sides in `.env`:

```dotenv
POLICY_TYPE=smolvla                 # or pi0, act, …
POLICY_PATH=lerobot/smolvla_base    # HF repo / checkpoint
POLICY_TASK=Pick up the cube
POLICY_SERVER_ADDRESS=192.168.1.50:8080   # GPU box, as seen from the Mac
SERVER_POLICY_DEVICE=cuda           # device on the GPU box
CLIENT_DEVICE=cpu                   # device on the Mac
```

Run it:

```bash
# On this Mac (arm + phone camera):
uv run python run_robot_client.py
```

### Deploy the server to the GPU box (Ansible)

Instead of setting up the box by hand, push + launch the server with the
playbook in `deploy/` (rsyncs the repo — no GitHub auth needed on the box —
installs uv, writes `.env.local` with your token, `uv sync`, and starts it):

```bash
HF_TOKEN=$(grep '^HF_TOKEN=' .env.local | cut -d= -f2-) \
  uvx --from ansible-core ansible-playbook -i deploy/inventory.ini deploy/playbook.yml
```

Edit `deploy/inventory.ini` for your box's Tailscale IP / user. The token is read
from the `HF_TOKEN` env var and written to the box's gitignored `.env.local` — it
is never stored in the playbook or committed.

`run_robot_client.py` assembles the LeRobot CLI from `.env`, including the phone
camera (`resolve_phone_url()` injects the IP Webcam credentials). The client tells
the server which policy to load.

## Optional: lowest latency over USB

Tether the phone over USB and forward the port with adb, then use `localhost`:

```bash
adb forward tcp:8080 tcp:8080
uv run python scripts/check_phone_stream.py http://localhost:8080/video
```

## Notes

- MJPEG over WiFi is typically ~100–200 ms latency. RTSP
  (`rtsp://<phone-ip>:8080/h264_ulaw.sdp`) is also available but tends to buffer
  more in OpenCV — prefer MJPEG.
