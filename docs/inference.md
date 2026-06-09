# Running a policy (inference)

Once you have a trained policy on the Hub, you can drive the SO-101 follower with it
two ways: **locally** on the Mac (small models), or **remotely** on a GPU box (the
real setup we use for the ACT model in the demo).

The policy is selected in [`.env`](../.env):

```dotenv
POLICY_TYPE=act                              # or smolvla, pi0, …
POLICY_PATH=nsuprun/merged-so101-49904152    # HF repo / checkpoint (our ACT model)
POLICY_TASK=Pick up the cube
```

> Our demo runs the merged **ACT** model
> [`nsuprun/merged-so101-49904152`](https://huggingface.co/nsuprun/merged-so101-49904152),
> trained on [`nsuprun/so101-pickup-merged`](https://huggingface.co/datasets/nsuprun/so101-pickup-merged).
> See **[datasets.md](datasets.md)** for how that data was collected.

## Local (single-machine) VLA inference

The dashboard's **Run inference** panel loads a Hugging Face policy *in this
process* (`policy_inference.py`) and drives the follower from a task prompt. Fine
for a quick local test on a Mac with a small policy (`lerobot/smolvla_base`);
heavy VLAs (pi0) want the remote setup below. Gated models need `HF_TOKEN` in
`.env.local` (gitignored — never commit it).

## Remote (two-machine) VLA inference

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

The dashboard launches `run_robot_client.py` for you when you hit **Run inference**;
running it by hand is the same thing without the UI.

## Deploy the server to the GPU box (Ansible)

Instead of setting up the box by hand, push + launch the server with the
playbook in `deploy/` (rsyncs the repo — no GitHub auth needed on the box —
installs uv, writes `.env.local` with your token, `uv sync`, and starts it):

```bash
HF_TOKEN=$(grep '^HF_TOKEN=' .env.local | cut -d= -f2-) \
  uvx --from ansible-core ansible-playbook -i deploy/inventory.ini deploy/playbook.yml
```

Or simply `make server-deploy` / `make server-logs`. Edit `deploy/inventory.ini` for
your box's Tailscale IP / user. The token is read from the `HF_TOKEN` env var and
written to the box's gitignored `.env.local` — it is never stored in the playbook or
committed.

`run_robot_client.py` assembles the LeRobot CLI from `.env`, including the phone
camera (`resolve_phone_url()` injects the IP Webcam credentials). The client tells
the server which policy to load.
