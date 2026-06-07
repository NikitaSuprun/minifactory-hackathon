"""Standalone drive dashboard for the atech car — forward / back / rotate / stop.

Confirmed firmware interface (one "car" program over dc_motor, latches commands
until changed; no deadman watchdog):

    motor_speed  <v>   v>0 -> forward, v<0 -> backward   (car_action: forward/backward)
    turn_left    <v>   rotate left in place  (0..255)     (car_action: turn_left)
    turn_right   <v>   rotate right in place (0..255)     (car_action: turn_right)
    stop               brake                              (car_action: braking/stopped)

Telemetry: car_action (state). car_speed exists but is a constant — ignore it.

Run:
    uv run python drive_dashboard.py                          # USB serial, http://localhost:8043
    ATECH_CAR_PORT=/dev/cu.usbmodem11201 uv run python drive_dashboard.py
    ATECH_CAR_HOST=car.local uv run python drive_dashboard.py # WiFi (no cable!)

Over USB: only one program can own the serial port — close the atech browser Web
Serial bridge (and any probe/monitor) first. Over WiFi: set ATECH_CAR_HOST to the
car's mDNS name (car.local) or IP; the car must be powered (battery) and on the
same WiFi (firmware: firmware/build_car_speaker.py).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from carlink import Car, connect_serial, connect_wifi

load_dotenv()

CAR_PORT = os.environ.get("ATECH_CAR_PORT") or None  # None = auto-discover
# Set ATECH_CAR_HOST (e.g. "car.local" or an IP) to drive over WiFi instead of
# the USB cable — needs the WiFi firmware flashed (firmware/build_car_speaker.py).
CAR_HOST = os.environ.get("ATECH_CAR_HOST") or None
# 224 is the empirical ceiling — above it the motor surge browns out the board's
# USB. Cap here so the slider/commands can't push past it.
MAX_SPEED = int(os.environ.get("CAR_MAX_SPEED", "224"))
DEFAULT_SPEED = int(os.environ.get("CAR_SPEED", str(MAX_SPEED)))  # fast by default
HTTP_PORT = int(os.environ.get("DRIVE_DASHBOARD_PORT", "8043"))

# Firmware action names (see module docstring). Turning uses dedicated actions;
# forward/back is signed motor_speed.
TURN_LEFT = "turn_left"
TURN_RIGHT = "turn_right"
STOP = "stop"

# Inversion used to replay a recorded path in reverse (retrace to start): play the
# segments in reverse order with each motion inverted — drive back, undo each turn.
INVERT = {
    "forward": "back",
    "back": "forward",
    "left": "right",
    "right": "left",
    "stop": "stop",
}

# Speaker instance name (atech speaker module -> <inst>_play_rtttl / _set_volume /
# _stop). Not yet confirmed on this firmware, so it's configurable + testable from
# the UI. Candidates to try by ear:
SPEAKER = os.environ.get("ATECH_SPEAKER", "spk")  # our firmware names it 'spk'

# RTTTL melodies (tune by ear from here — no reflash needed, just strings the
# dashboard sends). Each jingle is its own button.
JINGLES = {
    # "Erika" / "Auf der Heide blüht ein kleines Blümelein" (the Hemglass tune),
    # kept in octave 4-5 for the speaker.
    "erika": (
        "erika:d=8,o=5,b=125:"
        "e4,f4,g4,g4,g4,c5,c5,e5,e5,d5,4c5,4p,"  # Auf der Heide blüht ein kleines Blümelein
        "b4,c5,4d5,4p,"  # und das heißt
        "e5,d5,2c5"  # Erika
    ),
    # Draft "Dance of the Cuckoos" (Laurel & Hardy) — the other ice-cream tune.
    "cuckoos": (
        "cuckoos:d=4,o=5,b=180:"
        "8e6,8c6,8g,8e,8c6,8g,8e,8c,"
        "8d6,8b,8g,8d,8f6,8d6,8b,8g,2c6"
    ),
}
HONK = "honk:d=4,o=5,b=120:8a4,2a4"
TEST_TONE = "t:d=8,o=6,b=140:c,e,g"


class CarLink:
    """Owns one Car connection with open/close/reconnect + the 5 drive commands."""

    def __init__(self, port: str | None) -> None:
        self.port = port
        self.car: Car | None = None
        self.error: str | None = None
        self.last_cmd: tuple[str, int] | None = None  # replayed after a reconnect
        self.reconnects = 0
        self._connected_at = 0.0
        self._lock = threading.Lock()
        # Path record/replay: a path is a list of (command, speed, duration_s) segments.
        self.recording = False
        self.segments: list[tuple[str, int, float]] = []
        self._seg_name: str | None = None
        self._seg_speed = 0
        self._seg_t = 0.0
        self.replaying: str | None = None  # None | "forward" | "reverse"
        self._replay_thread: threading.Thread | None = None
        self._replay_abort = threading.Event()
        self._watchdog = threading.Thread(target=self._watch, daemon=True)
        self._watchdog.start()

    def connect(self) -> bool:
        with self._lock:
            self._close_locked()
            try:
                board = (
                    connect_wifi(CAR_HOST) if CAR_HOST else connect_serial(self.port)
                )
                self.car = Car(board, name="car")
                self.error = None
                self._connected_at = time.time()
                return True
            except Exception as e:  # noqa: BLE001
                self.car = None
                msg = str(e)
                if CAR_HOST:
                    msg += f" — can't reach the car at {CAR_HOST}:3333 (powered on & on WiFi?)"
                elif any(s in msg.lower() for s in ("busy", "resource", "access")):
                    msg += " — close the atech web bridge / serial monitor first."
                self.error = msg
                return False

    @staticmethod
    def _is_dead(err: str | None) -> bool:
        """A write/read failure that means the serial fd is gone (USB dropout)."""
        if not err:
            return False
        e = err.lower()
        return "not configured" in e or "write failed" in e or "device" in e

    def _watch(self) -> None:
        """Keep a live link: (re)connect whenever we're not connected — the initial
        connect (board still booting over WiFi), a USB dropout, or a TCP drop (motor
        brownout) — and replay the last command so driving resumes on its own."""
        while True:
            time.sleep(0.5)
            car = self.car
            stale = car is None or self._is_dead(car.last_send_error)
            if not stale and car is not None:
                # connected, but telemetry silent for too long (e.g. the board was
                # still settling when the socket opened) -> force a fresh connect.
                age = car.age_ms("car_action")
                since_conn = time.time() - self._connected_at
                if since_conn > 4.0 and (age is None or age > 3000):
                    stale = True
            if stale:
                if self.connect():
                    self.reconnects += 1
                    cmd = self.last_cmd
                    if cmd is not None:
                        try:
                            self.command(*cmd)
                        except Exception:  # noqa: BLE001
                            pass

    def _close_locked(self) -> None:
        if self.car is not None:
            try:
                self.car.send(STOP)  # brake on the way out
                self.car.close()
            except Exception:  # noqa: BLE001
                pass
            self.car = None

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def command(self, name: str, speed: int) -> None:
        """Send one latching drive command. The firmware holds it until changed."""
        speed = max(0, min(MAX_SPEED, int(speed)))  # cap below the brownout ceiling
        self.last_cmd = (name, speed)  # remembered so the watchdog can replay it
        if self.recording:
            self._record_step(name, speed)
        car = self.car
        if car is None or self._is_dead(car.last_send_error):
            if not self.connect():  # re-discovers the port
                raise RuntimeError(self.error or "not connected")
            car = self.car
            assert car is not None
        # Motors are wired reversed on this car, so forward = negative motor_speed.
        if name == "forward":
            car.drive(-speed)
        elif name == "back":
            car.drive(speed)
        elif name == "left":
            car.send(TURN_LEFT, speed)
        elif name == "right":
            car.send(TURN_RIGHT, speed)
        elif name == "stop":
            car.send(STOP)
        else:
            raise ValueError(f"unknown command {name!r}")

    def sound(self, inst: str, melody: str | None) -> None:
        """Play an RTTTL melody (or stop, melody=None) on speaker instance `inst`.

        Needs firmware that declares the atech speaker module — the current
        car-only firmware does not, so this is silent until a reflash adds it.
        """
        car = self.car
        if car is None or self._is_dead(car.last_send_error):
            if not self.connect():
                raise RuntimeError(self.error or "not connected")
            car = self.car
            assert car is not None
        if melody is None:
            car.send(f"{inst}_stop")
        else:
            car.send(f"{inst}_set_volume", 0.5)
            car.send(f"{inst}_play_rtttl", melody)

    # -- path record / replay --
    def start_record(self) -> None:
        self.stop_replay()
        self.recording = True
        self.segments = []
        self._seg_name = None

    def _record_step(self, name: str, speed: int) -> None:
        """Close the previous segment (with its held duration) and open a new one."""
        now = time.time()
        if self._seg_name is not None:
            self.segments.append((self._seg_name, self._seg_speed, now - self._seg_t))
        self._seg_name, self._seg_speed, self._seg_t = name, speed, now

    def stop_record(self) -> None:
        if self.recording and self._seg_name is not None:
            self.segments.append(
                (self._seg_name, self._seg_speed, time.time() - self._seg_t)
            )
        self.recording = False
        self._seg_name = None

    def start_replay(self, reverse: bool) -> None:
        if not self.segments:
            return
        self.stop_replay()
        self.recording = False
        self._replay_abort.clear()
        self.replaying = "reverse" if reverse else "forward"
        self._replay_thread = threading.Thread(
            target=self._replay_loop, args=(reverse,), daemon=True
        )
        self._replay_thread.start()

    def _replay_loop(self, reverse: bool) -> None:
        segs = list(reversed(self.segments)) if reverse else list(self.segments)
        try:
            for name, speed, dur in segs:
                if self._replay_abort.is_set():
                    break
                cmd = INVERT.get(name, name) if reverse else name
                try:
                    self.command(cmd, speed)
                except Exception:  # noqa: BLE001
                    pass
                end = time.time() + dur
                while time.time() < end and not self._replay_abort.is_set():
                    time.sleep(0.05)
        finally:
            try:
                self.command("stop", 0)
            except Exception:  # noqa: BLE001
                pass
            self.replaying = None

    def stop_replay(self) -> None:
        if self._replay_thread is not None and self._replay_thread.is_alive():
            self._replay_abort.set()
            self._replay_thread.join(timeout=2.0)
        self.replaying = None
        self._replay_abort.clear()

    def status(self) -> dict[str, Any]:
        car = self.car
        send_err = car.last_send_error if car else None
        alive = car is not None and not self._is_dead(send_err)
        return {
            "connected": alive,
            "port": (f"wifi {CAR_HOST}") if CAR_HOST else (self.port or "(auto)"),
            "error": self.error if not alive else None,
            "car_action": car.value("car_action") if car else None,
            "reconnects": self.reconnects,
            "recording": self.recording,
            "segments": len(self.segments),
            "replaying": self.replaying,
        }


link = CarLink(CAR_PORT)

app = FastAPI(title="atech drive dashboard")


@app.on_event("startup")
def _startup() -> None:
    link.connect()  # best-effort; UI shows the error and offers reconnect


@app.on_event("shutdown")
def _shutdown() -> None:
    link.close()


@app.post("/record/{action}")
def record(action: str) -> JSONResponse:
    if action == "start":
        link.start_record()
    elif action == "stop":
        link.stop_record()
    else:
        return JSONResponse(
            {"ok": False, "error": f"unknown record {action!r}"}, status_code=400
        )
    return JSONResponse({"ok": True, **link.status()})


@app.post("/replay/{mode}")
def replay(mode: str) -> JSONResponse:
    if mode == "forward":
        link.start_replay(reverse=False)
    elif mode == "reverse":
        link.start_replay(reverse=True)
    elif mode == "stop":
        link.stop_replay()
    else:
        return JSONResponse(
            {"ok": False, "error": f"unknown replay {mode!r}"}, status_code=400
        )
    return JSONResponse({"ok": True, **link.status()})


@app.post("/cmd/{name}")
def cmd(name: str, speed: int = DEFAULT_SPEED) -> JSONResponse:
    link.stop_replay()  # a manual command overrides/aborts an active replay
    try:
        link.command(name, speed)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, **link.status()})


@app.post("/sound/{kind}")
def sound(kind: str, inst: str = SPEAKER) -> JSONResponse:
    table = {**JINGLES, "honk": HONK, "test": TEST_TONE, "stop": None}
    if kind not in table:
        return JSONResponse(
            {"ok": False, "error": f"unknown sound {kind!r}"}, status_code=400
        )
    melody = table[kind]
    try:
        link.sound(inst, melody)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    return JSONResponse({"ok": True, "inst": inst, "kind": kind})


@app.post("/reconnect")
def reconnect() -> JSONResponse:
    ok = link.connect()
    return JSONResponse({"ok": ok, **link.status()})


@app.get("/status")
def status() -> JSONResponse:
    return JSONResponse(link.status())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (
        PAGE.replace("__SPEED__", str(DEFAULT_SPEED))
        .replace("__MAX__", str(MAX_SPEED))
        .replace("__SPEAKER__", SPEAKER)
    )


PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>atech car</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; font-family:system-ui,sans-serif; background:#0e1116; color:#e6edf3;
         display:flex; flex-direction:column; align-items:center; gap:18px; padding:24px; }
  h1 { font-size:18px; font-weight:600; margin:0; letter-spacing:.3px; }
  .pad { display:grid; grid-template-columns:repeat(3,92px); grid-template-rows:repeat(3,92px);
         gap:12px; }
  button.dir { font-size:30px; border:0; border-radius:14px; background:#1b2230; color:#e6edf3;
               cursor:pointer; transition:.06s; user-select:none; }
  button.dir:hover { background:#273141; }
  button.dir:active, button.dir.on { background:#2f81f7; color:#fff; }
  #stop { background:#7d1f25; }
  #stop:hover { background:#9b2530; } #stop:active { background:#da3633; }
  .fwd{grid-area:1/2;} .left{grid-area:2/1;} .stopc{grid-area:2/2;} .right{grid-area:2/3;} .back{grid-area:3/2;}
  .row { display:flex; align-items:center; gap:12px; }
  input[type=range]{ width:260px; }
  #spd { font-variant-numeric:tabular-nums; width:42px; text-align:right; }
  .stat { font-size:14px; color:#9aa7b4; }
  .stat b { color:#e6edf3; font-variant-numeric:tabular-nums; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#da3633; margin-right:6px; }
  .dot.ok { background:#3fb950; }
  .hint { font-size:12px; color:#6e7b8a; }
  button.mini { background:#1b2230; color:#e6edf3; border:0; border-radius:8px; padding:6px 10px; cursor:pointer; }
</style></head>
<body>
  <h1>atech car</h1>
  <div class="pad">
    <button class="dir fwd"   data-cmd="forward">▲</button>
    <button class="dir left"  data-cmd="left">⟲</button>
    <button class="dir stopc" id="stop" data-cmd="stop">■</button>
    <button class="dir right" data-cmd="right">⟳</button>
    <button class="dir back"  data-cmd="back">▼</button>
  </div>
  <div class="row">
    <span class="stat">speed</span>
    <input type="range" id="speed" min="80" max="__MAX__" value="__SPEED__">
    <span id="spd">__SPEED__</span>
  </div>
  <div class="row">
    <button class="mini" onclick="sound('erika')">🍦 erika</button>
    <button class="mini" onclick="sound('cuckoos')">🎬 cuckoos</button>
    <button class="mini" onclick="sound('honk')">📣 honk</button>
    <button class="mini" onclick="sound('stop')">🔇 stop</button>
    <span class="stat">speaker</span>
    <input id="spk" value="__SPEAKER__" style="width:78px">
    <button class="mini" onclick="sound('test')">test</button>
  </div>
  <div class="row">
    <button class="mini" id="recbtn" onclick="rec()">⏺ record</button>
    <button class="mini" onclick="replay('forward')">▶ replay</button>
    <button class="mini" onclick="replay('reverse')">◀ reverse</button>
    <button class="mini" onclick="replay('stop')">⏹ stop replay</button>
    <span class="stat" id="path">—</span>
  </div>
  <div class="stat"><span id="dot" class="dot"></span><span id="conn">…</span> &nbsp; state: <b id="action">—</b></div>
  <div class="hint">keys: W/↑ fwd · S/↓ back · A/← rotate L · D/→ rotate R · Space stop. Commands latch until changed.</div>
  <button class="mini" onclick="reconnect()">reconnect</button>

<script>
const speed = document.getElementById('speed');
const spd = document.getElementById('spd');
speed.oninput = () => spd.textContent = speed.value;

let lastCmd = null;
async function send(cmd){
  lastCmd = cmd;
  highlight(cmd);
  try { await fetch(`/cmd/${cmd}?speed=${speed.value}`, {method:'POST'}); }
  catch(e){}
}
function highlight(cmd){
  document.querySelectorAll('button.dir').forEach(b=>b.classList.toggle('on', b.dataset.cmd===cmd));
}
async function reconnect(){ await fetch('/reconnect',{method:'POST'}); }
async function sound(kind){
  const inst = encodeURIComponent(document.getElementById('spk').value.trim() || 'speaker');
  try { await fetch(`/sound/${kind}?inst=${inst}`, {method:'POST'}); } catch(e){}
}
let recording = false;
async function rec(){
  recording = !recording;
  try { await fetch(`/record/${recording?'start':'stop'}`, {method:'POST'}); } catch(e){}
}
async function replay(mode){
  if(recording){ recording=false; try{ await fetch('/record/stop',{method:'POST'}); }catch(e){} }
  try { await fetch(`/replay/${mode}`, {method:'POST'}); } catch(e){}
}

document.querySelectorAll('button.dir').forEach(b=>{
  b.addEventListener('click', ()=>send(b.dataset.cmd));
});

const KEYMAP = {KeyW:'forward',ArrowUp:'forward',KeyS:'back',ArrowDown:'back',
                KeyA:'left',ArrowLeft:'left',KeyD:'right',ArrowRight:'right',
                Space:'stop'};
document.addEventListener('keydown', e=>{
  const cmd = KEYMAP[e.code];
  if(!cmd || e.repeat) return;
  e.preventDefault();
  send(cmd);
});

async function poll(){
  try{
    const s = await (await fetch('/status')).json();
    document.getElementById('conn').textContent = s.connected ? ('connected '+s.port) : ('disconnected'+(s.error?' — '+s.error:''));
    document.getElementById('dot').classList.toggle('ok', s.connected);
    document.getElementById('action').textContent = s.car_action ?? '—';
    recording = !!s.recording;
    document.getElementById('recbtn').classList.toggle('on', recording);
    document.getElementById('recbtn').textContent = recording ? '⏹ stop rec' : '⏺ record';
    const p = s.replaying ? ('replaying '+s.replaying) : (recording ? 'recording…' : '');
    document.getElementById('path').textContent = `${p} · ${s.segments??0} segs`;
  }catch(e){}
}
setInterval(poll, 700); poll();
</script>
</body></html>"""


def main() -> int:
    print(
        f"drive dashboard on http://localhost:{HTTP_PORT}  (port={CAR_PORT or 'auto'})"
    )
    uvicorn.run(app, host="0.0.0.0", port=HTTP_PORT, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
