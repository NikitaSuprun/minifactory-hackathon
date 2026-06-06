"""Web backend + dashboard for the atech RC cars.

Run:
    uv run python car_dashboard.py        # http://localhost:8042

Backend (FastAPI) over the `carlink` stack (atech.Board + Car + PolicyRunner):
- Connect/disconnect every car in $ATECH_CARS. Each target is a serial path
  (USB now) or "gw:<project-id>" (the WiFi gateway, for later) — same code path.
- Manual driving: forward / backward / left / right / stop, a speed for
  motor_speed, plus heading control (turn_to_heading / tare) and enable/disable.
- Live telemetry: orientation (pitch/roll/heading), depth (min_distance mm),
  obstacle flag, status.
- Driving policy: start/stop StraightUntilObstacle (add your own in POLICIES).
- ABORT: one button stops every policy and brakes every car.

Only one program can own a USB serial port — disconnect the atech web bridge
(browser Web Serial) before Connect, or the open fails with "resource busy".
"""

from __future__ import annotations

import os
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from carlink import (
    Car,
    PolicyRunner,
    StraightUntilObstacle,
    connect_gateway,
    connect_serial,
)

load_dotenv()

# --- Configuration (from .env) ----------------------------------------------
ATECH_CARS = os.environ.get("ATECH_CARS", "")  # "car_a=/dev/...,car_b=gw:PROJECT_ID"
# Sim mode: run the whole dashboard on simulated cars (no hardware). Enable with
# `--sim` on the command line or ATECH_SIM=1. In sim, each car target is replaced
# by a SimCar, so $ATECH_CARS just defines the car names.
SIM = "--sim" in sys.argv or os.environ.get("ATECH_SIM") == "1"
POLICY_HZ = float(os.environ.get("POLICY_HZ", "20"))
# Keepalive interval: the firmware brakes if it hears nothing for 500ms, so ping
# every connected car well under that. 0 disables (e.g. if firmware has no deadman).
HEARTBEAT_MS = int(os.environ.get("HEARTBEAT_MS", "200"))
DASHBOARD_PORT = int(os.environ.get("CAR_DASHBOARD_PORT", "8042"))
DASHBOARD_USER = os.environ.get("DASHBOARD_USER", "admin")
DASHBOARD_PASS = os.environ.get("DASHBOARD_PASS", "")

# Selectable policies: name -> factory(**params). Add your own driving policies here.
POLICIES: dict[str, Any] = {
    "straight_until_obstacle": lambda **kw: StraightUntilObstacle(
        speed=int(kw.get("speed", 150)),
        stop_distance_mm=float(kw.get("stop_distance_mm", 300)),
    ),
}


def parse_cars(spec: str) -> dict[str, str]:
    """"car_a=/dev/ttyA,car_b=gw:PID" -> {"car_a": "/dev/ttyA", "car_b": "gw:PID"}."""
    cars: dict[str, str] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if chunk and "=" in chunk:
            name, target = chunk.split("=", 1)
            cars[name.strip()] = target.strip()
    return cars


def _build_slot(name: str, target: str) -> "CarSlot":
    """Open one car (sim / serial / gateway) and wrap it in a CarSlot."""
    sim = None
    if SIM:
        from carlink.sim import SimCar

        sim = SimCar(name).start()
        board = sim.board
    elif target.startswith("gw:"):
        board = connect_gateway(target[3:])
    else:
        board = connect_serial(target or None)
    car = Car(board, name=name)
    return CarSlot(car=car, runner=PolicyRunner(car, hz=POLICY_HZ), target=target, sim=sim)


# --- Authentication (HTTP Basic on every route) -----------------------------
_basic = HTTPBasic()


def require_auth(credentials: HTTPBasicCredentials = Depends(_basic)) -> str:
    ok = secrets.compare_digest(credentials.username, DASHBOARD_USER) and secrets.compare_digest(
        credentials.password, DASHBOARD_PASS
    )
    if not ok:
        raise HTTPException(401, "Unauthorized", {"WWW-Authenticate": "Basic"})
    return credentials.username


AUTH_ENABLED = bool(DASHBOARD_PASS)
_auth_deps = [Depends(require_auth)] if AUTH_ENABLED else []


# --- State ------------------------------------------------------------------
@dataclass
class CarSlot:
    car: Car
    runner: PolicyRunner
    target: str = ""
    sim: Any = None  # SimCar handle when running in --sim mode

    def close(self) -> None:
        try:
            self.runner.stop()  # brakes
        finally:
            self.car.close()  # stops the writer thread + closes the board
            if self.sim:
                self.sim.close()  # stops the sim feeder thread


@dataclass
class AppState:
    cars: dict[str, CarSlot] = field(default_factory=dict)
    error: str | None = None

    @property
    def connected(self) -> bool:
        return bool(self.cars)


state = AppState()
app = FastAPI(title="atech Car Dashboard", dependencies=_auth_deps)
if not AUTH_ENABLED:
    print("WARNING: DASHBOARD_PASS not set -> car dashboard is UNAUTHENTICATED.")


def _heartbeat_loop() -> None:
    """Ping every connected car so the firmware deadman never trips mid-drive.
    If the dashboard dies, pings stop and the firmware brakes — fail-safe."""
    period = HEARTBEAT_MS / 1000.0
    while True:
        for slot in list(state.cars.values()):
            try:
                slot.car.ping()
            except Exception:  # noqa: BLE001 - a dead link must not kill the heartbeat
                pass
        time.sleep(period)


if HEARTBEAT_MS > 0:
    threading.Thread(target=_heartbeat_loop, name="car-heartbeat", daemon=True).start()


def _slot(name: str) -> CarSlot:
    slot = state.cars.get(name)
    if slot is None:
        raise HTTPException(404, f"unknown car {name!r} (connect first)")
    return slot


def _car(name: str) -> Car:
    return _slot(name).car


def _manual(name: str) -> Car:
    """Get a car for a manual command, refusing if a policy owns it."""
    slot = _slot(name)
    if slot.runner.running:
        raise HTTPException(409, "a policy is running on this car; stop it first")
    return slot.car


# --- Connection -------------------------------------------------------------
@app.post("/connect")
def connect() -> dict[str, Any]:
    """Open every car in $ATECH_CARS. Partial: one car failing doesn't block the rest."""
    cars = parse_cars(ATECH_CARS)
    if not cars:
        raise HTTPException(400, "ATECH_CARS is empty in .env (name=target,...).")
    opened, failed = [], {}
    for name, target in cars.items():
        if name in state.cars:
            continue  # already connected
        try:
            state.cars[name] = _build_slot(name, target)
            opened.append(name)
        except Exception as e:  # noqa: BLE001
            failed[name] = str(e)
    state.error = (
        "; ".join(f"{n}: {m}" for n, m in failed.items()) if failed else None
    )
    return {"status": "connected", "cars": list(state.cars), "opened": opened, "failed": failed}


@app.post("/car/{name}/reconnect")
def reconnect(name: str) -> dict[str, str]:
    """Reopen a single car (e.g. after a serial drop) without touching the others."""
    target = state.cars[name].target if name in state.cars else dict(parse_cars(ATECH_CARS)).get(name)
    if target is None:
        raise HTTPException(404, f"{name} is not in $ATECH_CARS")
    if name in state.cars:
        state.cars.pop(name).close()
    try:
        state.cars[name] = _build_slot(name, target)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"reconnect {name} failed: {e}") from e
    return {"status": f"{name} reconnected"}


def _disconnect_all() -> None:
    for slot in state.cars.values():
        try:
            slot.close()
        except Exception as e:  # noqa: BLE001
            state.error = f"disconnect: {e}"
    state.cars.clear()


@app.post("/disconnect")
def disconnect() -> dict[str, str]:
    _disconnect_all()
    return {"status": "disconnected"}


# --- Monitoring -------------------------------------------------------------
@app.get("/status")
def status() -> JSONResponse:
    cars = {
        name: {
            **slot.car.snapshot(),
            "connected": True,
            "policy_running": slot.runner.running,
            "policy_name": slot.runner.policy_name,
            "policy_error": slot.runner.last_error,
        }
        for name, slot in state.cars.items()
    }
    return JSONResponse(
        {
            "connected": state.connected,
            "sim": SIM,
            "error": state.error,
            "policies": list(POLICIES),
            "cars": cars,
        }
    )


# --- Manual driving ---------------------------------------------------------
@app.post("/car/{name}/forward")
def forward(name: str, speed: int = 150) -> dict[str, str]:
    _manual(name).forward(speed)
    return {"status": f"{name} forward {speed}"}


@app.post("/car/{name}/backward")
def backward(name: str, speed: int = 150) -> dict[str, str]:
    _manual(name).backward(speed)
    return {"status": f"{name} backward {speed}"}


@app.post("/car/{name}/left")
def left(name: str, speed: int = 150) -> dict[str, str]:
    _manual(name).turn_left(speed)
    return {"status": f"{name} left {speed}"}


@app.post("/car/{name}/right")
def right(name: str, speed: int = 150) -> dict[str, str]:
    _manual(name).turn_right(speed)
    return {"status": f"{name} right {speed}"}


@app.post("/car/{name}/drive")
def drive(name: str, speed: int) -> dict[str, str]:
    """Generic throttle via motor_speed (-255..255; negative = reverse)."""
    _manual(name).drive(speed)
    return {"status": f"{name} drive {speed}"}


@app.post("/car/{name}/stop")
def stop(name: str) -> dict[str, str]:
    _car(name).stop()  # always allowed, even mid-policy
    return {"status": f"{name} stop"}


@app.post("/car/{name}/heading")
def heading(name: str, deg: float) -> dict[str, str]:
    _manual(name).turn_to_heading(deg)
    return {"status": f"{name} turn_to_heading {deg}"}


@app.post("/car/{name}/tare")
def tare(name: str) -> dict[str, str]:
    _car(name).tare_heading()
    return {"status": f"{name} heading tared"}


@app.post("/car/{name}/action")
def raw_action(name: str, action: str, value: str | None = None) -> dict[str, str]:
    """Send any action verbatim (the generic bridge)."""
    _car(name).send(action, value)
    return {"status": f"{name} {action}={value}"}


# --- Policy -----------------------------------------------------------------
@app.post("/car/{name}/policy/start")
def policy_start(
    name: str, policy: str, speed: int = 150, stop_distance_mm: float = 300
) -> dict[str, str]:
    slot = _slot(name)
    factory = POLICIES.get(policy)
    if factory is None:
        raise HTTPException(404, f"unknown policy {policy!r}; have {list(POLICIES)}")
    slot.runner.start(factory(speed=speed, stop_distance_mm=stop_distance_mm))
    return {"status": f"{name} policy {policy} started"}


@app.post("/car/{name}/policy/stop")
def policy_stop(name: str) -> dict[str, str]:
    _slot(name).runner.stop()  # brakes
    return {"status": f"{name} policy stopped"}


# --- Global abort (e-stop) --------------------------------------------------
@app.post("/abort")
def abort() -> dict[str, str]:
    """Stop every policy and brake every car. Never raises."""
    for slot in state.cars.values():
        try:
            slot.runner.abort()  # stops loop + brakes
        except Exception:  # noqa: BLE001
            try:
                slot.car.stop()
            except Exception:  # noqa: BLE001
                pass
    return {"status": "ABORTED — all cars braked, policies stopped"}


# --- UI ---------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>atech Car Dashboard</title>
<style>
  body{font-family:system-ui,sans-serif;margin:24px;background:#0f1115;color:#e6e6e6}
  h1{font-size:20px} h3{margin:0 0 6px}
  button{font-size:14px;padding:8px 12px;margin:3px;border:0;border-radius:6px;
    background:#2d6cdf;color:#fff;cursor:pointer}
  button.stop{background:#c0392b} button.ghost{background:#2a2f3a}
  button.abort{background:#e74c3c;font-size:18px;padding:12px 22px;font-weight:700}
  .card{background:#171a21;padding:16px;border-radius:10px;margin:12px 0;max-width:560px}
  .pad{display:grid;grid-template-columns:repeat(3,64px);gap:6px;justify-content:start;margin:8px 0}
  .pad .sp{visibility:hidden}
  .tele{display:grid;grid-template-columns:auto auto;gap:2px 16px;font-size:13px;margin:8px 0}
  .tele b{color:#7fd1ff} .warn{color:#ffb86b}
  input{width:64px;background:#0b0d11;color:#e6e6e6;border:1px solid #333;border-radius:4px;padding:5px}
  .muted{color:#8a93a3;font-size:12px}
</style></head><body>
<h1>atech Car Dashboard <span id="simbadge"></span></h1>
<div>
  <button onclick="post('/connect')">Connect</button>
  <button class="ghost" onclick="post('/disconnect')">Disconnect</button>
  <button class="abort" onclick="post('/abort')">■ ABORT ALL</button>
  <span class="muted">Speed <input id="spd" type="number" value="150" min="0" max="255"></span>
</div>
<div id="cars"></div>
<p class="muted">Polls /status every second. Manual driving is blocked while a policy runs (Stop/Abort always work).</p>
<script>
const spd = () => document.getElementById('spd').value;
async function post(p){
  try{const r=await fetch(p,{method:'POST'});
    if(!r.ok){alert((await r.json()).detail||r.statusText);}}
  catch(e){alert(e);} refresh();
}
function fmtOri(o){ return o ? `pitch ${o[0]}° roll ${o[1]}° hdg ${o[2]}°` : '—'; }
function stale(age){ return age==null || age>1500; }   // gray out telemetry older than 1.5s
function ageTag(age){ return age==null ? '' : ` <span class="muted">(${age} ms)</span>`; }
function modChips(m){
  if(!m) return '—';
  return Object.keys(m).map(k=>{
    const ok = m[k]==='ok', miss = m[k]==='missing';
    const col = ok?'#2e7d32':(miss?'#8a3030':'#333');
    return `<span style="background:${col};border-radius:4px;padding:1px 6px;margin-right:4px;font-size:11px">${k}</span>`;
  }).join('');
}
function card(name, c){
  const n = name;
  const a = c.ages_ms || {};
  const linkOk = c.last_rx_age_ms!=null && c.last_rx_age_ms<1500;
  return `<div class="card">
   <h3>${n} ${linkOk?'🟢':'🔴'} ${c.policy_running?('· policy: '+c.policy_name):''}
     <button class="ghost" style="font-size:12px;padding:3px 8px"
       onclick="post('/car/${n}/reconnect')">reconnect</button></h3>
   <div class="pad">
     <span class="sp"></span>
     <button onclick="post('/car/${n}/forward?speed='+spd())">▲</button>
     <span class="sp"></span>
     <button onclick="post('/car/${n}/left?speed='+spd())">◄</button>
     <button class="stop" onclick="post('/car/${n}/stop')">■</button>
     <button onclick="post('/car/${n}/right?speed='+spd())">►</button>
     <span class="sp"></span>
     <button onclick="post('/car/${n}/backward?speed='+spd())">▼</button>
     <span class="sp"></span>
   </div>
   <div class="tele">
     <span>orientation</span><span style="opacity:${stale(a.orientation)?0.45:1}"><b>${fmtOri(c.orientation)}</b>${ageTag(a.orientation)}</span>
     <span>depth (min)</span><span style="opacity:${stale(a.min_distance)?0.45:1}"><b>${c.distance_mm==null?'— (not connected)':c.distance_mm+' mm'}</b>${ageTag(a.min_distance)}</span>
     <span>obstacle</span><span><b class="${c.obstacle?'warn':''}">${c.obstacle==null?'—':(c.obstacle?'DETECTED':'clear')}</b></span>
     <span>status</span><span><b>${c.status||'—'}</b> ${(c.link_stale||!linkOk)?'<span class="warn">LINK STALE</span>':''}</span>
     <span>modules</span><span>${modChips(c.modules)}</span>
   </div>
   <div>
     heading <input id="hdg_${n}" type="number" value="0">
     <button onclick="post('/car/${n}/heading?deg='+document.getElementById('hdg_${n}').value)">turn to</button>
     <button class="ghost" onclick="post('/car/${n}/tare')">tare</button>
   </div>
   <div style="margin-top:8px">
     <button onclick="post('/car/${n}/policy/start?policy=straight_until_obstacle&speed='+spd()+'&stop_distance_mm=300')">Start policy</button>
     <button class="stop" onclick="post('/car/${n}/policy/stop')">Stop policy</button>
     ${c.policy_error?('<span class="muted">err: '+c.policy_error+'</span>'):''}
   </div></div>`;
}
async function refresh(){
  try{const r=await fetch('/status'); const s=await r.json();
    document.getElementById('simbadge').innerHTML =
      s.sim ? '<span class="muted" style="font-size:13px">— SIM MODE (no hardware)</span>' : '';
    const names=Object.keys(s.cars);
    document.getElementById('cars').innerHTML = names.length
      ? names.map(n=>card(n,s.cars[n])).join('') + (s.error?('<p class="warn">'+s.error+'</p>'):'')
      : '<p class="muted">No cars connected. Set $ATECH_CARS in .env, then Connect.</p>';
  }catch(e){}
}
setInterval(refresh,1000); refresh();
</script></body></html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT)
