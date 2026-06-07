"""Author + build + flash the car firmware (drive + speaker + WiFi) via the atech SDK.

Replicates the original car's serial interface (so drive_dashboard.py keeps
working unchanged), adds the I2S speaker, AND adds WiFi so the car can be driven
cable-free over the network. The exact same line-delimited JSON protocol runs
over BOTH USB serial and a TCP socket (port 3333):

  actions in : motor_speed <-255..255>  (sign = fwd/back, all 4 wheels)
               turn_left  <0..255>      (pivot left)
               turn_right <0..255>      (pivot right)
               stop                     (active brake all)
               spk_play_rtttl <string>  (play RTTTL melody, background)
               spk_set_volume <0..1>
               spk_stop
  events out : state  car_action = stopped|forward|backward|turn_left|turn_right
               state  wifi_ip    = <ip>   (emitted once WiFi connects)
               sensor car_speed  (source dc_motor)

Motors: fl=port1, rl=port6, fr=port9, rr=port14 (per firmware/PROMPT.md; right
side mirrored so +motor_speed drives all wheels forward). Speaker: SPEAKER_PORTS.
WiFi: AP mode — the car is its OWN hotspot (SSID CAR_AP_SSID, default "atech-car";
IP 192.168.4.1), so there's no router/DHCP/client-isolation to fight. Join that
network from the Mac, then run the dashboard with ATECH_CAR_HOST=192.168.4.1.

First-time setup (this uv-managed venv lacks pip, which PlatformIO needs to
install its esptool deps; esptool is also used for backup/restore):
    uv pip install pip esptool

Usage (port auto-discovers if omitted):
    uv run python firmware/build_car_speaker.py --generate        # write project + print main.cpp
    uv run python firmware/build_car_speaker.py --build           # compile (no flash)
    uv run python firmware/build_car_speaker.py --upload [PORT]   # build + flash

Restore the original anytime:
    uv run python -m esptool --port <PORT> --no-stub write-flash 0 firmware/backup/car_original.bin
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from atech import Project
from atech.build import run_build
from atech.upload import run_upload
from dotenv import load_dotenv

_REPO = Path(__file__).resolve().parent.parent
load_dotenv(_REPO / ".env.local")
load_dotenv(_REPO / ".env")

BOARD = "14port"
NAME = "car_speaker"
MOTORS = {"fl": 1, "rl": 6, "fr": 9, "rr": 14}  # instance -> port (PROMPT.md)
# Adjacent pair the speaker is plugged into. Valid FREE pairs on 14port (motors
# take 1/6/9/14): (2,3), (3,4), (4,5), (10,11). CONFIRM against the board!
SPEAKER_PORTS = (3, 4)
SPEAKER_INSTANCE = "spk"
TCP_PORT = 3333
# The car runs as its OWN WiFi access point (AP mode). Join this network from the
# Mac, then point the dashboard at 192.168.4.1. Password must be >=8 chars (WPA2).
AP_SSID = os.environ.get("CAR_AP_SSID", "atech-car")
AP_PASS = os.environ.get("CAR_AP_PASS", "minifactory")
BUILD_DIR = _REPO / "firmware" / "build" / NAME

# All behavior lives in loop(). The local SDK does NOT auto-handle module actions,
# so we parse the JSON ({"action":"..","value":".."}) ourselves and drive the
# in-scope module instances (fl/rl/fr/rr DCMotor, spk Speaker). The same protocol
# runs over Serial AND a TCP client (WiFi), so the car works tethered or wireless.
# No deadman: commands latch until changed. __AP_SSID__/__AP_PASS__ are substituted
# at build time (AP mode — the car is its own hotspot).
LOOP_CPP = r"""
static String rxS, rxW;
static unsigned long lastTele = 0;
static const char* curAction = "stopped";
static int curSpeed = 0;
static WiFiServer server(3333);
static WiFiClient client;
static bool wifiBegun = false;

// AP mode: the car is its OWN WiFi hotspot — no router, so no client isolation /
// DHCP / reachability problems (STA on a shared AP was unreliable). softAP() is
// synchronous, so the AP + TCP server are up immediately at 192.168.4.1.
if (!wifiBegun) {
    wifiBegun = true;
    WiFi.mode(WIFI_AP);
    WiFi.softAP("__AP_SSID__", "__AP_PASS__");
    WiFi.setSleep(false);   // keep the link low-latency / responsive
    server.begin();
    server.setNoDelay(true);
    MDNS.begin("car");
    if (Serial && Serial.availableForWrite() > 90) {
        Serial.print("{\"type\":\"event\",\"payload\":{\"event_type\":\"state\",\"key\":\"wifi_ip\",\"value\":\"");
        Serial.print(WiFi.softAPIP().toString());
        Serial.println("\"}}");
    }
}

// left side = fl,rl ; right side physically mirrored = fr,rr (negate)
auto setDrive = [](int left, int right) {
    fl.setSpeed(left);  rl.setSpeed(left);
    fr.setSpeed(-right); rr.setSpeed(-right);
};
auto brakeAll = []() { fl.brake(); rl.brake(); fr.brake(); rr.brake(); };
// emit one event line to BOTH transports. The serial write MUST never block: when
// the USB cable is plugged but no host drains the CDC, the TX buffer fills and a
// plain Serial.println() hangs the whole loop (so WiFi never even starts its
// server). availableForWrite() gates the write — if there's no room, we drop it.
auto sout = [&](const String& s) {
    if (Serial && Serial.availableForWrite() > (int)s.length() + 2) Serial.println(s);
};
auto emit = [&](const String& line) {
    sout(line);
    if (client && client.connected()) client.println(line);
};
auto handle = [&](const String& rx) {
    int ai = rx.indexOf("\"action\"");
    if (ai < 0) return;
    int q1 = rx.indexOf('"', rx.indexOf(':', ai) + 1);
    int q2 = rx.indexOf('"', q1 + 1);
    String action = rx.substring(q1 + 1, q2);
    String value = "";
    int vi = rx.indexOf("\"value\"");
    if (vi >= 0) {
        int p = rx.indexOf(':', vi) + 1;
        while (p < (int)rx.length() && rx[p] == ' ') p++;
        if (p < (int)rx.length() && rx[p] == '"') {
            int e = rx.indexOf('"', p + 1);
            value = rx.substring(p + 1, e);
        } else {
            int e = p;
            while (e < (int)rx.length() && rx[e] != ',' && rx[e] != '}') e++;
            value = rx.substring(p, e);
        }
    }
    int iv = value.toInt();
    if (iv > 255) iv = 255; if (iv < -255) iv = -255;
    if (action == "motor_speed") {
        setDrive(iv, iv);
        curAction = iv > 0 ? "forward" : (iv < 0 ? "backward" : "stopped");
        curSpeed = iv < 0 ? -iv : iv;
    } else if (action == "turn_left") {
        setDrive(-iv, iv); curAction = "turn_left"; curSpeed = iv;
    } else if (action == "turn_right") {
        setDrive(iv, -iv); curAction = "turn_right"; curSpeed = iv;
    } else if (action == "stop") {
        brakeAll(); curAction = "stopped"; curSpeed = 0;
    } else if (action == "spk_play_rtttl") {
        spk.playRTTTL(value.c_str());
    } else if (action == "spk_set_volume") {
        spk.setVolume(value.toFloat());
    } else if (action == "spk_stop") {
        spk.stop();
    }
};

// Accept new clients unconditionally: server.available() returns a fresh client
// only when one is actually pending (null otherwise), so calling it every loop
// picks up (re)connections immediately and we drop any old/stale client. This
// avoids the stale-connection deadlock of guarding the call on client state.
if (wifiBegun) {
    WiFiClient nc = server.available();
    if (nc) {
        if (client) client.stop();
        client = nc;
        client.setNoDelay(true);
        client.println("{\"type\":\"event\",\"payload\":{\"event_type\":\"log\",\"key\":\"hello\",\"value\":\"connected\"}}");
        sout("{\"type\":\"event\",\"payload\":{\"event_type\":\"log\",\"key\":\"wifi\",\"value\":\"client_connected\"}}");
    }
}

// drain action lines from Serial ...
while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') { handle(rxS); rxS = ""; }
    else if (c != '\r' && rxS.length() < 240) rxS += c;
}
// ... and from the TCP client
while (client && client.available()) {
    char c = (char)client.read();
    if (c == '\n') { handle(rxW); rxW = ""; }
    else if (c != '\r' && rxW.length() < 240) rxW += c;
}

// periodic telemetry to both transports (~2 Hz)
if (millis() - lastTele > 500) {
    lastTele = millis();
    emit(String("{\"type\":\"event\",\"payload\":{\"event_type\":\"state\",\"key\":\"car_action\",\"value\":\"")
         + curAction + "\"}}");
    emit(String("{\"type\":\"event\",\"payload\":{\"event_type\":\"sensor\",\"key\":\"car_speed\",\"value\":")
         + curSpeed + ",\"source\":\"dc_motor\"}}");
}
"""


def make_project() -> Project:
    print(
        f"AP mode: car hotspot SSID={AP_SSID!r} ({len(AP_PASS)}-char pass), IP 192.168.4.1"
    )
    p = Project(board=BOARD, name=NAME)
    for inst, port in MOTORS.items():
        p.add("dc_motor", port=port, instance=inst)
    p.add("speaker", ports=SPEAKER_PORTS, instance=SPEAKER_INSTANCE)
    loop = LOOP_CPP.replace("__AP_SSID__", AP_SSID).replace("__AP_PASS__", AP_PASS)
    p.set_loop(loop)
    issues = p.validate()
    if issues:
        print("Placement issues:")
        for i in issues:
            print("  -", i)
        raise SystemExit(1)
    return p


def generate_and_patch(p: Project) -> Path:
    """Generate the PlatformIO project, then patch main.cpp to add the WiFi/mDNS
    includes (the SDK has no file-scope include hook). Returns the project dir."""
    out = p.generate(BUILD_DIR)
    main_cpp = out / "src" / "main.cpp"
    text = main_cpp.read_text()
    if "#include <WiFi.h>" not in text:
        text = text.replace(
            "#include <Arduino.h>",
            "#include <Arduino.h>\n#include <WiFi.h>\n#include <ESPmDNS.h>",
            1,
        )
        main_cpp.write_text(text)
    return out


def main() -> int:
    args = sys.argv[1:]
    p = make_project()

    if "--generate" in args or not args:
        out = generate_and_patch(p)
        print(f"generated -> {out}\n")
        print((out / "src" / "main.cpp").read_text())
        return 0
    if "--build" in args:
        generate_and_patch(p)
        print("building (first build downloads the esp32 platform — be patient) ...")
        res = run_build(BUILD_DIR)
        print(res)
        return 0 if getattr(res, "success", True) else 1
    if "--upload" in args:
        rest = [a for a in args if not a.startswith("--")]
        port = rest[0] if rest else None
        generate_and_patch(p)
        print(f"build + flash to {port or '(auto)'} ...")
        res = run_upload(BUILD_DIR, port=port)
        print(res)
        return 0 if getattr(res, "success", True) else 1

    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
