"""Author + build + flash the car-with-speaker firmware via the atech SDK.

Replicates the original car's serial interface (so drive_dashboard.py keeps
working unchanged) AND adds the I2S speaker:

  actions in : motor_speed <-255..255>  (sign = fwd/back, all 4 wheels)
               turn_left  <0..255>      (pivot left)
               turn_right <0..255>      (pivot right)
               stop                     (active brake all)
               spk_play_rtttl <string>  (play RTTTL melody, background)
               spk_set_volume <0..1>
               spk_stop
  events out : state  car_action = stopped|forward|backward|turn_left|turn_right
               sensor car_speed  (source dc_motor)

Motors: fl=port1, rl=port6, fr=port9, rr=port14 (per firmware/PROMPT.md; right
side mirrored so +motor_speed drives all wheels forward). Speaker: set
SPEAKER_PORTS to the adjacent pair it's plugged into.

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

import sys
from pathlib import Path

from atech import Project

BOARD = "14port"
NAME = "car_speaker"
MOTORS = {"fl": 1, "rl": 6, "fr": 9, "rr": 14}  # instance -> port (PROMPT.md)
# Adjacent pair the speaker is plugged into. Valid FREE pairs on 14port (motors
# take 1/6/9/14): (2,3), (3,4), (4,5), (10,11). CONFIRM against the board!
SPEAKER_PORTS = (3, 4)
SPEAKER_INSTANCE = "spk"
BUILD_DIR = Path(__file__).resolve().parent / "build" / NAME

# All behavior lives in loop(). The local SDK does NOT auto-handle module actions,
# so we parse the serial JSON ({"action":"..","value":".."}) ourselves and drive
# the in-scope module instances (fl/rl/fr/rr DCMotor, spk Speaker). No deadman:
# commands latch until changed, matching the original firmware + the dashboard.
LOOP_CPP = r"""
static String rx;
static unsigned long lastTele = 0;
static const char* curAction = "stopped";
static int curSpeed = 0;

// left side = fl,rl ; right side physically mirrored = fr,rr (negate)
auto setDrive = [](int left, int right) {
    fl.setSpeed(left);  rl.setSpeed(left);
    fr.setSpeed(-right); rr.setSpeed(-right);
};
auto brakeAll = []() { fl.brake(); rl.brake(); fr.brake(); rr.brake(); };
auto emitState = [](const char* k, const char* v) {
    Serial.print("{\"type\":\"event\",\"payload\":{\"event_type\":\"state\",\"key\":\"");
    Serial.print(k); Serial.print("\",\"value\":\""); Serial.print(v);
    Serial.println("\"}}");
};

// drain whole lines from the host
while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
        int ai = rx.indexOf("\"action\"");
        if (ai >= 0) {
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
        }
        rx = "";
    } else if (c != '\r' && rx.length() < 240) {
        rx += c;
    }
}

// periodic telemetry so the dashboard's state/speed stays live (~2 Hz)
if (millis() - lastTele > 500) {
    lastTele = millis();
    emitState("car_action", curAction);
    Serial.print("{\"type\":\"event\",\"payload\":{\"event_type\":\"sensor\",\"key\":\"car_speed\",\"value\":");
    Serial.print(curSpeed);
    Serial.println(",\"source\":\"dc_motor\"}}");
}
"""


def make_project() -> Project:
    p = Project(board=BOARD, name=NAME)
    for inst, port in MOTORS.items():
        p.add("dc_motor", port=port, instance=inst)
    p.add("speaker", ports=SPEAKER_PORTS, instance=SPEAKER_INSTANCE)
    p.set_loop(LOOP_CPP)
    issues = p.validate()
    if issues:
        print("Placement issues:")
        for i in issues:
            print("  -", i)
        raise SystemExit(1)
    return p


def main() -> int:
    args = sys.argv[1:]
    p = make_project()

    if "--generate" in args or not args:
        out = p.generate(BUILD_DIR)
        main_cpp = (out / "src" / "main.cpp").read_text()
        print(f"generated -> {out}\n")
        print(main_cpp)
        return 0
    if "--build" in args:
        print("building (first build downloads the esp32 platform — be patient) ...")
        res = p.build(BUILD_DIR)
        print(res)
        return 0 if getattr(res, "ok", True) else 1
    if "--upload" in args:
        rest = [a for a in args if not a.startswith("--")]
        port = rest[0] if rest else None
        print(f"build + flash to {port or '(auto)'} ...")
        res = p.upload(port=port, out_dir=BUILD_DIR)
        print(res)
        return 0 if getattr(res, "ok", True) else 1

    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
