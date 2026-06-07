# atech car — drive, sound, WiFi, record/replay

The RC car is an **atech 14-port board (ESP32-S3)** with 4 DC motors, an I2S
speaker, a battery, and WiFi. We flash our **own** firmware
(`firmware/build_car_speaker.py`) and drive it from `drive_dashboard.py` over USB
serial **or** WiFi (same JSON protocol either way).

- Motors on ports **1, 6, 9, 14** (fl, rl, fr, rr); speaker on adjacent pair **3+4**
  (instance `spk`). Board MAC `80:f1:b2:cb:9a:84`.
- The firmware is already flashed. You only reflash to change behavior or WiFi.

## Quick start

**Wired (USB):**
```
uv run python drive_dashboard.py            # auto-discovers the USB port
```
Open http://localhost:8043. (Close the atech browser Web Serial bridge first, or
the port is "resource busy".)

**Wireless (WiFi, no cable) — AP mode:** the car is its OWN WiFi hotspot, so no
router/DHCP/client-isolation is involved (a shared AP turned out unreliable).
1. Power the car on **battery** (USB not needed). It starts its hotspot in ~1–2 s.
2. On the Mac, **join the car's WiFi**: SSID **`atech-car`**, password **`minifactory`**.
   ⚠️ One WiFi radio = no internet while on the car's hotspot. To keep internet too,
   use a **USB-Ethernet** adapter for internet (WiFi stays on the car), or build a
   **STA** variant that joins a phone hotspot (`WiFi.begin` — see git history).
3. Run:
   ```
   ATECH_CAR_HOST=192.168.4.1 uv run python drive_dashboard.py
   ```

Controls: arrows / **W A S D**, **Space** = stop. Speed slider (capped at 224 —
above that the motor surge browns out the board). Buttons for jingles (Erika /
Cuckoos), honk, and **record / replay / reverse**.

## Change the car's hotspot (SSID / password)

The car runs as its own AP (`AP_SSID`/`AP_PASS` in `firmware/build_car_speaker.py`,
default `atech-car` / `minifactory`; password must be ≥8 chars). To change it, set
`CAR_AP_SSID` / `CAR_AP_PASS` (env) or edit the constants, then reflash over USB:
```
CAR_AP_SSID=my-car CAR_AP_PASS=drivecar1 uv run python firmware/build_car_speaker.py --upload
```
The car's IP is always **192.168.4.1**.

**Want the car to join an existing WiFi instead (STA mode)** — e.g. so the Mac keeps
internet on a non-isolating network or a phone hotspot? Switch `WiFi.mode(WIFI_AP)`
/ `WiFi.softAP(...)` back to `WiFi.mode(WIFI_STA)` / `WiFi.begin(ssid, pass)` in
`LOOP_CPP` (see git history for the STA version), put creds in `.env.local`, reflash.

## Reflash / build firmware

```
uv pip install pip esptool                              # first time only (uv venv lacks pip)
uv run python firmware/build_car_speaker.py --generate  # write + print main.cpp (no flash)
uv run python firmware/build_car_speaker.py --build     # compile only
uv run python firmware/build_car_speaker.py --upload    # build + flash (port auto-discovers)
```
The first build downloads the ESP32 toolchain (a few minutes), then it's cached.
Edit `MOTORS` / `SPEAKER_PORTS` / `LOOP_CPP` in that file to change wiring or logic.

**Restore the original (pre-speaker) firmware** anytime:
```
uv run python -m esptool --port <PORT> --no-stub write-flash 0 firmware/backup/car_original.bin
```
(`--no-stub` is required — the stub flasher is unreliable over this board's native
USB on macOS.)

## Firmware interface (the wire protocol)

Line-delimited JSON, 115200 baud over USB **and** TCP port 3333 over WiFi.
- **Actions in:** `motor_speed <-255..255>` (sign = fwd/back), `turn_left <0..255>`,
  `turn_right <0..255>`, `stop`, `spk_play_rtttl <string>`, `spk_set_volume <0..1>`,
  `spk_stop`.
- **Events out:** `car_action` = stopped|forward|backward|turn_left|turn_right;
  `wifi_ip` = `<ip>` (once, on WiFi connect); `car_speed` (constant — ignore).
- Commands **latch** (no deadman): the car holds a command until you send another.

Probe a live board with `scripts/atech_probe.py` (`--list`, `--diagnostics`,
`--listen N`, `--send KEY VALUE`).

## Path record & replay

In the dashboard: **⏺ record** → drive a path → **⏹ stop rec**. Then **▶ replay**
repeats it; **◀ reverse** retraces back to start (segments reversed, each motion
inverted: forward↔back, left↔right). It's open-loop (no odometry), so long paths
drift. A manual drive command aborts an active replay.

## Sound / jingles

Speaker instance is `spk`. The dashboard sends RTTTL strings (edit `JINGLES` in
`drive_dashboard.py` — no reflash needed, just a string). `ATECH_SPEAKER` overrides
the instance name if you reflash with a different one.

## Find the car's IP (no serial)

```
for ip in $(seq 1 254); do ping -c1 -W200 192.168.100.$ip >/dev/null 2>&1 & done; wait
arp -an | grep -i 80:f1:b2:cb:9a:84     # -> the car's current IP
```
Or read it over USB serial on boot (the `wifi_ip` event), or use `car.local`.

## Troubleshooting

- **"resource busy" on USB:** another program owns the port — close the atech web
  bridge / any serial monitor.
- **Motors brown out / USB drops at speed:** keep speed ≤ 224 (the dashboard caps
  it). On battery the headroom is better. The dashboard auto-reconnects.
- **WiFi unreachable:** make sure the Mac actually joined `atech-car` (it may
  auto-rejoin a remembered network with internet); give the board a few seconds
  after power-on; the car is always at `192.168.4.1`. The dashboard watchdog
  reconnects on its own once you're on the car's hotspot.
- **Wrong drive direction:** this car's motors are wired reversed, so the dashboard
  sends negated `motor_speed` for forward (the on-screen state label reads inverted
  — cosmetic). Flip `INVERT`/the sign in `command()` if you rewire.

## For agents

- **Run wired:** `uv run python drive_dashboard.py` (background it; poll
  `curl -s localhost:8043/status`). **Run WiFi:** join the car's AP `atech-car`
  (`networksetup -setairportnetwork en0 atech-car minifactory`), then prefix
  `ATECH_CAR_HOST=192.168.4.1`. (On the Mac, joining the AP drops internet — single
  WiFi radio; use USB-ethernet for internet if you need both.)
- **Drive:** `POST /cmd/{forward|back|left|right|stop}?speed=N`. **Sound:**
  `POST /sound/{erika|cuckoos|honk|stop}`. **Record:** `POST /record/{start|stop}`,
  `POST /replay/{forward|reverse|stop}`. `GET /status` has connection + car_action +
  recording/replaying/segments.
- **Never reset the board mid-WiFi-test:** opening the USB serial port (pyserial)
  pulses DTR and **reboots** the board, dropping WiFi. Find the IP via ARP (above),
  not by opening serial.
- **`wifi_ip` is one-shot** (emitted once on connect) — don't wait for it on a board
  that's already up; you'll falsely conclude "no WiFi".
- **Firmware gotchas already handled** in `LOOP_CPP` (don't regress): start the TCP
  server only after `WL_CONNECTED`; accept clients unconditionally each loop;
  `WiFi.setSleep(false)` + `setAutoReconnect(true)`; gate **every** `Serial` write on
  `availableForWrite()` (a plugged-but-unread USB CDC otherwise blocks the whole
  loop and freezes WiFi).
- **AP mode** = the car is its own hotspot at a fixed `192.168.4.1` (no DHCP-from-
  router, no client isolation). A previous STA build chased a router IP via DHCP/mDNS
  and was unreliable on isolating networks — see git history if you need STA.
- **Secrets:** `WIFI_SSID`/`WIFI_PASS` live in `.env.local` (gitignored); the
  committed build script only has `__WIFI_SSID__` placeholders.
