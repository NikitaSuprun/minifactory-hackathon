# atech firmware prompt (cars)

Paste this into the atech hosted editor when (re)generating the car firmware. It
keeps the control interface our backend/dashboard (`carlink` + `car_dashboard.py`)
already speaks, and adds the hooks that make the host side robust.

> Keep the 14-port car: 4 DC motors (FL=port1, RL=port6, FR=port9, RR=port14,
> right side mirrored), the IMU, and the VL53L5CX depth sensor on port 11. Keep
> **USB-serial** transport for now (WiFi later). Keep the existing control
> actions and their string values (`atoi`/`atof`): `forward`, `backward`, `left`,
> `right` (each 0–255), `stop`, `motor_speed` (−255..255), `turn_to_heading`,
> `tare_heading`, `enable`, `disable`. Add the following:
>
> 1. **Depth sensor:** make the VL53L5CX on port 11 initialize and stream. Emit
>    `min_distance` (mm) every ~200ms as a sensor event. If `begin()` fails at
>    boot, keep retrying and emit a `state` event `depth_sensor` = `missing` /
>    `ok` so the host knows whether it's connected. (Bonus: also emit the 8×8
>    zone grid so the host can steer around obstacles, not only stop.)
>
> 2. **Deadman watchdog:** if no action has arrived from the host for **500ms**,
>    automatically brake all motors. Reset the timer on every received action.
>    Accept a no-op `ping` action as a keepalive (resets the timer, does nothing
>    else). Emit a `state` event `link` = `stale` when it trips and `link` = `ok`
>    when commands resume.
>
> 3. **Module presence / health:** on boot, emit one `state` event per module of
>    the form key=`module.<instance>` value=`ok` or `missing` (e.g.
>    `module.vl53l5cx`, `module.imu`, `module.fl` …). Repeat the full set every
>    ~5s. This lets the dashboard show exactly what's physically connected.
>
> 4. **Readiness:** emit `status` = `ready` on boot **and** repeat it every ~2s,
>    so a host that connects after boot still sees readiness. Keep streaming
>    `orientation` (pitch,roll,heading) and the `obstacle` (detected/clear) state.
>
> Don't change the JSON envelope shape: `{"type":"event","payload":{"event_type":
> "...","key":"...","value":...,"source":"..."}}`.

## After you reflash, tell me

- Any **action names / event keys** that changed, and the **keepalive action name
  + watchdog timeout** — I'll match `Car` and the dashboard heartbeat to it.
- Or just paste the regenerated `.cpp` and I'll read the interface off it.

## Add the SPEAKER (for sound / jingles)

The car board physically has the I2S speaker (MAX98357A), but the current firmware
doesn't declare it, so `*_play_rtttl` does nothing. To enable sound, regenerate in
the atech editor with this added (keep everything above unchanged):

> Add the **speaker** module (`speaker`, I2S MAX98357A). It is a **double module**
> — it occupies an **adjacent pair of ports**; place it on a free pair (the motors
> use ports 1, 6, 9, 14, so e.g. **[2,3]** or **[7,8]** is free) and wire it there.
> Give it the instance name **`spk`**. In `setup()` call `spk.begin()` and
> `spk.setVolume(0.4f)`. Do **not** play anything on boot. Expose the standard
> speaker actions over the existing JSON envelope so the host can drive it:
> `spk_play_rtttl` (string RTTTL, plays in the background), `spk_set_volume`
> (0.0–1.0 float), and `spk_stop`. Audio must play in the background FreeRTOS task
> (non-blocking) so driving keeps working while a melody plays.

After reflash, tell me the **speaker instance name** you used (the spec says `spk`).
The dashboard already has 🍦 Jingle / 📣 Honk / Test buttons + a speaker-name box —
set the box to that name (or `ATECH_SPEAKER=<name>`), hit Test, and it'll sound.

## Host side (I implement, no firmware needed)

- `--sim` mode for the dashboard (run everything on `atech.MockTransport`, no car).
- Surface module-presence + telemetry **age** in `/status` and the UI.
- Per-car **partial connect** + reconnect (one car failing shouldn't block the other).
- **Heartbeat sender** that pings each car every ~200ms so the deadman never trips
  in normal use.
