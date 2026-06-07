"""Discover & poke an atech board over USB serial — the bring-up keystone.

The firmware on each board defines its own action/event names (e.g. `fl_speed`
vs a global `forward`, the speaker instance, whatever the proximity sensor
emits). Before we can drive/sound/sense anything, we need to learn those names
from the live board. This tool does exactly that, using the official SDK
(`atech.Board`, line-delimited JSON @ 115200).

Usage:
    # which serial ports look like atech boards? (tells the two boards apart)
    uv run python scripts/atech_probe.py --list

    # reboot a board and print its module list (instance names + ok flags)
    uv run python scripts/atech_probe.py /dev/cu.usbmodem11201 --diagnostics

    # stream every inbound event for 8s (reveals real telemetry keys)
    uv run python scripts/atech_probe.py /dev/cu.usbmodem11201 --listen 8

    # send one action and watch the board react (confirm drive/speaker names)
    uv run python scripts/atech_probe.py /dev/cu.usbmodem11201 --send fl_speed 150
    uv run python scripts/atech_probe.py /dev/cu.usbmodem11201 --send spk_play_rtttl "honk:d=4,o=5,b=160:c"

    # combine: diagnostics, then listen, then drive briefly (all on one open)
    uv run python scripts/atech_probe.py /dev/cu.usbmodem11201 --diagnostics --listen 5

Note: only ONE program can own a serial port. Close the atech browser Web
Serial bridge first, or the open fails with "resource busy".
"""

from __future__ import annotations

import argparse
import sys
import time

from atech import Board
from atech.runtime.transport import discover_ports


def cmd_list() -> int:
    ports = discover_ports()
    if not ports:
        print("No atech-like serial ports found.")
        print("  - Plug a board in via a DATA usb cable (not power-only).")
        print("  - On macOS check: ls /dev/cu.usbmodem*")
        return 1
    print(f"{len(ports)} candidate port(s), most-likely first:\n")
    for i, p in enumerate(ports):
        vid = f"{p.vid:#06x}" if p.vid is not None else "?"
        print(f"  [{i}] {p.device}")
        print(f"      desc   : {p.description}")
        print(f"      usb vid: {vid}   serial: {p.serial_number or '?'}")
    return 0


def _open(port: str) -> Board:
    try:
        return Board.connect(port)
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        print(f"Could not open {port}: {e}", file=sys.stderr)
        if "busy" in msg or "resource" in msg or "access" in msg:
            print(
                "  -> The port is held by another program. Close the atech browser\n"
                "     Web Serial bridge (or any serial monitor) and try again.",
                file=sys.stderr,
            )
        raise SystemExit(1)


def cmd_diagnostics(board: Board) -> None:
    print(">>> rebooting board and waiting for boot diagnostics ...")
    report = board.diagnostics(reset=True, timeout=6.0)
    if not report:
        print(
            "    No boot report (firmware too old/silent, or board still booting).\n"
            "    Try --listen instead to see whatever it streams."
        )
        return
    print(f"    reset_reason: {report.get('reset_reason')}")
    print(f"    free_heap   : {report.get('free_heap')}")
    modules = report.get("modules") or []
    if not modules:
        print("    modules     : (none reported)")
        return
    print(f"    modules     : {len(modules)}")
    for m in modules:
        ok = m.get("ok")
        mark = "ok " if ok else "MISSING"
        det = "" if m.get("detectable", True) else "  (not detectable)"
        print(f"      - {m.get('instance'):<10} {m.get('module'):<14} {mark}{det}")


def cmd_listen(board: Board, seconds: float) -> None:
    print(f">>> listening for events for {seconds:.0f}s (Ctrl-C to stop early) ...")
    print(f"    {'event_type':<12} {'key':<22} {'source':<12} value")
    deadline = time.time() + seconds
    seen: set[str] = set()
    try:
        for ev in board.events(poll_interval=0.25):
            print(
                f"    {ev.type:<12} {ev.key:<22} {(ev.module_type or ''):<12} {ev.value!r}"
            )
            seen.add(ev.key)
            if time.time() >= deadline:
                break
    except KeyboardInterrupt:
        pass
    print(
        f"\n    distinct keys seen ({len(seen)}): {', '.join(sorted(seen)) or '(none)'}"
    )


def cmd_send(board: Board, key: str, raw_value: str) -> None:
    # Pass numbers through as numbers, everything else as a string. The firmware
    # reads action values as char* and atoi/atof's them, so a string always works.
    value: object = raw_value
    try:
        value = int(raw_value)
    except ValueError:
        try:
            value = float(raw_value)
        except ValueError:
            pass
    print(f">>> send  action={key!r}  value={value!r}")
    board.send(key, value)
    time.sleep(0.2)  # let the write flush before we (maybe) close


def main() -> int:
    ap = argparse.ArgumentParser(description="Probe an atech board over USB serial.")
    ap.add_argument("port", nargs="?", help="serial device, e.g. /dev/cu.usbmodem11201")
    ap.add_argument(
        "--list", action="store_true", help="list candidate serial ports and exit"
    )
    ap.add_argument(
        "--diagnostics", action="store_true", help="reboot + print module list"
    )
    ap.add_argument(
        "--listen", type=float, metavar="SECONDS", help="stream events for N seconds"
    )
    ap.add_argument(
        "--send", nargs=2, metavar=("KEY", "VALUE"), help="send one action, then exit"
    )
    args = ap.parse_args()

    if args.list or (
        not args.port and not any([args.diagnostics, args.listen, args.send])
    ):
        return cmd_list()

    if not args.port:
        ports = discover_ports()
        if not ports:
            print(
                "No port given and none auto-discovered. Run with --list.",
                file=sys.stderr,
            )
            return 1
        args.port = ports[0].device
        print(f"(auto-selected {args.port})")

    board = _open(args.port)
    try:
        if args.diagnostics:
            cmd_diagnostics(board)
        if args.send:
            cmd_send(board, args.send[0], args.send[1])
        if args.listen:
            cmd_listen(board, args.listen)
    finally:
        board.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
