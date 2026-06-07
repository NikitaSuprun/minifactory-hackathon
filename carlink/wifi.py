"""Direct-LAN WiFi transport for the atech runtime — drive the car cable-free.

Our custom car firmware runs a TCP server (port 3333) speaking the *exact same*
line-delimited JSON as its USB serial link. This transport is the host side of
that: it implements the same `atech.runtime.Transport` Protocol as
`SerialTransport`, but over a socket, so it drops straight into `atech.Board`
(and therefore `carlink.Car` + the dashboard) with nothing else changing.

    from carlink import connect_wifi, Car
    car = Car(connect_wifi("car.local"))   # or an IP
    car.drive(150)

Unlike `gateway.py` (which relays through atech's cloud), this talks directly to
the board on the LAN — lower latency, no cloud dependency. It reuses the official
`encode_action` / `decode_event` so the wire format always matches the SDK.
"""

from __future__ import annotations

import queue
import socket
import threading
import time
from typing import Any

from atech import Action, Board, Event
from atech.runtime.transport import decode_event, encode_action

DEFAULT_TCP_PORT = 3333


class TcpTransport:
    """`atech.runtime.Transport` over a raw TCP socket to the car's WiFi server.

    A background thread reads the socket, splits on newlines, parses events, caches
    the latest per key, and feeds a queue for ``recv`` — the same model as
    SerialTransport, so ``Board.latest()`` works without anyone draining events.
    """

    def __init__(
        self, host: str, port: int = DEFAULT_TCP_PORT, connect_timeout: float = 5.0
    ):
        self.host = host
        self.port = port
        self._sock = socket.create_connection((host, port), timeout=connect_timeout)
        self._sock.settimeout(0.2)
        self._buf = b""
        self._q: "queue.Queue[Event]" = queue.Queue()
        self._latest: dict[str, Event] = {}
        self._lock = threading.Lock()
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._reader, name="atech-wifi", daemon=True
        )
        self._thread.start()

    # -- background reader --
    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._sock.recv(1024)
            except socket.timeout:
                continue
            except OSError as e:
                self.last_error = f"recv: {e}"
                time.sleep(0.2)
                continue
            if not chunk:  # peer closed
                self.last_error = "connection closed by board"
                break
            self._buf += chunk
            while b"\n" in self._buf:
                line, self._buf = self._buf.split(b"\n", 1)
                evt = decode_event(line.decode("utf-8", "replace"))
                if evt is None:
                    continue
                with self._lock:
                    self._latest[evt.key] = evt
                self._q.put(evt)

    # -- Transport protocol --
    def send(self, action: Action) -> None:
        self._sock.sendall(encode_action(action))

    def recv(self, timeout: float | None = None) -> Event | None:
        try:
            return self._q.get(timeout=timeout) if timeout else self._q.get_nowait()
        except queue.Empty:
            return None

    def latest(self, key: str) -> Event | None:
        with self._lock:
            return self._latest.get(key)

    def boot_report(
        self, *, reset: bool = True, timeout: float = 5.0
    ) -> dict[str, Any] | None:
        return None  # no reset/boot diagnostics over WiFi

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass


def connect_wifi(
    host: str, port: int = DEFAULT_TCP_PORT, connect_timeout: float = 5.0
) -> Board:
    """Open an atech.Board to the car over WiFi (LAN TCP), by host/IP or mDNS name.

    ``connect_timeout`` bounds the initial TCP connect — keep it short when WiFi is
    only *tried* before falling back to USB serial."""
    return Board(TcpTransport(host, port, connect_timeout=connect_timeout))
