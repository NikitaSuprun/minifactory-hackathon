"""WiFi transport for the official atech runtime, via gateway.atech.dev.

The official SDK only ships serial (`atech.SerialTransport`). This implements the
same `atech.runtime.Transport` Protocol (send/recv/latest/boot_report/close) over
the gateway's WebSocket, so it drops straight into `atech.Board`:

    from carlink import connect_gateway
    board = connect_gateway("my-project-id")   # -> atech.Board over WiFi
    board.send("fl_speed", 200)
    for ev in board.events():
        print(ev.key, ev.value)

Gateway protocol (https://gateway.atech.dev):
  receive: {"type":"device_event","payload":{...}}  + device_connected/disconnected
  send:    {"type":"send_to_device","device_id":<pid>,"payload":{action,value}}

The inner ``payload`` is identical to the serial envelope's payload, so events
map to ``atech.Event`` exactly as the serial transport does
(``type=event_type``, ``module_type=source``).
"""

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Optional

from atech import Action, Board, Event

GATEWAY_HOST = "gateway.atech.dev"


def _wire_value(value) -> str:
    """Stringify an action value for firmware that reads value as char*."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    return json.dumps(value, separators=(",", ":"))


def _event_from_payload(payload: dict) -> Optional[Event]:
    """Build an atech.Event from a gateway device_event payload (mirrors the
    official serial parser: type=event_type, module_type=source)."""
    event_type = payload.get("event_type")
    key = payload.get("key", "")
    if not event_type or not isinstance(key, str):
        return None
    try:
        return Event(
            type=event_type,
            key=key,
            value=payload.get("value"),
            module_type=payload.get("source") or None,
            received_at=time.time(),
        )
    except Exception:  # noqa: BLE001 - malformed payload
        return None


class GatewayTransport:
    """`atech.runtime.Transport` implementation over the gateway WebSocket.

    A background thread reads the socket, parses device events, caches the latest
    per key, and feeds a queue for ``recv`` — the same model as SerialTransport,
    so ``Board.latest()`` works without anyone draining ``events()``.
    """

    def __init__(self, project_id: str, host: str = GATEWAY_HOST, timeout: float = 1.0):
        import websocket  # websocket-client

        self.project_id = project_id
        self.url = f"wss://{host}/ws/live/{project_id}"
        self._ws = websocket.create_connection(self.url, timeout=timeout)
        self._timeout_exc = websocket.WebSocketTimeoutException
        self._q: "queue.Queue[Event]" = queue.Queue()
        self._latest: dict[str, Event] = {}
        self._lock = threading.Lock()
        self.connected_devices = False
        self.last_error: Optional[str] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._reader, name="atech-gw", daemon=True
        )
        self._thread.start()

    # -- background reader --
    def _reader(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._ws.recv()
            except self._timeout_exc:
                continue
            except Exception as e:  # noqa: BLE001 - surface, back off
                self.last_error = f"recv: {e}"
                time.sleep(0.2)
                continue
            if not raw:
                continue
            self._ingest(
                raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
            )

    def _ingest(self, text: str) -> None:
        try:
            frame = json.loads(text)
        except json.JSONDecodeError:
            return
        ftype = frame.get("type")
        if ftype == "device_connected":
            self.connected_devices = True
            return
        if ftype == "device_disconnected":
            self.connected_devices = False
            return
        if ftype != "device_event":
            return
        payload = frame.get("payload")
        if not isinstance(payload, dict):
            return
        evt = _event_from_payload(payload)
        if evt is None:
            return
        with self._lock:
            self._latest[evt.key] = evt
        self._q.put(evt)

    # -- Transport protocol --
    def send(self, action: Action) -> None:
        # Firmware reads value as char* (handleMessage does atoi(value)), so send
        # it as a string — matching the official serial encoder's convention.
        frame = {
            "type": "send_to_device",
            "device_id": self.project_id,
            "payload": {"action": action.key, "value": _wire_value(action.value)},
        }
        self._ws.send(json.dumps(frame, separators=(",", ":")))

    def recv(self, timeout: Optional[float] = None) -> Optional[Event]:
        try:
            return self._q.get(timeout=timeout) if timeout else self._q.get_nowait()
        except queue.Empty:
            return None

    def latest(self, key: str) -> Optional[Event]:
        with self._lock:
            return self._latest.get(key)

    def boot_report(
        self, *, reset: bool = True, timeout: float = 5.0
    ) -> Optional[dict]:
        return None  # no boot diagnostics over the gateway

    def close(self) -> None:
        self._stop.set()
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001
            pass


def connect_gateway(project_id: str, host: str = GATEWAY_HOST) -> Board:
    """Open an atech.Board for a car over the WiFi gateway, by project ID."""
    return Board(GatewayTransport(project_id, host))
