# minifactory-hackathon

Stream an Android phone's camera to this Mac over WiFi and use it as a
[LeRobot](https://github.com/huggingface/lerobot) camera for model inference and
dataset recording.

The phone runs the **IP Webcam** app and serves an MJPEG stream; LeRobot's
`OpenCVCamera` opens that URL directly — no macOS driver or virtual camera needed.

## 1. Phone setup (IP Webcam)

1. Install **IP Webcam** (by Pavel Khlebovich) from the Play Store.
2. Open the app and, for low latency, set **Video preferences**:
   - Resolution: `640x480` (modest is fine; lower = lower latency)
   - Quality: ~50%
   - FPS limit: 30
   - Disable audio streaming
3. (Optional) Protect the stream: under **Connection → Login/password**, set a
   **Login** and **Password** (e.g. `admin` / `123123`). The stream then requires
   HTTP Basic Auth.
4. Scroll to the bottom and tap **Start server**.
5. The app shows a URL like `http://192.168.1.42:8080`. The MJPEG stream is that
   address + `/video`, e.g. `http://192.168.1.42:8080/video`.

## 2. Network check

Phone and Mac must be on the **same WiFi** (guest networks / "client isolation"
will block this). Confirm by opening `http://<phone-ip>:8080` in the Mac's browser —
you should see the live video and controls.

## 3. Configure `.env`

All connection settings live in the committed [`.env`](.env). Edit
`PHONE_CAM_HOST` to your phone's IP (and the login/password if you set them):

```dotenv
PHONE_CAM_HOST=192.168.1.42
PHONE_CAM_PORT=8080
PHONE_CAM_PATH=/video
PHONE_CAM_USER=admin
PHONE_CAM_PASS=123123
```

The URL is assembled from these as `http://USER:PASS@HOST:PORT/PATH`. Set
`PHONE_CAM_URL` instead if you want to override the whole thing.

> The `.env` is committed on purpose for this hackathon. That means the password
> is in git history — keep it a throwaway, LAN-only value, never a real secret.

## 4. Verify ingestion via LeRobot

```bash
uv run python scripts/check_phone_stream.py            # uses .env
# or pass an explicit URL (overrides .env):
uv run python scripts/check_phone_stream.py http://admin:123123@<phone-ip>:8080/video
```

Expected: it prints the detected resolution, achieved FPS, and a latency proxy,
then saves `stream_sample.png` showing the phone's camera view. That confirms
LeRobot's `OpenCVCamera` is ingesting the stream.

> `opencv-python-headless` (pulled in by LeRobot) has no GUI, so there's no live
> preview window — use the phone's browser page for live view; the script proves
> ingestion via stats + the saved snapshot.

## 5. Use the phone camera in LeRobot (later)

```python
from phone_camera import build_phone_camera_config

robot_config.cameras = {
    "phone": build_phone_camera_config("http://192.168.1.42:8080/video"),
}
```

`build_phone_camera_config` leaves `fps/width/height` unset so LeRobot auto-detects
the stream profile — set resolution/FPS in the IP Webcam app, not in code.

For a password-protected stream, embed credentials in the URL (optionally via the
`with_credentials` helper):

```python
from phone_camera import build_phone_camera_config, with_credentials

url = with_credentials("http://192.168.1.42:8080/video", "admin", "123123")
robot_config.cameras = {"phone": build_phone_camera_config(url)}
```

## Optional: lowest latency over USB

Tether the phone over USB and forward the port with adb, then use `localhost`:

```bash
adb forward tcp:8080 tcp:8080
uv run python scripts/check_phone_stream.py http://localhost:8080/video
```

## Notes

- MJPEG over WiFi is typically ~100–200 ms latency. RTSP
  (`rtsp://<phone-ip>:8080/h264_ulaw.sdp`) is also available but tends to buffer
  more in OpenCV — prefer MJPEG.
