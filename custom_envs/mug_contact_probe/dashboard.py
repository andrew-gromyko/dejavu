"""Dashboard composition and presentation backends for mug contact probe."""

from __future__ import annotations

import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Mapping, Optional

import cv2
import numpy as np


@dataclass
class DashboardPayload:
    video_rgb: np.ndarray
    control_step: int
    max_steps: int
    tactile_rgbs: Mapping[str, Optional[np.ndarray]]
    audio_waveforms: Mapping[str, Optional[np.ndarray]]
    audio_play_start_times: Mapping[str, Optional[float]]
    contact_active: Mapping[str, bool]
    force_norm_n: Mapping[str, float]
    is_grasping: bool


def _draw_audio_waveform(w: int, h: int, waveform: Optional[np.ndarray], playhead_ratio: float = -1.0) -> np.ndarray:
    img = np.full((h, w, 3), 25, dtype=np.uint8)
    mid_y = h // 2
    cv2.line(img, (0, mid_y), (w, mid_y), (60, 60, 60), 1)

    if waveform is not None and len(waveform) > 0:
        bin_size = max(1, len(waveform) // w)
        for x in range(w):
            start = x * bin_size
            end = min(start + bin_size, len(waveform))
            if start < end:
                val_min = float(np.min(waveform[start:end]))
                val_max = float(np.max(waveform[start:end]))
            else:
                val_min = 0.0
                val_max = 0.0
            y_min = int(mid_y - val_min * (mid_y - 5))
            y_max = int(mid_y - val_max * (mid_y - 5))
            cv2.line(img, (x, y_min), (x, y_max), (235, 160, 30), 1)

        if 0.0 <= playhead_ratio <= 1.0:
            px = int(playhead_ratio * w)
            cv2.line(img, (px, 0), (px, h), (0, 0, 255), 2)
    else:
        cv2.putText(
            img,
            "AWAITING IMPACT...",
            (w // 2 - 80, h // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 100, 100),
            1,
            cv2.LINE_AA,
        )
    return img


def _draw_pad_panel(
    pad_label: str,
    tactile_rgb: Optional[np.ndarray],
    audio_waveform: Optional[np.ndarray],
    audio_play_start_time: Optional[float],
    contact_active: bool,
    force_norm_n: float,
) -> tuple[np.ndarray, Optional[float]]:
    panel = np.full((330, 480, 3), 18, dtype=np.uint8)
    panel[0:34, :] = 28

    cv2.putText(panel, pad_label.upper(), (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
    status = "CONTACT" if contact_active else "NO CONTACT"
    status_color = (0, 220, 90) if contact_active else (150, 150, 150)
    cv2.putText(panel, status, (350, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)

    if tactile_rgb is None:
        tactile_bgr = np.full((120, 160, 3), 127, dtype=np.uint8)
    else:
        tactile_bgr = cv2.cvtColor(tactile_rgb, cv2.COLOR_RGB2BGR)
    tactile_img = cv2.resize(tactile_bgr, (220, 220), interpolation=cv2.INTER_LINEAR)
    panel[50:270, 14:234] = tactile_img

    playhead_ratio = -1.0
    updated_audio_start = audio_play_start_time
    if audio_play_start_time is not None:
        elapsed = time.time() - audio_play_start_time
        if elapsed < 3.0:
            playhead_ratio = elapsed / 3.0
        else:
            updated_audio_start = None

    waveform_img = _draw_audio_waveform(220, 220, audio_waveform, playhead_ratio)
    panel[50:270, 246:466] = waveform_img

    cv2.putText(panel, "TACTILE", (14, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1, cv2.LINE_AA)
    if playhead_ratio >= 0.0 and updated_audio_start is not None:
        elapsed = time.time() - updated_audio_start
        audio_text = f"AUDIO {elapsed:.1f}s"
        audio_color = (0, 0, 255)
    else:
        audio_text = "AUDIO IDLE"
        audio_color = (150, 150, 150)
    cv2.putText(panel, audio_text, (246, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.38, audio_color, 1, cv2.LINE_AA)

    press_depth_mm = 0.0
    if contact_active:
        press_depth = 0.0005 + (0.0020 - 0.0005) * min(force_norm_n / 4.0, 1.0)
        press_depth_mm = press_depth * 1000.0
    force_text = f"FORCE {force_norm_n:.2f} N | DEPTH {press_depth_mm:.2f} mm"
    cv2.putText(panel, force_text, (14, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)

    cv2.rectangle(panel, (13, 49), (235, 271), (55, 55, 55), 1)
    cv2.rectangle(panel, (245, 49), (467, 271), (55, 55, 55), 1)
    cv2.line(panel, (0, 329), (480, 329), (55, 55, 55), 1)
    return panel, updated_audio_start


def compose_dashboard_frame(payload: DashboardPayload) -> tuple[np.ndarray, dict[str, Optional[float]]]:
    env_rgb = payload.video_rgb
    if env_rgb.shape[:2] != (600, 800):
        env_rgb = cv2.resize(env_rgb, (800, 600), interpolation=cv2.INTER_AREA)
    env_bgr = cv2.cvtColor(env_rgb, cv2.COLOR_RGB2BGR)

    dashboard = np.zeros((720, 1280, 3), dtype=np.uint8)
    dashboard[0:60, :] = 20
    cv2.putText(
        dashboard,
        "MANISKILL PAD-LEVEL MULTIMODAL DASHBOARD",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    step_text = f"STEP: {payload.control_step}/{payload.max_steps}"
    cv2.putText(dashboard, step_text, (600, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    grasp_text = f"GRASPING: {'ACTIVE' if payload.is_grasping else 'IDLE'}"
    grasp_color = (0, 255, 0) if payload.is_grasping else (0, 255, 255)
    cv2.putText(dashboard, grasp_text, (790, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, grasp_color, 1, cv2.LINE_AA)
    total_force = sum(float(v) for v in payload.force_norm_n.values())
    active_count = sum(1 for v in payload.contact_active.values() if v)
    force_text = f"ACTIVE PADS: {active_count}/2 | TOTAL FORCE: {total_force:.2f} N"
    cv2.putText(dashboard, force_text, (980, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    dashboard[60:660, 0:800] = env_bgr
    video_bar = np.full((60, 800, 3), 16, dtype=np.uint8)
    cv2.putText(video_bar, "VIDEO", (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
    dashboard[660:720, 0:800] = video_bar

    updated_audio_starts: dict[str, Optional[float]] = {}
    pad_specs = (("left", "Left Pad"), ("right", "Right Pad"))
    for row, (pad_id, pad_label) in enumerate(pad_specs):
        panel, updated_start = _draw_pad_panel(
            pad_label=pad_label,
            tactile_rgb=payload.tactile_rgbs.get(pad_id),
            audio_waveform=payload.audio_waveforms.get(pad_id),
            audio_play_start_time=payload.audio_play_start_times.get(pad_id),
            contact_active=bool(payload.contact_active.get(pad_id, False)),
            force_norm_n=float(payload.force_norm_n.get(pad_id, 0.0)),
        )
        y0 = 60 + row * 330
        dashboard[y0 : y0 + 330, 800:1280] = panel
        updated_audio_starts[pad_id] = updated_start

    cv2.line(dashboard, (0, 60), (1280, 60), (50, 50, 50), 2)
    cv2.line(dashboard, (800, 60), (800, 720), (50, 50, 50), 2)
    return dashboard, updated_audio_starts


class BrowserDashboard:
    """Tiny no-dependency browser presenter for lossless dashboard frames."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._condition = threading.Condition()
        self._frame_png: Optional[bytes] = None
        self._stop_requested = False
        self._httpd = ThreadingHTTPServer((host, port), self._make_handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address
        return f"http://{host}:{port}/"

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_requested = True
        with self._condition:
            self._condition.notify_all()
        self._httpd.shutdown()
        self._httpd.server_close()

    def update(self, dashboard_bgr: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".png", dashboard_bgr)
        if not ok:
            return
        with self._condition:
            self._frame_png = encoded.tobytes()
            self._condition.notify_all()

    def _make_handler(self):
        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/":
                    body = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ManiSkill Probe Dashboard</title>
  <style>
    html, body { margin: 0; background: #111; color: #eee; font-family: sans-serif; }
    main { display: grid; place-items: center; min-height: 100vh; gap: 12px; }
	    img { width: min(100vw, 1280px); height: auto; image-rendering: auto; }
    button { padding: 8px 14px; border: 0; border-radius: 6px; background: #ddd; cursor: pointer; }
  </style>
</head>
<body>
  <main>
    <img src="/stream" alt="ManiSkill Probe Dashboard">
    <button onclick="fetch('/stop')">Stop probe</button>
  </main>
</body>
</html>""".encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == "/stop":
                    dashboard._stop_requested = True
                    with dashboard._condition:
                        dashboard._condition.notify_all()
                    self.send_response(204)
                    self.end_headers()
                    return

                if self.path != "/stream":
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()

                last_frame = None
                while not dashboard._stop_requested:
                    with dashboard._condition:
                        dashboard._condition.wait(timeout=1.0)
                        frame = dashboard._frame_png
                    if frame is None or frame is last_frame:
                        continue
                    last_frame = frame
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/png\r\n")
                        self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break

            def log_message(self, format: str, *args) -> None:
                return

        return Handler


class DashboardPresenter:
    def __init__(self, enabled: bool, backend: str, window_name: str = "ManiSkill 3-Modality Probe Dashboard"):
        self.enabled = enabled
        self.backend = backend
        self.window_name = window_name
        self.browser: Optional[BrowserDashboard] = None
        self._opencv_frame: Optional[np.ndarray] = None

    def start(self) -> None:
        if not self.enabled:
            return
        if self.backend == "browser":
            self.browser = BrowserDashboard()
            self.browser.start()
            print(f"Dashboard available at {self.browser.url}")
            webbrowser.open(self.browser.url)
        else:
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def render(self, payload: DashboardPayload) -> tuple[bool, dict[str, Optional[float]]]:
        if not self.enabled:
            return True, dict(payload.audio_play_start_times)
        dashboard_frame, updated_audio_start = compose_dashboard_frame(payload)

        if self.backend == "browser":
            assert self.browser is not None
            self.browser.update(dashboard_frame)
            if self.browser.stop_requested:
                print("Visualization closed by user.")
                return False, None
            return True, updated_audio_start

        self._opencv_frame = dashboard_frame.copy()
        cv2.imshow(self.window_name, self._opencv_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("Visualization closed by user.")
            return False, None
        return True, updated_audio_start

    def close(self) -> None:
        if self.browser is not None:
            self.browser.stop()
        cv2.destroyAllWindows()
