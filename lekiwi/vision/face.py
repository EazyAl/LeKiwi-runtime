from __future__ import annotations

import collections
import dataclasses
import math
import time
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class FaceLandmarker:
    def __init__(self, max_faces: int = 1):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def infer(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.face_mesh.process(rgb)

    def close(self):
        self.face_mesh.close()


def compute_face_box(landmarks, image_w: int, image_h: int, scale: float = 1.2):
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cx = (min_x + max_x) * 0.5
    cy = (min_y + max_y) * 0.5
    w = (max_x - min_x) * scale
    h = (max_y - min_y) * scale
    x0 = max(0, int((cx - w * 0.5) * image_w))
    y0 = max(0, int((cy - h * 0.5) * image_h))
    x1 = min(image_w, int((cx + w * 0.5) * image_w))
    y1 = min(image_h, int((cy + h * 0.5) * image_h))
    return (x0, y0, x1, y1)


def _ear(pts):
    p2_p6 = math.dist(pts[1], pts[5])
    p3_p5 = math.dist(pts[2], pts[4])
    p1_p4 = math.dist(pts[0], pts[3])
    return (p2_p6 + p3_p5) / (2.0 * p1_p4 + 1e-6)


def compute_eye_metrics(landmarks, image_w: int, image_h: int) -> Dict[str, float]:
    left_idx = [33, 160, 158, 133, 153, 144]
    right_idx = [362, 385, 387, 263, 373, 380]

    def _landmark_xy(idx):
        lm = landmarks[idx]
        return (lm.x * image_w, lm.y * image_h)

    left_pts = [_landmark_xy(i) for i in left_idx]
    right_pts = [_landmark_xy(i) for i in right_idx]

    left_ear = _ear(left_pts)
    right_ear = _ear(right_pts)
    ear_mean = 0.5 * (left_ear + right_ear)

    eyes_open_prob = float(np.clip((ear_mean - 0.15) / 0.15, 0.0, 1.0))

    return {
        "left_ear": float(left_ear),
        "right_ear": float(right_ear),
        "ear_mean": float(ear_mean),
        "eyes_open_prob": eyes_open_prob,
    }


@dataclasses.dataclass
class BlinkState:
    last_state: bool = False
    last_change_ts: float = 0.0
    blink_count: int = 0


class BlinkAwakeEstimator:
    def __init__(
        self,
        window_seconds: float = 30.0,
        ear_close_thresh: float = 0.18,
        ear_open_thresh: float = 0.22,
    ):
        self.window_seconds = window_seconds
        self.ear_close_thresh = ear_close_thresh
        self.ear_open_thresh = ear_open_thresh
        self.samples: Deque[Tuple[float, float]] = collections.deque()
        self.state = BlinkState()

    def update(self, ear_mean: float, timestamp: Optional[float] = None) -> Dict[str, float]:
        ts = timestamp or time.time()
        self.samples.append((ts, ear_mean))
        cutoff = ts - self.window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

        is_closed = ear_mean < self.ear_close_thresh
        if is_closed != self.state.last_state and is_closed:
            self.state.blink_count += 1
        if is_closed != self.state.last_state:
            self.state.last_state = is_closed
            self.state.last_change_ts = ts

        closed_time = 0.0
        if len(self.samples) >= 2:
            for i in range(1, len(self.samples)):
                t0, ear0 = self.samples[i - 1]
                t1, ear1 = self.samples[i]
                mid_closed = 1.0 if (ear0 + ear1) * 0.5 < self.ear_close_thresh else 0.0
                closed_time += (t1 - t0) * mid_closed

        total_time = self.samples[-1][0] - self.samples[0][0] if len(self.samples) >= 2 else 0.0
        perclos = (closed_time / total_time) if total_time > 0 else 0.0

        blinks_per_min = (self.state.blink_count / total_time * 60.0) if total_time > 0 else 0.0
        awake_likelihood = float(np.clip(1.0 - perclos, 0.0, 1.0))

        return {
            "blinks_per_min": float(blinks_per_min),
            "perclos": float(perclos),
            "awake_likelihood": awake_likelihood,
        }


class RPpgHeartEstimator:
    def __init__(self, buffer_seconds: float = 12.0, fs_hint: float = 30.0):
        self.buffer_seconds = buffer_seconds
        self.fs_hint = fs_hint
        self.samples: Deque[Tuple[float, float]] = collections.deque()

    def _bandpass_hr(self, ts_values, signal):
        if len(signal) < 20:
            return None, 0.0
        t0, t1 = ts_values[0], ts_values[-1]
        duration = t1 - t0
        if duration < 5.0:
            return None, 0.0
        fs = max(15.0, min(60.0, len(signal) / duration))
        n = int(duration * fs)
        if n < 50:
            return None, 0.0
        uniform_t = np.linspace(t0, t1, n)
        uniform_sig = np.interp(uniform_t, ts_values, signal)
        uniform_sig = uniform_sig - np.mean(uniform_sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        fft_vals = np.abs(np.fft.rfft(uniform_sig))
        band_mask = (freqs >= 0.75) & (freqs <= 3.0)
        if not np.any(band_mask):
            return None, 0.0
        band_freqs = freqs[band_mask]
        band_fft = fft_vals[band_mask]
        peak_idx = int(np.argmax(band_fft))
        peak_freq = band_freqs[peak_idx]
        peak_power = band_fft[peak_idx]
        total_power = np.sum(band_fft) + 1e-6
        quality = float(np.clip(peak_power / total_power, 0.0, 1.0))
        hr_bpm = peak_freq * 60.0
        return hr_bpm, quality

    def update(self, frame_bgr, face_box, timestamp: Optional[float] = None) -> Dict[str, float]:
        ts = timestamp or time.time()
        x0, y0, x1, y1 = face_box
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return {"hr_bpm": -1.0, "hr_quality": 0.0, "method": "fft_green"}

        green = roi[:, :, 1].astype(np.float32)
        mean_green = float(np.mean(green))
        self.samples.append((ts, mean_green))

        cutoff = ts - self.buffer_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

        ts_values = [t for t, _ in self.samples]
        signal = [v for _, v in self.samples]
        hr_bpm, quality = self._bandpass_hr(ts_values, signal)
        if hr_bpm is None:
            return {"hr_bpm": -1.0, "hr_quality": 0.0, "method": "fft_green"}
        return {"hr_bpm": float(hr_bpm), "hr_quality": quality, "method": "fft_green"}
