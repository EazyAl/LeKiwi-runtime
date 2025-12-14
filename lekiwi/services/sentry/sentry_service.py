from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import cv2
import numpy as np
import os

from lekiwi.services.base import ServiceBase
from lekiwi.services.pose_detection.pose_service import CameraStream, FallDetector, PoseEstimator
from lekiwi.viz.rerun_viz import NullViz


# MediaPipe pose landmark indices (same as `nav/nav.py`)
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26


def _calculate_thigh_midpoint(landmarks, frame_width: int, frame_height: int):
    """
    Return (mid_x, mid_y, conf, side) or (None, None, 0.0, None).
    Uses hip+knee landmarks visibility to pick the most confident thigh.
    """
    if not landmarks:
        return None, None, 0.0, None

    candidates = []

    # Left thigh
    if (
        len(landmarks) > LEFT_HIP
        and len(landmarks) > LEFT_KNEE
        and getattr(landmarks[LEFT_HIP], "visibility", 0.0) > 0.5
        and getattr(landmarks[LEFT_KNEE], "visibility", 0.0) > 0.5
    ):
        hip_x = float(landmarks[LEFT_HIP].x) * frame_width
        hip_y = float(landmarks[LEFT_HIP].y) * frame_height
        knee_x = float(landmarks[LEFT_KNEE].x) * frame_width
        knee_y = float(landmarks[LEFT_KNEE].y) * frame_height
        mid_x = (hip_x + knee_x) * 0.5
        mid_y = (hip_y + knee_y) * 0.5
        conf = (float(landmarks[LEFT_HIP].visibility) + float(landmarks[LEFT_KNEE].visibility)) * 0.5
        candidates.append((mid_x, mid_y, conf, "left"))

    # Right thigh
    if (
        len(landmarks) > RIGHT_HIP
        and len(landmarks) > RIGHT_KNEE
        and getattr(landmarks[RIGHT_HIP], "visibility", 0.0) > 0.5
        and getattr(landmarks[RIGHT_KNEE], "visibility", 0.0) > 0.5
    ):
        hip_x = float(landmarks[RIGHT_HIP].x) * frame_width
        hip_y = float(landmarks[RIGHT_HIP].y) * frame_height
        knee_x = float(landmarks[RIGHT_KNEE].x) * frame_width
        knee_y = float(landmarks[RIGHT_KNEE].y) * frame_height
        mid_x = (hip_x + knee_x) * 0.5
        mid_y = (hip_y + knee_y) * 0.5
        conf = (float(landmarks[RIGHT_HIP].visibility) + float(landmarks[RIGHT_KNEE].visibility)) * 0.5
        candidates.append((mid_x, mid_y, conf, "right"))

    if not candidates:
        return None, None, 0.0, None

    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0]

def _calculate_hip_midpoint(landmarks, frame_width: int, frame_height: int):
    """
    Return (mid_x, mid_y, conf, "hips") or (None, None, 0.0, None).
    This is a robust fallback when knees are occluded (common when close to camera).
    """
    if not landmarks:
        return None, None, 0.0, None
    if (
        len(landmarks) <= RIGHT_HIP
        or getattr(landmarks[LEFT_HIP], "visibility", 0.0) <= 0.0
        or getattr(landmarks[RIGHT_HIP], "visibility", 0.0) <= 0.0
    ):
        return None, None, 0.0, None
    lx = float(landmarks[LEFT_HIP].x) * frame_width
    ly = float(landmarks[LEFT_HIP].y) * frame_height
    rx = float(landmarks[RIGHT_HIP].x) * frame_width
    ry = float(landmarks[RIGHT_HIP].y) * frame_height
    mid_x = (lx + rx) * 0.5
    mid_y = (ly + ry) * 0.5
    conf = (float(getattr(landmarks[LEFT_HIP], "visibility", 0.0)) + float(getattr(landmarks[RIGHT_HIP], "visibility", 0.0))) * 0.5
    return mid_x, mid_y, conf, "hips"


def _calculate_thigh_offset_norm(thigh_x: Optional[float], frame_width: int) -> float:
    if thigh_x is None or frame_width <= 0:
        return 0.0
    frame_center_x = float(frame_width) * 0.5
    offset_pixels = float(thigh_x) - frame_center_x
    return float(offset_pixels / (frame_center_x + 1e-6))


def _pixels_to_angle_deg(offset_pixels: float, frame_width: int, camera_fov_deg: float = 20.0) -> float:
    """
    Convert pixel offset to angular offset (deg).
    This mirrors `nav/nav.py` (20deg default).
    """
    if frame_width <= 0:
        return 0.0
    fov_rad = np.radians(float(camera_fov_deg))
    angle_per_pixel = fov_rad / float(frame_width)
    angle_rad = float(offset_pixels) * float(angle_per_pixel)
    return float(np.degrees(angle_rad))


@dataclass
class SentryDetails:
    score: float
    ratio: float
    timestamp: float


FallCallback = Callable[[Dict[str, Any]], None]


class SentryService(ServiceBase):
    """
    Always-on sentry loop:
      - continuously track the user's thigh and rotate the base to keep them centered
      - detect falls (torso ratio) and trigger an orchestrator callback exactly once per fall

    IMPORTANT: On fall we DO NOT stop/join other worker threads here (we are a worker thread).
    We only invoke the callback; the orchestrator should schedule quiescing on its asyncio thread.
    """

    def __init__(
        self,
        *,
        robot: Any,
        robot_lock: Any,
        on_fall: FallCallback,
        frame_subscription: Optional[Any] = None,
        camera_index: Optional[int] = None,
        rotate_180: bool = True,
        fps: int = 20,
        rotation_kp: float = 5.0,
        rotation_deadzone: float = 0.08,
        max_theta_vel: float = 20.0,
        min_conf: float = 0.5,
        fall_freeze_s: float = 3.0,
        viz: Optional[Any] = None,
    ) -> None:
        super().__init__("sentry")
        self.robot = robot
        self.robot_lock = robot_lock
        self.on_fall = on_fall

        self.frame_subscription = frame_subscription
        self._use_external_frames = frame_subscription is not None
        self.rotate_180 = bool(rotate_180)
        self.fps = max(1, int(fps))

        self.rotation_kp = float(rotation_kp)
        self.rotation_deadzone = float(rotation_deadzone)
        self.max_theta_vel = float(max_theta_vel)
        self.min_conf = float(min_conf)

        self._tracking_enabled = True
        self.fall_freeze_s = float(fall_freeze_s)
        self._fall_freeze_until = 0.0
        self._fall_latched = False
        self._last_event: Optional[SentryDetails] = None

        self.pose = PoseEstimator()
        self.fall_detector = FallDetector()
        self.camera = None if self._use_external_frames else CameraStream(index=int(camera_index or 0))

        self.viz = viz if viz is not None else NullViz()

    def start(self):
        if not self._use_external_frames:
            try:
                assert self.camera is not None
                self.camera.start()
            except Exception as e:
                self.logger.error(f"Failed to start sentry camera: {e}")
                return
        try:
            self.logger.info(
                "SentryService: tracking_enabled=%s rotate_180=%s fps=%d kp=%.2f deadzone=%.3f min_conf=%.2f",
                str(self._tracking_enabled),
                str(self.rotate_180),
                int(self.fps),
                float(self.rotation_kp),
                float(self.rotation_deadzone),
                float(self.min_conf),
            )
        except Exception:
            pass
        super().start()

    def stop(self, timeout: float = 5.0):
        # Stop loop first.
        super().stop(timeout)
        # Release camera + pose resources.
        try:
            if not self._use_external_frames and self.camera is not None:
                self.camera.stop()
        except Exception:
            pass
        try:
            self.pose.close()
        except Exception:
            pass
        # Ensure the base doesn't keep rotating on a stale command.
        self._send_base_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})

    def handle_event(self, event_type: str, payload: Any):
        if event_type == "set_tracking_enabled":
            self._tracking_enabled = bool(payload)
            if not self._tracking_enabled:
                self._send_base_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        elif event_type == "reset_fall_latch":
            self._fall_latched = False
        else:
            self.logger.warning(f"Unknown event type: {event_type}")

    def _pull_frame(self, timeout: float = 0.2) -> Optional[Tuple[float, np.ndarray]]:
        if self._use_external_frames and self.frame_subscription:
            pulled = self.frame_subscription.pull(timeout=timeout)
            if pulled is None:
                return None
            return pulled
        if self.camera is None:
            return None
        frame = self.camera.read()
        return time.time(), frame

    def _maybe_rotate(self, frame: np.ndarray) -> np.ndarray:
        if not self.rotate_180:
            return frame
        try:
            return cv2.rotate(frame, cv2.ROTATE_180)
        except Exception:
            return frame

    def _send_base_action(self, action: Dict[str, float]) -> None:
        try:
            with self.robot_lock:
                self.robot.send_base_action(action)
        except Exception:
            # Best-effort safety: never raise out of the sentry loop.
            pass

    def _event_loop(self):
        dt = 1.0 / float(self.fps)
        debug = os.getenv("LEKIWI_SENTRY_DEBUG", "0").strip() == "1"
        last_debug = 0.0
        last_nonzero_theta = 0.0
        last_frame_ok = 0.0

        while self._running.is_set():
            # --- inbound control events (same pattern as PoseDetectionService) ---
            if self._event_available.wait(timeout=0):
                with self._event_lock:
                    if self._current_event:
                        service_event = self._current_event
                    else:
                        service_event = None
                if service_event is not None:
                    try:
                        self.handle_event(service_event.event_type, service_event.payload)
                    except Exception as e:
                        self.logger.error(f"Error handling inbound event {service_event.event_type}: {e}")
                    finally:
                        with self._event_lock:
                            self._current_event = None
                            self._event_available.clear()
            # -------------------------------------------------------------------

            pulled = None
            try:
                pulled = self._pull_frame(timeout=0.2)
            except Exception:
                pulled = None

            if pulled is None:
                time.sleep(0.01)
                continue

            frame_ts, frame = pulled
            last_frame_ok = time.time()
            frame = self._maybe_rotate(frame)

            # Best-effort viz of front feed (keeps parity with prior pose service UX).
            try:
                if self.viz:
                    self.viz.log_front_rgb(frame, frame_ts)
            except Exception:
                pass

            # Pose inference once; reuse landmarks for both thigh tracking + fall detection.
            try:
                result = self.pose.infer(frame)
                landmarks = (
                    result.pose_landmarks.landmark if result and result.pose_landmarks else None
                )
            except Exception:
                landmarks = None

            h, w = frame.shape[:2]

            # --- Always-on base tracking (rotation only) ---
            if self._tracking_enabled:
                # Primary: thigh midpoint (hip->knee), fallback: hip-center (more reliable).
                target_x, _target_y, conf, src = _calculate_thigh_midpoint(landmarks, w, h)
                if (target_x is None) or (conf < self.min_conf):
                    hx, hy, hconf, hsrc = _calculate_hip_midpoint(landmarks, w, h)
                    if hx is not None and hconf > 0.0:
                        target_x, _target_y, conf, src = hx, hy, hconf, hsrc

                off_norm = _calculate_thigh_offset_norm(target_x, w)

                if conf < self.min_conf:
                    # Don't hunt on low-confidence detections.
                    self._send_base_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
                    if debug and (time.time() - last_debug) > 1.0:
                        last_debug = time.time()
                        self.logger.info(
                            "tracking: no target (conf=%.2f < %.2f) rotate_180=%s",
                            float(conf),
                            float(self.min_conf),
                            str(self.rotate_180),
                        )
                elif abs(off_norm) <= self.rotation_deadzone:
                    self._send_base_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
                    if debug and (time.time() - last_debug) > 1.0:
                        last_debug = time.time()
                        self.logger.info(
                            "tracking: centered src=%s (conf=%.2f off_norm=%.3f dz=%.3f)",
                            str(src),
                            float(conf),
                            float(off_norm),
                            float(self.rotation_deadzone),
                        )
                else:
                    offset_pixels = off_norm * (float(w) / 2.0)
                    rotation_angle = _pixels_to_angle_deg(offset_pixels, w)
                    theta_vel = -rotation_angle * self.rotation_kp
                    theta_vel = float(np.clip(theta_vel, -self.max_theta_vel, self.max_theta_vel))
                    self._send_base_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": float(theta_vel)})
                    if abs(theta_vel) > 1e-3:
                        last_nonzero_theta = time.time()
                    if debug and (time.time() - last_debug) > 1.0:
                        last_debug = time.time()
                        self.logger.info(
                            "tracking: rotate src=%s (conf=%.2f off_norm=%.3f theta_vel=%.2f)",
                            str(src),
                            float(conf),
                            float(off_norm),
                            float(theta_vel),
                        )

            # If debug is on, emit a simple camera heartbeat even when pose is bad.
            if debug and (time.time() - last_debug) > 1.0:
                # Don't overwrite the more detailed tracking logs above; only log if they didn't fire.
                last_debug = time.time()
                age = time.time() - last_frame_ok if last_frame_ok else -1.0
                self.logger.info("heartbeat: frame_ok_age_s=%.2f shape=%s", float(age), str((h, w)))

            # --- Fall detection ---
            event = None
            try:
                if landmarks:
                    event = self.fall_detector.detect(landmarks)
            except Exception:
                event = None

            now = time.time()
            if now < self._fall_freeze_until:
                is_fall = True
            else:
                is_fall = bool(event.is_fall) if event else False
                if is_fall:
                    self._fall_freeze_until = now + max(0.0, self.fall_freeze_s)

            if event is not None:
                self._last_event = SentryDetails(score=float(event.score), ratio=float(event.ratio), timestamp=float(event.timestamp))

            if is_fall and not self._fall_latched:
                self._fall_latched = True
                # Stop rotating before we hand off.
                self._send_base_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
                # Disable tracking until orchestrator restarts us.
                self._tracking_enabled = False

                details: Dict[str, Any] = {
                    "score": float(self._last_event.score) if self._last_event else (float(event.score) if event else 1.0),
                    "ratio": float(self._last_event.ratio) if self._last_event else (float(event.ratio) if event else 0.0),
                    "timestamp": float(self._last_event.timestamp) if self._last_event else time.time(),
                }
                try:
                    self.on_fall(details)
                except Exception as e:
                    self.logger.error(f"on_fall callback failed: {e}")

            # Cooperative pacing
            time.sleep(dt)


