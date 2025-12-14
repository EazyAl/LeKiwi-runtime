"""
Combined Navigation: Face thigh direction, then drive forward to target distance.

Combines thigh tracking for alignment with distance-based driving for approach.
Phase 1: Rotate to face person's thigh
Phase 2: Drive forward until reaching target distance
"""

import cv2
import time
import torch
import numpy as np
import argparse
import mediapipe as mp
import os
import threading

import logging
import sys
from typing import Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
from lekiwi.robot.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
from lekiwi.services.pose_detection.pose_service import PoseEstimator

# Import MonoPilot from combined_viewer (package-relative import)
from .combined_viewer import MonoPilot

# MediaPipe pose landmark indices
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26


def calculate_thigh_midpoint(landmarks, frame_width, frame_height):
    """
    Calculate thigh midpoint from pose landmarks.

    Args:
        landmarks: MediaPipe pose landmarks
        frame_width: Width of camera frame in pixels
        frame_height: Height of camera frame in pixels

    Returns:
        tuple: (midpoint_x, midpoint_y, confidence_score, side) or (None, None, 0, None) if not detected
    """
    if not landmarks:
        return None, None, 0, None

    # Try to get both left and right thigh midpoints
    candidates = []

    # Left thigh
    if (
        len(landmarks) > LEFT_HIP
        and len(landmarks) > LEFT_KNEE
        and landmarks[LEFT_HIP].visibility > 0.5
        and landmarks[LEFT_KNEE].visibility > 0.5
    ):

        hip_x = landmarks[LEFT_HIP].x * frame_width
        hip_y = landmarks[LEFT_HIP].y * frame_height
        knee_x = landmarks[LEFT_KNEE].x * frame_width
        knee_y = landmarks[LEFT_KNEE].y * frame_height

        midpoint_x = (hip_x + knee_x) / 2
        midpoint_y = (hip_y + knee_y) / 2
        confidence = (
            landmarks[LEFT_HIP].visibility + landmarks[LEFT_KNEE].visibility
        ) / 2

        candidates.append((midpoint_x, midpoint_y, confidence, "left"))

    # Right thigh
    if (
        len(landmarks) > RIGHT_HIP
        and len(landmarks) > RIGHT_KNEE
        and landmarks[RIGHT_HIP].visibility > 0.5
        and landmarks[RIGHT_KNEE].visibility > 0.5
    ):

        hip_x = landmarks[RIGHT_HIP].x * frame_width
        hip_y = landmarks[RIGHT_HIP].y * frame_height
        knee_x = landmarks[RIGHT_KNEE].x * frame_width
        knee_y = landmarks[RIGHT_KNEE].y * frame_height

        midpoint_x = (hip_x + knee_x) / 2
        midpoint_y = (hip_y + knee_y) / 2
        confidence = (
            landmarks[RIGHT_HIP].visibility + landmarks[RIGHT_KNEE].visibility
        ) / 2

        candidates.append((midpoint_x, midpoint_y, confidence, "right"))

    if not candidates:
        return None, None, 0, None

    # Choose the most confident/recently seen thigh
    # Sort by confidence, pick the highest
    candidates.sort(key=lambda x: x[2], reverse=True)
    best_x, best_y, best_conf, side = candidates[0]

    return best_x, best_y, best_conf, side


def calculate_thigh_offset(thigh_x, frame_width):
    """
    Calculate horizontal offset of thigh midpoint from frame center.

    Args:
        thigh_x: X coordinate of thigh midpoint
        frame_width: Width of camera frame in pixels

    Returns:
        offset_pixels: Horizontal pixel offset from center (positive = thigh right of center)
        normalized_offset: Offset normalized to [-1, 1] range
    """
    if thigh_x is None:
        return 0, 0

    frame_center_x = frame_width / 2
    offset_pixels = thigh_x - frame_center_x
    normalized_offset = offset_pixels / (frame_width / 2)

    return offset_pixels, normalized_offset


def pixels_to_angle(offset_pixels, frame_width, camera_fov_deg=20):
    """
    Convert pixel offset to angular offset in degrees.

    Args:
        offset_pixels: Pixel offset from frame center
        frame_width: Frame width in pixels
        camera_fov_deg: Camera horizontal field of view in degrees

    Returns:
        angle_deg: Angular offset in degrees (positive = rotate clockwise/right)
    """
    fov_rad = np.radians(camera_fov_deg)
    angle_per_pixel = fov_rad / frame_width
    angle_rad = offset_pixels * angle_per_pixel
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def estimate_distance_from_depth(depth_value):
    """
    Convert MiDaS depth value to approximate distance in centimeters.

    Based on empirical testing with MiDaS_small model:
    - ~800+ : Very close (< 30cm)
    - ~600-800 : Close (30-50cm)
    - ~400-600 : Medium distance (50-100cm)
    - ~200-400 : Far (100cm+)

    Args:
        depth_value: Raw depth value from MiDaS

    Returns:
        distance_cm: Estimated distance in centimeters
    """
    # Empirically derived mapping (needs calibration for your setup)
    if depth_value > 750:
        distance_cm = max(15, 50 - (depth_value - 750) * 0.2)  # 15-50cm
    elif depth_value > 600:
        distance_cm = 50 + (750 - depth_value) * 0.1  # 50-65cm
    elif depth_value > 400:
        distance_cm = 65 + (600 - depth_value) * 0.15  # 65-95cm
    elif depth_value > 200:
        distance_cm = 95 + (400 - depth_value) * 0.25  # 95-145cm
    else:
        distance_cm = 145 + (200 - depth_value) * 0.5  # 145cm+

    return distance_cm


def get_center_distance(depth_map, center_region_ratio=0.3):
    """
    Get the average depth distance in the center region of the depth map.

    Args:
        depth_map: Depth map from MiDaS
        center_region_ratio: Fraction of frame to consider as "center" (0.3 = 30%)

    Returns:
        avg_depth: Average depth value in center region
    """
    h, w = depth_map.shape

    # Define center region
    center_y_start = int(h * (0.5 - center_region_ratio / 2))
    center_y_end = int(h * (0.5 + center_region_ratio / 2))
    center_x_start = int(w * (0.5 - center_region_ratio / 2))
    center_x_end = int(w * (0.5 + center_region_ratio / 2))

    # Extract center region
    center_region = depth_map[center_y_start:center_y_end, center_x_start:center_x_end]

    # Return average depth (higher values = closer)
    return np.mean(center_region)


def get_mode_distance(depth_map, center_region_ratio=0.3, bin_size=10):
    """
    Get the mode (most common) depth distance in the center region.

    Args:
        depth_map: Depth map from MiDaS
        center_region_ratio: Fraction of frame to consider as "center" (0.3 = 30%)
        bin_size: Size of bins for histogram (smaller = more precise)

    Returns:
        mode_depth: Most common depth value in center region
    """
    h, w = depth_map.shape

    # Define center region
    center_y_start = int(h * (0.5 - center_region_ratio / 2))
    center_y_end = int(h * (0.5 + center_region_ratio / 2))
    center_x_start = int(w * (0.5 - center_region_ratio / 2))
    center_x_end = int(w * (0.5 + center_region_ratio / 2))

    # Extract center region
    center_region = depth_map[center_y_start:center_y_end, center_x_start:center_x_end]

    # Flatten and create histogram to find mode
    flat_depths = center_region.flatten()

    # Create bins and find the most common depth value
    if len(flat_depths) == 0:
        return 0

    # Use histogram to find mode
    hist, bin_edges = np.histogram(
        flat_depths, bins=np.arange(0, np.max(flat_depths) + bin_size, bin_size)
    )
    mode_bin_idx = np.argmax(hist)
    mode_depth = (bin_edges[mode_bin_idx] + bin_edges[mode_bin_idx + 1]) / 2

    return mode_depth


class CombinedNavigator:
    """
    Importable navigation helper based on this script's logic (align to thigh, then approach).

    Designed for orchestrators like `sentry.py`:
    - No argparse / no OpenCV UI
    - Can consume frames from a `CameraHub` subscription (preferred), or open its own cv2 camera.
    """

    def __init__(
        self,
        robot: Any,
        *,
        robot_lock: Optional[Any] = None,
        frame_subscription: Optional[Any] = None,
        rotate_180: bool = True,
        viz: Optional[Any] = None,
        camera_index: Optional[int] = None,
    ) -> None:
        self.robot = robot
        self.robot_lock = robot_lock
        self.frame_subscription = frame_subscription
        self._use_external_frames = frame_subscription is not None
        self.rotate_180 = rotate_180
        self.viz = viz

        # Models
        self.pose = PoseEstimator()
        self.mono_pilot = MonoPilot()

        # Tracking state (updated by `track_step`; can be used as a warm-start for navigation)
        self._track_lock = threading.Lock()
        self._track_ts: Optional[float] = None
        self._track_frame_shape: Optional[Tuple[int, int]] = None  # (h, w)
        self._track_side: Optional[str] = None
        self._track_conf: float = 0.0
        self._track_thigh_xy: Optional[Tuple[float, float]] = None
        self._track_offset_norm: float = 0.0
        self._track_est_distance_cm: Optional[float] = None

        # Local camera (only if we are not provided external frames)
        self.cap: Optional[cv2.VideoCapture] = None
        if not self._use_external_frames:
            if camera_index is None:
                # Prefer env override; fall back to the historic default used by this script.
                camera_index = int(os.getenv("NAV_CAMERA_INDEX", "4"))
            self.cap = cv2.VideoCapture(int(camera_index))
            try:
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            except Exception:
                pass

    # --- Tracking-only mode (no driving) ---
    def track_step(self, frame_bgr: np.ndarray, ts: Optional[float] = None) -> None:
        """
        Update internal tracking estimates from a frame.

        This method NEVER commands the robot. It is meant to run continuously so models stay
        warm and we have a last-known thigh position/offset and approximate distance ready
        when a fall event triggers navigation.
        """
        if frame_bgr is None:
            return
        if ts is None:
            ts = time.time()

        frame = self._maybe_rotate(frame_bgr)
        h, w = frame.shape[:2]

        # Pose-based thigh tracking
        result = self.pose.infer(frame)
        landmarks = (
            result.pose_landmarks.landmark if result and result.pose_landmarks else None
        )
        thigh_x, thigh_y, conf, side = calculate_thigh_midpoint(landmarks, w, h)
        _offset_px, offset_norm = calculate_thigh_offset(thigh_x, w)

        # Depth-based distance (sample around thigh midpoint when possible)
        est_cm: Optional[float] = None
        try:
            _, _, _, depth_map, _scores = self.mono_pilot.process_frame(frame)
            use_x = thigh_x
            use_y = thigh_y
            if use_x is not None and use_y is not None:
                tx, ty = int(use_x), int(use_y)
                y_min, y_max = max(0, ty - 5), min(h, ty + 5)
                x_min, x_max = max(0, tx - 5), min(w, tx + 5)
                region = depth_map[y_min:y_max, x_min:x_max]
                if region.size > 0:
                    depth_val = float(np.mean(region))
                    est_cm = float(estimate_distance_from_depth(depth_val))
            # Optional best-effort viz
            self._log_depth_best_effort(depth_map, ts)
        except Exception:
            est_cm = None

        with self._track_lock:
            self._track_ts = ts
            self._track_frame_shape = (h, w)
            if side:
                self._track_side = side
            self._track_conf = float(conf or 0.0)
            if thigh_x is not None and thigh_y is not None:
                self._track_thigh_xy = (float(thigh_x), float(thigh_y))
            self._track_offset_norm = float(offset_norm or 0.0)
            if est_cm is not None:
                self._track_est_distance_cm = float(est_cm)

    def get_tracking_snapshot(self) -> dict:
        """Return a shallow snapshot of the last tracking values (for logging/debug)."""
        with self._track_lock:
            return {
                "ts": self._track_ts,
                "frame_shape": self._track_frame_shape,
                "side": self._track_side,
                "conf": self._track_conf,
                "thigh_xy": self._track_thigh_xy,
                "offset_norm": self._track_offset_norm,
                "est_distance_cm": self._track_est_distance_cm,
            }

    def center_person_step(
        self,
        *,
        rotation_kp: float = 1.5,
        rotation_deadzone: float = 0.08,
        max_theta_vel: float = 20.0,
        min_conf: float = 0.5,
        tracking_fresh_s: float = 0.5,
    ) -> bool:
        """
        Use the most recent tracking snapshot to rotate the robot base to keep the
        person centered. This method ONLY rotates (no translation).

        Returns:
            bool: True if a command was sent, False otherwise.
        """
        snap = self.get_tracking_snapshot()
        ts = snap.get("ts")
        shape = snap.get("frame_shape")
        conf = float(snap.get("conf") or 0.0)
        off_norm = float(snap.get("offset_norm") or 0.0)

        if ts is None or shape is None:
            return False
        try:
            if (time.time() - float(ts)) > tracking_fresh_s:
                return False
        except Exception:
            return False

        if conf < min_conf:
            # Don't hunt/oscillate on low-confidence detections.
            self._send_base_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
            return False

        if abs(off_norm) <= rotation_deadzone:
            # Already centered: send explicit zero to prevent stale rotation commands.
            self._send_base_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
            return True

        h, w = shape
        offset_pixels = off_norm * (float(w) / 2.0)
        rotation_angle = pixels_to_angle(offset_pixels, float(w))
        theta_vel = -rotation_angle * float(rotation_kp)
        theta_vel = float(np.clip(theta_vel, -max_theta_vel, max_theta_vel))

        self._send_base_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": theta_vel})
        return True

    def _pull_frame(self, timeout: float = 0.2) -> Optional[Tuple[float, np.ndarray]]:
        if self._use_external_frames and self.frame_subscription:
            pulled = self.frame_subscription.pull(timeout=timeout)
            if pulled is None:
                return None
            return pulled

        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return time.time(), frame

    def _maybe_rotate(self, frame: np.ndarray) -> np.ndarray:
        if not self.rotate_180:
            return frame
        try:
            return cv2.rotate(frame, cv2.ROTATE_180)
        except Exception:
            return frame

    def _send_base_action(self, action: dict) -> None:
        if self.robot_lock is None:
            self.robot.send_base_action(action)
            return
        try:
            with self.robot_lock:
                self.robot.send_base_action(action)
        except TypeError:
            # If robot_lock isn't a context manager, fall back to unlocked.
            self.robot.send_base_action(action)

    def _stop_base(self) -> None:
        if self.robot_lock is None:
            self.robot.stop_base()
            return
        try:
            with self.robot_lock:
                self.robot.stop_base()
        except TypeError:
            self.robot.stop_base()

    def _send_stop(self) -> None:
        """
        Prefer sending a zero-velocity command over `stop_base()` during normal operation.
        `stop_base()` is still used for final cleanup as a safety net.
        """
        try:
            self._send_base_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        except Exception:
            pass

    def _log_depth_best_effort(self, depth_map: np.ndarray, ts: float) -> None:
        if self.viz is None or depth_map is None:
            return
        try:
            depth_norm = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
            self.viz.log_depth(depth_color, ts=ts)
        except Exception:
            pass

    def navigate_to_person(
        self,
        *,
        target_distance_cm: float = 25.0,
        drive_speed: float = 0.15,
        rotation_kp: float = 1.5,
        rotation_deadzone: float = 0.08,
        consecutive_frames: int = 1,
        alignment_timeout_s: float = 10.0,
        approach_timeout_s: float = 6.5,
        backoff_duration_s: float = 1.0,
        hard_stop_on_finish: bool = False,
        warm_start_from_tracking: bool = True,
        tracking_fresh_s: float = 0.75,
    ) -> str:
        """
        Align to a detected thigh, then drive forward until within target distance.

        Returns a human-readable status string.
        """
        last_thigh_x = None
        last_thigh_y = None
        last_tracked_side = None
        did_stop_base = False
        success = False

        try:
            # --- Optional warm-start: if tracking is fresh + already aligned, skip alignment ---
            if warm_start_from_tracking:
                snap = self.get_tracking_snapshot()
                snap_ts = snap.get("ts")
                snap_off = snap.get("offset_norm", 0.0) or 0.0
                if (
                    isinstance(snap_ts, (int, float))
                    and (time.time() - float(snap_ts)) <= tracking_fresh_s
                    and abs(float(snap_off)) <= rotation_deadzone
                ):
                    # Already aligned recently; proceed to approach.
                    pass
                else:
                    snap = None
            else:
                snap = None

            # --- Phase 1: Alignment ---
            start_align = time.monotonic()
            if snap is None:
                while (time.monotonic() - start_align) < alignment_timeout_s:
                    pulled = self._pull_frame(timeout=0.2)
                    if pulled is None:
                        continue
                    frame_ts, frame = pulled
                    frame = self._maybe_rotate(frame)

                    h, w = frame.shape[:2]
                    result = self.pose.infer(frame)
                    landmarks = (
                        result.pose_landmarks.landmark
                        if result and result.pose_landmarks
                        else None
                    )

                    thigh_x, thigh_y, confidence, side = calculate_thigh_midpoint(
                        landmarks, w, h
                    )

                    if side:
                        last_tracked_side = side
                    if thigh_x is not None and thigh_y is not None:
                        last_thigh_x, last_thigh_y = thigh_x, thigh_y

                    if thigh_x is None or confidence <= 0:
                        time.sleep(0.03)
                        continue

                    offset_pixels, normalized_offset = calculate_thigh_offset(thigh_x, w)
                    if abs(normalized_offset) <= rotation_deadzone:
                        break

                    rotation_angle = pixels_to_angle(offset_pixels, w)
                    rotation_speed = -rotation_angle * rotation_kp
                    rotation_speed = float(np.clip(rotation_speed, -20.0, 20.0))
                    self._send_base_action(
                        {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": rotation_speed}
                    )
                    time.sleep(0.05)
                else:
                    return "Alignment timeout - could not find/align to person"

            # Stop any residual rotation before approach (avoid spamming stop_base logs)
            self._send_stop()

            # --- Phase 2: Approach ---
            start_approach = time.monotonic()
            below = 0
            while True:
                if (time.monotonic() - start_approach) > approach_timeout_s:
                    # --- Phase 3: Safety backoff ---
                    backoff_start = time.monotonic()
                    while (time.monotonic() - backoff_start) < backoff_duration_s:
                        self._send_base_action(
                            {"x.vel": -abs(drive_speed), "y.vel": 0.0, "theta.vel": 0.0}
                        )
                        time.sleep(0.05)
                    self._send_stop()
                    return "Approach timeout - backed off for safety"

                pulled = self._pull_frame(timeout=0.2)
                if pulled is None:
                    continue
                frame_ts, frame = pulled
                frame = self._maybe_rotate(frame)
                h, w = frame.shape[:2]

                # Update thigh tracking (best-effort)
                result = self.pose.infer(frame)
                landmarks = (
                    result.pose_landmarks.landmark
                    if result and result.pose_landmarks
                    else None
                )
                thigh_x, thigh_y, confidence, side = calculate_thigh_midpoint(
                    landmarks, w, h
                )
                if side:
                    last_tracked_side = side
                if thigh_x is not None and thigh_y is not None:
                    last_thigh_x, last_thigh_y = thigh_x, thigh_y

                # Depth inference
                _, _, _, depth_map, _scores = self.mono_pilot.process_frame(frame)
                self._log_depth_best_effort(depth_map, frame_ts)

                # Sample depth around the thigh point (or last known)
                use_x = thigh_x if thigh_x is not None else last_thigh_x
                use_y = thigh_y if thigh_y is not None else last_thigh_y
                if use_x is None or use_y is None:
                    # No target point yet; keep creeping slowly forward.
                    self._send_base_action(
                        {"x.vel": drive_speed, "y.vel": 0.0, "theta.vel": 0.0}
                    )
                    time.sleep(0.05)
                    continue

                tx, ty = int(use_x), int(use_y)
                y_min, y_max = max(0, ty - 5), min(h, ty + 5)
                x_min, x_max = max(0, tx - 5), min(w, tx + 5)
                region = depth_map[y_min:y_max, x_min:x_max]
                if region.size == 0:
                    self._send_base_action(
                        {"x.vel": drive_speed, "y.vel": 0.0, "theta.vel": 0.0}
                    )
                    time.sleep(0.05)
                    continue

                depth_val = float(np.mean(region))
                est_cm = float(estimate_distance_from_depth(depth_val))

                if est_cm <= target_distance_cm:
                    below += 1
                else:
                    below = 0

                if below >= max(1, int(consecutive_frames)):
                    self._send_stop()
                    success = True
                    return (
                        f"Successfully navigated to person (side={last_tracked_side}, "
                        f"distance={est_cm:.1f}cm)"
                    )

                # Continue approach
                self._send_base_action(
                    {"x.vel": drive_speed, "y.vel": 0.0, "theta.vel": 0.0}
                )
                time.sleep(0.05)

        except Exception as e:
            # Best-effort stop on error (avoid duplicates)
            self._send_stop()
            if not did_stop_base:
                try:
                    self._stop_base()
                    did_stop_base = True
                except Exception:
                    pass
            return f"Navigation failed: {e}"
        finally:
            # Best-effort cleanup:
            # - Always send a 0-velocity command so the robot doesn't continue on stale commands.
            # - Only issue a "hard stop" (stop_base) if requested or if we did not succeed.
            self._send_stop()
            if (hard_stop_on_finish or not success) and not did_stop_base:
                try:
                    self._stop_base()
                    did_stop_base = True
                except Exception:
                    pass
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(
        description="Combined navigation: align with thigh, then approach"
    )
    parser.add_argument("--flip", action="store_true", help="Flip camera 180 degrees")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for the robot",
    )
    parser.add_argument("--id", type=str, default="biden_kiwi", help="ID of the robot")
    parser.add_argument(
        "--target_distance",
        type=float,
        default=75.0,
        help="Target distance in cm to stop at",
    )
    parser.add_argument(
        "--drive_speed",
        type=float,
        default=0.15,
        help="Forward driving speed (m/s)",
    )
    parser.add_argument(
        "--rotation_kp",
        type=float,
        default=5,
        help="Proportional gain for thigh rotation control",
    )
    parser.add_argument(
        "--rotation_deadzone",
        type=float,
        default=0.08,
        help="Deadzone for thigh rotation (normalized offset)",
    )
    parser.add_argument(
        "--consecutive_frames",
        type=int,
        default=1,
        help="Number of consecutive frames mode distance must be below threshold",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibration mode - shows depth values without driving",
    )
    args = parser.parse_args()

    print("Initializing Combined Navigation System...")

    # Initialize camera
    cap = cv2.VideoCapture(4)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize mediapipe pose detection
    pose = PoseEstimator()

    # Initialize MiDaS depth estimation
    print("Initializing MiDaS depth model...")
    mono_pilot = MonoPilot()

    # Initialize LeKiwi robot
    try:
        config = LeKiwiConfig(port=args.port, id=args.id, cameras={})
        robot = LeKiwi(config)
        robot.connect()
        logger.info(f"Robot connected successfully on port {args.port}")
    except Exception as e:
        # `lerobot-find-port` identifies the *device path* by unplug/replug, but
        # doesn't guarantee this process can open it (permissions/busy port) or
        # that the controller is responding (protocol/baud).
        logger.exception(f"Failed to connect to robot on port {args.port}")

        def _find_in_exc_chain(
            exc: BaseException, predicate
        ) -> Optional[BaseException]:
            cur: Optional[BaseException] = exc
            seen: set[int] = set()
            while cur is not None and id(cur) not in seen:
                seen.add(id(cur))
                try:
                    if predicate(cur):
                        return cur
                except Exception:
                    pass
                cur = cur.__cause__ or cur.__context__
            return None

        # Best-effort Linux diagnostics (never crash on non-Linux).
        try:
            if os.path.exists(args.port):
                can_rw = os.access(args.port, os.R_OK | os.W_OK)
                logger.info(
                    "Serial device exists: %s (read/write access for current user: %s)",
                    args.port,
                    can_rw,
                )
                perm = _find_in_exc_chain(
                    e,
                    lambda ex: isinstance(ex, PermissionError)
                    or ("Permission denied" in str(ex))
                    or (getattr(ex, "errno", None) == 13),
                )
                if perm is not None or not can_rw:
                    logger.info(
                        "Hint: this looks like a permissions issue opening %s.\n"
                        "Fix (recommended): add your user to the serial group, then re-login:\n"
                        "  sudo usermod -aG dialout $USER\n"
                        "  # log out/in (or reboot)\n"
                        "Temporary (until unplug/replug):\n"
                        "  sudo chmod a+rw %s\n"
                        "If your distro uses a different group, check with:\n"
                        "  ls -l %s",
                        args.port,
                        args.port,
                        args.port,
                    )
            else:
                logger.info("Serial device path does not exist: %s", args.port)
        except Exception:
            pass
        logger.info("Running in camera-only mode (no robot control)")
        robot = None

    # Navigation states
    PHASE_ALIGNMENT = "ALIGNMENT"  # Rotating to face thigh
    PHASE_APPROACH = "APPROACH"  # Driving forward to target
    PHASE_BACKOFF = "BACKOFF"  # Safety reverse after timeout
    PHASE_COMPLETE = "COMPLETE"  # Finished

    current_phase = PHASE_ALIGNMENT
    last_tracked_side = None
    last_thigh_x = None
    last_thigh_y = None
    consecutive_below_threshold = 0
    alignment_start_time = time.time()
    alignment_timeout = 15.0  # Give up alignment after 10 seconds

    # Use monotonic time for safety timers (robust to system clock changes)
    approach_start_time = None
    backoff_start_time = None
    approach_timeout = 8.0  # Max time to drive forward
    backoff_duration = 1  # Duration to reverse

    logger.info(
        "Starting combined navigation. Press 'q' to quit, 's' to stop immediately."
    )

    try:
        while cap.isOpened():
            now_mono = time.monotonic()
            ok, frame = cap.read()
            if not ok:
                print("Error: Could not read frame from camera")
                break

            # Flip camera frame 180 degrees
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            h, w = frame.shape[:2]

            # Pose detection for thigh tracking
            result = pose.infer(frame)
            landmarks = (
                result.pose_landmarks.landmark
                if result and result.pose_landmarks
                else None
            )

            # Calculate thigh midpoint
            thigh_x, thigh_y, confidence, side = calculate_thigh_midpoint(
                landmarks, w, h
            )

            # Update tracking continuity
            if side:
                last_tracked_side = side

            # Update last known thigh position
            if thigh_x is not None and thigh_y is not None:
                last_thigh_x = thigh_x
                last_thigh_y = thigh_y

            # Get depth information for distance measurement
            vx, vy, omega, depth_map, scores = mono_pilot.process_frame(frame)

            # Calculate center distance
            center_depth = get_center_distance(depth_map)

            # Use the average depth of the 10 pixels surrounding the thigh midpoint
            target_depth = 0.0

            # Determine which coordinates to use: current or last known
            use_x, use_y = None, None
            if thigh_x is not None and thigh_y is not None:
                use_x, use_y = thigh_x, thigh_y
            elif (
                current_phase == PHASE_APPROACH
                and last_thigh_x is not None
                and last_thigh_y is not None
            ):
                # Fallback to last known position during approach
                use_x, use_y = last_thigh_x, last_thigh_y
                logger.debug(
                    f"Lost thigh detection, using last known pos: ({use_x:.1f}, {use_y:.1f})"
                )

            if use_x is not None and use_y is not None:
                # Defined as a 10x10 region (+/- 5 pixels) around the point
                tx, ty = int(use_x), int(use_y)
                y_min, y_max = max(0, ty - 5), min(h, ty + 5)
                x_min, x_max = max(0, tx - 5), min(w, tx + 5)

                region = depth_map[y_min:y_max, x_min:x_max]
                if region.size > 0:
                    target_depth = np.mean(region)

            mode_depth = target_depth
            estimated_distance = estimate_distance_from_depth(center_depth)
            mode_distance = estimate_distance_from_depth(mode_depth)

            # Calculate thigh position offset
            offset_pixels, normalized_offset = calculate_thigh_offset(thigh_x, w)
            rotation_angle = pixels_to_angle(offset_pixels, w)

            # Control logic
            robot_command = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

            if not args.calibrate and robot is not None:
                if current_phase == PHASE_ALIGNMENT:
                    # Phase 1: Align with thigh
                    if thigh_x is not None and confidence > 0:
                        # Person detected - rotate to face them
                        if abs(normalized_offset) > args.rotation_deadzone:
                            rotation_speed = -rotation_angle * args.rotation_kp
                            rotation_speed = np.clip(rotation_speed, -20.0, 20.0)

                            logger.debug(
                                f"Aligning: Offset={normalized_offset:.3f}, CmdRot={rotation_speed:.1f}"
                            )

                            robot_command = {
                                "x.vel": 0.0,
                                "y.vel": 0.0,
                                "theta.vel": rotation_speed,
                            }
                        else:
                            # Aligned! Switch to approach phase
                            logger.info(
                                f"Alignment complete! Switching to approach phase. Offset={normalized_offset:.3f}"
                            )
                            current_phase = PHASE_APPROACH
                            approach_start_time = now_mono
                    elif time.time() - alignment_start_time > alignment_timeout:
                        # Timeout - no person detected, give up
                        logger.warning(
                            f"Alignment timeout ({alignment_timeout}s) - no person detected"
                        )
                        current_phase = PHASE_COMPLETE

                elif current_phase == PHASE_APPROACH:
                    # If we somehow entered APPROACH without a start time, set it.
                    if approach_start_time is None:
                        approach_start_time = now_mono

                    # Check for timeout
                    if (now_mono - approach_start_time) > approach_timeout:
                        logger.warning(
                            f"Approach timeout ({approach_timeout}s) reached! Initiating safety backoff."
                        )
                        current_phase = PHASE_BACKOFF
                        backoff_start_time = now_mono

                    # Phase 2: Drive forward to target distance
                    elif mode_distance < 20.0:
                        consecutive_below_threshold += 1
                        logger.debug(
                            f"Close distance detected ({mode_distance:.1f}cm) - {consecutive_below_threshold}/{args.consecutive_frames} frames"
                        )
                    else:
                        consecutive_below_threshold = 0  # Reset counter

                    # Stop only after consecutive frames below threshold
                    if (
                        current_phase == PHASE_APPROACH
                    ):  # Check again in case timeout happened
                        if consecutive_below_threshold >= args.consecutive_frames:
                            logger.info(
                                f"Target reached at {mode_distance:.1f}cm after {consecutive_below_threshold} consecutive frames"
                            )
                            robot_command = {
                                "x.vel": 0.0,
                                "y.vel": 0.0,
                                "theta.vel": 0.0,
                            }
                            current_phase = PHASE_COMPLETE
                        else:
                            # Continue driving forward
                            robot_command = {
                                "x.vel": args.drive_speed,
                                "y.vel": 0.0,
                                "theta.vel": 0.0,
                            }

                elif current_phase == PHASE_BACKOFF:
                    if backoff_start_time is None:
                        backoff_start_time = now_mono

                    if (now_mono - backoff_start_time) < backoff_duration:
                        # Drive backwards
                        robot_command = {
                            "x.vel": -args.drive_speed,  # Negative speed
                            "y.vel": 0.0,
                            "theta.vel": 0.0,
                        }
                    else:
                        logger.info("Safety backoff complete. Navigation finished.")
                        robot_command = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
                        current_phase = PHASE_COMPLETE

                # Send command to robot
                try:
                    # Always send the latest command (including STOP) so the robot doesn't
                    # keep executing a stale command when we enter COMPLETE.
                    robot.send_base_action(robot_command)
                except Exception as e:
                    logger.error(f"Error sending robot command: {e}")

            # Visualization
            vis_frame = frame.copy()

            # Create depth visualization
            depth_norm = cv2.normalize(
                depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

            # Draw center region rectangle
            center_region_ratio = 0.3
            center_y_start = int(h * (0.5 - center_region_ratio / 2))
            center_y_end = int(h * (0.5 + center_region_ratio / 2))
            center_x_start = int(w * (0.5 - center_region_ratio / 2))
            center_x_end = int(w * (0.5 + center_region_ratio / 2))

            cv2.rectangle(
                vis_frame,
                (center_x_start, center_y_start),
                (center_x_end, center_y_end),
                (0, 255, 0),
                2,
            )
            cv2.rectangle(
                depth_color,
                (center_x_start, center_y_start),
                (center_x_end, center_y_end),
                (0, 255, 0),
                2,
            )

            # Draw thigh midpoint if detected (or fallback in approach phase)
            draw_x, draw_y = thigh_x, thigh_y
            is_fallback = False

            if (
                draw_x is None
                and current_phase == PHASE_APPROACH
                and last_thigh_x is not None
            ):
                draw_x, draw_y = last_thigh_x, last_thigh_y
                is_fallback = True

            if draw_x is not None and draw_y is not None:
                # Draw depth sampling region (10x10)
                tx, ty = int(draw_x), int(draw_y)
                box_color = (
                    (0, 0, 255) if not is_fallback else (0, 165, 255)
                )  # Red normally, Orange if fallback

                cv2.rectangle(
                    vis_frame,
                    (tx - 5, ty - 5),
                    (tx + 5, ty + 5),
                    box_color,
                    2,
                )
                cv2.rectangle(
                    depth_color,
                    (tx - 5, ty - 5),
                    (tx + 5, ty + 5),
                    box_color,
                    2,
                )

                # Draw midpoint
                cv2.circle(vis_frame, (int(draw_x), int(draw_y)), 8, (0, 255, 0), -1)

                if is_fallback:
                    cv2.putText(
                        vis_frame,
                        "LOST TRACK - USING LAST POS",
                        (tx + 10, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2,
                    )

                # Draw line from hip to knee (only if real detection)
                if not is_fallback and thigh_x is not None:
                    if side == "left" and landmarks:
                        hip_x_pos = landmarks[LEFT_HIP].x * w
                        hip_y_pos = landmarks[LEFT_HIP].y * h
                        knee_x_pos = landmarks[LEFT_KNEE].x * w
                        knee_y_pos = landmarks[LEFT_KNEE].y * h

                        cv2.line(
                            vis_frame,
                            (int(hip_x_pos), int(hip_y_pos)),
                            (int(knee_x_pos), int(knee_y_pos)),
                            (255, 0, 0),
                            3,
                        )
                    elif side == "right" and landmarks:
                        hip_x_pos = landmarks[RIGHT_HIP].x * w
                        hip_y_pos = landmarks[RIGHT_HIP].y * h
                        knee_x_pos = landmarks[RIGHT_KNEE].x * w
                        knee_y_pos = landmarks[RIGHT_KNEE].y * h

                        cv2.line(
                            vis_frame,
                            (int(hip_x_pos), int(hip_y_pos)),
                            (int(knee_x_pos), int(knee_y_pos)),
                            (255, 0, 0),
                            3,
                        )

                # Draw center line and offset
                frame_center_x = w // 2
                cv2.line(
                    vis_frame,
                    (frame_center_x, 0),
                    (frame_center_x, h),
                    (255, 255, 255),
                    1,
                )

                # Offset line from center to thigh midpoint
                cv2.line(
                    vis_frame,
                    (frame_center_x, int(draw_y)),
                    (int(draw_x), int(draw_y)),
                    (0, 255, 255),
                    2,
                )

            # Add distance overlay on depth map
            cv2.putText(
                depth_color,
                f"Thigh: {mode_distance:.1f}cm",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                depth_color,
                f"Avg: {estimated_distance:.1f}cm",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2,
            )

            # Status text
            if args.calibrate:
                status_color = (255, 255, 0)  # Yellow for calibration
                status_text = "CALIBRATION MODE"
                key_help = "Press 'q' to quit"
            else:
                if current_phase == PHASE_COMPLETE:
                    status_color = (0, 255, 0)  # Green for complete
                elif current_phase == PHASE_APPROACH:
                    status_color = (0, 255, 255)  # Yellow for approach
                else:
                    status_color = (255, 255, 255)  # White for alignment

                phase_status = {
                    PHASE_ALIGNMENT: f"ALIGNING ({last_tracked_side or 'None'})",
                    PHASE_APPROACH: "APPROACHING",
                    PHASE_BACKOFF: "SAFETY BACKOFF",
                    PHASE_COMPLETE: "COMPLETE",
                }[current_phase]

                status_text = phase_status
                key_help = "Press 'q' to quit, 's' to stop immediately"

            status_lines = [
                f"Thigh Offset: {offset_pixels:.1f}px ({normalized_offset:.3f})",
                f"Distance: {mode_distance:.1f}cm",
                f"Consecutive: {consecutive_below_threshold}/{args.consecutive_frames}",
                f"Phase: {status_text}",
                f"Robot: {args.port}" if robot else "Robot: DISCONNECTED",
                key_help,
            ]

            for i, line in enumerate(status_lines):
                cv2.putText(
                    vis_frame,
                    line,
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    status_color,
                    2,
                )

            cv2.imshow("Combined Navigation - Camera", vis_frame)
            cv2.imshow("Combined Navigation - Depth Map", depth_color)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif not args.calibrate and key == ord("s"):
                logger.info("Emergency stop requested")
                if robot:
                    robot.stop_base()
                current_phase = PHASE_COMPLETE
                consecutive_below_threshold = 0

            # Exit if phase is complete
            if current_phase == PHASE_COMPLETE and not args.calibrate:
                logger.info("Navigation complete! Press 'q' to exit.")
                # Keep running to show final status

    finally:
        cap.release()
        cv2.destroyAllWindows()

        if robot is not None:
            print("Stopping robot...")
            robot.stop_base()
            robot.disconnect()
            print("Robot disconnected.")


if __name__ == "__main__":
    main()
