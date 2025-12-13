"""
Navigation module for LeKiwi robot - handles person detection, alignment, and approach.
"""

import cv2
import time
import torch
import numpy as np
import mediapipe as mp

# LeKiwi imports
from lekiwi.services.pose_detection.pose_service import PoseEstimator

# Import MonoPilot from combined_viewer
from nav.combined_viewer import MonoPilot

# MediaPipe pose landmark indices
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26


class Navigator:
    """
    Navigation controller for LeKiwi robot.

    Handles complete navigation sequence: find person, align with thigh, approach to distance.
    Designed to be called synchronously from workflow tools.
    """

    def __init__(self, robot):
        """
        Initialize navigator with robot instance.

        Args:
            robot: Connected LeKiwi robot instance
        """
        self.robot = robot

        # Initialize components
        self.pose_estimator = PoseEstimator()
        self.mono_pilot = MonoPilot()
        self.cap = cv2.VideoCapture(0)  # Use camera 0 for workflow integration
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # Navigation parameters (can be made configurable later)
        self.target_distance = 75.0  # cm
        self.drive_speed = 0.15  # m/s
        self.rotation_kp = 1.5
        self.rotation_deadzone = 0.08
        self.consecutive_frames = 1
        self.alignment_timeout = 10.0  # seconds

    def navigate_to_person(self) -> str:
        """
        Complete navigation sequence: find person, align with thigh, approach to distance.

        This method blocks until navigation is complete or fails.

        Returns:
            str: Success message or failure reason
        """
        try:
            print("Starting navigation to person...")

            # Phase 1: Find and align with person
            if not self._find_and_align_person():
                return "Could not find or align with person"

            # Phase 2: Approach to safe distance
            if not self._approach_to_safe_distance():
                return "Could not approach to safe distance"

            return "Successfully navigated to person"

        except Exception as e:
            return f"Navigation failed: {str(e)}"
        finally:
            self.cap.release()

    def _find_and_align_person(self) -> bool:
        """
        Find person and rotate to face their thigh.

        Returns:
            bool: True if alignment successful, False if timeout or failure
        """
        print("Phase 1: Finding and aligning with person...")
        start_time = time.time()
        last_tracked_side = None

        while time.time() - start_time < self.alignment_timeout:
            ok, frame = self.cap.read()
            if not ok:
                print("Camera read failed during alignment")
                return False

            h, w = frame.shape[:2]

            # Pose detection
            result = self.pose_estimator.infer(frame)
            landmarks = (
                result.pose_landmarks.landmark
                if result and result.pose_landmarks
                else None
            )

            # Calculate thigh midpoint
            thigh_x, thigh_y, confidence, side = self._calculate_thigh_midpoint(landmarks, w, h)

            if side:
                last_tracked_side = side

            # Check if we have a valid person detection
            if thigh_x is not None and confidence > 0:
                # Calculate offset and rotation
                offset_pixels, normalized_offset = self._calculate_thigh_offset(thigh_x, w)

                # Check if we're aligned
                if abs(normalized_offset) <= self.rotation_deadzone:
                    print(f"Alignment complete! Aligned with {last_tracked_side} thigh.")
                    return True

                # Rotate towards the person
                rotation_angle = self._pixels_to_angle(offset_pixels, w)
                rotation_speed = -rotation_angle * self.rotation_kp
                rotation_speed = np.clip(rotation_speed, -20.0, 20.0)

                # Send rotation command
                try:
                    self.robot.send_base_action({
                        "x.vel": 0.0,
                        "y.vel": 0.0,
                        "theta.vel": rotation_speed,
                    })
                except Exception as e:
                    print(f"Error sending rotation command: {e}")
                    return False

            # Small delay to prevent overwhelming the robot
            time.sleep(0.05)

        # Timeout
        print(".1f")
        return False

    def _approach_to_safe_distance(self) -> bool:
        """
        Drive forward until reaching target distance.

        Returns:
            bool: True if approach successful, False if failure
        """
        print("Phase 2: Approaching to safe distance...")
        consecutive_below_threshold = 0

        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("Camera read failed during approach")
                return False

            # Get depth information
            vx, vy, omega, depth_map, scores = self.mono_pilot.process_frame(frame)

            # Calculate distance
            mode_depth = self._get_mode_distance(depth_map)
            mode_distance = self._estimate_distance_from_depth(mode_depth)

            # Check distance condition
            if mode_distance < 60.0:
                consecutive_below_threshold += 1
                print(".1f")
            else:
                consecutive_below_threshold = 0  # Reset counter

            # Check if we've reached the target
            if consecutive_below_threshold >= self.consecutive_frames:
                print(".1f")
                return True

            # Continue driving forward
            try:
                self.robot.send_base_action({
                    "x.vel": self.drive_speed,
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                })
            except Exception as e:
                print(f"Error sending drive command: {e}")
                return False

            # Small delay
            time.sleep(0.05)

    def _calculate_thigh_midpoint(self, landmarks, frame_width, frame_height):
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
        candidates.sort(key=lambda x: x[2], reverse=True)
        best_x, best_y, best_conf, side = candidates[0]

        return best_x, best_y, best_conf, side

    def _calculate_thigh_offset(self, thigh_x, frame_width):
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

    def _pixels_to_angle(self, offset_pixels, frame_width, camera_fov_deg=60):
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

    def _estimate_distance_from_depth(self, depth_value):
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

    def _get_mode_distance(self, depth_map, center_region_ratio=0.3, bin_size=10):
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
