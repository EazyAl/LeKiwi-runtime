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

import logging
import sys
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
from lekiwi.robot.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
from lekiwi.services.pose_detection.pose_service import PoseEstimator

# Import MonoPilot from combined_viewer
from nav.combined_viewer import MonoPilot

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


def main():
    parser = argparse.ArgumentParser(
        description="Combined navigation: align with thigh, then approach"
    )
    parser.add_argument("--flip", action="store_true", help="Flip camera 180 degrees")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/tty.usbmodem58760432781",
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
        default=1.5,
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

        def _find_in_exc_chain(exc: BaseException, predicate) -> Optional[BaseException]:
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
    PHASE_BACKOFF = "BACKOFF"    # Safety reverse after timeout
    PHASE_COMPLETE = "COMPLETE"  # Finished

    current_phase = PHASE_ALIGNMENT
    last_tracked_side = None
    last_thigh_x = None
    last_thigh_y = None
    consecutive_below_threshold = 0
    alignment_start_time = time.time()
    alignment_timeout = 10.0  # Give up alignment after 10 seconds
    
    # Use monotonic time for safety timers (robust to system clock changes)
    approach_start_time = None
    backoff_start_time = None
    approach_timeout = 8.0 # Max time to drive forward
    backoff_duration = 1 # Duration to reverse

    logger.info("Starting combined navigation. Press 'q' to quit, 's' to stop immediately.")

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
            elif current_phase == PHASE_APPROACH and last_thigh_x is not None and last_thigh_y is not None:
                # Fallback to last known position during approach
                use_x, use_y = last_thigh_x, last_thigh_y
                logger.debug(f"Lost thigh detection, using last known pos: ({use_x:.1f}, {use_y:.1f})")

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
                            
                            logger.debug(f"Aligning: Offset={normalized_offset:.3f}, CmdRot={rotation_speed:.1f}")
                            
                            robot_command = {
                                "x.vel": 0.0,
                                "y.vel": 0.0,
                                "theta.vel": rotation_speed,
                            }
                        else:
                            # Aligned! Switch to approach phase
                            logger.info(f"Alignment complete! Switching to approach phase. Offset={normalized_offset:.3f}")
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
                        logger.warning(f"Approach timeout ({approach_timeout}s) reached! Initiating safety backoff.")
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
                    if current_phase == PHASE_APPROACH: # Check again in case timeout happened
                        if consecutive_below_threshold >= args.consecutive_frames:
                            logger.info(
                                f"Target reached at {mode_distance:.1f}cm after {consecutive_below_threshold} consecutive frames"
                            )
                            robot_command = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
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
                            "x.vel": -args.drive_speed, # Negative speed
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
            
            if draw_x is None and current_phase == PHASE_APPROACH and last_thigh_x is not None:
                draw_x, draw_y = last_thigh_x, last_thigh_y
                is_fallback = True

            if draw_x is not None and draw_y is not None:
                # Draw depth sampling region (10x10)
                tx, ty = int(draw_x), int(draw_y)
                box_color = (0, 0, 255) if not is_fallback else (0, 165, 255) # Red normally, Orange if fallback
                
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
                    cv2.putText(vis_frame, "LOST TRACK - USING LAST POS", (tx + 10, ty), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
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
