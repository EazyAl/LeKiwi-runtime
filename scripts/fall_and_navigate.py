"""
Fall Detection and Navigation Script

Combines fall detection with autonomous navigation to fallen persons.
First detects falls using pose detection, then navigates to the person using thigh tracking and depth estimation.

Usage:
    python scripts/fall_and_navigate.py --camera-index 4 --port /dev/tty.usbmodem58760432781

States:
1. FALL_DETECTION: Continuously monitor for falls
2. NAVIGATION: Navigate to the fallen person
"""

import argparse
import cv2
import time
import torch
import numpy as np
import logging
import sys

import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Import components from pose detection
from lekiwi.services.pose_detection.pose_service import (
    CameraStream,
    PoseEstimator,
    FallDetector,
    default_visualizer,
)

# Import robot and navigation components
from lekiwi.robot.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

# Import navigation utilities from nav.py
from nav.nav import (
    calculate_thigh_midpoint,
    calculate_thigh_offset,
    pixels_to_angle,
    estimate_distance_from_depth,
    get_center_distance,
    get_mode_distance,
)

# Import MonoPilot from combined_viewer
from nav.combined_viewer import MonoPilot

# MediaPipe pose landmark indices
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26


def main():
    parser = argparse.ArgumentParser(
        description="Fall detection and autonomous navigation to fallen persons"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=4,
        help="OpenCV camera index (default: 4)",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for the robot",
    )
    parser.add_argument("--id", type=str, default="biden_kiwi", help="ID of the robot")
    parser.add_argument(
        "--target-distance",
        type=float,
        default=50.0,
        help="Target distance in cm to stop at during navigation",
    )
    parser.add_argument(
        "--drive-speed",
        type=float,
        default=0.15,
        help="Forward driving speed (m/s)",
    )
    parser.add_argument(
        "--rotation-kp",
        type=float,
        default=1.5,
        help="Proportional gain for thigh rotation control",
    )
    parser.add_argument(
        "--rotation-deadzone",
        type=float,
        default=0.08,
        help="Deadzone for thigh rotation (normalized offset)",
    )
    parser.add_argument(
        "--consecutive-frames",
        type=int,
        default=5,
        help="Number of consecutive frames mode distance must be below threshold",
    )
    parser.add_argument(
        "--fall-confirmation-frames",
        type=int,
        default=3,
        help="Number of consecutive frames a fall must be detected to trigger navigation",
    )
    parser.add_argument(
        "--navigation-timeout",
        type=float,
        default=8.0,
        help="Maximum time to spend navigating to person",
    )
    args = parser.parse_args()

    print("Initializing Fall Detection and Navigation System...")

    # States
    STATE_FALL_DETECTION = "FALL_DETECTION"
    STATE_NAVIGATION = "NAVIGATION"
    STATE_COMPLETE = "COMPLETE"

    # Navigation phases (from nav.py)
    PHASE_ALIGNMENT = "ALIGNMENT"
    PHASE_APPROACH = "APPROACH"
    PHASE_BACKOFF = "BACKOFF"
    PHASE_COMPLETE = "COMPLETE"

    current_state = STATE_FALL_DETECTION
    navigation_phase = PHASE_ALIGNMENT

    # Initialize camera and pose detection components
    print(f"Using camera index: {args.camera_index}")
    camera = CameraStream(index=args.camera_index)
    pose_estimator = PoseEstimator()
    fall_detector = FallDetector()

    # Initialize MiDaS depth estimation for navigation
    print("Initializing MiDaS depth model...")
    mono_pilot = MonoPilot()

    # Initialize LeKiwi robot
    robot = None
    try:
        config = LeKiwiConfig(port=args.port, id=args.id, cameras={})
        robot = LeKiwi(config)
        robot.connect()
        logger.info(f"Robot connected successfully on port {args.port}")
    except Exception as e:
        logger.error(f"Failed to connect to robot on port {args.port}: {e}")
        logger.info("Running in camera-only mode (no robot control)")
        robot = None

    # State variables for fall detection
    consecutive_fall_frames = 0
    fall_detected_timestamp = None

    # Navigation state variables
    last_tracked_side = None
    last_thigh_x = None
    last_thigh_y = None
    consecutive_below_threshold = 0
    alignment_start_time = time.time()
    approach_start_time = None
    backoff_start_time = None

    # Navigation timeouts
    alignment_timeout = 8.0
    approach_timeout = args.navigation_timeout
    backoff_duration = 1.5

    logger.info("Starting fall detection. Press 'q' to quit.")

    try:
        # Start the camera
        camera.start()

        while True:
            now_mono = time.monotonic()
            # Read frame from camera
            frame = camera.read()

            # Rotate camera frame 180 degrees (flip both vertically and horizontally)
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            h, w = frame.shape[:2]

            # Perform pose estimation
            result = pose_estimator.infer(frame)
            landmarks = (
                result.pose_landmarks.landmark
                if result and result.pose_landmarks
                else None
            )

            if current_state == STATE_FALL_DETECTION:
                # Fall detection mode
                event = None
                is_fall = False
                if landmarks:
                    event = fall_detector.detect(landmarks)
                    if event:
                        is_fall = event.is_fall

                # Check for fall confirmation
                if is_fall:
                    consecutive_fall_frames += 1
                    if consecutive_fall_frames >= args.fall_confirmation_frames:
                        logger.info(f"Fall detected! Confirmed over {consecutive_fall_frames} frames. Switching to navigation mode.")
                        current_state = STATE_NAVIGATION
                        navigation_phase = PHASE_ALIGNMENT
                        fall_detected_timestamp = time.time()
                        alignment_start_time = time.time()
                        consecutive_fall_frames = 0
                else:
                    consecutive_fall_frames = 0

                # Calculate FPS for visualization
                now = time.time()
                fps = 1.0 / max(now - time.time(), 1e-6)  # Simplified

                # Use the default visualizer to display results
                should_stop = default_visualizer(
                    frame, result, event, is_fall, fps
                )

                # Add robot status overlay for fall detection mode
                if robot is None:
                    cv2.putText(frame, "ROBOT DISCONNECTED - Camera Only Mode",
                               (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, "Navigation will show robot commands but won't move",
                               (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                if should_stop:
                    break

            elif current_state == STATE_NAVIGATION:
                # Navigation mode - use logic from nav.py

                # Calculate thigh midpoint for navigation
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
                elif navigation_phase == PHASE_APPROACH and last_thigh_x is not None and last_thigh_y is not None:
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

                if robot is not None:
                    if navigation_phase == PHASE_ALIGNMENT:
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
                                navigation_phase = PHASE_APPROACH
                                approach_start_time = now_mono
                        elif time.time() - alignment_start_time > alignment_timeout:
                            # Timeout - no person detected, give up
                            logger.warning(
                                f"Alignment timeout ({alignment_timeout}s) - no person detected"
                            )
                            current_state = STATE_COMPLETE

                    elif navigation_phase == PHASE_APPROACH:
                        # If we somehow entered APPROACH without a start time, set it.
                        if approach_start_time is None:
                            approach_start_time = now_mono

                        # Check for timeout
                        if (now_mono - approach_start_time) > approach_timeout:
                            logger.warning(f"Approach timeout ({approach_timeout}s) reached! Initiating safety backoff.")
                            navigation_phase = PHASE_BACKOFF
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
                        if navigation_phase == PHASE_APPROACH:  # Check again in case timeout happened
                            if consecutive_below_threshold >= args.consecutive_frames:
                                logger.info(
                                    f"Target reached at {mode_distance:.1f}cm after {consecutive_below_threshold} consecutive frames"
                                )
                                robot_command = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
                                current_state = STATE_COMPLETE
                            else:
                                # Continue driving forward
                                robot_command = {
                                    "x.vel": args.drive_speed,
                                    "y.vel": 0.0,
                                    "theta.vel": 0.0,
                                }

                    elif navigation_phase == PHASE_BACKOFF:
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
                            current_state = STATE_COMPLETE

                    # Send command to robot
                    try:
                        robot.send_base_action(robot_command)
                    except Exception as e:
                        logger.error(f"Error sending robot command: {e}")

                # Navigation visualization
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

                if draw_x is None and navigation_phase == PHASE_APPROACH and last_thigh_x is not None:
                    draw_x, draw_y = last_thigh_x, last_thigh_y
                    is_fallback = True

                if draw_x is not None and draw_y is not None:
                    # Draw depth sampling region (10x10)
                    tx, ty = int(draw_x), int(draw_y)
                    box_color = (0, 0, 255) if not is_fallback else (0, 165, 255)  # Red normally, Orange if fallback

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
                    if not is_fallback and thigh_x is not None and landmarks:
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

                # Status text for navigation
                if current_state == STATE_COMPLETE:
                    status_color = (0, 255, 0)  # Green for complete
                elif navigation_phase == PHASE_APPROACH:
                    status_color = (0, 255, 255)  # Yellow for approach
                else:
                    status_color = (255, 255, 255)  # White for alignment

                phase_status = {
                    PHASE_ALIGNMENT: f"ALIGNING ({last_tracked_side or 'None'})",
                    PHASE_APPROACH: "APPROACHING",
                    PHASE_BACKOFF: "SAFETY BACKOFF",
                    PHASE_COMPLETE: "COMPLETE",
                }[navigation_phase]

                status_text = phase_status

                # Determine robot status message
                if robot is None:
                    robot_status = "Robot: DISCONNECTED (camera-only mode)"
                    robot_color = (0, 0, 255)  # Red for disconnected
                else:
                    robot_status = f"Robot: {args.port} (connected)"
                    robot_color = (0, 255, 0)  # Green for connected

                status_lines = [
                    f"State: {current_state}",
                    f"Thigh Offset: {offset_pixels:.1f}px ({normalized_offset:.3f})",
                    f"Distance: {mode_distance:.1f}cm",
                    f"Consecutive: {consecutive_below_threshold}/{args.consecutive_frames}",
                    f"Phase: {status_text}",
                    robot_status,
                    "Press 'q' to quit, 's' to stop immediately",
                ]

                for i, line in enumerate(status_lines):
                    # Use different color for robot status line
                    line_color = robot_color if "Robot:" in line else status_color
                    cv2.putText(
                        vis_frame,
                        line,
                        (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        line_color,
                        2,
                    )

                cv2.imshow("Fall Detection & Navigation - Camera", vis_frame)
                cv2.imshow("Fall Detection & Navigation - Depth Map", depth_color)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s") and robot:
                logger.info("Emergency stop requested")
                robot.stop_base()
                current_state = STATE_COMPLETE
                navigation_phase = PHASE_COMPLETE
                consecutive_below_threshold = 0

            # Exit if state is complete
            if current_state == STATE_COMPLETE:
                logger.info("Task complete! Press 'q' to exit.")
                # Keep running to show final status

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up resources
        camera.stop()
        pose_estimator.close()
        cv2.destroyAllWindows()

        if robot is not None:
            print("Stopping robot...")
            robot.stop_base()
            robot.disconnect()
            print("Robot disconnected.")

        print("Resources cleaned up")


if __name__ == "__main__":
    main()