"""
Thigh Tracker: Rotates LeKiwi robot base to keep thigh midpoint centered horizontally.

Uses mediapipe pose detection to find hip and knee landmarks, calculates the midpoint
between hip and knee for the most visible/recently seen leg, and rotates the robot
base to keep this point centered.
"""

import cv2
import time
import argparse
import numpy as np
import mediapipe as mp

# LeKiwi imports
from lekiwi.robot.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
from lekiwi.services.pose_detection.pose_service import PoseEstimator


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


def pixels_to_angle(offset_pixels, frame_width, camera_fov_deg=60):
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


def main():
    parser = argparse.ArgumentParser(description="Thigh tracking robot controller")
    parser.add_argument("--flip", action="store_true", help="Flip camera 180 degrees")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/tty.usbmodem58760432781",
        help="Serial port for the robot",
    )
    parser.add_argument("--id", type=str, default="biden_kiwi", help="ID of the robot")
    parser.add_argument(
        "--kp", type=float, default=1.5, help="Proportional gain for rotation control"
    )
    parser.add_argument(
        "--max_rot_speed",
        type=float,
        default=20.0,
        help="Maximum rotation speed (deg/s)",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.08,
        help="Deadzone for rotation (normalized offset)",
    )
    args = parser.parse_args()

    print("Initializing Thigh Tracker...")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize mediapipe pose detection
    pose = PoseEstimator()

    # Initialize LeKiwi robot
    try:
        config = LeKiwiConfig(port=args.port, id=args.id, cameras={})
        robot = LeKiwi(config)
        robot.connect()
        print(f"Robot connected successfully on port {args.port}")
    except Exception as e:
        print(f"Failed to connect to robot on port {args.port}: {e}")
        print("Running in camera-only mode (no robot control)")
        robot = None

    print("Starting thigh tracking. Press 'q' to quit.")

    # Track which leg we're following for continuity
    last_tracked_side = None

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Error: Could not read frame from camera")
                break

            # Optional camera flip
            if args.flip:
                frame = cv2.rotate(frame, cv2.ROTATE_180)

            h, w = frame.shape[:2]

            # Pose detection
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

            # Calculate position offset
            offset_pixels, normalized_offset = calculate_thigh_offset(thigh_x, w)

            # Convert to rotation command
            rotation_angle = pixels_to_angle(offset_pixels, w)
            rotation_speed = 0.0

            # Apply deadzone and proportional control
            if abs(normalized_offset) > args.deadzone and confidence > 0:
                rotation_speed = (
                    -rotation_angle * args.kp
                )  # Negative because we want to rotate towards the thigh

                # Clamp to maximum speed
                rotation_speed = np.clip(
                    rotation_speed, -args.max_rot_speed, args.max_rot_speed
                )

            # Send rotation command to robot
            if robot is not None:
                try:
                    robot.send_base_action(
                        {
                            "x.vel": 0.0,  # No forward/backward movement
                            "y.vel": 0.0,  # No lateral movement
                            "theta.vel": rotation_speed,  # Rotation in deg/s
                        }
                    )
                except Exception as e:
                    print(f"Error sending robot command: {e}")

            # Visualization
            vis_frame = frame.copy()

            # Pose landmarks visualization removed to avoid import issues
            # Core functionality works without full pose skeleton drawing

            # Draw thigh midpoint if detected
            if thigh_x is not None and thigh_y is not None:
                # Draw midpoint
                cv2.circle(vis_frame, (int(thigh_x), int(thigh_y)), 8, (0, 255, 0), -1)

                # Draw line from hip to knee
                if side == "left" and landmarks:
                    hip_x = landmarks[LEFT_HIP].x * w
                    hip_y = landmarks[LEFT_HIP].y * h
                    knee_x = landmarks[LEFT_KNEE].x * w
                    knee_y = landmarks[LEFT_KNEE].y * h

                    cv2.line(
                        vis_frame,
                        (int(hip_x), int(hip_y)),
                        (int(knee_x), int(knee_y)),
                        (255, 0, 0),
                        3,
                    )
                elif side == "right" and landmarks:
                    hip_x = landmarks[RIGHT_HIP].x * w
                    hip_y = landmarks[RIGHT_HIP].y * h
                    knee_x = landmarks[RIGHT_KNEE].x * w
                    knee_y = landmarks[RIGHT_KNEE].y * h

                    cv2.line(
                        vis_frame,
                        (int(hip_x), int(hip_y)),
                        (int(knee_x), int(knee_y)),
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
                    (frame_center_x, int(thigh_y)),
                    (int(thigh_x), int(thigh_y)),
                    (0, 255, 255),
                    2,
                )

            # Status text
            robot_status = f"Robot: {args.port}" if robot else "Robot: DISCONNECTED"
            tracking_status = (
                f"Tracking: {last_tracked_side or 'None'} ({confidence:.2f})"
                if confidence > 0
                else "Tracking: None"
            )
            status_lines = [
                f"Thigh Offset: {offset_pixels:.1f}px ({normalized_offset:.3f})",
                f"Rotation: {rotation_speed:.1f} deg/s",
                f"Deadzone: {args.deadzone:.3f}",
                tracking_status,
                robot_status,
                "Press 'q' to quit",
            ]

            for i, line in enumerate(status_lines):
                cv2.putText(
                    vis_frame,
                    line,
                    (10, 30 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow("Thigh Tracker", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

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
