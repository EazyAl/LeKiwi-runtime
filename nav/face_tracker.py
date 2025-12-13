"""
Face Tracker: Rotates LeKiwi robot base to keep face centered horizontally.

Uses mediapipe face detection to find face landmarks, calculates horizontal offset
from frame center, and rotates the robot base to keep the face centered.
"""

import cv2
import time
import argparse
import numpy as np
import mediapipe as mp

# LeKiwi imports
from lekiwi.robot.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
from lekiwi.vision import FaceLandmarker, compute_face_box


def calculate_face_center_offset(face_box, frame_width):
    """
    Calculate horizontal offset of face center from frame center.

    Args:
        face_box: Tuple (x0, y0, x1, y1) - face bounding box
        frame_width: Width of camera frame in pixels

    Returns:
        offset_pixels: Horizontal pixel offset from center (positive = face right of center)
        normalized_offset: Offset normalized to [-1, 1] range
    """
    if not face_box:
        return 0, 0

    # Face center X coordinate
    face_center_x = (face_box[0] + face_box[2]) / 2
    frame_center_x = frame_width / 2

    # Pixel offset (positive = face is to the right of center)
    offset_pixels = face_center_x - frame_center_x

    # Normalized offset (-1 = face at left edge, +1 = face at right edge)
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
    # Convert FOV to radians
    fov_rad = np.radians(camera_fov_deg)

    # Angular offset per pixel
    angle_per_pixel = fov_rad / frame_width

    # Calculate angular offset
    angle_rad = offset_pixels * angle_per_pixel
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def main():
    parser = argparse.ArgumentParser(description="Face tracking robot controller")
    parser.add_argument("--flip", action="store_true", help="Flip camera 180 degrees")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/tty.usbmodem58760432781",
        help="Serial port for the robot",
    )
    parser.add_argument("--id", type=str, default="biden_kiwi", help="ID of the robot")
    parser.add_argument(
        "--kp", type=float, default=2.0, help="Proportional gain for rotation control"
    )
    parser.add_argument(
        "--max_rot_speed",
        type=float,
        default=30.0,
        help="Maximum rotation speed (deg/s)",
    )
    parser.add_argument(
        "--deadzone",
        type=float,
        default=0.05,
        help="Deadzone for rotation (normalized offset)",
    )
    args = parser.parse_args()

    print("Initializing Face Tracker...")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize mediapipe face detection
    face_lm = FaceLandmarker()

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

    print("Starting face tracking. Press 'q' to quit.")

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

            # Face detection
            face_box = None
            face_result = face_lm.infer(frame)
            if face_result.multi_face_landmarks:
                face_lms = face_result.multi_face_landmarks[0].landmark
                face_box = compute_face_box(face_lms, w, h)

            # Calculate face position
            offset_pixels, normalized_offset = calculate_face_center_offset(face_box, w)

            # Convert to rotation command
            rotation_angle = pixels_to_angle(offset_pixels, w)
            rotation_speed = 0.0

            # Apply deadzone and proportional control
            if abs(normalized_offset) > args.deadzone:
                # Proportional control with gain
                rotation_speed = (
                    -rotation_angle * args.kp
                )  # Negative because we want to rotate towards the face

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

            # Draw face box if detected
            if face_box:
                x0, y0, x1, y1 = face_box
                cv2.rectangle(vis_frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

                # Draw center line and face center point
                frame_center_x = w // 2
                face_center_x = int((x0 + x1) / 2)

                # Vertical center line
                cv2.line(
                    vis_frame,
                    (frame_center_x, 0),
                    (frame_center_x, h),
                    (255, 255, 255),
                    1,
                )

                # Face center point
                cv2.circle(
                    vis_frame, (face_center_x, int((y0 + y1) / 2)), 5, (0, 0, 255), -1
                )

                # Offset line
                cv2.line(
                    vis_frame,
                    (frame_center_x, h // 2),
                    (face_center_x, h // 2),
                    (0, 255, 255),
                    2,
                )

            # Status text
            robot_status = f"Robot: {args.port}" if robot else "Robot: DISCONNECTED"
            status_lines = [
                f"Face Offset: {offset_pixels:.1f}px ({normalized_offset:.3f})",
                f"Rotation: {rotation_speed:.1f} deg/s",
                f"Deadzone: {args.deadzone:.3f}",
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

            cv2.imshow("Face Tracker", vis_frame)

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
