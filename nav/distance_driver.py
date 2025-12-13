"""
Distance Driver: Drives forward until approximately 30cm from objects in front.

Uses MiDaS depth estimation to measure distance to objects directly ahead and
drives forward at a controlled speed until reaching the target distance.
"""

import cv2
import time
import torch
import numpy as np
import argparse

# LeKiwi imports
from lekiwi.robot.lekiwi import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

# Import MonoPilot from combined_viewer
from combined_viewer import MonoPilot


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
    parser = argparse.ArgumentParser(description="Distance-based forward driver")
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
        "--speed",
        type=float,
        default=0.2,
        help="Forward driving speed (m/s)",
    )
    parser.add_argument(
        "--depth_threshold",
        type=float,
        default=1000.0,
        help="Depth threshold for stopping (higher = closer, ~750 = ~50cm)",
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

    print(f"Distance Driver - Target: {args.target_distance}cm, Speed: {args.speed}m/s")

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize MiDaS depth estimation
    print("Initializing MiDaS depth model...")
    mono_pilot = MonoPilot()

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

    if args.calibrate:
        print("Calibration mode: Showing depth values. Press 'q' to quit.")
        print("Move objects to different distances and note the depth values.")
    else:
        print(
            "Starting distance-based driving. Press 'q' to quit, 's' to stop immediately."
        )

    driving_forward = False
    target_reached = False
    consecutive_below_threshold = 0

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

            # Get depth information
            vx, vy, omega, depth_map, scores = mono_pilot.process_frame(frame)

            # Calculate center distance
            center_depth = get_center_distance(depth_map)
            mode_depth = get_mode_distance(depth_map)
            estimated_distance = estimate_distance_from_depth(center_depth)
            mode_distance = estimate_distance_from_depth(mode_depth)

            # Control logic (skip if in calibration mode)
            robot_command = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

            if not args.calibrate and robot is not None and not target_reached:
                # Check if mode distance is below 60cm
                if mode_distance < 60.0:
                    consecutive_below_threshold += 1
                    print(
                        f"Close distance detected ({mode_distance:.1f}cm) - {consecutive_below_threshold}/{args.consecutive_frames} frames"
                    )
                else:
                    consecutive_below_threshold = 0  # Reset counter

                # Stop only after consecutive frames below threshold
                if consecutive_below_threshold >= args.consecutive_frames:
                    # Too close - stop and mark target reached
                    print(
                        f"Target reached at {mode_distance:.1f}cm after {consecutive_below_threshold} consecutive frames"
                    )
                    robot_command = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
                    target_reached = True
                    driving_forward = False
                elif not driving_forward:
                    # Start driving forward
                    print(f"Starting to drive forward at {args.speed:.2f} m/s")
                    robot_command = {
                        "x.vel": args.speed,
                        "y.vel": 0.0,
                        "theta.vel": 0.0,
                    }
                    driving_forward = True
                else:
                    # Continue driving forward
                    robot_command = {
                        "x.vel": args.speed,
                        "y.vel": 0.0,
                        "theta.vel": 0.0,
                    }

                # Send command to robot
                try:
                    robot.send_base_action(robot_command)
                except Exception as e:
                    print(f"Error sending robot command: {e}")

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

            # Add distance overlay on depth map
            cv2.putText(
                depth_color,
                f"Mode: {mode_distance:.1f}cm",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                depth_color,
                f"Avg: {estimated_distance:.1f}cm",
                (10, 60),
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
                status_color = (
                    (0, 255, 0)
                    if target_reached
                    else (0, 255, 255) if driving_forward else (255, 255, 255)
                )
                status_text = f"{'STOPPED - TARGET REACHED' if target_reached else 'DRIVING FORWARD' if driving_forward else 'WAITING'}"
                key_help = "Press 'q' to quit, 's' to stop immediately"

            status_lines = [
                ".1f",
                ".1f" ".0f" f"Depth: {center_depth:.0f}",
                f"Consecutive: {consecutive_below_threshold}/{args.consecutive_frames}",
                f"Status: {status_text}",
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

            cv2.imshow("Distance Driver - Camera", vis_frame)
            cv2.imshow("Distance Driver - Depth Map", depth_color)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif not args.calibrate and key == ord("s"):
                print("Emergency stop requested")
                if robot:
                    robot.stop_base()
                target_reached = True
                driving_forward = False
                consecutive_below_threshold = 0

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
