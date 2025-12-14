"""
Standalone viewer for pose detection and fall detection.
Uses only components from pose_service.py. Uses the default webcam (index 0). Press 'q' to quit.
"""

import argparse
import cv2
import time
import mediapipe as mp

from lekiwi.services.pose_detection.pose_service import (
    CameraStream,
    PoseEstimator,
    FallDetector,
    default_visualizer,
)


def main():
    parser = argparse.ArgumentParser(description="Pose detection viewer")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0)",
    )
    args = parser.parse_args()

    # Initialize camera and pose detection components
    print(f"Using camera index: {args.camera_index} (override via --camera-index)")
    camera = CameraStream(index=args.camera_index)
    pose_estimator = PoseEstimator()
    fall_detector = FallDetector()

    # State variables for fall detection
    last_event = None
    last_is_fall = False

    try:
        # Start the camera
        camera.start()
        print("Camera started. Press 'q' to quit.")

        prev_time = time.time()

        while True:
            # Read frame from camera
            frame = camera.read()

            # Rotate camera frame 180 degrees (flip both vertically and horizontally)
            frame = cv2.flip(frame, -1)

            # Perform pose estimation
            result = pose_estimator.infer(frame)
            landmarks = (
                result.pose_landmarks.landmark
                if result and result.pose_landmarks
                else None
            )

            # Perform fall detection if landmarks are available
            event = None
            is_fall = last_is_fall
            if landmarks:
                event = fall_detector.detect(landmarks)
                if event:
                    is_fall = event.is_fall
                    last_event = event
                    last_is_fall = is_fall

            # Calculate FPS
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            # Use the default visualizer to display results
            should_stop = default_visualizer(
                frame, result, event or last_event, is_fall, fps
            )

            if should_stop:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up resources
        camera.stop()
        pose_estimator.close()
        cv2.destroyAllWindows()
        print("Resources cleaned up")


if __name__ == "__main__":
    main()
