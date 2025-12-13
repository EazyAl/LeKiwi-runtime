"""\
Rerun viewer for pose + fall detection.

This is a drop-in alternative to OpenCV windows when those can't be shown
(e.g. when a service runs in a worker thread).

Usage:
  uv run scripts/test_pose_rerun.py --camera-index 1

Controls:
  - Close the Rerun viewer window, or Ctrl+C in the terminal to stop.

What you get in Rerun:
  - RGB camera frame
  - Pose landmarks (2D)
  - Pose skeleton connections (2D)
  - Fall detection status + metrics
"""

import argparse
import time

import cv2
import numpy as np
import rerun as rr
import mediapipe as mp

from lekiwi.services.pose_detection.pose_service import PoseEstimator, FallDetector


def _landmarks_to_pixels(landmarks, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Convert mediapipe landmarks to Nx2 pixel coords + Nx visibility."""
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks], dtype=np.float32)
    vis = np.array(
        [getattr(lm, "visibility", 0.0) for lm in landmarks], dtype=np.float32
    )
    return pts, vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun pose viewer")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index (default: 0)",
    )
    args = parser.parse_args()

    print(f"Using camera index: {args.camera_index} (override via --camera-index)")

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    pose = PoseEstimator()
    fall = FallDetector()

    rr.init("lekiwi_pose_viewer")
    rr.spawn(memory_limit="25%")

    prev_t = time.time()
    last_event = None
    last_is_fall = False
    frame_idx = 0

    pose_edges = list(mp.solutions.pose.POSE_CONNECTIONS)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Error: Could not read frame from camera")
                break

            now = time.time()
            fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]

            # Run pose
            result = pose.infer(frame_bgr)
            landmarks = (
                result.pose_landmarks.landmark
                if result and result.pose_landmarks
                else None
            )

            # Fall detection
            event = None
            is_fall = last_is_fall
            if landmarks:
                event = fall.detect(landmarks)
                if event:
                    is_fall = event.is_fall
                    last_event = event
                    last_is_fall = is_fall

            # Timeline
            rr.set_time_sequence("frame", frame_idx)
            rr.set_time_seconds("time", now)

            # Image
            rr.log("camera/rgb", rr.Image(frame_rgb))

            # Status
            label = "FALL" if is_fall else "OK"
            rr.log("pose/status", rr.TextLog(label))
            rr.log("pose/fps", rr.Scalars(fps))

            ev = event or last_event
            if ev is not None:
                rr.log("pose/torso_ratio", rr.Scalars(float(ev.ratio)))
                rr.log("pose/fall_score", rr.Scalars(float(ev.score)))

            # Pose viz
            if landmarks:
                pts, vis = _landmarks_to_pixels(landmarks, w, h)

                colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
                good = vis >= 0.5
                colors[good] = np.array([0, 255, 0], dtype=np.uint8)
                colors[~good] = np.array([255, 0, 0], dtype=np.uint8)

                rr.log(
                    "pose/landmarks",
                    rr.Points2D(pts, colors=colors, radii=3.0),
                )

                segments: list[np.ndarray] = []
                for a, b in pose_edges:
                    if (
                        a < len(vis)
                        and b < len(vis)
                        and vis[a] >= 0.5
                        and vis[b] >= 0.5
                    ):
                        segments.append(np.array([pts[a], pts[b]], dtype=np.float32))

                if segments:
                    rr.log(
                        "pose/skeleton",
                        rr.LineStrips2D(segments, colors=[0, 200, 255], radii=1.5),
                    )

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        pose.close()


if __name__ == "__main__":
    main()
