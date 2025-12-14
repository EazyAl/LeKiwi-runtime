"""
Standalone viewer for pose + fall detection + quality metrics.
Uses the default webcam (index 0). Press 'q' to quit.
"""

import argparse
import cv2
import time
import mediapipe as mp
import numpy as np

from lekiwi.services.pose_detection.pose_service import (
    PoseEstimator,
    FallDetector,
)
from lekiwi.vision import (
    compute_quality_metrics,
    FaceLandmarker,
    compute_face_box,
    compute_eye_metrics,
    BlinkAwakeEstimator,
    RPpgHeartEstimator,
)


def draw_overlay(
    frame, result, event, is_fall: bool, fps: float, quality: dict, face_stats: dict
) -> bool:
    """Custom viewer with larger text overlays."""
    label = "FALL" if is_fall else "OK"
    color = (0, 0, 255) if is_fall else (0, 200, 0)
    ratio_txt = f"{event.ratio:.2f}" if event else "--"
    score_txt = f"{event.score:.2f}" if event else "--"

    if result and result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
        )

    cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 5)
    cv2.putText(
        frame,
        f"torso ratio {ratio_txt}  score {score_txt}",
        (10, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        3,
    )
    cv2.putText(
        frame,
        f"bright {quality['brightness']:.0f}  blur {quality['blur']:.0f}  motion {quality['motion']:.1f}",
        (10, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"vis_min {quality['visibility_min']:.2f}  vis_mean {quality['visibility_mean']:.2f}",
        (10, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"FPS {fps:.1f}",
        (10, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    # Face stats always visible (with defaults)
    cv2.putText(
        frame,
        f"awake {face_stats.get('awake_likelihood', -1):.2f}  blinks/min {face_stats.get('blinks_per_min', 0):.1f}  perclos {face_stats.get('perclos', 0):.2f}  eyes_open {face_stats.get('eyes_open_prob', 0):.2f}",
        (10, 250),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"HR {face_stats.get('hr_bpm', -1):.0f} bpm  q {face_stats.get('hr_quality', 0):.2f}  method {face_stats.get('hr_method', '')}",
        (10, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"face_status {face_stats.get('face_status', 'no_face')}",
        (10, 310),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
    )

    if face_stats.get("face_box"):
        x0, y0, x1, y1 = face_stats["face_box"]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)
    if face_stats.get("face_landmarks"):
        # Some mediapipe versions lack get_default_face_mesh_landmarks_style; omit landmark spec.
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            face_stats["face_landmarks"],
            mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )

    cv2.imshow("Fall Detection Viewer", frame)
    return bool(cv2.waitKey(1) & 0xFF == ord("q"))


def main():
    parser = argparse.ArgumentParser(
        description="Pose + fall detection + quality metrics viewer"
    )
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

    pose = PoseEstimator()
    detector = FallDetector()
    face_lm = FaceLandmarker()
    blink_awake = BlinkAwakeEstimator()
    rppg = RPpgHeartEstimator()

    prev_gray = None
    last_time = time.time()
    last_event = None
    last_is_fall = False

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Error: Could not read frame from camera")
            break

        result = pose.infer(frame)
        landmarks = (
            result.pose_landmarks.landmark if result and result.pose_landmarks else None
        )

        quality = compute_quality_metrics(
            frame, prev_gray, landmarks, downscale_width=320
        )
        prev_gray = quality.pop("gray", None)

        event = None
        is_fall = last_is_fall
        if landmarks:
            event = detector.detect(landmarks)
            if event:
                is_fall = event.is_fall
                last_event = event
                last_is_fall = is_fall

        now = time.time()
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now

        face_stats = {
            "face_status": "no_face",
            "hr_method": "",
        }
        face_result = face_lm.infer(frame)
        if face_result.multi_face_landmarks:
            face_lms = face_result.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            face_box = compute_face_box(face_lms, w, h)
            eyes = compute_eye_metrics(face_lms, w, h)
            awake = blink_awake.update(eyes["ear_mean"], timestamp=now)
            hr = rppg.update(frame, face_box, timestamp=now)

            face_stats = {
                **eyes,
                **awake,
                **hr,
                "face_box": face_box,
                "face_landmarks": face_result.multi_face_landmarks[0],
                "face_status": "ok",
                "hr_method": hr.get("method", ""),
            }

        stop = draw_overlay(
            frame, result, event or last_event, is_fall, fps, quality, face_stats
        )
        if stop:
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main()
