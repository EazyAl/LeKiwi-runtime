"""
Combined viewer for MonoDepth (Obstacle Avoidance) + Pose/Fall/Face Detection.
Merges logic from nav/mono-test.py and scripts/test_pose_viewer.py
"""

import cv2
import time
import torch
import numpy as np
import mediapipe as mp
import argparse

# --- Imports from LeKiwi Codebase ---
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


# --- MonoPilot Class (from nav/mono-test.py) ---
class MonoPilot:
    def __init__(self):
        print("Loading MiDaS AI model...")
        # Load small model (fastest)
        model_type = "MiDaS_small"
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)

        # Move to GPU if available
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.midas.to(self.device)
        self.midas.eval()

        # Setup Transform (resize/normalize image for AI)
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.small_transform

        print(f"Model loaded on {self.device}")

    def get_depth_map(self, frame):
        """
        Input: BGR Frame (from OpenCV)
        Output: Depth Map (numpy array), Higher Value = CLOSER
        """
        input_batch = self.transform(frame).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            # Resize to original image size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map

    def process_frame(self, frame):
        """
        Returns a control command (vx, vy, omega) based on depth.
        """
        depth_map = self.get_depth_map(frame)
        h, w = depth_map.shape

        # --- Heuristic Logic ---
        # 1. Look at the center strip (Horizon)
        strip = depth_map[int(h * 0.4) : int(h * 0.7), :]

        # 2. Divide into Left / Center / Right
        third = w // 3
        left_zone = strip[:, :third]
        center_zone = strip[:, third : 2 * third]
        right_zone = strip[:, 2 * third :]

        # 3. Calculate "Closeness" (Mean value)
        l_score = np.mean(left_zone)
        c_score = np.mean(center_zone)
        r_score = np.mean(right_zone)

        # 4. Calibration (Assume ~500 is "Too Close" for MiDaS small)
        THRESHOLD = 500.0

        vx, vy, omega = 0.0, 0.0, 0.0

        if c_score > THRESHOLD:
            # CENTER BLOCKED!
            if l_score < r_score:
                omega = 0.5  # Turn Left
            else:
                omega = -0.5  # Turn Right
            vy = -0.1
        else:
            # PATH CLEAR
            vy = 0.3
            push = (l_score - r_score) * 0.001
            vx = push

        return vx, vy, omega, depth_map, (l_score, c_score, r_score)


# --- Visualization Helper (from scripts/test_pose_viewer.py) ---
def draw_pose_overlay(
    frame, result, event, is_fall: bool, fps: float, quality: dict, face_stats: dict
):
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
        f"FPS {fps:.1f}",
        (10, 215),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    # Simplified stats for combined view
    if face_stats.get("face_box"):
        x0, y0, x1, y1 = face_stats["face_box"]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)


# --- Main Combined Loop ---
def main(*, flip_180: bool = False):
    # Initialize Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Initialize AI Models
    print("Initializing Models...")
    pose = PoseEstimator()
    detector = FallDetector()
    face_lm = FaceLandmarker()
    blink_awake = BlinkAwakeEstimator()
    rppg = RPpgHeartEstimator()
    mono_pilot = MonoPilot()

    prev_gray = None
    last_time = time.time()
    last_event = None
    last_is_fall = False

    print("Starting Main Loop. Press 'q' to quit.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            print("Error: Could not read frame from camera")
            break

        # Optional: flip camera upside down (180° rotation)
        if flip_180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # --- 1. Run MonoDepth (Obstacle Avoidance) ---
        vx, vy, omega, dmap, scores = mono_pilot.process_frame(frame)

        # --- 2. Run Pose/Fall Detection ---
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

        # --- 3. Run Face Detection ---
        face_stats = {}
        now = time.time()
        face_result = face_lm.infer(frame)
        if face_result.multi_face_landmarks:
            face_lms = face_result.multi_face_landmarks[0].landmark
            h, w, _ = frame.shape
            face_box = compute_face_box(face_lms, w, h)
            eyes = compute_eye_metrics(face_lms, w, h)
            awake = blink_awake.update(eyes["ear_mean"], timestamp=now)
            hr = rppg.update(frame, face_box, timestamp=now)
            face_stats = {**eyes, **awake, **hr, "face_box": face_box}

        # --- 4. Visualization ---
        fps = 1.0 / max(now - last_time, 1e-6)
        last_time = now

        # Window 1: Camera Feed with Pose/Face Overlay
        vis_frame = frame.copy()
        draw_pose_overlay(
            vis_frame, result, event or last_event, is_fall, fps, quality, face_stats
        )

        # --- Visualization: Navigation Arrow & Warning ---
        h_img, w_img = vis_frame.shape[:2]
        center_x, center_y = w_img // 2, h_img // 2

        # 1. Obstacle Warning
        THRESHOLD = 500.0  # Must match MonoPilot threshold
        if scores[1] > THRESHOLD:
            cv2.putText(
                vis_frame,
                "OBSTACLE DETECTED!",
                (center_x - 150, center_y - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
            )
            cv2.rectangle(vis_frame, (10, 10), (w_img - 10, h_img - 10), (0, 0, 255), 5)

        # 2. Draw Velocity Arrow (Steering)
        # Scale arrow length by speed, angle by rotation
        arrow_len = 100 * (abs(vy) + 0.1)
        angle_rad = -omega * 2.0  # Invert for display logic (left turn = arrow left)
        end_x = int(center_x + arrow_len * np.sin(angle_rad))
        end_y = int(center_y - arrow_len * np.cos(angle_rad))

        arrow_color = (0, 255, 0) if scores[1] <= THRESHOLD else (0, 0, 255)
        cv2.arrowedLine(
            vis_frame, (center_x, center_y + 50), (end_x, end_y), arrow_color, 4
        )

        # Add MonoPilot Nav Info
        cv2.putText(
            vis_frame,
            f"NAV: vx={vx:.2f} vy={vy:.2f} rot={omega:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            vis_frame,
            f"Depth: L={scores[0]:.0f} C={scores[1]:.0f} R={scores[2]:.0f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("LeKiwi Perception", vis_frame)

        # Window 2: Depth Map with Zones
        dmap_norm = cv2.normalize(dmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dmap_color = cv2.applyColorMap(dmap_norm, cv2.COLORMAP_MAGMA)

        # Draw vertical lines for zones
        third = w_img // 3
        cv2.line(dmap_color, (third, 0), (third, h_img), (255, 255, 255), 1)
        cv2.line(dmap_color, (2 * third, 0), (2 * third, h_img), (255, 255, 255), 1)

        # Label zones with scores
        cv2.putText(
            dmap_color,
            f"{scores[0]:.0f}",
            (third // 2 - 20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            dmap_color,
            f"{scores[1]:.0f}",
            (third + third // 2 - 20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            dmap_color,
            f"{scores[2]:.0f}",
            (2 * third + third // 2 - 20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Depth Map", dmap_color)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combined viewer for MonoDepth + Pose/Fall + Face detection."
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Rotate camera frames 180° (useful if the camera is mounted upside down).",
    )
    args = parser.parse_args()
    main(flip_180=args.flip)
