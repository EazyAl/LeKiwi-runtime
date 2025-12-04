"""
Capture video from the camera and draw pose landmarks on the frame using MediaPipe.

You can connect a camera to the computer for simple testing. Note that you might need to change
the camera index in cv2.VideoCapture() to the correct index of your camera.

Exit the program by pressing 'q' on the keyboard.
"""

import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils  # type: ignore[attr-defined]

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame from camera")
        break

    # Convert BGR to RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw pose landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Releases capture so it's no longer in use
cap.release()
cv2.destroyAllWindows()
