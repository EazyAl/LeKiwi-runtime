#!/usr/bin/env python3
"""Test LeKiwi cameras - capture and display without saving"""

import cv2
import matplotlib.pyplot as plt


def find_cameras(max_test=10):
    """Find all available cameras"""
    cameras = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras


def capture_from_camera(index):
    """Capture a single frame from camera"""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        return None

    # Let camera warm up
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if ret:
        # Convert BGR to RGB for matplotlib
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


if __name__ == "__main__":
    print("Searching for cameras...")
    cameras = find_cameras()

    if not cameras:
        print("No cameras found!")
        exit(1)

    print(f"Found {len(cameras)} camera(s): {cameras}")

    # Capture from all cameras
    frames = {}
    for cam_idx in cameras:
        print(f"Capturing from camera {cam_idx}...")
        frame = capture_from_camera(cam_idx)
        if frame is not None:
            frames[cam_idx] = frame
            print(f"  ✓ Camera {cam_idx}: {frame.shape}")
        else:
            print(f"  ✗ Camera {cam_idx}: Failed to capture")

    # Display all frames
    if frames:
        num_cameras = len(frames)
        fig, axes = plt.subplots(1, num_cameras, figsize=(6 * num_cameras, 6))
        if num_cameras == 1:
            axes = [axes]

        for ax, (cam_idx, frame) in zip(axes, frames.items()):
            ax.imshow(frame)
            ax.set_title(f"Camera {cam_idx}", fontsize=14, fontweight="bold")
            ax.axis("off")

        plt.tight_layout()
        print("\nDisplaying images... Close window to exit.")
        plt.show()
    else:
        print("No frames captured!")
