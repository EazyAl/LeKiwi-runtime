#!/usr/bin/env python3
"""
Record arm motions directly with SO100 arm connected to laptop (no Pi/LeKiwi needed).
Saves joint positions to CSV for replay, optionally records video.

Usage:
    python -m scripts.operate.record_arm_direct --name wave_hello
    python -m scripts.operate.record_arm_direct --name wave_hello --record-video
"""

import argparse
import csv
import os
import sys
import time
import threading

import cv2

from lerobot.robots.so100 import SO100Robot, SO100RobotConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep


def main():
    parser = argparse.ArgumentParser(
        description="Record arm motions directly (no Pi needed)"
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Name of the recording"
    )
    parser.add_argument(
        "--follower-port",
        type=str,
        default="/dev/tty.usbmodem58760432781",
        help="Serial port for the follower arm",
    )
    parser.add_argument(
        "--follower-id",
        type=str,
        default="my_follower",
        help="ID of the follower arm",
    )
    parser.add_argument(
        "--leader-port",
        type=str,
        default="/dev/tty.usbmodem5AB90687441",
        help="Serial port for the leader arm",
    )
    parser.add_argument(
        "--leader-id",
        type=str,
        default="my_leader",
        help="ID of the leader arm",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for recording (default: 30)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Also record video from cameras",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index for video recording (default: 0)",
    )
    args = parser.parse_args()

    # Set up recordings directory
    recordings_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "lekiwi", "recordings", "arm"
    )
    os.makedirs(recordings_dir, exist_ok=True)

    csv_filename = os.path.join(recordings_dir, f"{args.name}.csv")
    video_filename = os.path.join(recordings_dir, f"{args.name}.mp4") if args.record_video else None

    # Connect to follower arm
    print(f"Connecting to follower arm on {args.follower_port}...")
    follower_config = SO100RobotConfig(port=args.follower_port, id=args.follower_id)
    follower = SO100Robot(follower_config)
    follower.connect()
    print("Follower arm connected.")

    # Connect to leader arm
    print(f"Connecting to leader arm on {args.leader_port}...")
    leader_config = SO100LeaderConfig(port=args.leader_port, id=args.leader_id)
    leader = SO100Leader(leader_config)
    leader.connect()
    print("Leader arm connected.")

    # Set up video recording if requested
    video_writer = None
    cap = None
    if args.record_video:
        print(f"Opening camera {args.camera_index}...")
        cap = cv2.VideoCapture(args.camera_index)
        if not cap.isOpened():
            print(f"Warning: Could not open camera {args.camera_index}, continuing without video")
            args.record_video = False
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_filename, fourcc, args.fps, (width, height))
            print(f"Video recording enabled: {width}x{height} @ {args.fps}fps")

    # Arm joint keys (matching the original format)
    arm_keys = [
        "arm_shoulder_pan.pos",
        "arm_shoulder_lift.pos",
        "arm_elbow_flex.pos",
        "arm_wrist_flex.pos",
        "arm_wrist_roll.pos",
        "arm_gripper.pos",
    ]

    input("\nPress Enter to start recording (Ctrl+C to stop)...")
    print("Recording started!")

    with open(csv_filename, "w", newline="") as csvfile:
        csv_writer = csv.DictWriter(csvfile, fieldnames=["timestamp"] + arm_keys)
        csv_writer.writeheader()

        frame_count = 0
        try:
            while True:
                t0 = time.perf_counter()

                # Get leader arm position
                leader_action = leader.get_action()
                
                # Convert to arm_ prefixed keys
                obs = {f"arm_{key}": val for key, val in leader_action.items()}

                # Send to follower arm
                follower.send_action(leader_action)

                # Record to CSV
                row = {"timestamp": t0}
                for key in arm_keys:
                    row[key] = obs.get(key, 0.0)
                csv_writer.writerow(row)
                csvfile.flush()

                # Record video frame if enabled
                if args.record_video and cap is not None and video_writer is not None:
                    ret, frame = cap.read()
                    if ret:
                        video_writer.write(frame)

                frame_count += 1
                if frame_count % args.fps == 0:
                    print(f"Recorded {frame_count} frames ({frame_count / args.fps:.1f}s)...")

                # Maintain frame rate
                elapsed = time.perf_counter() - t0
                sleep_time = max(1.0 / args.fps - elapsed, 0.0)
                precise_sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping recording...")

        finally:
            # Clean up
            follower.disconnect()
            leader.disconnect()
            
            if video_writer is not None:
                video_writer.release()
            if cap is not None:
                cap.release()

            print(f"\n=== Recording Complete ===")
            print(f"CSV saved to: {csv_filename}")
            print(f"Total frames: {frame_count}")
            print(f"Duration: {frame_count / args.fps:.1f}s")
            if video_filename and args.record_video:
                print(f"Video saved to: {video_filename}")


if __name__ == "__main__":
    main()
