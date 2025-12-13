#!/usr/bin/env python
"""
Record episodes with LeKiwi robot for training.
Captures: Pi cameras (front, wrist) + Mac top-down camera + arm actions.
Saves episodes to disk incrementally, then uploads to Hugging Face.

Controls:
  - ENTER: Start recording
  - Ctrl+C: End episode
  - After episode: 's' to save, 'r' to re-record, 'q' to quit
"""

import argparse
import time
import os
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from huggingface_hub import HfApi, create_repo

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import (
    KeyboardTeleop,
    KeyboardTeleopConfig,
)
from lerobot.utils.robot_utils import precise_sleep

FPS = 30


def save_episode_to_disk(episode_data, episode_idx, output_dir):
    """Save a single episode to disk immediately to avoid memory issues."""
    episode_dir = output_dir / f"episode_{episode_idx:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Save images
    for cam_name in ["front", "wrist", "top_down"]:
        cam_dir = episode_dir / cam_name
        cam_dir.mkdir(exist_ok=True)
        key = f"observation.image.{cam_name}"
        for i, img in enumerate(episode_data[key]):
            img_path = cam_dir / f"frame_{i:06d}.jpg"
            Image.fromarray(img).save(img_path, quality=85)

    # Save metadata (state, action, timestamp)
    metadata = {
        "state": episode_data["observation.state"],
        "action": episode_data["action"],
        "timestamp": episode_data["timestamp"],
        "num_frames": len(episode_data["timestamp"]),
    }
    with open(episode_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)

    return episode_dir


def upload_dataset(output_dir, repo_id, num_episodes):
    """Upload saved episodes to Hugging Face."""
    print(f"\nUploading {num_episodes} episodes to HuggingFace...")

    api = HfApi()

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Note: {e}")

    # Upload the entire output directory
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )

    print(f"Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Record episodes for LeKiwi robot")
    parser.add_argument("--ip", type=str, default="172.20.10.2", help="Robot IP")
    parser.add_argument("--id", type=str, default="biden_kiwi", help="Robot ID")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/tty.usbmodem5AB90687441",
        help="Leader arm port",
    )
    parser.add_argument(
        "--leader_id", type=str, default="obama_leader", help="Leader arm ID"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., username/dataset_name)",
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10, help="Number of episodes to record"
    )
    parser.add_argument(
        "--max_episode_length",
        type=int,
        default=9000,
        help="Max frames per episode (default 9000 = 5min at 30fps)",
    )
    parser.add_argument(
        "--top_down_camera", type=int, default=0, help="Mac top-down camera index"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./recorded_episodes",
        help="Local directory to save episodes",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        response = input(
            f"Output directory {output_dir} exists. Delete and start fresh? (y/n): "
        )
        if response.lower() == "y":
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup robot and teleoperators
    robot_config = LeKiwiClientConfig(remote_ip=args.ip, id=args.id)
    robot = LeKiwiClient(robot_config)

    leader_config = SO100LeaderConfig(port=args.port, id=args.leader_id)
    leader_arm = SO100Leader(leader_config)

    keyboard_config = KeyboardTeleopConfig(id="my_laptop_keyboard")
    keyboard = KeyboardTeleop(keyboard_config)

    # Connect
    print("Connecting to robot...")
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    # Open Mac's top-down camera
    top_down_cam = cv2.VideoCapture(args.top_down_camera)
    if not top_down_cam.isOpened():
        print(f"Warning: Could not open top-down camera (index {args.top_down_camera})")

    print(f"\nReady to record {args.num_episodes} episodes")
    print(f"Episodes will be saved to: {output_dir}")
    print("Controls:")
    print("  - Press ENTER to start recording")
    print("  - Press Ctrl+C to end episode")
    print("  - After episode: type 's' to save, 'r' to re-record, 'q' to quit\n")

    episode_idx = 0
    saved_episodes = 0

    try:
        while episode_idx < args.num_episodes:
            input(
                f"\nEpisode {episode_idx + 1}/{args.num_episodes} - Press ENTER to start recording..."
            )

            episode_data = {
                "observation.image.front": [],
                "observation.image.wrist": [],
                "observation.image.top_down": [],
                "observation.state": [],
                "action": [],
                "timestamp": [],
            }

            print(f"Recording episode {episode_idx + 1}... (Press Ctrl+C to end)")
            frame_idx = 0
            start_time = time.time()

            try:
                while frame_idx < args.max_episode_length:
                    t0 = time.perf_counter()

                    # Get observation from robot (includes Pi cameras)
                    observation = robot.get_observation()

                    # Get leader arm action
                    arm_action = leader_arm.get_action()
                    arm_action = {f"arm_{k}": v for k, v in arm_action.items()}

                    # Get keyboard action for base
                    keyboard_keys = keyboard.get_action()
                    base_action = robot._from_keyboard_to_base_action(keyboard_keys)

                    action = (
                        {**arm_action, **base_action}
                        if len(base_action) > 0
                        else arm_action
                    )

                    # Send action to robot
                    robot.send_action(action)

                    # Get top-down camera frame
                    ret, top_frame = top_down_cam.read()
                    if ret:
                        top_frame_rgb = cv2.cvtColor(top_frame, cv2.COLOR_BGR2RGB)
                    else:
                        top_frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

                    # Extract camera images from observation
                    front_img = observation.get(
                        "observation.image.front",
                        np.zeros((480, 640, 3), dtype=np.uint8),
                    )
                    wrist_img = observation.get(
                        "observation.image.wrist",
                        np.zeros((480, 640, 3), dtype=np.uint8),
                    )

                    # Extract state
                    state = [
                        observation.get("observation.state.arm_shoulder_pan.pos", 0.0),
                        observation.get("observation.state.arm_shoulder_lift.pos", 0.0),
                        observation.get("observation.state.arm_elbow_flex.pos", 0.0),
                        observation.get("observation.state.arm_wrist_flex.pos", 0.0),
                        observation.get("observation.state.arm_wrist_roll.pos", 0.0),
                        observation.get("observation.state.arm_gripper.pos", 0.0),
                    ]

                    # Extract action values
                    action_values = [
                        action.get("arm_shoulder_pan.pos", 0.0),
                        action.get("arm_shoulder_lift.pos", 0.0),
                        action.get("arm_elbow_flex.pos", 0.0),
                        action.get("arm_wrist_flex.pos", 0.0),
                        action.get("arm_wrist_roll.pos", 0.0),
                        action.get("arm_gripper.pos", 0.0),
                        action.get("x.vel", 0.0),
                        action.get("y.vel", 0.0),
                        action.get("theta.vel", 0.0),
                    ]

                    # Store data in memory (will save to disk after episode)
                    episode_data["observation.image.front"].append(front_img)
                    episode_data["observation.image.wrist"].append(wrist_img)
                    episode_data["observation.image.top_down"].append(top_frame_rgb)
                    episode_data["observation.state"].append(state)
                    episode_data["action"].append(action_values)
                    episode_data["timestamp"].append(time.time() - start_time)

                    frame_idx += 1

                    # Show progress every second
                    if frame_idx % 30 == 0:
                        elapsed = time.time() - start_time
                        print(
                            f"\r  Recording... {elapsed:.1f}s ({frame_idx} frames)",
                            end="",
                            flush=True,
                        )

                    precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

            except KeyboardInterrupt:
                print(f"\n\nEpisode ended with {frame_idx} frames")

            if frame_idx < 10:
                print("Episode too short, discarding...")
                continue

            # Ask what to do with this episode
            while True:
                choice = input("Save (s), Re-record (r), or Quit (q)? ").lower().strip()
                if choice == "s":
                    # Save to disk immediately
                    print(f"Saving episode {episode_idx + 1} to disk...")
                    save_episode_to_disk(episode_data, saved_episodes, output_dir)
                    print(f"Episode {episode_idx + 1} saved to disk!")
                    episode_idx += 1
                    saved_episodes += 1
                    # Clear memory
                    del episode_data
                    break
                elif choice == "r":
                    print("Re-recording episode...")
                    del episode_data
                    break
                elif choice == "q":
                    del episode_data
                    raise KeyboardInterrupt
                else:
                    print("Please enter 's', 'r', or 'q'")

    except KeyboardInterrupt:
        print("\n\nRecording session ended.")

    # Cleanup
    top_down_cam.release()
    robot.disconnect()
    leader_arm.disconnect()
    keyboard.disconnect()

    if saved_episodes > 0:
        # Save dataset info
        info = {
            "num_episodes": saved_episodes,
            "fps": FPS,
            "cameras": ["front", "wrist", "top_down"],
        }
        with open(output_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"\n{saved_episodes} episodes saved locally to {output_dir}")

        # Ask to upload
        upload = input("Upload to HuggingFace now? (y/n): ").lower().strip()
        if upload == "y":
            upload_dataset(output_dir, args.repo_id, saved_episodes)
        else:
            print(f"You can upload later with:")
            print(
                f"  huggingface-cli upload {args.repo_id} {output_dir} --repo-type dataset"
            )
    else:
        print("No episodes recorded")


if __name__ == "__main__":
    main()
