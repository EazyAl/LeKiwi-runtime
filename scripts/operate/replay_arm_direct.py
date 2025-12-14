#!/usr/bin/env python3
"""
Replay recorded arm motions directly with SO100 arm connected to laptop (no Pi/LeKiwi needed).
Reads joint positions from CSV and plays them back on the follower arm.

Usage:
    python -m scripts.operate.replay_arm_direct --name wave_hello
"""

import argparse
import csv
import os
import time

from lerobot.robots.so100 import SO100Robot, SO100RobotConfig
from lerobot.utils.robot_utils import precise_sleep


def main():
    parser = argparse.ArgumentParser(
        description="Replay arm motions directly (no Pi needed)"
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Name of the recording to replay"
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
        "--fps",
        type=int,
        default=30,
        help="Frames per second for replay (default: 30)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop the recording continuously",
    )
    args = parser.parse_args()

    # Find the recording
    recordings_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "lekiwi", "recordings", "arm"
    )
    csv_filename = os.path.join(recordings_dir, f"{args.name}.csv")

    if not os.path.exists(csv_filename):
        print(f"Error: Recording not found: {csv_filename}")
        print(f"\nAvailable recordings:")
        if os.path.exists(recordings_dir):
            for f in os.listdir(recordings_dir):
                if f.endswith(".csv"):
                    print(f"  - {f[:-4]}")
        return

    # Load the recording
    print(f"Loading recording: {args.name}")
    with open(csv_filename, "r") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        actions = list(csv_reader)

    print(f"Loaded {len(actions)} frames ({len(actions) / args.fps:.1f}s)")

    # Connect to follower arm
    print(f"\nConnecting to follower arm on {args.follower_port}...")
    follower_config = SO100RobotConfig(port=args.follower_port, id=args.follower_id)
    follower = SO100Robot(follower_config)
    follower.connect()
    print("Follower arm connected.")

    input("\nPress Enter to start replay (Ctrl+C to stop)...")

    try:
        loop_count = 0
        while True:
            loop_count += 1
            print(f"\n--- Playing (loop {loop_count}) ---")

            for idx, row in enumerate(actions):
                t0 = time.perf_counter()

                # Extract arm action data (exclude timestamp, remove arm_ prefix)
                action = {}
                for key, value in row.items():
                    if key != "timestamp" and key.startswith("arm_"):
                        # Remove "arm_" prefix for the robot
                        clean_key = key[4:]  # Remove "arm_" prefix
                        action[clean_key] = float(value)

                # Send to follower arm
                follower.send_action(action)

                # Progress indicator
                if idx % args.fps == 0:
                    print(f"  Frame {idx}/{len(actions)} ({idx / args.fps:.1f}s)")

                # Maintain frame rate
                elapsed = time.perf_counter() - t0
                sleep_time = max(1.0 / args.fps - elapsed, 0.0)
                precise_sleep(sleep_time)

            print(f"Replay complete!")

            if not args.loop:
                break

    except KeyboardInterrupt:
        print("\nStopping replay...")

    finally:
        follower.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
