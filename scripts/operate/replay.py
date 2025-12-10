# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import os
import sys
import time

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.utils.robot_utils import precise_sleep


def main():
    parser = argparse.ArgumentParser(
        description="Replay recorded arm or wheels movements from CSV file"
    )
    parser.add_argument(
        "--ip", type=str, default="172.20.10.2", help="Remote IP for the LeKiwi robot"
    )
    parser.add_argument(
        "--id", type=str, default="biden_kiwi", help="ID of the LeKiwi robot"
    )
    parser.add_argument(
        "--name", type=str, required=True, help="Name of the recording to replay"
    )
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=["arm", "wheels"],
        help="Type of recording: 'arm' or 'wheels'",
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for replay (default: 30)"
    )
    args = parser.parse_args()

    # Initialize the robot config
    robot_config = LeKiwiClientConfig(remote_ip=args.ip, id=args.id)
    robot = LeKiwiClient(robot_config)

    # Try to find the CSV file in both arm and wheels directories
    recordings_base_dir = os.path.join(
        os.path.dirname(__file__), "..", "lekiwi", "recordings"
    )
    csv_filename = f"{args.name}.csv"

    recording_type = args.type
    csv_path = os.path.join(recordings_base_dir, recording_type, csv_filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Recording not found: {csv_path}")

    # Read CSV file and replay actions
    with open(csv_path, "r") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        actions = list(csv_reader)

    # Connect to the robot
    robot.connect()

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print(f"Replaying {len(actions)} actions from {csv_path} (type: {recording_type})")
    print("Starting replay loop...")

    for row in actions:
        t0 = time.perf_counter()

        if recording_type == "arm":
            # Extract arm action data (exclude timestamp column)
            arm_action = {
                key: float(value) for key, value in row.items() if key != "timestamp"
            }

            # Add empty base velocities (robot expects both arm and base actions)
            action = {
                **arm_action,
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0,
            }
        else:  # wheels
            # Extract base velocities (exclude timestamp column)
            base_action = {
                key: float(value) for key, value in row.items() if key != "timestamp"
            }

            present_position = robot.get_observation()
            # Keep existing arm positions
            arm_action = {
                key: float(value)
                for key, value in present_position.items()
                if key.endswith(".pos")
            }

            # Keep existing arm position
            action = {
                **base_action,
                **arm_action,
            }

        # Send action to robot
        _ = robot.send_action(action)

        precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

    print("Replay complete!")
    robot.disconnect()


if __name__ == "__main__":
    # How to run (with defaults):
    # python -m scripts.operate.replay --name test_arm --type arm
    # python -m scripts.operate.replay --name test_wheels --type wheels
    # Or override defaults:
    # python -m scripts.operate.replay --name test_arm --type arm --ip 172.20.10.3
    main()
