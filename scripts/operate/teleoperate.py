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
import sys
import time

import cv2
import rerun as rr

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import (
    KeyboardTeleop,
    KeyboardTeleopConfig,
)
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

FPS = 30


def main():
    parser = argparse.ArgumentParser(
        description="Teleoperate LeKiwi with leader arm and keyboard"
    )
    parser.add_argument(
        "--ip", type=str, default="172.20.10.2", help="Remote IP for the LeKiwi robot"
    )
    parser.add_argument(
        "--id", type=str, default="biden_kiwi", help="ID of the LeKiwi robot"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/tty.usbmodem5AB90687441",
        help="Serial port for the leader arm",
    )
    parser.add_argument(
        "--leader_id",
        type=str,
        default="obama_leader",
        help="ID of the leader arm",
    )
    parser.add_argument(
        "--keyboard_id",
        type=str,
        default="my_laptop_keyboard",
        help="ID of the keyboard teleoperator",
    )
    args = parser.parse_args()

    # Create the robot and teleoperator configurations
    robot_config = LeKiwiClientConfig(remote_ip=args.ip, id=args.id)
    teleop_arm_config = SO100LeaderConfig(port=args.port, id=args.leader_id)
    keyboard_config = KeyboardTeleopConfig(id=args.keyboard_id)

    # Initialize the robot and teleoperator
    robot = LeKiwiClient(robot_config)
    leader_arm = SO100Leader(teleop_arm_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # Connect to the robot and teleoperator
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    # Init rerun viewer
    init_rerun(session_name="lekiwi_teleop")

    # Open Mac's top-down camera (external USB camera)
    top_down_cam = cv2.VideoCapture(0)
    if not top_down_cam.isOpened():
        print("Warning: Could not open top-down camera (index 1)")

    if (
        not robot.is_connected
        or not leader_arm.is_connected
        or not keyboard.is_connected
    ):
        raise ValueError("Robot or teleop is not connected!")

    print("Starting teleop loop...")
    while True:
        t0 = time.perf_counter()

        # Get robot observation
        observation = robot.get_observation()

        # Get teleop action
        # Arm
        arm_action = leader_arm.get_action()
        arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
        # Keyboard
        keyboard_keys = keyboard.get_action()
        base_action = robot._from_keyboard_to_base_action(keyboard_keys)

        action = {**arm_action, **base_action} if len(base_action) > 0 else arm_action

        # Send action to robot
        _ = robot.send_action(action)

        # Visualize
        log_rerun_data(observation=observation, action=action)

        # Show top-down camera from Mac in Rerun
        ret, top_frame = top_down_cam.read()
        if ret:
            # Convert BGR to RGB for Rerun
            top_frame_rgb = cv2.cvtColor(top_frame, cv2.COLOR_BGR2RGB)
            rr.log("observation/top_down", rr.Image(top_frame_rgb))

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    # Cleanup
    top_down_cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
