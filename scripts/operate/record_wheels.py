import argparse
import csv
import os
import sys
import time

import numpy as np

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardTeleopConfig
from lerobot.utils.robot_utils import precise_sleep


def main():
    parser = argparse.ArgumentParser(
        description="Record an action sequence for LeKiwi's wheels using keyboard control"
    )
    parser.add_argument(
        "--ip", type=str, default="172.20.10.2", help="Remote IP for the LeKiwi robot"
    )
    parser.add_argument(
        "--id", type=str, default="biden_kiwi", help="ID of the LeKiwi robot"
    )
    parser.add_argument("--name", type=str, required=True, help="Name of the recording")
    parser.add_argument(
        "--keyboard_id",
        type=str,
        default="my_laptop_keyboard",
        help="ID of the keyboard teleoperator",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for recording (default: 30)",
    )
    args = parser.parse_args()

    # Create the robot configuration
    robot_config = LeKiwiClientConfig(remote_ip=args.ip, id=args.id, cameras={})
    robot = LeKiwiClient(robot_config)
    robot.connect()  # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`

    keyboard_config = KeyboardTeleopConfig(id=args.keyboard_id)
    keyboard = KeyboardTeleop(keyboard_config)
    keyboard.connect()

    input("Press Enter to start recording...")

    # Save into the repo's lekiwi/recordings/wheels folder.
    recordings_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "lekiwi", "recordings", "wheels"
    )
    os.makedirs(recordings_dir, exist_ok=True)

    csv_filename = os.path.join(recordings_dir, f"{args.name}.csv")

    # TODO(Victor): consider using 'make_default_processors()' from HF to use the HF dataset format as practice
    # Would use this together with record_loop from lerobot_record.py to record the actions
    wheel_keys = [
        "x.vel",
        "y.vel",
        "theta.vel",
    ]

    with open(csv_filename, "w", newline="") as csvfile:
        csv_writer = None

        try:
            while True:
                t0 = time.perf_counter()

                keyboard_keys = keyboard.get_action()
                # Convert dict keys to numpy array for _from_keyboard_to_base_action
                pressed_keys_array = np.array(list(keyboard_keys.keys()))
                base_action = robot._from_keyboard_to_base_action(pressed_keys_array)

                # Keep existing arm positions
                present_position = robot.get_observation()
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
                robot.send_action(action)

                if csv_writer is None:
                    fieldnames = ["timestamp"] + wheel_keys
                    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    csv_writer.writeheader()

                row = {"timestamp": t0}
                for key in wheel_keys:
                    row[key] = base_action.get(key, 0.0)
                csv_writer.writerow(row)
                csvfile.flush()

                precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

        except KeyboardInterrupt:
            print("Shutting down teleop...")
        finally:
            robot.disconnect()
            if keyboard is not None:
                keyboard.disconnect()
            print(f"Recording saved to {csv_filename}")


if __name__ == "__main__":
    # How to run (with defaults):
    # python -m scripts.operate.record_wheels --name test_wheels
    # Or override defaults:
    # python -m scripts.operate.record_wheels --name test_wheels --ip 172.20.10.3
    main()
