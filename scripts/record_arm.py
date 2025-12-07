import argparse
import time
import csv
import os

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.robot_utils import precise_sleep


def main():
    parser = argparse.ArgumentParser(
        description="Record an action sequence for LeKiwi's leader arm"
    )
    parser.add_argument(
        "--ip", type=str, required=True, help="Remote ip for the lekiwi robot"
    )
    parser.add_argument("--id", type=str, required=True, help="ID of the lekiwi robot")
    parser.add_argument("--name", type=str, required=True, help="Name of the recording")
    # Teleoperator
    parser.add_argument(
        "--port",
        type=str,
        required=True,
        help="Serial port for the leader arm (if teleoperating)",
    )
    parser.add_argument(
        "--leader_id",
        type=str,
        required=True,
        help="ID of the leader arm (if teleoperating)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for recording (default: 30)",
    )
    args = parser.parse_args()

    # Create the robot configuration
    robot_config = LeKiwiClientConfig(remote_ip=args.ip, id=args.id)
    robot = LeKiwiClient(robot_config)
    robot.connect()  # To connect you already should have this script running on LeKiwi: `python -m lerobot.robots.lekiwi.lekiwi_host --robot.id=my_awesome_kiwi`

    leader_arm_config = SO100LeaderConfig(port=args.port, id=args.leader_id)
    leader_arm = SO100Leader(leader_arm_config)
    leader_arm.connect()

    input("Press Enter to start recording...")

    recordings_dir = os.path.join(
        os.path.dirname(__file__), "..", "lekiwi", "recordings", "arm"
    )
    os.makedirs(recordings_dir, exist_ok=True)

    csv_filename = os.path.join(recordings_dir, f"{args.name}.csv")

    # TODO(Victor): consider using 'make_default_processors()' from HF to use the HF dataset format as practice
    # Would use this together with record_loop from lerobot_record.py to record the actions
    arm_keys = [
        "arm_shoulder_pan.pos",
        "arm_shoulder_lift.pos",
        "arm_elbow_flex.pos",
        "arm_wrist_flex.pos",
        "arm_wrist_roll.pos",
        "arm_gripper.pos",
    ]

    with open(csv_filename, "w", newline="") as csvfile:
        csv_writer = None

        try:
            while True:
                t0 = time.perf_counter()

                leader_action = leader_arm.get_action()  # type: ignore[attribute-error]
                obs = {f"arm_{key}": val for key, val in leader_action.items()}
                # Add empty base velocities (robot expects both arm and base actions)
                action = {
                    **obs,
                    "x.vel": 0.0,
                    "y.vel": 0.0,
                    "theta.vel": 0.0,
                }
                robot.send_action(action)

                if csv_writer is None:
                    fieldnames = ["timestamp"] + arm_keys
                    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    csv_writer.writeheader()

                row = {"timestamp": t0}
                for key in arm_keys:
                    row[key] = obs.get(key, 0.0)
                csv_writer.writerow(row)
                csvfile.flush()

                precise_sleep(max(1.0 / args.fps - (time.perf_counter() - t0), 0.0))

        except KeyboardInterrupt:
            print("Shutting down teleop...")
        finally:
            robot.disconnect()
            if leader_arm is not None:
                leader_arm.disconnect()
            print(f"Recording saved to {csv_filename}")


if __name__ == "__main__":
    # How to run:
    # python -m scripts.record_arm --ip 172.20.10.2 --id biden_kiwi --name test_arm --port /dev/tty.usbmodem5AB90687441 --leader_id obama_leader
    main()
