import argparse
import csv
import os
import sys
import time

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.teleoperators.so100_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.errors import DeviceNotConnectedError
from lerobot.utils.robot_utils import precise_sleep

from lekiwi.robot import LeKiwi


def main():
    parser = argparse.ArgumentParser(
        description="Record an action sequence for LeKiwi's leader arm"
    )
    parser.add_argument(
        "--connect",
        choices=["auto", "host", "direct"],
        default="auto",
        help="How to connect to the robot. 'host' requires lekiwi/lerobot host running. "
        "'direct' uses local USB serial. 'auto' tries host then falls back to direct.",
    )
    parser.add_argument(
        "--ip", type=str, default="172.20.10.2", help="Remote IP for the LeKiwi robot"
    )
    parser.add_argument(
        "--id", type=str, default="biden_kiwi", help="ID of the LeKiwi robot"
    )
    parser.add_argument(
        "--robot-port",
        type=str,
        default="/dev/tty.usbmodem58760432781",
        help="Serial port for the LeKiwi robot (direct mode only)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run motor calibration on connect (direct mode only). Default: off.",
    )
    parser.add_argument("--name", type=str, required=True, help="Name of the recording")
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
        "--fps",
        type=int,
        default=30,
        help="Frames per second for recording (default: 30)",
    )
    args = parser.parse_args()

    # Create and connect robot (host or direct USB).
    robot = None
    if args.connect in ("host", "auto"):
        try:
            robot_config = LeKiwiClientConfig(remote_ip=args.ip, id=args.id, cameras={})
            robot = LeKiwiClient(robot_config)
            robot.connect()
        except DeviceNotConnectedError:
            if args.connect == "host":
                raise
            # Fall back to direct mode when no host is running.
            robot = None

    if robot is None:
        robot_config = LeKiwiConfig(port=args.robot_port, id=args.id, cameras={})
        robot = LeKiwi(robot_config)
        # Default calibrate=False to avoid interactive prompts during recording.
        robot.connect(calibrate=bool(args.calibrate))

    leader_arm_config = SO100LeaderConfig(port=args.port, id=args.leader_id)
    leader_arm = SO100Leader(leader_arm_config)
    leader_arm.connect()

    input("Press Enter to start recording...")

    # Save into the repo's lekiwi/recordings/arm folder.
    recordings_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "lekiwi", "recordings", "arm"
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
            try:
                if robot is not None:
                    robot.disconnect()
            except Exception:
                pass
            if leader_arm is not None:
                leader_arm.disconnect()
            print(f"Recording saved to {csv_filename}")


if __name__ == "__main__":
    # How to run (with defaults):
    # python -m scripts.operate.record_arm --name test_arm
    # Or override defaults:
    # python -m scripts.operate.record_arm --name test_arm --ip 172.20.10.3 --port /dev/ttyUSB0
    main()
