import sys
import os
import argparse

sys.path.append(os.path.dirname(__file__))

from lerobot.robots.lekiwi import LeKiwiConfig, LeKiwi


def turn_off(port: str, robot_id: str):
    # Initialize robot connection
    robot_config = LeKiwiConfig(port=port, id=robot_id, cameras={})
    robot = LeKiwi(robot_config)

    try:
        # Connect to robot
        print(f"Connecting to robot on port {port} with ID {robot_id}...")
        robot.connect(calibrate=False)
        print("Robot connected successfully")

        # Stop base motors before disconnecting
        print("Stopping base motors...")
        robot.stop_base()

        print("Turn off complete")

    except Exception as e:
        print(f"Error during turn off: {e}")
    finally:
        # Clean up connections - disconnect() will stop base and disable torque
        if robot.is_connected:
            print("Disconnecting robot...")
            robot.disconnect()
            print("Robot disconnected")


def main():
    parser = argparse.ArgumentParser(
        description="Turn off LeKiwi robot motors and disconnect"
    )
    parser.add_argument(
        "--id", type=str, default="biden_kiwi", help="ID of the robot"
    )
    parser.add_argument(
        "--port", type=str, default="/dev/ttyACM0", help="Serial port for the robot"
    )
    args = parser.parse_args()

    turn_off(args.port, args.id)


if __name__ == "__main__":
    print("Turning off robot...")

    main()
