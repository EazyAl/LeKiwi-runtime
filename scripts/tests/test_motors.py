import argparse
import sys
import time
from pathlib import Path

# Add lekiwi to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from lekiwi.services.motors import ArmsService, WheelsService


def test_arms():
    """Test ArmsService - play available arm recordings"""
    print("\n=== Testing Arms Service ===")

    arms_service = ArmsService(port=args.port, robot_id=args.id)
    arms_service.start()

    try:
        print("Getting available arm recordings...")
        recordings = arms_service.get_available_recordings()
        print(f"Available: {recordings}")

        if recordings:
            # Play first recording
            print(f"\nPlaying: {recordings[0]}")
            arms_service.dispatch("play", recordings[0])
            time.sleep(3)  # Let it play for a bit
            print("Arm test completed!")
        else:
            print("No arm recordings found.")
    finally:
        arms_service.stop()


def test_wheels():
    """Test WheelsService - play available wheel recordings"""
    print("\n=== Testing Wheels Service ===")

    wheels_service = WheelsService(port=args.port, robot_id=args.id)
    wheels_service.start()

    try:
        print("Getting available wheel recordings...")
        recordings = wheels_service.get_available_recordings()
        print(f"Available: {recordings}")

        if recordings:
            # Play first recording
            print(f"\nPlaying: {recordings[0]}")
            wheels_service.dispatch("play", recordings[0])
            wheels_service.wait_until_idle(timeout=30)
            print("Wheels test completed!")
        else:
            print("No wheel recordings found.")
    finally:
        wheels_service.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LeKiwi motor services")
    parser.add_argument("--id", type=str, default="biden_kiwi", help="Robot ID")
    parser.add_argument("--port", type=str, default="/dev/ttyACM0", help="Serial port")
    parser.add_argument(
        "--test",
        type=str,
        choices=["arms", "wheels", "both"],
        default="both",
        help="Which service to test",
    )
    args = parser.parse_args()

    print(f"Testing motors for {args.id} on {args.port}")

    if args.test in ["arms", "both"]:
        test_arms()

    if args.test in ["wheels", "both"]:
        test_wheels()

    print("\n✓ Motor tests completed!")
