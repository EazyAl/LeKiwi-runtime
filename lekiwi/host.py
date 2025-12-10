"""
Run with: python -m lekiwi.host
Use this if you want to control the robot from the computer client and not autonomously.

LeKiwi Host - ZMQ server for remote robot control
This wraps lerobot's lekiwi_host functionality for use from LeKiwi-runtime
"""

import sys
from lerobot.robots.lekiwi.lekiwi_host import main

# TODO: investigate host cycle limit of 30 seconds
if __name__ == "__main__":
    # Set default values for command-line arguments if not provided
    if not any("robot.port" in arg for arg in sys.argv):
        sys.argv.append("--robot.port=/dev/ttyACM0")
    if not any("robot.id" in arg for arg in sys.argv):
        sys.argv.append("--robot.id=biden_kiwi")
    if not any("host.connection_time_s" in arg for arg in sys.argv):
        sys.argv.append("--host.connection_time_s=600")

    main()  # type: ignore
