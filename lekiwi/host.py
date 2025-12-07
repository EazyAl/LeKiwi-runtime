"""
Run with: lekiwi-host --robot.id=biden_kiwi
Use this if you want to control the robot from the computer client and not autonomously.

LeKiwi Host - ZMQ server for remote robot control
This wraps lerobot's lekiwi_host functionality for use from LeKiwi-runtime
"""

from lerobot.robots.lekiwi.lekiwi_host import main

# TODO: investigate host cycle limit of 30 seconds
if __name__ == "__main__":
    # Simply call lerobot's host main function
    # All command-line arguments are passed through (set connection time to 10 minutes for longer operation)
    # run with: lekiwi-host --robot.id=biden_kiwi --host.connection_time_s=600
    main()  # type: ignore
