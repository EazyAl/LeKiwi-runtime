"""
LeKiwi Host - ZMQ server for remote robot control
This wraps lerobot's lekiwi_host functionality for use from LeKiwi-runtime
"""

from lerobot.robots.lekiwi.lekiwi_host import main

if __name__ == "__main__":
    # Simply call lerobot's host main function
    # All command-line arguments are passed through
    main()  # type: ignore
