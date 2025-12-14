import os
import csv
import time
import threading
from typing import Any, List, Dict, Optional
from ..base import ServiceBase
from lerobot.robots.lekiwi import LeKiwiConfig
from lekiwi.robot import LeKiwi


class WheelsService(ServiceBase):
    def __init__(
        self,
        port: str = None,
        robot_id: str = None,
        robot: LeKiwi = None,
        robot_lock: threading.Lock = None,
        fps: int = 30,
    ):
        super().__init__("motors")
        if robot is not None:
            # Use provided robot instance (shared connection)
            self.robot = robot
            self.robot_lock = robot_lock if robot_lock is not None else threading.Lock()
            self._owns_robot = False
            self.port = None
            self.robot_id = None
        else:
            # Create own robot connection (backward compatibility)
            if port is None or robot_id is None:
                raise ValueError(
                    "Either robot instance or (port, robot_id) must be provided"
                )
            self.robot = None
            self._owns_robot = True
            self.port = port
            self.robot_id = robot_id
            self.robot_lock = threading.Lock()  # Own lock if creating own connection
            self.robot_config = LeKiwiConfig(
                port=port, id=robot_id, cameras={}
            )  # TODO: add cameras later if needed
        self.fps = fps
        self.recordings_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "recordings", "wheels"
        )

    def start(self):
        super().start()
        if self._owns_robot:
            self.robot = LeKiwi(self.robot_config)
            self.robot.connect(calibrate=False)

            # Suppress verbose DEBUG logs from lerobot library
            import logging

            logging.getLogger("lerobot.robots.lekiwi.lekiwi").setLevel(logging.INFO)

            self.logger.info(f"Wheels service connected to {self.port}")
        else:
            self.logger.info("Wheels service using shared robot connection")

    def stop(self, timeout: float = 5.0):
        if self._owns_robot and self.robot:
            self.robot.disconnect()
            self.robot = None
        super().stop(timeout)

    def handle_event(self, event_type: str, payload: Any):
        if event_type == "play":
            self._handle_play(payload)
        else:
            self.logger.warning(f"Unknown event type: {event_type}")

    def _handle_play(self, recording_name: str):
        """Play a recording by name"""
        if not self.robot:
            self.logger.error("Robot not connected")
            return

        csv_filename = f"{recording_name}.csv"
        csv_path = os.path.join(self.recordings_dir, csv_filename)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Recording not found: {csv_path}")

        try:
            with open(csv_path, "r") as csvfile:
                csv_reader = csv.DictReader(csvfile)
                actions = list(csv_reader)

            self.logger.info(f"Playing {len(actions)} actions from {recording_name}")

            for idx, row in enumerate(actions):
                t0 = time.perf_counter()

                # Extract action data (exclude timestamp column)
                base_action = {
                    key: float(value)
                    for key, value in row.items()
                    if key != "timestamp"
                }

                # Send base action directly using the new method (with lock for thread safety)
                with self.robot_lock:
                    self.robot.send_base_action(base_action)

                # Use time.sleep instead of busy_wait to avoid blocking other threads
                sleep_time = 1.0 / self.fps - (time.perf_counter() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            self.logger.info(f"Finished playing recording: {recording_name}")

        except Exception as e:
            self.logger.error(f"Error playing recording {recording_name}: {e}")

    def get_available_recordings(self) -> List[str]:
        """Get list of recording names available for this lamp ID"""
        if not os.path.exists(self.recordings_dir):
            return []

        recordings = []
        suffix = f".csv"

        for filename in os.listdir(self.recordings_dir):
            if filename.endswith(suffix):
                # Remove the lamp_id suffix to get the recording name
                recording_name = filename[: -len(suffix)]
                recordings.append(recording_name)

        return sorted(recordings)
