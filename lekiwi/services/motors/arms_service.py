import os
import csv
import time
import threading
from typing import Any, List, Dict, Optional, Tuple
from lerobot.robots.lekiwi import LeKiwiConfig
from lekiwi.robot import LeKiwi


class ArmsService:
    def __init__(
        self,
        port: str | None = None,
        robot_id: str | None = None,
        robot: LeKiwi = None,
        robot_lock: threading.Lock = None,
        fps: int = 30,
        duration: float = 0.6,
        idle_recording: str = "idle",
    ):
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
        self.duration = duration
        self.idle_recording = idle_recording
        self.recordings_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "recordings", "arm"
        )

        # State management
        self._recording_cache: Dict[str, List[Dict[str, float]]] = {}
        self._current_state: Optional[Dict[str, float]] = None
        self._current_recording: Optional[str] = None
        self._current_frame_index: int = 0
        self._current_actions: List[Dict[str, float]] = []
        self._interpolation_frames: int = 0
        self._interpolation_total_frames: int = 0
        self._interpolation_target: Optional[Dict[str, float]] = None

        # Custom event handling
        self._running = threading.Event()
        self._paused = threading.Event()  # For temporarily pausing playback
        self._event_queue = []
        self._event_lock = threading.Lock()
        self._event_thread: Optional[threading.Thread] = None

    def start(self, *, start_idle: bool = True, preload: Optional[List[str]] = None):
        if self._owns_robot:
            self.robot = LeKiwi(self.robot_config)
            self.robot.connect(calibrate=False)
            print(f"Arms service connected to {self.port}")
        else:
            print("Arms service using shared robot connection")

        # Preload recordings into memory to avoid disk IO on first play.
        if preload:
            for name in preload:
                try:
                    self._load_recording(name)
                except Exception as e:
                    print(f"Arms service failed to preload '{name}': {e}")

        # Start event processing thread
        self._running.set()
        self._event_thread = threading.Thread(target=self._event_loop, daemon=True)
        self._event_thread.start()

        # Initialize with idle recording via self dispatch (optional)
        if start_idle:
            self.dispatch("play", self.idle_recording)

    def stop(self, timeout: float = 5.0):
        # Stop event processing
        self._running.clear()
        if self._event_thread and self._event_thread.is_alive():
            self._event_thread.join(timeout=timeout)

        if self._owns_robot and self.robot:
            self.robot.disconnect()
            self.robot = None

    def dispatch(self, event_type: str, payload: Any):
        """Dispatch an event - same interface as ServiceBase"""
        if not self._running.is_set():
            print(f"Arms service is not running, ignoring event {event_type}")
            return

        with self._event_lock:
            self._event_queue.append((event_type, payload))

    def pause(self):
        """Pause arm playback temporarily (e.g., during epipen administration)"""
        self._paused.set()
        print("Arms service paused")

    def resume(self):
        """Resume arm playback after pause"""
        self._paused.clear()
        print("Arms service resumed")

    def _event_loop(self):
        """Custom event loop that supports interruption"""
        while self._running.is_set():
            # Skip playback if paused (but still process events)
            if self._paused.is_set():
                time.sleep(0.05)  # Sleep while paused
                continue

            # Check for events
            with self._event_lock:
                if self._event_queue:
                    event_type, payload = self._event_queue.pop(0)
                else:
                    event_type, payload = None, None

            if event_type:
                try:
                    self.handle_event(event_type, payload)
                except Exception as e:
                    print(f"Error handling event {event_type}: {e}")

            # Continue current playback
            self._continue_playback()

            time.sleep(1.0 / self.fps)  # Frame rate timing

    def handle_event(self, event_type: str, payload: Any):
        if event_type == "play":
            self._handle_play(payload)
        else:
            print(f"Unknown event type: {event_type}")

    def _handle_play(self, recording_name: str):
        """Start playing a recording with interpolation from current state"""
        if not self.robot:
            print("Robot not connected")
            return

        # Load the recording
        actions = self._load_recording(recording_name)
        if actions is None:
            return

        print(f"Starting {recording_name} with interpolation")

        # Set up new playback
        self._current_recording = recording_name
        self._current_actions = actions
        self._current_frame_index = 0

        # If we have a current state, set up interpolation to the first frame
        if self._current_state is not None:
            self._interpolation_total_frames = max(1, int(self.duration * self.fps))
            self._interpolation_frames = self._interpolation_total_frames
            self._interpolation_target = actions[0]
        else:
            self._interpolation_frames = 0
            self._interpolation_total_frames = 0
            self._interpolation_target = None

    def _continue_playback(self):
        """Continue current playback - called every frame"""
        if not self._current_recording or not self._current_actions:
            return

        if not self.robot:
            print("Robot not connected")
            return

        try:
            # Handle interpolation to first frame
            if (
                self._interpolation_frames > 0
                and self._interpolation_target is not None
            ):
                # Calculate interpolation progress
                total = max(1, int(self._interpolation_total_frames))
                progress = 1.0 - (self._interpolation_frames / total)
                progress = max(0.0, min(1.0, progress))

                # Interpolate between current state and target
                interpolated_action = {}
                for joint in self._interpolation_target.keys():
                    current_val = self._current_state.get(joint, 0)
                    target_val = self._interpolation_target[joint]
                    interpolated_action[joint] = (
                        current_val + (target_val - current_val) * progress
                    )

                # Send arm action directly using the new method (with lock for thread safety)
                with self.robot_lock:
                    self.robot.send_arm_action(interpolated_action)
                self._current_state = interpolated_action.copy()
                self._interpolation_frames -= 1
                return

            # Play current frame
            if self._current_frame_index < len(self._current_actions):
                action = self._current_actions[self._current_frame_index]
                # Send arm action directly using the new method (with lock for thread safety)
                with self.robot_lock:
                    self.robot.send_arm_action(action)
                self._current_state = action.copy()
                self._current_frame_index += 1
            else:
                # Recording finished
                if self._current_recording != self.idle_recording:
                    # Interpolate back to idle
                    idle_actions = self._load_recording(self.idle_recording)
                    if idle_actions is not None and len(idle_actions) > 0:
                        self._current_recording = self.idle_recording
                        self._current_actions = idle_actions
                        self._current_frame_index = 0
                        # Set up interpolation back to idle
                        if self._current_state is not None:
                            self._interpolation_total_frames = max(
                                1, int(self.duration * self.fps)
                            )
                            self._interpolation_frames = (
                                self._interpolation_total_frames
                            )
                            self._interpolation_target = idle_actions[0]
                else:
                    # Loop idle recording
                    self._current_frame_index = 0

        except Exception as e:
            print(f"Error in playback: {e}")
            # Reset to safe state
            self._current_recording = None
            self._current_actions = []
            self._current_frame_index = 0

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

    def _load_recording(self, recording_name: str) -> Optional[List[Dict[str, float]]]:
        """Load a recording from cache or file"""
        # Check cache first
        if recording_name in self._recording_cache:
            return self._recording_cache[recording_name]

        csv_filename = f"{recording_name}.csv"
        csv_path = os.path.join(self.recordings_dir, csv_filename)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Recording not found: {csv_path}")

        try:
            with open(csv_path, "r") as csvfile:
                csv_reader = csv.DictReader(csvfile)
                actions = []
                for row in csv_reader:
                    # Extract action data (exclude timestamp column)
                    action = {
                        key: float(value)
                        for key, value in row.items()
                        if key != "timestamp"
                    }
                    actions.append(action)

            # Cache the recording
            self._recording_cache[recording_name] = actions
            return actions

        except Exception as e:
            print(f"Error loading recording {recording_name}: {e}")
            return None
