import os
import threading
import sys
import asyncio
import logging
from pathlib import Path
import time
import csv
from typing import Optional, Dict, Any, List, Callable

from dotenv import load_dotenv

# Disable tokenizers parallelism to avoid fork warnings with PI0 policy
# Must be set before any HuggingFace imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress verbose DEBUG logging from lerobot cameras and robot before any imports
# Must be done early before the loggers are created
logging.getLogger("lerobot.cameras.opencv.camera_opencv").setLevel(logging.WARNING)
logging.getLogger("lerobot.cameras").setLevel(logging.WARNING)
logging.getLogger("lekiwi.robot.lekiwi").setLevel(logging.INFO)

logger = logging.getLogger(__name__)
from livekit import agents
from livekit.agents import (
    Agent,
    RoomInputOptions,
    AgentSession,
    function_tool,
)
from livekit.plugins import (
    openai,
    noise_cancellation,
)

# LeKiwi robot imports
from lekiwi.robot import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

from lekiwi.services.motors.arms_service import ArmsService
from lekiwi.services.motors.wheels_service import WheelsService
from lekiwi.services.pose_detection.pose_service import (
    PoseDetectionService,
    CameraStream,
    default_visualizer,
)
from lekiwi.vision.camera_hub import CameraHub
from lekiwi.viz.rerun_viz import create_viz, NullViz

load_dotenv()

_RUN_ARGS = {
    "front_idx": 0,
    "wrist_idx": 2,
    "viz_enabled": False,
}


def _load_system_prompt() -> str:
    """Load the system prompt from the personality/system.txt file."""
    # Get the directory where this file is located
    current_dir = Path(__file__).parent
    system_prompt_path = current_dir / "lekiwi" / "personality" / "system.txt"

    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"System prompt file not found at {system_prompt_path}. "
            "Please ensure the file exists."
        )


class LeTars(Agent):
    def __init__(
        self,
        port: str = "/dev/tty.usbmodem58760432781",
        robot_id: str = "biden_kiwi",
        viz_enabled: bool = False,
        front_camera_index: int = 0,
        wrist_camera_index: int = 2,
    ):
        super().__init__(instructions=_load_system_prompt())
        # Bound by the asyncio entrypoint thread after `session.start()`.
        # Used so worker threads can request sentry transitions safely.
        self._runtime_loop: Optional[asyncio.AbstractEventLoop] = None
        self._session: Optional[AgentSession] = None
        self._sentry_task: Optional[asyncio.Task] = None
        self._sentry_schedule_lock = threading.Lock()

        # Initialize single shared robot connection
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        self.robot_config = LeKiwiConfig(
            port=port,
            id=robot_id,
            cameras={
                "front": OpenCVCameraConfig(
                    index_or_path=0, width=1280, height=720, fps=30
                ),
                "wrist": OpenCVCameraConfig(
                    index_or_path=1, width=1280, height=720, fps=30
                ),
            },
        )
        self.robot = LeKiwi(self.robot_config)
        self.robot.connect(calibrate=False)

        # Visualization wiring
        self.viz_enabled = viz_enabled
        self.viz = create_viz(viz_enabled, app_id="lekiwi_viz")
        self.camera_hub = None
        self.front_sub_pose = None
        self.front_sub_nav = None
        self.front_sub_viz = None
        self.wrist_sub_viz = None
        self._viz_pump_stop = threading.Event()
        self._viz_pump_threads: List[threading.Thread] = []
        if self.viz_enabled:
            self.camera_hub = CameraHub(
                front_index=front_camera_index,
                wrist_index=wrist_camera_index,
                fps=30,
            )
            self.camera_hub.start()
            self.front_sub_pose = self.camera_hub.subscribe_front(max_queue=2)
            self.front_sub_nav = self.camera_hub.subscribe_front(max_queue=2)
            self.wrist_sub_viz = self.camera_hub.subscribe_wrist(max_queue=2)
            self._start_camera_pumps()
            self.viz.set_status("normal", ts=time.time())
        else:
            self.viz = NullViz()
            self.camera_hub = None
            self.front_sub_pose = None
            self.front_sub_nav = None
            self.wrist_sub_viz = None

        # Lock to serialize access to robot motor commands (serial port is not thread-safe)
        self.robot_lock = threading.Lock()

        # Three services running on separate threads, with agent dispatching events to them
        self.wheels_service = WheelsService(
            robot=self.robot, robot_lock=self.robot_lock
        )
        self.arms_service = ArmsService(robot=self.robot, robot_lock=self.robot_lock)
        self.pose_service = PoseDetectionService(
            status_callback=self._handle_pose_status,
            camera=CameraStream(index=front_camera_index),
            visualizer=None if self.viz_enabled else default_visualizer,
            frame_subscription=self.front_sub_pose,
            viz=self.viz,
        )

        # Initialize operational mode (normal or sentry)
        self.is_sentry = False
        self._push_status_to_viz()

        # Start robot services
        self.wheels_service.start()
        self.arms_service.start()
        self.pose_service.start()

        # Wake up animation
        self.arms_service.dispatch("play", "wake_up")

    def _start_camera_pumps(self):
        """Pump camera frames into viz in background threads."""
        if not self.viz_enabled or not self.camera_hub:
            return

        sub = self.wrist_sub_viz
        if not sub:
            return

        def pump_wrist():
            while not self._viz_pump_stop.is_set():
                pulled = sub.pull(timeout=0.5)
                if pulled:
                    ts, frame = pulled
                    self.viz.log_wrist_rgb(frame, ts=ts)

        t = threading.Thread(target=pump_wrist, name="viz-wrist-pump", daemon=True)
        self._viz_pump_threads.append(t)
        t.start()

    def _stop_camera_pumps(self, timeout: float = 1.0) -> None:
        self._viz_pump_stop.set()
        for t in list(self._viz_pump_threads):
            try:
                if t.is_alive():
                    t.join(timeout=timeout)
            except Exception:
                pass
        self._viz_pump_threads.clear()

    def _push_status_to_viz(self) -> None:
        if not getattr(self, "viz", None):
            return
        try:
            self.viz.set_status(self.status, ts=time.time())
        except Exception:
            pass

    @property
    def status(self) -> str:
        return "sentry" if self.is_sentry else "normal"

    def bind_runtime(
        self, *, loop: asyncio.AbstractEventLoop, session: AgentSession
    ) -> None:
        """Bind asyncio loop + session (called from async entrypoint thread)."""
        self._runtime_loop = loop
        self._session = session

    def _log_tool(
        self, name: str, message: str, level: str = "info", emoji: str = "🧰"
    ):
        if not getattr(self, "viz", None):
            return
        try:
            self.viz.log_tool_call(
                name, message, level=level, emoji=emoji, ts=time.time()
            )
        except Exception:
            pass

    def __del__(self):
        """Cleanup: disconnect robot when agent is destroyed"""
        if hasattr(self, "robot") and self.robot:
            try:
                self.robot.disconnect()
            except:
                pass  # Ignore errors during cleanup
        try:
            self._stop_camera_pumps()
            hub = getattr(self, "camera_hub", None)
            if hub is not None:
                hub.stop()
            viz = getattr(self, "viz", None)
            if viz is not None:
                viz.close()
        except Exception:
            pass

    def _handle_pose_status(self, status_type: str, details: dict):
        """
        Callback method to receive status updates from the PoseDetectionService.
        This runs in the context of the service's worker thread, but is called by it.
        """
        print(
            f"LeKiwi: Received pose status update - Type: {status_type}, Details: {details}"
        )

        if status_type == "PERSON_FALLEN":
            # This callback runs on the pose worker thread.
            # Best practice: do NOT stop/join other threads from here.
            self._request_enter_sentry(details)
        elif status_type == "PERSON_STABLE":
            self._request_exit_sentry(details)

    def _request_enter_sentry(self, details: Dict[str, Any]) -> None:
        loop = self._runtime_loop
        if loop is None:
            self.is_sentry = True
            self._push_status_to_viz()
            logger.warning("LeKiwi: sentry requested but runtime loop not bound yet")
            return

        def _schedule() -> None:
            with self._sentry_schedule_lock:
                if self._sentry_task is not None and not self._sentry_task.done():
                    return
                self._sentry_task = asyncio.create_task(self._enter_sentry(details))

        loop.call_soon_threadsafe(_schedule)

    def _request_exit_sentry(self, details: Dict[str, Any]) -> None:
        loop = self._runtime_loop
        if loop is None:
            self.is_sentry = False
            self._push_status_to_viz()
            return

        def _apply() -> None:
            if self.is_sentry:
                self.is_sentry = False
                self._push_status_to_viz()
                logger.info("LeKiwi: Person stable detected, switching to normal mode")

        loop.call_soon_threadsafe(_apply)

    def _stop_workers_for_sentry(self) -> None:
        """
        Quiesce all background threads we own (cooperative stop + timeouts).
        This ensures sentry mode runs without any of our worker threads operating.
        """
        try:
            self._stop_camera_pumps(timeout=1.0)
        except Exception:
            pass

        try:
            self.pose_service.stop(timeout=2.0)
        except Exception:
            pass

        try:
            self.wheels_service.stop(timeout=2.0)
        except Exception:
            pass
        try:
            self.arms_service.stop(timeout=2.0)
        except Exception:
            pass

        try:
            if self.camera_hub is not None:
                self.camera_hub.stop()
        except Exception:
            pass

    def _start_workers_after_sentry(self) -> None:
        """Restart background workers after sentry sequence completes."""
        try:
            if self.viz_enabled and self.camera_hub is not None:
                self.camera_hub.start()
                self._viz_pump_stop.clear()
                self._start_camera_pumps()
        except Exception:
            pass

        try:
            self.wheels_service.start()
        except Exception:
            pass
        try:
            self.arms_service.start()
        except Exception:
            pass
        try:
            self.pose_service.start()
        except Exception:
            pass

    async def _speak(self, text: str) -> None:
        if self._session is None:
            logger.warning(f"LeKiwi: (no session) would say: {text}")
            return
        try:
            await self._session.generate_reply(instructions=f"Say: '{text}'")
        except Exception as e:
            logger.exception(f"LeKiwi: failed to speak: {e}")

    @staticmethod
    def _load_csv_actions(csv_path: str) -> List[Dict[str, float]]:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            actions: List[Dict[str, float]] = []
            for row in reader:
                action = {k: float(v) for k, v in row.items() if k and k != "timestamp"}
                actions.append(action)
            return actions

    async def _play_actions(
        self,
        actions: List[Dict[str, float]],
        *,
        fps: int,
        send_fn: Callable[[Dict[str, float]], Any],
    ) -> None:
        dt = 1.0 / max(1, int(fps))
        for action in actions:
            with self.robot_lock:
                send_fn(action)
            await asyncio.sleep(dt)

    async def _run_sentry_tasks(self, details: Dict[str, Any]) -> None:
        """
        Add your deterministic step sequence here.
        Runs on the asyncio thread while worker threads are quiesced.
        """
        # Example tasks (safe because we hold robot_lock for every write):
        try:
            base_csv = os.path.join(
                os.path.dirname(__file__),
                "lekiwi",
                "recordings",
                "wheels",
                "wiggle.csv",
            )
            arm_csv = os.path.join(
                os.path.dirname(__file__), "lekiwi", "recordings", "arm", "greeting.csv"
            )
            base_actions = self._load_csv_actions(base_csv)
            arm_actions = self._load_csv_actions(arm_csv)

            await self._play_actions(
                base_actions, fps=30, send_fn=lambda a: self.robot.send_base_action(a)
            )
            await self._play_actions(
                arm_actions, fps=30, send_fn=lambda a: self.robot.send_arm_action(a)
            )
        except Exception as e:
            logger.exception(f"LeKiwi: sentry tasks failed: {e}")

    async def _enter_sentry(self, details: Dict[str, Any]) -> None:
        """
        Sentry sequence:
        - stop all worker threads we own
        - say the phrase first
        - run tasks on main thread
        - restart workers afterwards
        """
        if not self.is_sentry:
            self.is_sentry = True
            self._push_status_to_viz()

        logger.info("LeKiwi: entering sentry mode (quiescing workers)")
        await asyncio.to_thread(self._stop_workers_for_sentry)

        await self._speak("Oh no! Is everything okay?")
        await self._run_sentry_tasks(details)

        logger.info("LeKiwi: sentry sequence complete (restarting workers)")
        await asyncio.to_thread(self._start_workers_after_sentry)

        # Exit sentry mode after we’ve resumed normal worker operation.
        self.is_sentry = False
        self._push_status_to_viz()

    @function_tool
    async def get_available_recordings(self) -> str:
        """
        Use this tool to double check what recordings (i.e. physical expressions) you can input to the play_recording tool.
        Use this when you're curious about what physical expressions you can perform, or when someone
        asks about your capabilities. Each recording is a choreographed movement that shows personality -
        like head tilts, nods, excitement wiggles, or confused gestures. Check this regularly to remind
        yourself of your expressive range!

        Returns:
            List of available physical expression recordings you can perform.
        """
        print("LeKiwi: get_available_recordings function called")
        self._log_tool("get_available_recordings", "call")
        try:
            recordings = self.arms_service.get_available_recordings()
            recordings += self.wheels_service.get_available_recordings()

            if recordings:
                result = f"Available recordings: {', '.join(recordings)}"
                level = "info"
            else:
                result = "No recordings found."
                level = "info"
        except Exception as e:
            result = f"Error getting recordings: {str(e)}"
            level = "error"

        self._log_tool("get_available_recordings", result, level=level)
        return result

    @function_tool
    async def play_recording(self, recording_name: str, type: str = "arm") -> str:
        """
        Use this constantly to show personality and emotion.
        Perfect for: greeting gestures, excited bounces, confused head tilts, thoughtful nods,
        celebratory wiggles, disappointed slouches, or any emotional response that needs body language.
        Use this tool frequently to show you're alive and have personality.
        Don't just talk, MOVE!

        You have the following recordings available:
        - For arms: greeting, wake_up
        - For wheels: charge_forwards, spin, wiggle

        Args:
            recording_name: Name of the physical expression to perform (listed above)
            type: arm or wheels
        """
        print(
            f"LeKiwi: play_recording function called with recording_name: {recording_name}"
        )
        self._log_tool("play_recording", f"call {recording_name}")
        try:
            # Send play event to animation service
            if type == "arm":
                # self.arms_service.dispatch("play", recording_name)
                return "Arm service is currently disabled"
            elif type == "wheels":
                self.wheels_service.dispatch("play", recording_name)
            else:
                result = f"Error: type must be either 'arms' or 'wheels', got '{type}'"
                self._log_tool("play_recording", result, level="error")
                return result
            result = f"Started playing recording: {recording_name}"
        except Exception as e:
            result = f"Error playing recording {recording_name}: {str(e)}"
        level = "error" if result.lower().startswith("error") else "info"
        self._log_tool("play_recording", result, level=level)
        return result


# Entry to the agent
async def entrypoint(ctx: agents.JobContext):
    # Use pre-parsed arguments from __main__
    front_idx = _RUN_ARGS["front_idx"]
    wrist_idx = _RUN_ARGS["wrist_idx"]
    viz_enabled = _RUN_ARGS["viz_enabled"]

    # Initialize agent with streaming enabled if requested
    agent = LeTars(
        viz_enabled=viz_enabled,
        front_camera_index=front_idx,
        wrist_camera_index=wrist_idx,
    )

    session = AgentSession(llm=openai.realtime.RealtimeModel(voice="verse"))

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            audio_enabled=True,
            audio_sample_rate=16000,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Bind runtime so worker threads can request sentry mode safely.
    agent.bind_runtime(loop=asyncio.get_running_loop(), session=session)

    await session.generate_reply(
        instructions=f"""When you wake up, greet with: 'Hello.'"""
    )


if __name__ == "__main__":
    # Run with: python main_workflows.py dev
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--viz", action="store_true", help="Enable Rerun visualization (default off)"
    )
    parser.add_argument(
        "--front-camera-index",
        type=int,
        default=1,
        help="Override front camera index (default 0)",
    )
    parser.add_argument(
        "--wrist-camera-index",
        type=int,
        default=0,
        help="Override wrist camera index (default 2)",
    )
    args, unknown = parser.parse_known_args()

    # Capture custom flags for use in entrypoint and strip them from argv before LiveKit parses.
    _RUN_ARGS["viz_enabled"] = bool(args.viz)
    _RUN_ARGS["front_idx"] = int(args.front_camera_index)
    _RUN_ARGS["wrist_idx"] = int(args.wrist_camera_index)

    # Remove parsed args so LiveKit CLI doesn't see them as unknown
    sys.argv = [sys.argv[0]] + unknown

    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, num_idle_processes=1)  # type: ignore[arg-type]
    )
