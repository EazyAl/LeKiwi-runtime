import sys
from pathlib import Path
import time

from dotenv import load_dotenv
from livekit import rtc, agents
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
from lekiwi.services.motors import ArmsService, WheelsService
from lekiwi.services.pose_detection import (
    PoseDetectionService,
    CameraStream,
    PoseEstimator,
    FallDetector,
)
from lekiwi.viz.rerun_viz import create_viz, NullViz
from lekiwi.vision.camera_hub import CameraHub

load_dotenv()

_RUN_ARGS = {
    "stream_enabled": False,
    "stream_port": 5556,
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
        stream_data: bool = False,
        stream_port: int = 5556,
        viz_enabled: bool = False,
        front_camera_index: int = 0,
        wrist_camera_index: int = 2,
    ):
        super().__init__(instructions=_load_system_prompt())
        self.viz_enabled = viz_enabled
        self.viz = create_viz(viz_enabled, app_id="lekiwi_viz")
        self.camera_hub = None
        self.front_sub_pose = None
        self.wrist_sub_viz = None
        self.status = "normal"
        if viz_enabled:
            self.camera_hub = CameraHub(
                front_index=front_camera_index,
                wrist_index=wrist_camera_index,
                fps=30,
            )
            self.camera_hub.start()
            self.front_sub_pose = self.camera_hub.subscribe_front(max_queue=2)
            self.wrist_sub_viz = self.camera_hub.subscribe_wrist(max_queue=2)
            self._start_camera_pumps()
            self._push_status_to_viz()
        else:
            self.viz = NullViz()
            self.camera_hub = None
            self.front_sub_pose = None
            self.wrist_sub_viz = None
            self._push_status_to_viz()
        # Three services running on separate threads, with LeKiwi agent dispatching events to them
        self.wheels_service = WheelsService(port=port, robot_id=robot_id)
        self.arms_service = ArmsService(port=port, robot_id=robot_id)
        camera_stream = CameraStream()
        pose_estimator = PoseEstimator()
        fall_detector = FallDetector()
        self.pose_service = PoseDetectionService(
            camera=camera_stream,
            pose=pose_estimator,
            detector=fall_detector,
            status_callback=self._handle_pose_status,  # callback method
            frame_subscription=self.front_sub_pose,
            viz=self.viz,
        )

        self.wheels_service.start()
        self.arms_service.start()
        self.pose_service.start()

        # Wake up
        self.arms_service.dispatch("play", "wake_up")

    def __del__(self):
        try:
            hub = getattr(self, "camera_hub", None)
            if hub is not None:
                hub.stop()
            viz = getattr(self, "viz", None)
            if viz is not None:
                viz.close()
        except Exception:
            pass

    def _start_camera_pumps(self):
        if not self.viz_enabled or not self.camera_hub:
            return

        sub = self.wrist_sub_viz
        if not sub:
            return

        def pump_wrist():
            while True:
                pulled = sub.pull(timeout=0.5)
                if pulled:
                    ts, frame = pulled
                    self.viz.log_wrist_rgb(frame, ts=ts)

        import threading

        threading.Thread(target=pump_wrist, name="viz-wrist-pump", daemon=True).start()

    def _push_status_to_viz(self):
        if not self.viz:
            return
        try:
            self.viz.set_status(self.status, ts=time.time())
        except Exception:
            pass

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

    def _handle_pose_status(self, status_type: str, details: dict):
        """
        Callback method to receive status updates from the PoseDetectionService.
        This runs in the context of the service's worker thread, but is called by it.
        """
        print(
            f"LeKiwi: Received pose status update - Type: {status_type}, Details: {details}"
        )

        if status_type == "PERSON_FALLEN":
            # Example 2: Dispatch a HIGH-priority motor action (e.g., look up, check)
            # administer the epipen
            # log it
            print(f"LeKiwi: Person fallen detected, dispatching spin action")
            self.status = "concerned"
            self._push_status_to_viz()

        elif status_type == "PERSON_STABLE":
            self.status = "normal"
            self._push_status_to_viz()

            # Example 3: Log the event for the main LLM loop to pick up (complex, but robust)
            # You might set a flag or put an event in a queue monitored by the agent's reply loop.
            pass

    @function_tool
    async def get_available_recordings(self) -> str:
        """
        Discover your physical expressions! Get your repertoire of motor movements for body language.
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
    async def play_recording(self, recording_name: str) -> str:
        """
        Express yourself through physical movement! Use this constantly to show personality and emotion.
        Perfect for: greeting gestures, excited bounces, confused head tilts, thoughtful nods,
        celebratory wiggles, disappointed slouches, or any emotional response that needs body language.
        Combine with RGB colors for maximum expressiveness! Your movements are like a dog wagging its tail -
        use them frequently to show you're alive, engaged, and have personality. Don't just talk, MOVE!

        Args:
            recording_name: Name of the physical expression to perform (use get_available_recordings first)
        """
        print(
            f"LeKiwi: play_recording function called with recording_name: {recording_name}"
        )
        self._log_tool("play_recording", f"call {recording_name}")
        try:
            # Send play event to animation service
            self.arms_service.dispatch("play", recording_name)
            result = f"Started playing recording: {recording_name}"
        except Exception as e:
            result = f"Error playing recording {recording_name}: {str(e)}"
        level = "error" if result.lower().startswith("error") else "info"
        self._log_tool("play_recording", result, level=level)
        return result

    @function_tool
    async def get_configuration(self) -> str:
        """
        Get the status of the robot.
        """
        # TODO: Implement this with proper configuration checking and return as json () - see https://github.com/TARS-AI-Community/TARS-AI/blob/V2/src/character/TARS/persona.ini
        result = "Status: Nominal"
        self._log_tool("get_configuration", result)
        return result


# Entry to the agent
async def entrypoint(ctx: agents.JobContext):
    # Use pre-parsed args from __main__
    stream_enabled = _RUN_ARGS["stream_enabled"]
    stream_port = _RUN_ARGS["stream_port"]
    front_idx = _RUN_ARGS["front_idx"]
    wrist_idx = _RUN_ARGS["wrist_idx"]
    viz_enabled = _RUN_ARGS["viz_enabled"]

    agent = LeTars(
        stream_data=stream_enabled,
        stream_port=stream_port,
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

    await session.generate_reply(
        instructions=f"""When you wake up, greet with: 'Hello.'"""
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--stream", action="store_true", help="Enable data streaming for visualization"
    )
    parser.add_argument(
        "--stream-port", type=int, default=5556, help="Port for ZMQ data streaming"
    )
    parser.add_argument(
        "--viz", action="store_true", help="Enable rerun visualization (off by default)"
    )
    parser.add_argument(
        "--front-camera-index",
        type=int,
        default=0,
        help="Override front camera index (default 0)",
    )
    parser.add_argument(
        "--wrist-camera-index",
        type=int,
        default=2,
        help="Override wrist camera index (default 2)",
    )
    args, unknown = parser.parse_known_args()

    _RUN_ARGS["stream_enabled"] = bool(args.stream)
    _RUN_ARGS["stream_port"] = int(args.stream_port)
    _RUN_ARGS["viz_enabled"] = bool(args.viz)
    _RUN_ARGS["front_idx"] = int(args.front_camera_index)
    _RUN_ARGS["wrist_idx"] = int(args.wrist_camera_index)

    # Strip parsed args so LiveKit CLI doesn't see them
    sys.argv = [sys.argv[0]] + unknown

    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, num_idle_processes=1)
    )
