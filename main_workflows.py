import os
import threading
import sys
import asyncio
import logging
from pathlib import Path
import time
from typing import Optional, Dict, Any

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

# LeKiwi robot imports
from lekiwi.robot import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

# from lekiwi.services import Priority
from lekiwi.services.motors.arms_service import ArmsService
from lekiwi.services.motors.wheels_service import WheelsService
from lekiwi.services.pose_detection.pose_service import (
    PoseDetectionService,
    CameraStream,
    default_visualizer,
)
from lekiwi.workflows.workflows import WorkflowService
from lekiwi.services.navigation import Navigator
from lekiwi.vision.camera_hub import CameraHub
from lekiwi.viz.rerun_viz import create_viz, NullViz

# import zmq

load_dotenv()

_RUN_ARGS = {
    "front_idx": 0,
    "wrist_idx": 2,
    "viz_enabled": False,
}


async def _generate_reply(session: AgentSession, instructions: str) -> None:
    # Wrapper so we always pass a coroutine to run_coroutine_threadsafe().
    await session.generate_reply(instructions=instructions)


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


# Parse workflow arguments from environment variable (LiveKit CLI intercepts command-line args)
def parse_workflow_args():
    """
    Parse which workflows to preload from environment variable.
    LiveKit CLI intercepts command-line arguments, so we use environment variables instead.

    Usage:
        # Single workflow (production/console mode):
        WORKFLOWS=help uv run main_workflows.py console

        # Multiple workflows:
        WORKFLOWS=help,emergency uv run main_workflows.py console

        # All workflows (if WORKFLOWS not set):
        uv run main_workflows.py console

        # Development mode (for testing with LiveKit dev environment):
        WORKFLOWS=help uv run main_workflows.py dev
    """
    env_workflows = os.getenv("WORKFLOWS")
    if env_workflows:
        workflows = [w.strip() for w in env_workflows.split(",") if w.strip()]
        logger.debug(f"[CONFIG] Loading workflows from WORKFLOWS env var: {workflows}")
        return workflows
    else:
        logger.debug(
            "[CONFIG] No WORKFLOWS env var set, will load all available workflows"
        )
        return None


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

        # LiveKit session bridge (set in entrypoint after session.start()).
        # Needed because pose callbacks run on a worker thread and cannot directly
        # "return" data into the LLM context.
        self._lk_session: Optional[AgentSession] = None
        self._lk_loop: Optional[asyncio.AbstractEventLoop] = None
        self._lk_ready: bool = False
        self._pending_llm_instructions: list[str] = []
        self._pending_llm_lock = threading.Lock()

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
        # Pass shared robot instance and lock to avoid multiple connections and serial port conflicts
        self.wheels_service = WheelsService(
            robot=self.robot, robot_lock=self.robot_lock
        )
        # Commented out for testing.
        # self.arms_service = ArmsService(robot=self.robot, robot_lock=self.robot_lock)

        # Pose service camera selection:
        self.pose_service = PoseDetectionService(
            status_callback=self._handle_pose_status,
            camera=CameraStream(index=front_camera_index),
            visualizer=None if self.viz_enabled else default_visualizer,
            frame_subscription=self.front_sub_pose,
            viz=self.viz,
        )

        # Main_thread (blocking) services
        # Initialize navigator with robot instance and lock
        self.navigator = Navigator(
            self.robot,
            robot_lock=self.robot_lock,
            viz=self.viz,
            frame_subscription=self.front_sub_nav,
        )

        # Initialize epipen service with robot instance
        from lekiwi.services.epipen import EpipenService

        self.epipen_service = EpipenService(self.robot)

        # Initialize workflow service
        self.workflow_service = WorkflowService()
        self.workflow_service.set_agent(self)

        # Initialize operational mode (normal, concerned, or emergency)
        self.status = "normal"
        self._push_status_to_viz()

        # Start robot services
        self.wheels_service.start()
        # self.arms_service.start()
        self.pose_service.start()

        # Wake up animation
        # self.arms_service.dispatch("play", "wake_up")

    def _start_camera_pumps(self):
        """Pump camera frames into viz in background threads."""
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

    def _enqueue_or_send_llm_instructions(self, instructions: str) -> None:
        """
        Thread-safe bridge: enqueue instructions until LiveKit session is ready,
        otherwise schedule a generate_reply() on the LiveKit asyncio loop.
        """
        # If session isn't ready yet, queue the message.
        if not self._lk_ready or self._lk_session is None or self._lk_loop is None:
            with self._pending_llm_lock:
                self._pending_llm_instructions.append(instructions)
            return

        # Schedule async call from this (possibly non-async) context safely.
        try:
            fut = asyncio.run_coroutine_threadsafe(
                _generate_reply(self._lk_session, instructions),
                self._lk_loop,
            )
            # Don't block; optionally force exception propagation in logs.
            fut.add_done_callback(
                lambda f: f.exception()
                and logger.error(
                    "LeKiwi: generate_reply failed from callback: %s", f.exception()
                )
            )
        except Exception as e:
            logger.error("LeKiwi: failed to schedule generate_reply: %s", e)

    def __del__(self):
        """Cleanup: disconnect robot when agent is destroyed"""
        if hasattr(self, "robot") and self.robot:
            try:
                self.robot.disconnect()
            except:
                pass  # Ignore errors during cleanup
        try:
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
            # Switch to concerned mode when person falls
            if self.status != "concerned":
                self.status = "concerned"
                self._push_status_to_viz()
                print(f"LeKiwi: Person fallen detected, switching to concerned mode")
                # Start the help workflow automatically
                try:
                    self.workflow_service.start_workflow("help")
                    logger.info("LeKiwi: Started help workflow due to person fallen")

                    # Pull the next step immediately and inject into the LLM context.
                    # This callback runs on a worker thread; use the session bridge.
                    next_step = self.workflow_service.get_next_step()
                    self._enqueue_or_send_llm_instructions(
                        "PERSON_FALLEN detected. Entering concerned mode.\n\n"
                        "[HELP WORKFLOW STARTED]\n"
                        f"Next step:\n{next_step}\n\n"
                        "Follow the workflow strictly. After completing the step, call complete_step()."
                    )
                except Exception as e:
                    logger.error(f"LeKiwi: Error starting help workflow: {e}")
        elif status_type == "PERSON_STABLE":
            # Switch back to normal mode when person is stable
            if self.status != "normal":
                self.status = "normal"
                self._push_status_to_viz()
                print(f"LeKiwi: Person stable detected, switching to normal mode")
                self._enqueue_or_send_llm_instructions(
                    "PERSON_STABLE detected. Returning to normal mode."
                )

    @function_tool
    async def get_available_recordings(self) -> str:
        """
        Use this tool to double check what recordings (i.e. physical expressions) you can input to
        the play_recording tool.

        Returns:
            List of available physical expression recordings you can perform.
        """
        print("LeKiwi: get_available_recordings function called")
        self._log_tool("get_available_recordings", "call")
        try:
            # recordings = self.arms_service.get_available_recordings()
            recordings = self.wheels_service.get_available_recordings()

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
    async def play_recording(self, recording_name: str, type: str = "arms") -> str:
        """
        Use this tool to play a prerecorded movement (arms or wheels).
        Call get_available_recordings() first to see valid names.

        Args:
            recording_name: Name of the recording to perform.
            type: "arms" or "wheels".
        """
        print(
            f"LeKiwi: play_recording function called with recording_name: {recording_name}"
        )
        self._log_tool("play_recording", f"call {recording_name}")
        try:
            # Send play event to animation service
            if type == "arms":
                # self.arms_service.dispatch("play", recording_name)
                return "Arms service is currently disabled"
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

    @function_tool
    async def administer_epipen(self) -> str:
        """
        Use this tool to administer the epipen via the epipen service.
        This blocks until the epipen administration is complete.

        Returns:
            Confirmation message indicating successful epipen administration or failure reason.
        """

        # Check if service is ready
        if not hasattr(self, "epipen_service") or not self.epipen_service.is_ready():
            return "Error: Epipen service not available. Check π₀.₅ installation."

        # Execute synchronous epipen administration
        try:
            result = self.epipen_service.administer_epipen()
            logger.info(f"LeKiwi: administer_epipen completed with result: {result}")
            return result
        except Exception as e:
            error_msg = f"Epipen administration failed: {str(e)}"
            logger.error(f"LeKiwi: administer_epipen error: {error_msg}")
            return error_msg

    @function_tool
    async def get_configuration(self) -> str:
        """
        Use this tool to get the robot status/configuration (currently a stub).
        """
        # TODO: Implement this with proper configuration checking and return as json () - see https://github.com/TARS-AI-Community/TARS-AI/blob/V2/src/character/TARS/persona.ini
        result = "Status: Nominal"
        self._log_tool("get_configuration", result)
        return result

    @function_tool
    async def toggle_state(self, mode: str) -> str:
        """
        Use this tool to set the robot operational mode.
        If switching to "concerned", the "help" workflow will automatically start.

        Args:
            mode: "emergency", "normal", or "concerned".

        Returns:
            Confirmation message indicating the current mode.
        """
        try:
            self._log_tool("toggle_state", f"call {mode}")
            if mode not in ["emergency", "normal", "concerned"]:
                result = f"Error: mode must be either 'emergency' or 'normal' or 'concerned', got '{mode}'"
                self._log_tool("toggle_state", result, level="error")
                return result

            # Store the mode as an instance variable
            self.status = mode
            self._push_status_to_viz()
            result = f"Switched to {mode} mode"
            logger.debug(f"LeKiwi: {result}")

            # If switching to concerned mode, automatically start the help workflow
            if mode == "concerned":
                try:
                    # Start the help workflow
                    self.workflow_service.start_workflow("help")

                    # Get the first step of the help workflow
                    next_step = self.workflow_service.get_next_step()

                    # Return the workflow step information in the response
                    # The LLM will naturally process this and act on it
                    result += f"\n\n[HELP WORKFLOW STARTED] \n First step: \n{next_step}\n\n Please follow the workflow instructions and execute the required actions. Start by reading the step carefully and then taking the appropriate action."

                except Exception as e:
                    error_msg = f"Error starting help workflow: {str(e)}"
                    import traceback

                    logger.debug(traceback.format_exc())
                    result += f". Warning: {error_msg}"

            self._log_tool("toggle_state", result, level="info")
            return result
        except Exception as e:
            result = f"Error toggling state: {str(e)}"
            logger.error(f"LeKiwi: {result}")
            self._log_tool("toggle_state", result, level="error")
            return result

    @function_tool
    async def get_available_workflows(self) -> str:
        """
        Use this tool to list workflows you can start with start_workflow().

        Returns:
            List of available workflow names you can execute.
        """
        logger.debug("LeKiwi: get_available_workflows function called")
        self._log_tool("get_available_workflows", "call")
        try:
            workflows = self.workflow_service.get_available_workflows()

            if workflows:
                result = f"Available workflows: {', '.join(workflows)}"
                level = "info"
            else:
                result = "No workflows found."
                level = "info"
        except Exception as e:
            result = f"Error getting workflows: {str(e)}"
            level = "error"

        self._log_tool("get_available_workflows", result, level=level)
        return result

    @function_tool
    async def start_workflow(self, workflow_name: str) -> str:
        """
        Use this tool to start a workflow by name.
        After starting, call get_next_step() and complete_step() until the workflow completes.

        Args:
            workflow_name: Name of the workflow to start (see get_available_workflows()).
        """
        logger.debug(
            f"LeKiwi: start_workflow function called with workflow_name: {workflow_name}"
        )
        self._log_tool("start_workflow", f"call {workflow_name}")
        try:
            self.workflow_service.start_workflow(workflow_name)
            result = f"Started the workflow: {workflow_name}. You can now call the get_next_step function to get the next step."
        except Exception as e:
            result = f"Error starting workflow {workflow_name}: {str(e)}"
            self._log_tool("start_workflow", result, level="error")
            return result

        self._log_tool("start_workflow", result, level="info")
        return result

    @function_tool
    async def get_next_step(self) -> str:
        """
        Use this tool to fetch the next step for the active workflow.
        After you complete the instructions, call complete_step() to advance.

        Returns:
            The next step instructions (including context and optional state variables).
        """
        logger.debug(f"\n{'='*60}")
        logger.debug(f"LeKiwi: get_next_step called")
        logger.debug(f"  Active workflow: {self.workflow_service.active_workflow}")
        logger.debug(f"{'='*60}\n")
        self._log_tool("get_next_step", "call")

        try:
            if self.workflow_service.active_workflow is None:
                result = "Error: No active workflow. Call start_workflow first."
                self._log_tool("get_next_step", result, level="error")
                return result

            next_step = self.workflow_service.get_next_step()

            logger.debug(f"\n{'='*60}")
            logger.debug(f"LeKiwi: get_next_step RESULT:")
            logger.debug(f"{next_step}")
            logger.debug(f"{'='*60}\n")

            self._log_tool("get_next_step", next_step, level="info")
            return next_step
        except Exception as e:
            error_msg = f"Error getting next step: {str(e)}"
            logger.error(f"[ERROR] {error_msg}")
            import traceback

            logger.debug(traceback.format_exc())
            self._log_tool("get_next_step", error_msg, level="error")
            return error_msg

    @function_tool
    async def complete_step(
        self, state_updates: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Use this tool to mark the current workflow step complete and advance.
        Optionally provide state updates to influence workflow routing.

        Args:
            state_updates: Optional dict of state updates (leave empty if none).

        Returns:
            Information about the next step or workflow completion message.
        """
        import json
        import inspect

        logger.debug(f"\n{'='*60}")
        logger.debug(f"LeKiwi: complete_step called")
        self._log_tool("complete_step", "call")

        # Debug: Check what we actually received
        frame = inspect.currentframe()
        if frame and frame.f_back:
            local_vars = frame.f_back.f_locals
            logger.debug(f"  All local variables: {list(local_vars.keys())}")
            if "state_updates" in local_vars:
                logger.debug(
                    f"  state_updates from locals: {local_vars['state_updates']}"
                )

        logger.debug(f"  Raw state_updates parameter: {state_updates}")
        logger.debug(f"  Type: {type(state_updates)}")
        logger.debug(f"  Repr: {repr(state_updates)}")

        # Handle case where state_updates might come as a string or need parsing
        original_state_updates = state_updates
        if state_updates is not None:
            if isinstance(state_updates, str):
                try:
                    state_updates = json.loads(state_updates)
                    logger.debug(f"  ✓ Parsed JSON string to dict: {state_updates}")
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"  ❌ Warning: Could not parse state_updates as JSON: {e}"
                    )
                    logger.debug(
                        f"     String value was: {repr(original_state_updates)}"
                    )
                    state_updates = None
        else:
            logger.debug(
                f"  ⚠️  state_updates is None - this might indicate LiveKit didn't parse the parameter"
            )

        logger.debug(f"  Final state_updates: {state_updates}")
        logger.debug(f"{'='*60}\n")

        try:
            if self.workflow_service.active_workflow is None:
                result = "Error: No active workflow."
                self._log_tool("complete_step", result, level="error")
                return result

            result = self.workflow_service.complete_step(state_updates)

            logger.debug(f"\n{'='*60}")
            logger.debug(f"LeKiwi: complete_step RESULT:")
            logger.debug(f"{result}")
            logger.debug(f"{'='*60}\n")

            self._log_tool("complete_step", result, level="info")
            return result
        except Exception as e:
            error_msg = f"Error completing step: {str(e)}"
            logger.error(f"[ERROR] {error_msg}")
            import traceback

            logger.debug(traceback.format_exc())
            self._log_tool("complete_step", error_msg, level="error")
            return error_msg


# Entry to the agent
async def entrypoint(ctx: agents.JobContext):
    # Use pre-parsed arguments from __main__
    front_idx = _RUN_ARGS["front_idx"]
    wrist_idx = _RUN_ARGS["wrist_idx"]
    viz_enabled = _RUN_ARGS["viz_enabled"]

    # Parse which workflows to preload
    workflow_names = parse_workflow_args()

    # Initialize agent
    agent = LeTars(
        viz_enabled=viz_enabled,
        front_camera_index=front_idx,
        wrist_camera_index=wrist_idx,
    )

    # Ensure agent instance is set (should already be set in __init__, but double-check)
    if agent.workflow_service.agent_instance is None:
        logger.warning("[MAIN] Warning: Agent instance not set, setting it now...")
        agent.workflow_service.set_agent(agent)

    # Preload workflow tools BEFORE creating the session
    # LiveKit scans for tools when AgentSession is instantiated, so we must register before that
    if workflow_names:
        logger.debug(f"[MAIN] Preloading tools from workflows: {workflow_names}")
    else:
        logger.debug(f"[MAIN] Preloading tools from all available workflows")
    agent.workflow_service.preload_workflow_tools(workflow_names)

    session = AgentSession(llm=openai.realtime.RealtimeModel(voice="verse"))

    # Store session + loop on the agent so background threads (pose callbacks)
    # can safely schedule LLM replies once the session is started.
    agent._lk_session = session
    agent._lk_loop = asyncio.get_running_loop()

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            audio_enabled=True,
            audio_sample_rate=16000,
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Mark session ready and flush any queued instructions from early callbacks.
    agent._lk_ready = True
    with agent._pending_llm_lock:
        pending = list(agent._pending_llm_instructions)
        agent._pending_llm_instructions.clear()
    for msg in pending:
        # Schedule on the running loop; we're already inside it.
        try:
            await session.generate_reply(instructions=msg)
        except Exception as e:
            logger.error("LeKiwi: failed to flush queued instruction: %s", e)

    await session.generate_reply(
        instructions=f"""When you wake up, greet with: 'Systems nominal. What's the plan?' or 'All systems operational. Nice to see you sir.'"""
    )


if __name__ == "__main__":
    # Run with: WORKFLOWS=help python main_workflows.py dev
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
