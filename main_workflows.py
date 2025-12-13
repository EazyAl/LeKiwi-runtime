import os
import asyncio
import threading
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

# import zmq

load_dotenv()


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
        stream_data: bool = False,
        stream_port: int = 5556,
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
                "front": OpenCVCameraConfig(index_or_path=0, width=1280, height=720, fps=30),
                "wrist": OpenCVCameraConfig(index_or_path=1, width=1280, height=720, fps=30),
            },
        )
        self.robot = LeKiwi(self.robot_config)
        self.robot.connect(calibrate=False)

        # Lock to serialize access to robot motor commands (serial port is not thread-safe)
        self.robot_lock = threading.Lock()  

        # Three services running on separate threads, with agent dispatching events to them
        # Pass shared robot instance and lock to avoid multiple connections and serial port conflicts
        self.wheels_service = WheelsService(
            robot=self.robot, robot_lock=self.robot_lock
        )
        #Commented out for testing. 
        # self.arms_service = ArmsService(robot=self.robot, robot_lock=self.robot_lock)

        # Pose service camera selection:
        # - Defaults to index 0
        # - Override at runtime via POSE_CAMERA_INDEX=1 (or 2, 3, ...)
        pose_camera_index = int(os.getenv("POSE_CAMERA_INDEX", "1"))
        self.pose_service = PoseDetectionService(
            status_callback=self._handle_pose_status,
            camera=CameraStream(index=pose_camera_index),
            visualizer=default_visualizer,
        )

        # Main_thread (blocking) services
        # Initialize navigator with robot instance and lock
        self.navigator = Navigator(self.robot, robot_lock=self.robot_lock)

        # Initialize epipen service with robot instance
        from lekiwi.services.epipen import EpipenService

        self.epipen_service = EpipenService(self.robot)

        # Initialize workflow service
        self.workflow_service = WorkflowService()
        self.workflow_service.set_agent(self)

        # Initialize operational mode (normal, concerned, or emergency)
        self.operational_mode = "normal"

        # Start robot services
        self.wheels_service.start()
        # self.arms_service.start()
        self.pose_service.start()

        # Wake up animation
        # self.arms_service.dispatch("play", "wake_up")

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
            if self.operational_mode != "concerned":
                self.operational_mode = "concerned"
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
            if self.operational_mode != "normal":
                self.operational_mode = "normal"
                print(f"LeKiwi: Person stable detected, switching to normal mode")
                self._enqueue_or_send_llm_instructions(
                    "PERSON_STABLE detected. Returning to normal mode."
                )
        

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
        try:
            # recordings = self.arms_service.get_available_recordings()
            recordings = self.wheels_service.get_available_recordings()

            if recordings:
                result = f"Available recordings: {', '.join(recordings)}"
                return result
            else:
                result = "No recordings found."
                return result
        except Exception as e:
            result = f"Error getting recordings: {str(e)}"
            return result

    @function_tool
    async def play_recording(self, recording_name: str, type: str = "arms") -> str:
        """
        Express yourself through physical movement! Use this constantly to show personality and emotion.
        Perfect for: greeting gestures, excited bounces, confused head tilts, thoughtful nods,
        celebratory wiggles, disappointed slouches, or any emotional response that needs body language.
        Combine with RGB colors for maximum expressiveness! Your movements are like a dog wagging its tail -
        use them frequently to show you're alive, engaged, and have personality. Don't just talk, MOVE!

        Args:
            recording_name: Name of the physical expression to perform (use get_available_recordings first)
            type: arms or wheels
        """
        print(
            f"LeKiwi: play_recording function called with recording_name: {recording_name}"
        )
        try:
            # Send play event to animation service
            if type == "arms":
                # self.arms_service.dispatch("play", recording_name)
                return "Arms service is currently disabled"
            elif type == "wheels":
                self.wheels_service.dispatch("play", recording_name)
            else:
                return f"Error: type must be either 'arms' or 'wheels', got '{type}'"
            result = f"Started playing recording: {recording_name}"
            return result
        except Exception as e:
            result = f"Error playing recording {recording_name}: {str(e)}"
            return result

    @function_tool
    async def administer_epipen(self) -> str:
        """
        Administer the epipen to the person in need using advanced VLA control.
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
        Get the status of the robot.
        """
        # TODO: Implement this with proper configuration checking and return as json () - see https://github.com/TARS-AI-Community/TARS-AI/blob/V2/src/character/TARS/persona.ini
        return "Status: Nominal"

    @function_tool
    async def toggle_state(self, mode: str) -> str:
        """
        Toggle between emergency mode, concerned mode, and normal mode. Use this to switch the robot's operational state.
        This affects how the robot behaves and responds to situations.

        When switching to "concerned" mode, the help workflow will automatically start.

        Args:
            mode: Either "emergency" to enter emergency mode, "normal" to return to normal mode,
                  or "concerned" to enter concerned mode (which automatically starts the help workflow).

        Returns:
            Confirmation message indicating the current mode.
        """
        try:
            if mode not in ["emergency", "normal", "concerned"]:
                return f"Error: mode must be either 'emergency' or 'normal' or 'concerned', got '{mode}'"

            # Store the mode as an instance variable
            self.operational_mode = mode
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

            return result
        except Exception as e:
            result = f"Error toggling state: {str(e)}"
            logger.error(f"LeKiwi: {result}")
            return result

    @function_tool
    async def get_available_workflows(self) -> str:
        """
        Discover what workflows you can execute! Get your repertoire of user-defined step workflows.
        Use this when someone asks you about your capabilities or when they ask you to execute a workflow.
        Each workflow is a user-defined graph or general instructions -
        like emergency response, assistance routines, or specific operational sequences.

        Returns:
            List of available workflow names you can execute.
        """
        logger.debug("LeKiwi: get_available_workflows function called")
        try:
            workflows = self.workflow_service.get_available_workflows()

            if workflows:
                result = f"Available workflows: {', '.join(workflows)}"
                return result
            else:
                result = "No workflows found."
                return result
        except Exception as e:
            result = f"Error getting workflows: {str(e)}"
            return result

    @function_tool
    async def start_workflow(self, workflow_name: str) -> str:
        f"""
        Start a workflow called {workflow_name}. This sets the workflow_service's active workflow.
        In order to perform the workflow you will need to iteratively call the get_next_step function until the workflow is complete.
        
        Args:
            workflow_name: Name of the workflow to start. Check the available workflows with the get_available_workflows function first.
        """
        logger.debug(
            f"LeKiwi: start_workflow function called with workflow_name: {workflow_name}"
        )
        try:
            self.workflow_service.start_workflow(workflow_name)
            return f"Started the workflow: {workflow_name}. You can now call the get_next_step function to get the next step."
        except Exception as e:
            result = f"Error starting workflow {workflow_name}: {str(e)}"
            return result

    @function_tool
    async def get_next_step(self) -> str:
        """
        Get the current step in the active workflow with full context.
        Shows you what to do, what tools to use, and what state variables you can update.
        After fulfilling the instructions of the this step, call complete_step() to advance.

        Returns:
            Your next instruction to fulfill, written in plain language, possibly with some suggested tools to use. It will also provide context about available the workflows state variables that you can update.
        """
        logger.debug(f"\n{'='*60}")
        logger.debug(f"LeKiwi: get_next_step called")
        logger.debug(f"  Active workflow: {self.workflow_service.active_workflow}")
        logger.debug(f"{'='*60}\n")

        try:
            if self.workflow_service.active_workflow is None:
                return "Error: No active workflow. Call start_workflow first."

            next_step = self.workflow_service.get_next_step()

            logger.debug(f"\n{'='*60}")
            logger.debug(f"LeKiwi: get_next_step RESULT:")
            logger.debug(f"{next_step}")
            logger.debug(f"{'='*60}\n")

            return next_step
        except Exception as e:
            error_msg = f"Error getting next step: {str(e)}"
            logger.error(f"[ERROR] {error_msg}")
            import traceback

            logger.debug(traceback.format_exc())
            return error_msg

    @function_tool
    async def complete_step(
        self, state_updates: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Complete the current workflow step and advance to the next one.
        Optionally update state variables that affect workflow routing.

        Args:
            state_updates: Optional dict of state updates, e.g. {"user_response_detected": true, "attempt_count": 1}
                          Leave empty if no state needs updating.

        Returns:
            Information about the next step or workflow completion message.
        """
        import json
        import inspect

        logger.debug(f"\n{'='*60}")
        logger.debug(f"LeKiwi: complete_step called")

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
                return "Error: No active workflow."

            result = self.workflow_service.complete_step(state_updates)

            logger.debug(f"\n{'='*60}")
            logger.debug(f"LeKiwi: complete_step RESULT:")
            logger.debug(f"{result}")
            logger.debug(f"{'='*60}\n")

            return result
        except Exception as e:
            error_msg = f"Error completing step: {str(e)}"
            logger.error(f"[ERROR] {error_msg}")
            import traceback

            logger.debug(traceback.format_exc())
            return error_msg


# Entry to the agent
async def entrypoint(ctx: agents.JobContext):
    # Parse command-line args to get stream settings
    import sys

    stream_enabled = "--stream" in sys.argv
    stream_port = 5556
    for i, arg in enumerate(sys.argv):
        if arg == "--stream-port" and i + 1 < len(sys.argv):
            stream_port = int(sys.argv[i + 1])

    # Parse which workflows to preload
    workflow_names = parse_workflow_args()

    # Initialize agent with streaming enabled if requested
    agent = LeTars(stream_data=stream_enabled, stream_port=stream_port)

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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stream", action="store_true", help="Enable data streaming for visualization"
    )
    parser.add_argument(
        "--stream-port", type=int, default=5556, help="Port for ZMQ data streaming"
    )
    args, unknown = parser.parse_known_args()
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, num_idle_processes=1)
    )
