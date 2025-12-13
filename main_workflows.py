import os
import sys

import logging
import asyncio
from pathlib import Path
import time
from typing import Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from livekit.agents import AgentSession

from dotenv import load_dotenv

# ---- Logging configuration ----
# Many dependencies (draccus, gitpython, livekit) can be extremely chatty at DEBUG.
# Configure logging once, early, and allow override via LOG_LEVEL env var.
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,  # override any handlers configured by imported libs
)
# Silence especially noisy loggers by default.
for _name, _lvl in {
    "draccus": logging.WARNING,
    "git": logging.WARNING,
    "git.cmd": logging.WARNING,
    "asyncio": logging.INFO,
    "lekiwi.workflows.workflows": logging.INFO,
}.items():
    logging.getLogger(_name).setLevel(_lvl)

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
    PoseEstimator,
    FallDetector,
)
from lekiwi.workflows.workflows import WorkflowService
from lekiwi.services.navigation import Navigator

# import zmq

load_dotenv()


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
        # Three services running on separate threads, with agent dispatching events to them
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
        )

        # Initialize robot connection
        self.robot_config = LeKiwiConfig(port=port, id=robot_id, cameras={})
        self.robot = LeKiwi(self.robot_config)
        self.robot.connect()

        # Initialize navigator with robot instance
        self.navigator = Navigator(self.robot)

        # Initialize epipen service with robot instance
        from lekiwi.services.epipen import EpipenService

        self.epipen_service = EpipenService(self.robot)

        # Initialize workflow service
        self.workflow_service = WorkflowService()
        # Pass agent instance to workflow service for dynamic tool registration
        self.workflow_service.set_agent(self)

        # Initialize operational mode (normal or emergency)
        self.operational_mode = "normal"

        # Session reference for triggering LLM responses (set after session creation)
        # Using _agent_session to avoid conflict with Agent base class's session property
        self._agent_session: Optional["AgentSession"] = None
        # Filled in after session creation (used by tools)
        self.current_room_name: Optional[str] = None

        # Start robot services
        self.wheels_service.start()
        self.arms_service.start()
        self.pose_service.start()

        # Optional data streaming (to anyone listening)
        # TODO: This should probably exist in the pose detection worker thread instead
        self.stream_data = stream_data
        self.zmq_pub = None
        if stream_data:
            import zmq

            context = zmq.Context()
            self.zmq_pub = context.socket(zmq.PUB)
            self.zmq_pub.setsockopt(zmq.CONFLATE, 1)
            self.zmq_pub.bind(f"tcp://*:{stream_port}")
            print(f"ZMQ Publisher on LeKiwi bound to port {stream_port}")

        # Wake up animation
        self.arms_service.dispatch("play", "wake_up")

    def _publish_sensor_data(self, data_type: str, data: dict):
        """Publish sensor data to ZMQ stream if enabled."""
        if self.zmq_pub:
            message = {"type": data_type, "timestamp": time.time(), "data": data}
            self.zmq_pub.send_json(message)

    def _handle_pose_status(self, status_type: str, details: dict):
        """
        Callback method to receive status updates from the PoseDetectionService.
        This runs in the context of the service's worker thread, but is called by it.
        """
        print(
            f"LeKiwi: Received pose status update - Type: {status_type}, Details: {details}"
        )

        # Stream pose data if enabled
        if self.stream_data:
            self._publish_sensor_data(
                "pose",
                {
                    "status": status_type,
                    "score": details.get("score", 0.0),
                    "ratio": details.get("ratio", 0.0),
                },
            )

        if status_type == "PERSON_FALLEN":
            # The main thread (LiveKit orchestrator) decides what to do
            # In an Agent, this often means generating a reply or dispatching a motor action.

            # Example 1: Use the LLM to generate an urgent reply
            # You would need a mechanism to break into the current LLM flow.
            # For simplicity, let's dispatch an action for now.

            # Example 2: Dispatch a HIGH-priority motor action (e.g., look up, check)
            # log it
            print(f"LeKiwi: Person fallen detected, dispatching concerned mode")

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
        try:
            recordings = self.arms_service.get_available_recordings()

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
        try:
            # Send play event to animation service
            self.arms_service.dispatch("play", recording_name)
            result = f"Started playing recording: {recording_name}"
            return result
        except Exception as e:
            result = f"Error playing recording {recording_name}: {str(e)}"
            return result

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

    # Make room/session info available to tools that need to target the active room
    agent.current_room_name = getattr(ctx, "room", None) and getattr(
        ctx.room, "name", None
    )
    agent._agent_session = session

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
