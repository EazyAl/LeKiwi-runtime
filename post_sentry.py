import asyncio
import base64
import json
import logging
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Disable tokenizers parallelism to avoid fork warnings with PI0 policy
# Must be set before any HuggingFace imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress verbose DEBUG logging from lerobot cameras and robot before any imports
logging.getLogger("lerobot.cameras.opencv.camera_opencv").setLevel(logging.WARNING)
logging.getLogger("lerobot.cameras").setLevel(logging.WARNING)
logging.getLogger("lekiwi.robot.lekiwi").setLevel(logging.INFO)

logger = logging.getLogger(__name__)
from livekit import agents
from livekit.agents import Agent, AgentSession, RoomInputOptions, function_tool
from livekit.plugins import noise_cancellation, openai

# LeKiwi robot imports
from lekiwi.robot import LeKiwi
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

from lekiwi.services.motors.arms_service import ArmsService
from lekiwi.services.motors.wheels_service import WheelsService
from lekiwi.viz.rerun_viz import NullViz, create_viz

load_dotenv()

_RUN_ARGS = {
    "viz_enabled": False,
}


def _load_system_prompt() -> str:
    """Load the system prompt from the personality/system.txt file."""
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
    """
    LiveKit agent that does NOT monitor pose; instead it exposes
    callable SIP outbound to alert a human when an external trigger fires.
    """

    def __init__(
        self,
        port: str = "/dev/tty.usbmodem58760432781",
        robot_id: str = "biden_kiwi",
        viz_enabled: bool = False,
    ):
        super().__init__(instructions=_load_system_prompt())
        self._runtime_loop: Optional[asyncio.AbstractEventLoop] = None
        self._session: Optional[AgentSession] = None
        self._call_task: Optional[asyncio.Task] = None
        self._call_schedule_lock = threading.Lock()

        # Robot connection
        self.robot_config = LeKiwiConfig(port=port, id=robot_id, cameras={})
        self.robot = LeKiwi(self.robot_config)
        self.robot.connect(calibrate=False)

        # Visualization (kept minimal; no cameras)
        self.viz_enabled = viz_enabled
        self.viz = create_viz(viz_enabled, app_id="lekiwi_viz") if viz_enabled else NullViz()

        # Lock to serialize access to robot motor commands
        self.robot_lock = threading.Lock()

        # Services (motors only; pose removed)
        self.wheels_service = WheelsService(
            robot=self.robot, robot_lock=self.robot_lock
        )
        self.arms_service = ArmsService(robot=self.robot, robot_lock=self.robot_lock)

        # Initialize operational mode (call/idle)
        self.is_calling = False
        self._push_status_to_viz()

        # Start robot services
        self.wheels_service.start()
        self.arms_service.start()

        # Wake up animation
        self.arms_service.dispatch("play", "wake_up")

        # SIP config
        self.livekit_url = _normalize_base_url(os.getenv("LIVEKIT_URL"))
        self.livekit_api_key = os.getenv("LIVEKIT_API_KEY")
        self.livekit_api_secret = os.getenv("LIVEKIT_API_SECRET")
        self.livekit_sip_trunk_id = os.getenv("LIVEKIT_SIP_TRUNK_ID")
        self.default_destination = os.getenv("LIVEKIT_SIP_TO")
        # Bearer token (JWT) support: prefer TOKEN, fallback to LIVEKIT_BEARER_TOKEN
        self.livekit_bearer_token = os.getenv("TOKEN") or os.getenv(
            "LIVEKIT_BEARER_TOKEN"
        )

    def _push_status_to_viz(self) -> None:
        if not getattr(self, "viz", None):
            return
        try:
            self.viz.set_status(self.status, ts=time.time())
        except Exception:
            pass

    @property
    def status(self) -> str:
        return "calling" if self.is_calling else "idle"

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
            viz = getattr(self, "viz", None)
            if viz is not None:
                viz.close()
        except Exception:
            pass

    def _request_start_call(self, details: Dict[str, Any]) -> None:
        loop = self._runtime_loop
        if loop is None:
            logger.warning("LeKiwi: call requested but runtime loop not bound yet")
            return

        def _schedule() -> None:
            with self._call_schedule_lock:
                if self._call_task is not None and not self._call_task.done():
                    return
                self._call_task = asyncio.create_task(self._enter_call(details))

        loop.call_soon_threadsafe(_schedule)

    async def _speak(self, text: str) -> None:
        if self._session is None:
            logger.warning(f"LeKiwi: (no session) would say: {text}")
            return
        try:
            await self._session.generate_reply(instructions=f"Say: '{text}'")
        except Exception as e:
            logger.exception(f"LeKiwi: failed to speak: {e}")

    async def _run_call_tasks(self, details: Dict[str, Any]) -> str:
        """
        Place an outbound SIP call via LiveKit SIP CreateSIPParticipant.
        """
        destination = details.get("to") or self.default_destination
        wait_until_answered = bool(details.get("wait_until_answered", True))
        room_name = details.get("room_name", "sentry-alert-room")
        participant_identity = details.get("participant_identity", "sip-alert")
        participant_name = details.get("participant_name", "Sentry Alert")

        missing = [
            name
            for name, val in [
                ("LIVEKIT_URL", self.livekit_url),
                ("LIVEKIT_API_KEY", self.livekit_api_key),
                ("LIVEKIT_API_SECRET", self.livekit_api_secret),
                ("LIVEKIT_SIP_TRUNK_ID", self.livekit_sip_trunk_id),
                ("destination", destination),
            ]
            if not val
        ]
        if missing:
            msg = f"Missing required config for SIP call: {', '.join(missing)}"
            logger.error("LeKiwi: %s", msg)
            return msg

        payload = {
            "sip_trunk_id": self.livekit_sip_trunk_id,
            "sip_call_to": destination,
            "room_name": room_name,
            "participant_identity": participant_identity,
            "participant_name": participant_name,
            "wait_until_answered": wait_until_answered,
        }

        logger.info(
            "LeKiwi: initiating SIP call (dest=%s room=%s trunk_set=%s)",
            _mask_phone_number(str(destination)),
            room_name,
            bool(self.livekit_sip_trunk_id),
        )

        ok, result = await create_sip_participant(
            base_url=self.livekit_url,
            api_key=self.livekit_api_key,  # type: ignore[arg-type]
            api_secret=self.livekit_api_secret,  # type: ignore[arg-type]
            payload=payload,
            bearer_token=self.livekit_bearer_token,
        )
        if ok:
            logger.debug("LeKiwi: SIP participant created: %s", result)
            return "SIP call initiated."
        logger.error("LeKiwi: failed to initiate SIP call: %s", result)
        return f"Failed to initiate SIP call: {result}"

    async def _enter_call(self, details: Dict[str, Any]) -> None:
        """
        Call sequence:
        - set status
        - optionally speak in the room
        - initiate SIP outbound
        """
        if not self.is_calling:
            self.is_calling = True
            self._push_status_to_viz()

        logger.info("LeKiwi: starting outbound SIP alert")
        await self._speak("I detected a fall. I am calling for help.")
        result = await self._run_call_tasks(details)
        await self._speak(result)

        self.is_calling = False
        self._push_status_to_viz()

    @function_tool
    async def start_sip_call(
        self,
        to_number: Optional[str] = None,
        room_name: str = "sentry-alert-room",
        participant_identity: str = "sip-alert",
        participant_name: str = "Sentry Alert",
    ) -> str:
        """
        Trigger an outbound SIP call via LiveKit using the configured SIP trunk.
        The default destination comes from LIVEKIT_SIP_TO if not provided.
        """
        details = {
            "to": to_number,
            "room_name": room_name,
            "participant_identity": participant_identity,
            "participant_name": participant_name,
            "wait_until_answered": True,
        }
        self._log_tool("start_sip_call", f"call {to_number or 'default'}")
        self._request_start_call(details)
        return "SIP call requested."

    def trigger_call(
        self,
        *,
        to_number: Optional[str] = None,
        room_name: str = "sentry-alert-room",
        participant_identity: str = "sip-alert",
        participant_name: str = "Sentry Alert",
        wait_until_answered: bool = True,
    ) -> None:
        """
        Public entrypoint for external code to initiate the call sequence.
        Safe to invoke from non-async threads.
        """
        details = {
            "to": to_number,
            "room_name": room_name,
            "participant_identity": participant_identity,
            "participant_name": participant_name,
            "wait_until_answered": wait_until_answered,
        }
        self._request_start_call(details)


def _mask_phone_number(phone: str) -> str:
    if len(phone) <= 4:
        return phone
    return f"{'*' * (len(phone) - 4)}{phone[-4:]}"


def _normalize_base_url(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    base = raw.rstrip("/")
    if base.startswith("wss://"):
        base = "https://" + base[len("wss://") :]
    elif base.startswith("ws://"):
        base = "http://" + base[len("ws://") :]
    return base


async def create_sip_participant(
    *,
    base_url: Optional[str],
    api_key: str,
    api_secret: str,
    payload: Dict[str, Any],
    bearer_token: Optional[str] = None,
) -> tuple[bool, str | Dict[str, Any]]:
    """
    Fire-and-forget helper around LiveKit SIP CreateSIPParticipant.
    """
    if not base_url:
        return False, "LIVEKIT_URL is not set or invalid"

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        _create_sip_participant_sync,
        base_url,
        api_key,
        api_secret,
        payload,
        bearer_token,
    )


def _create_sip_participant_sync(
    base_url: str,
    api_key: str,
    api_secret: str,
    payload: Dict[str, Any],
    bearer_token: Optional[str] = None,
) -> tuple[bool, str | Dict[str, Any]]:
    url = f"{base_url}/twirp/livekit.SIP/CreateSIPParticipant"
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        logger.debug("LeKiwi: Using Bearer auth for SIP request")
        headers["Authorization"] = f"Bearer {bearer_token}"
    else:
        logger.debug("LeKiwi: Using Basic auth for SIP request")
        auth = base64.b64encode(f"{api_key}:{api_secret}".encode("utf-8")).decode("utf-8")
        headers["Authorization"] = f"Basic {auth}"
    safe_payload = dict(payload)
    if "sip_call_to" in safe_payload:
        safe_payload["sip_call_to"] = _mask_phone_number(str(safe_payload["sip_call_to"]))
    logger.debug("LeKiwi: SIP CreateSIPParticipant POST %s payload=%s", url, safe_payload)

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            try:
                parsed = json.loads(body)
                logger.info(
                    "LeKiwi: SIP CreateSIPParticipant succeeded (http=%s)",
                    getattr(resp, "status", "unknown"),
                )
                return True, parsed
            except json.JSONDecodeError:
                logger.warning(
                    "LeKiwi: SIP CreateSIPParticipant returned non-JSON body (len=%s)",
                    len(body),
                )
                return True, {"raw": body}
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        logger.error(
            "LeKiwi: SIP CreateSIPParticipant HTTPError (http=%s, body_len=%s)",
            e.code,
            len(error_body),
        )
        return False, f"HTTP {e.code}: {error_body}"
    except Exception as e:
        logger.exception("LeKiwi: SIP CreateSIPParticipant transport error")
        return False, str(e)


# Entry to the agent
async def entrypoint(ctx: agents.JobContext):
    viz_enabled = _RUN_ARGS["viz_enabled"]

    # Initialize agent with streaming enabled if requested
    agent = LeTars(
        viz_enabled=viz_enabled,
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
    args, unknown = parser.parse_known_args()

    # Capture custom flags for use in entrypoint and strip them from argv before LiveKit parses.
    _RUN_ARGS["viz_enabled"] = bool(args.viz)

    # Remove parsed args so LiveKit CLI doesn't see them as unknown
    sys.argv = [sys.argv[0]] + unknown

    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, num_idle_processes=1)  # type: ignore[arg-type]
    )
