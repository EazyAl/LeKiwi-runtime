import asyncio
import base64
import json
import logging
import os
import urllib.error
import urllib.request

from livekit.agents import function_tool

logger = logging.getLogger(__name__)

ALLOWED_DEMO_NUMBER = os.getenv("DEMO_EMERGENCY_NUMBER", "+33753823988")

# Workflow-specific tools for the help/emergency workflow


# TODO(ALI): not needed if you replace it
def _mask_phone_number(num: str | None) -> str:
    """Mask a phone number for logs (avoid leaking full destination)."""
    if not num:
        return "<none>"
    s = str(num)
    if len(s) <= 4:
        return "****"
    return f"{s[:-4]}****"


def _guard_emergency_only(agent) -> str | None:
    """
    Prevent emergency-only tools from executing during normal operation.

    Returns:
        None if execution is allowed, otherwise a user/LLM-facing message describing why it was blocked.
    """
    mode = getattr(agent, "status", "normal")
    if mode == "normal":
        msg = (
            "Blocked: this tool is emergency-only and will not run while status='normal'. "
            "Switch to 'concerned' or 'emergency' mode first (e.g., call toggle_state('concerned'))."
        )
        logger.warning(f"[HELP TOOL GUARD] {msg}")
        return msg
    return None


@function_tool
async def navigate_to_person(self) -> str:
    """
    Navigate to the person in need. Move the robot to the person's location
    so the epipen can be administered. This will:
    1. Locate the person using pose detection
    2. Rotate to their thigh for proper orientation
    3. Drive forward to an appropriate interaction distance
    4. Block until the complete sequence finishes

    Returns:
        Confirmation message indicating successful navigation to the person.
    """
    guard_msg = _guard_emergency_only(self)
    if guard_msg is not None:
        return guard_msg

    logger.debug("LeKiwi: navigate_to_person function called")

    # Run navigation synchronously (blocking). This is intentional for the demo.
    try:
        result = self.navigator.navigate_to_person()
        logger.info(f"LeKiwi: navigate_to_person completed with result: {result}")
        return result
    except Exception as e:
        error_msg = f"Navigation failed: {str(e)}"
        logger.error(f"LeKiwi: navigate_to_person failed: {error_msg}")
        return error_msg


# TODO(ALI): replace with working ER call
@function_tool
async def call_emergency_services(self) -> str:
    """
    Call 911 emergency services. This initiates contact with emergency responders.
    The function should return success status which will be stored in call_911_success state.

    Returns:
        Confirmation message indicating whether the 911 call was successful or failed.
    """
    guard_msg = _guard_emergency_only(self)
    if guard_msg is not None:
        logger.info(
            "LeKiwi: call_911_emergency blocked by guard (status=%s)",
            getattr(self, "status", "normal"),
        )
        return guard_msg
    logger.debug("LeKiwi: call_911_emergency function called")

    # Hard allowlist to prevent real emergency dialing
    destination = ALLOWED_DEMO_NUMBER
    if destination != "+33753823988" and os.getenv("DEMO_EMERGENCY_NUMBER") is None:
        # If someone changed the default without setting the env, refuse
        logger.warning(
            "LeKiwi: call_911_emergency blocked by allowlist (destination=%s, env_set=%s)",
            _mask_phone_number(destination),
            False,
        )
        return (
            "Blocked: destination number not allowlisted. "
            "Set DEMO_EMERGENCY_NUMBER to the approved demo target."
        )

    livekit_url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    sip_trunk_id = os.getenv("LIVEKIT_SIP_TRUNK_ID")
    missing = [
        name
        for name, val in [
            ("LIVEKIT_URL", livekit_url),
            ("LIVEKIT_API_KEY", api_key),
            ("LIVEKIT_API_SECRET", api_secret),
            ("LIVEKIT_SIP_TRUNK_ID", sip_trunk_id),
        ]
        if not val
    ]
    if missing:
        logger.error(
            "LeKiwi: call_911_emergency missing required env vars: %s",
            ", ".join(missing),
        )
        return f"Missing required env vars for SIP call: {', '.join(missing)}"
    # Type narrowing for static checkers (os.getenv returns Optional[str])
    assert livekit_url and api_key and api_secret and sip_trunk_id

    room_name = "emergency-help-room"

    payload = {
        "sip_trunk_id": sip_trunk_id,
        "sip_call_to": destination,
        "room_name": room_name,
        "participant_identity": "sip-operator",
        "participant_name": "Demo 911",
        "wait_until_answered": True,
    }

    logger.info(
        "LeKiwi: initiating demo emergency SIP call (destination=%s, room=%s, trunk_set=%s, livekit_base=%s)",
        _mask_phone_number(destination),
        room_name,
        bool(sip_trunk_id),
        _normalize_base_url(livekit_url),
    )

    ok, result = await _create_sip_participant(
        base_url=_normalize_base_url(livekit_url),
        api_key=api_key,
        api_secret=api_secret,
        payload=payload,
    )
    if ok:
        logger.debug("LeKiwi: SIP participant created: %s", result)
        return "911 emergency call initiated (demo number), waiting for operator."
    else:
        logger.error("LeKiwi: failed to create SIP participant: %s", result)
        return f"Failed to initiate emergency call: {result}"


# TODO(ALI): not needed if you replace it
def _normalize_base_url(raw: str | None) -> str | None:
    if not raw:
        return None
    base = raw.rstrip("/")
    if base.startswith("wss://"):
        base = "https://" + base[len("wss://") :]
    elif base.startswith("ws://"):
        base = "http://" + base[len("ws://") :]
    return base


# TODO(ALI): not needed if you replace it
async def _create_sip_participant(
    base_url: str | None, api_key: str, api_secret: str, payload: dict
) -> tuple[bool, str | dict]:
    if not base_url:
        return False, "LIVEKIT_URL is not set or invalid"

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, _create_sip_participant_sync, base_url, api_key, api_secret, payload
    )


# TODO(ALI): not needed if you replace it
def _create_sip_participant_sync(
    base_url: str, api_key: str, api_secret: str, payload: dict
) -> tuple[bool, str | dict]:
    url = f"{base_url}/twirp/livekit.SIP/CreateSIPParticipant"
    data = json.dumps(payload).encode("utf-8")
    auth = base64.b64encode(f"{api_key}:{api_secret}".encode("utf-8")).decode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Basic {auth}",
    }
    safe_payload = dict(payload)
    if "sip_call_to" in safe_payload:
        safe_payload["sip_call_to"] = _mask_phone_number(
            str(safe_payload["sip_call_to"])
        )
    logger.debug(
        "LeKiwi: SIP CreateSIPParticipant POST %s payload=%s", url, safe_payload
    )
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
    except Exception as e:  # pragma: no cover - network/transport errors
        logger.exception("LeKiwi: SIP CreateSIPParticipant transport error")
        return False, str(e)


@function_tool
async def administer_epipen(self) -> str:
    """
    Administer the epipen to the person in need using advanced VLA control.
    This blocks until the epipen administration is complete.

    Returns:
        Confirmation message indicating successful epipen administration or failure reason.
    """
    guard_msg = _guard_emergency_only(self)
    if guard_msg is not None:
        return guard_msg

    # Check if service is ready
    if not hasattr(self, "epipen_service") or not self.epipen_service.is_ready():
        return "Error: Epipen service not available. Check π₀.₅ installation."

    # Pause arms service to prevent conflicting motor commands during ACT policy execution
    if hasattr(self, "arms_service"):
        self.arms_service.pause()

    # Execute synchronous epipen administration
    try:
        result = self.epipen_service.administer_epipen()
        logger.info(f"LeKiwi: administer_epipen completed with result: {result}")
        return result
    except Exception as e:
        error_msg = f"Epipen administration failed: {str(e)}"
        logger.error(f"LeKiwi: administer_epipen error: {error_msg}")
        return error_msg
    finally:
        # Always resume arms service after epipen administration (success or failure)
        if hasattr(self, "arms_service"):
            self.arms_service.resume()
