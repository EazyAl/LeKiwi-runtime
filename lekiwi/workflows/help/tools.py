import logging

from livekit.agents import function_tool

logger = logging.getLogger(__name__)

# Workflow-specific tools for the help/emergency workflow
# These are dummy implementations - actual implementations will be added later


def _guard_emergency_only(agent) -> str | None:
    """
    Prevent emergency-only tools from executing during normal operation.

    Returns:
        None if execution is allowed, otherwise a user/LLM-facing message describing why it was blocked.
    """
    mode = getattr(agent, "operational_mode", "normal")
    if mode == "normal":
        msg = (
            "Blocked: this tool is emergency-only and will not run while operational_mode='normal'. "
            "Switch to 'concerned' or 'emergency' mode first (e.g., call toggle_state('concerned'))."
        )
        logger.warning(f"[HELP TOOL GUARD] {msg}")
        return msg
    return None


@function_tool
async def navigate_to_epipen_location(self) -> str:
    """
    Navigate to the epipen administration point. Move the robot to the location
    where the epipen is stored so it can be administered.

    Returns:
        Confirmation message indicating successful navigation to epipen location.
    """
    guard_msg = _guard_emergency_only(self)
    if guard_msg is not None:
        return guard_msg
    logger.debug("LeKiwi: navigate_to_epipen_location function called")
    # TODO: Implement actual navigation logic
    return "Navigated to epipen administration point"


@function_tool
async def call_911_emergency(self) -> str:
    """
    Call 911 emergency services. This initiates contact with emergency responders.
    The function should return success status which will be stored in call_911_success state.

    Returns:
        Confirmation message indicating whether the 911 call was successful or failed.
    """
    guard_msg = _guard_emergency_only(self)
    if guard_msg is not None:
        return guard_msg
    logger.debug("LeKiwi: call_911_emergency function called")
    # TODO: Implement actual 911 calling logic
    # This should return success/failure status
    return "911 emergency call initiated"


@function_tool
async def administer_epipen(self) -> str:
    """
    Administer the epipen to the person in need. This is a critical emergency action
    that should be performed when instructed by 911 operator or as a fallback if 911 call fails.

    Returns:
        Confirmation message indicating successful epipen administration.
    """
    guard_msg = _guard_emergency_only(self)
    if guard_msg is not None:
        return guard_msg
    logger.debug("LeKiwi: administer_epipen function called")
    # TODO: Implement actual epipen administration logic
    return "Epipen administered successfully"
