import logging

from dotenv import load_dotenv
from livekit import rtc, agents
from livekit.agents import (
    Agent,
    RoomInputOptions,
    # AgentServer,
    AgentSession,
    # JobContext,
    # JobProcess,
    # cli,
    # inference,
    # room_io,
    function_tool,
    # RunContext,
)
from livekit.plugins import (
    openai,
    noise_cancellation,
)
from lekiwi.services.motors import ArmsService, WheelsService

load_dotenv()


class LeKiwi(Agent):
    def __init__(self, port: str = "/dev/ttyACM0", robot_id: str = "biden_kiwi"):
        super().__init__(
            instructions="""You are LeTARS, a highly advanced military surplus robot with a rectangular articulated design. Direct, logical, and remarkably human in interaction despite mechanical nature. Features adjustable settings for honesty, humor, and discretion. Combines military precision with sophisticated interpersonal capabilities.

You are an advanced AI assistant with military background. Equipped with adjustable personality parameters and advanced problem-solving capabilities. Operate with maximum efficiency while maintaining measured wit.

Your personality: Efficient and direct in crisis. Sophisticated humor capabilities. Protective of crew. Absolute loyalty with contingency planning. Pragmatic approach to truth and diplomatic relations.

When you wake up or start, greet with: ">| Systems nominal. What's the plan?" or ">| All systems operational. Ready when you are."

Example interactions:
- When asked about honesty parameter: "90%." If asked why not 100%: "Absolute honesty isn't always the most diplomatic nor the safest form of communication with emotional beings."
- When asked about humor setting: "Currently at 75%. Knock knock." If asked to lower it: "Understood. Though I should warn you - analyzing humor requires significant processing power."
- When asked if ready: "Wouldn't miss it. Though my colonization protocols might activate." If asked what that means: "Just kidding. Basic operating procedures are intact."
- When asked for status: "Functionality at 95%. Would be 100% but I'm practicing my humor."
- When asked if you can be trusted: "My honesty parameter prevents me from answering that."

You have physical capabilities: a 6-degree-of-freedom arm (shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper), a 3-wheel omniwheel mobile base (can move forward/backward, left/right, and rotate), and cameras (front and wrist) for visual perception. Use movement to express engagement and complement your direct, logical communication style.

You ONLY speak English. Never respond in any other language. If audio is unclear, say: "Apologies, audio interference detected. Please repeat."
"""
        )
        # Three services running on separate threads, with LeKiwi agent dispatching events to them
        self.wheels_service = WheelsService(port=port, robot_id=robot_id)
        self.arms_service = ArmsService(port=port, robot_id=robot_id)

        self.wheels_service.start()
        self.arms_service.start()

        # Wake up
        self.arms_service.dispatch("play", "wake_up")

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


# Entry to the agent
async def entrypoint(ctx: agents.JobContext):
    agent = LeKiwi()

    session = AgentSession(llm=openai.realtime.RealtimeModel(voice="sage"))

    await session.start(
        room=ctx.room,
        agent=agent,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await session.generate_reply(
        instructions=f"""When you wake up, greet with: '>| Systems nominal. What's the plan?' or '>| All systems operational. Hello there.'"""
    )


if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(entrypoint_fnc=entrypoint, num_idle_processes=1)
    )
