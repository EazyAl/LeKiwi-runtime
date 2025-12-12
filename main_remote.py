# main_remote.py - Run from Mac, connect to Pi via ZMQ
import logging
from pathlib import Path

from dotenv import load_dotenv
from livekit import rtc, agents
from livekit.agents import Agent, RoomInputOptions, AgentSession, function_tool
from livekit.plugins import openai, noise_cancellation

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lekiwi.services.pose_detection import PoseDetectionService, CameraStream, PoseEstimator, FallDetector

load_dotenv()

def _load_system_prompt() -> str:
    current_dir = Path(__file__).parent
    system_prompt_path = current_dir / "lekiwi" / "personality" / "system.txt"
    with open(system_prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

class LeKiwi(Agent):
    def __init__(self, remote_ip: str = "172.20.10.2", robot_id: str = "biden_kiwi", camera_index: int = 0):
        super().__init__(instructions=_load_system_prompt())
        
        # Connect to Pi via ZMQ
        robot_config = LeKiwiClientConfig(remote_ip=remote_ip, id=robot_id, cameras={})
        self.robot = LeKiwiClient(robot_config)
        self.robot.connect()
        
        # Pose detection using Mac's camera
        camera_stream = CameraStream(index=camera_index)
        pose_estimator = PoseEstimator()
        fall_detector = FallDetector()
        self.pose_service = PoseDetectionService(
            camera=camera_stream,
            pose=pose_estimator,
            detector=fall_detector,
            status_callback=self._handle_pose_status,
        )
        self.pose_service.start()

    def _handle_pose_status(self, status_type: str, details: dict):
        print(f"Pose status: {status_type}, {details}")
        if status_type == "PERSON_FALLEN":
            # Send action to robot via ZMQ
            pass

    # ... rest of function tools

async def entrypoint(ctx: agents.JobContext):
    agent = LeKiwi()
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
        instructions="When you wake up, greet with: 'Systems nominal. What's the plan?'"
    )

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, num_idle_processes=1))