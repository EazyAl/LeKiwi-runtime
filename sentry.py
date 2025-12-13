import time
import logging
import threading
import sys
import queue
import numpy as np
import sounddevice as sd
import os
import tempfile
import scipy.io.wavfile as wav

# Set logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sentry")

# Imports
from lekiwi.robot.lekiwi import LeKiwi
# We need to access LeKiwiConfig to initialize the robot
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

from lekiwi.services.pose_detection.pose_service import PoseDetectionService
from lekiwi.services.navigation.navigator import Navigator
from lekiwi.services.epipen.epipen_service import EpipenService
from lekiwi.vision.camera_hub import CameraHub
from lekiwi.viz.rerun_viz import create_viz

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
AUDIO_DEVICE = 0

class MockRobot:
    """Mock robot for testing logic without hardware."""
    def __init__(self):
        self.is_connected = False
        
    def connect(self, calibrate=False):
        self.is_connected = True
        logger.info("[MockRobot] Connected")
        
    def disconnect(self):
        self.is_connected = False
        logger.info("[MockRobot] Disconnected")
        
    def stop_base(self):
        logger.info("[MockRobot] stop_base called")
        
    def send_base_action(self, action):
        logger.info(f"[MockRobot] send_base_action: {action}")
        
    def send_action(self, action):
        logger.info(f"[MockRobot] send_action: {action}")
        
    def get_observation(self):
        logger.info("[MockRobot] get_observation called")
        # Return dummy dict to prevent immediate crash, though ACT policy might still fail
        return {}

class Sentry:
    def __init__(self):
        self.state = "NORMAL" # NORMAL, CONCERN, EMERGENCY
        self.running = True
        
        # --- Robot Setup ---
        # Initialize LeKiwi without cameras (handled by CameraHub for vision services)
        # Using default port/id from main.py
        self.robot_config = LeKiwiConfig(
            port="/dev/tty.usbmodem58760432781", 
            id="biden_kiwi",
            cameras={} # No cameras managed by LeKiwi directly
        )
        self.robot = LeKiwi(self.robot_config)
        # self.robot = MockRobot()
        
        # --- Vision Setup ---
        # CameraHub manages the camera devices and distributes frames
        self.camera_hub = CameraHub(front_index=0, wrist_index=1)
        
        # --- Visualization Setup ---
        self.viz = create_viz(enable=True, app_id="lekiwi_sentry")
        # Subscribe to wrist camera for visualization
        self.wrist_sub = self.camera_hub.subscribe_wrist(max_queue=2)
        
        # --- Services Setup ---
        
        # Pose Detection Service
        # Subscribes to front camera frames
        self.pose_sub = self.camera_hub.subscribe_front(max_queue=2)
        self.pose_service = PoseDetectionService(
            status_callback=self._handle_pose_status,
            frame_subscription=self.pose_sub,
            viz=self.viz
        )
        
        # Navigator Service
        # Also subscribes to front camera frames
        self.nav_sub = self.camera_hub.subscribe_front(max_queue=2)
        self.navigator = Navigator(
            robot=self.robot,
            frame_subscription=self.nav_sub,
            viz=self.viz
        )
        
        # Epipen Service
        self.epipen_service = EpipenService(
            robot=self.robot,
            front_cam_sub=self.camera_hub.subscribe_front(max_queue=1),
            wrist_cam_sub=self.camera_hub.subscribe_wrist(max_queue=1)
        )
        
        # State management
        self.fall_event = threading.Event()
        
    def start(self):
        logger.info("Starting Sentry System...")
        
        # Connect Robot
        try:
            logger.info("Connecting to robot...")
            self.robot.connect(calibrate=False)
        except Exception as e:
            logger.error(f"Failed to connect robot: {e}")
            return

        # Start Camera Hub
        logger.info("Starting cameras...")
        self.camera_hub.start()
        # Start wrist camera visualization pump
        threading.Thread(target=self._pump_wrist, daemon=True, name="wrist-viz").start()
        
        # Start Pose Service
        logger.info("Starting pose detection...")
        self.pose_service.start()
        
        # Main State Machine Loop
        try:
            self._push_status_to_viz()
            while self.running:
                logger.info(f"Current State: {self.state}")
                
                if self.state == "NORMAL":
                    self._run_normal()
                elif self.state == "CONCERN":
                    self._run_concern()
                elif self.state == "EMERGENCY":
                    self._run_emergency()
                    # End run after emergency sequence
                    self.running = False
                
                self._push_status_to_viz()
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Stopping Sentry...")
        finally:
            self.stop()
            
    def stop(self):
        logger.info("Stopping services...")
        self.pose_service.stop()
        self.camera_hub.stop()
        if self.viz:
            self.viz.close()
        if self.robot.is_connected:
            self.robot.disconnect()
        logger.info("Sentry stopped.")

    def _pump_wrist(self):
        """Continuously pull wrist frames and send to visualization."""
        while self.running:
            pulled = self.wrist_sub.pull(timeout=0.1)
            if pulled:
                ts, frame = pulled
                if self.viz:
                    self.viz.log_wrist_rgb(frame, ts=ts)
            else:
                # Sleep briefly if no frame to avoid tight loop
                time.sleep(0.05)

    def _handle_pose_status(self, status_type: str, details: dict):
        """Callback for pose detection service."""
        if status_type == "PERSON_FALLEN":
            if self.state == "NORMAL":
                logger.warning(f"FALL DETECTED! Details: {details}")
                self.fall_event.set()

    def _push_status_to_viz(self):
        if self.viz:
            try:
                self.viz.set_status(self.state, ts=time.time())
            except Exception:
                pass

    def _run_normal(self):
        """Monitor for falls."""
        # Check if fall event occurred
        if self.fall_event.is_set():
            logger.info("Fall detected in NORMAL state. Switching to CONCERN.")
            self.state = "CONCERN"
            self._push_status_to_viz()
            self.fall_event.clear()
        else:
            time.sleep(0.1)

    def _run_concern(self):
        """Handle potential emergency: Navigate, Assess, React."""
        logger.info("Entering CONCERN mode")
        
        # 1. Navigate to person
        logger.info("Navigating to person...")
        nav_result = self.navigator.navigate_to_person()
        logger.info(f"Navigation result: {nav_result}")
        
        # 2. Verbal confirmation
        logger.info("Playing audio challenge (Simulated)...")
        self._play_audio_challenge()
        
        # 3. Record response
        logger.info("Recording response (5s)...")
        audio_data = self._record_audio(duration=5)
        
        # 4. Check response
        if self._is_loud_enough(audio_data):
            logger.info("Response detected. Analyzing speech...")
            text = self._speech_to_text(audio_data)
            logger.info(f"Transcribed text: '{text}'")
            
            if self._is_confirmation_okay(text):
                logger.info("Human confirmed OK. Reverting to NORMAL.")
                self.state = "NORMAL"
                self._push_status_to_viz()
            else:
                logger.warning("Human not OK or ambiguous. Switching to EMERGENCY.")
                self.state = "EMERGENCY"
                self._push_status_to_viz()
        else:
            logger.warning("No response (too quiet). Switching to EMERGENCY.")
            self.state = "EMERGENCY"
            self._push_status_to_viz()

    def _run_emergency(self):
        """Handle emergency: Administer Epipen."""
        logger.error("ENTERING EMERGENCY MODE")
        
        # Run Epipen Service
        # Note: ACT policy might require camera images in observation. 
        # Since we initialized LeKiwi without cameras, this might fail if the policy is vision-dependent.
        # Future TODO: Augment observation with CameraHub frames.
        
        result = self.epipen_service.administer_epipen()
        logger.info(f"Epipen Service Result: {result}")

    # --- Audio Helpers ---
    def _play_audio_challenge(self):
        """Play a sound to challenge the fallen person."""
        # Placeholder: Generate a tone 
        # Ideally play a wav file like "Are you okay?"
        fs = 16000
        duration = 1.0
        t = np.linspace(0, duration, int(fs * duration), endpoint=False)
        tone = 0.5 * np.sin(2 * np.pi * 440 * t) # A4 tone
        try:
            sd.play(tone, samplerate=fs)
            sd.wait()
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")

    def _record_audio(self, duration=5):
        """Record audio for specified duration."""
        try:
            recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
            sd.wait()
            return recording
        except Exception as e:
            logger.error(f"Audio recording failed: {e}")
            return np.zeros((100, 1))

    def _is_loud_enough(self, audio_data, threshold=0.02):
        """Check if audio volume exceeds threshold (RMS)."""
        rms = np.sqrt(np.mean(audio_data**2))
        logger.info(f"Audio RMS: {rms:.4f} (Threshold: {threshold})")
        return rms > threshold

    def _speech_to_text(self, audio_data):
        """Transcribe audio using OpenAI Whisper if available."""
        # Try importing openai
        try:
            from openai import OpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("No OPENAI_API_KEY. Cannot perform STT.")
                return ""
                
            client = OpenAI(api_key=api_key)
            
            # Save to temp wav
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav.write(tmp.name, SAMPLE_RATE, audio_data)
                tmp_path = tmp.name
                
            with open(tmp_path, "rb") as audio_file:
                logger.info("Sending audio to OpenAI Whisper...")
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file
                )
            
            os.remove(tmp_path)
            return transcript.text.lower()
            
        except ImportError:
            logger.warning("OpenAI library not found. STT unavailable.")
            return ""
        except Exception as e:
            logger.error(f"STT failed: {e}")
            return ""

    def _is_confirmation_okay(self, text):
        """Check if text indicates status is OK."""
        # Basic keyword matching
        positive_keywords = ["okay", "ok", "fine", "good", "alive", "yes", "i'm good"]
        negative_keywords = ["help", "pain", "hurt", "no", "emergency", "dying"]
        
        text = text.lower()
        
        if any(word in text for word in negative_keywords):
            return False
            
        return any(word in text for word in positive_keywords)

if __name__ == "__main__":
    sentry = Sentry()
    sentry.start()
