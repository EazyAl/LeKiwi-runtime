"""
Epipen administration service using ACT policy for precise medical procedure execution.
Uses lerobot's official preprocessor/postprocessor pipeline for proper normalization.
"""
import numpy as np
import torch
import time
import logging
from typing import Optional
from lekiwi.robot.lekiwi import LeKiwi

# lerobot imports for policy inference with proper normalization
try:
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import make_robot_action
    from lerobot.datasets.utils import hw_to_dataset_features, build_dataset_frame
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.control_utils import predict_action

    LEROBOT_AVAILABLE = True
except ImportError as e:
    LEROBOT_AVAILABLE = False
    logging.warning(f"lerobot not available: {e}")

logger = logging.getLogger(__name__)


class EpipenService:
    """
    Synchronous epipen administration service using ACT policy.

    Handles the complete epipen administration sequence with precise control.
    Uses lerobot's official preprocessor/postprocessor for proper normalization.
    Designed to be called synchronously from workflow tools.
    """

    def __init__(
        self,
        robot: LeKiwi,
        front_cam_sub=None,
        wrist_cam_sub=None,
        policy_path: str = "CRPlab/letars_test_policy_2",
        *,
        arm_only: Optional[bool] = None,
        output_rgb: bool = True,
        front_rotate_code: Optional[int] = None,
        wrist_rotate_code: Optional[int] = None,
    ):
        """
        Initialize epipen service with robot and ACT policy.

        Args:
            robot: Connected LeKiwi robot instance
            front_cam_sub: Optional front camera subscription for visual input
            wrist_cam_sub: Optional wrist camera subscription for visual input
            policy_path: Path to ACT policy (default: CRPlab/lekiwi_test_act_policy_2)
        """
        self.robot = robot
        self.front_cam_sub = front_cam_sub
        self.wrist_cam_sub = wrist_cam_sub
        # If None, we auto-detect from the policy's expected feature shapes.
        self.arm_only: Optional[bool] = arm_only
        self.output_rgb = output_rgb

        # Defaults chosen to match LeRobot's official LeKiwi OpenCV camera config:
        # - front: ROTATE_180
        # - wrist: ROTATE_90_CLOCKWISE (OpenCV uses clockwise rotation constant)
        # We keep these configurable because physical camera mounting can differ.
        try:
            import cv2  # type: ignore

            self._front_rotate_code = cv2.ROTATE_180 if front_rotate_code is None else front_rotate_code
            self._wrist_rotate_code = (
                cv2.ROTATE_90_CLOCKWISE if wrist_rotate_code is None else wrist_rotate_code
            )
        except Exception:
            # If cv2 isn't available at init time, we'll just skip rotation.
            self._front_rotate_code = front_rotate_code
            self._wrist_rotate_code = wrist_rotate_code

        self.policy = None
        self.preprocess = None
        self.postprocess = None
        self.dataset_features = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        if not LEROBOT_AVAILABLE:
            logger.error("lerobot not available. Cannot initialize EpipenService.")
        else:
            try:
                logger.info(f"Loading ACT policy from {policy_path}")
                self.policy = ACTPolicy.from_pretrained(policy_path)
                self.policy.eval()
                
                # Create preprocessor and postprocessor for proper normalization
                self.preprocess, self.postprocess = make_pre_post_processors(
                    self.policy.config,
                    policy_path,
                    preprocessor_overrides={"device_processor": {"device": str(self.device)}},
                )
                
                # Build dataset features from robot + cameras (and auto-match policy expected dims)
                # Image shape: (height, width, channels) -> (480, 640, 3)
                self._image_height = 480
                self._image_width = 640
                self._rebuild_dataset_features(robot)
                
                logger.info(f"ACT policy loaded successfully on device: {self.device}")
            except Exception as e:
                logger.error(f"Failed to load ACT policy: {e}")
                self.policy = None

        # Task description matching the training data
        self._task_description = "pick up object and stab object"

        # Robot type for multi-embodiment support
        self._robot_type = "lekiwi"

        # Control parameters
        self.max_administration_time = 60  # Maximum 60 seconds for safety
        self.inference_fps = 30  # Match dataset fps

    def administer_epipen(self) -> str:
        """
        Execute complete epipen administration sequence using VLA control.

        This method BLOCKS until administration is complete, fails, or times out.
        Uses lerobot's preprocessor/postprocessor for proper normalization.

        Returns:
            str: Success message or failure reason
        """
        if self.policy is None:
            return "Error: ACT policy not available. Check installation."

        try:
            logger.info("Starting epipen administration sequence...")
            print("Starting epipen administration sequence...")

            # Match `lerobot_record` behavior: reset policy + processors at the start of a run
            try:
                if hasattr(self.policy, "reset"):
                    self.policy.reset()
                if self.preprocess is not None and hasattr(self.preprocess, "reset"):
                    self.preprocess.reset()
                if self.postprocess is not None and hasattr(self.postprocess, "reset"):
                    self.postprocess.reset()
            except Exception as e:
                logger.warning(f"Failed to reset policy/processors (continuing): {e}")

            start_time = time.time()
            last_inference_time = 0

            while time.time() - start_time < self.max_administration_time:
                current_time = time.time()

                # Throttle inference to target FPS
                if current_time - last_inference_time < (1.0 / self.inference_fps):
                    time.sleep(0.01)  # Small delay
                    continue

                last_inference_time = current_time

                try:
                    # Get current observation from robot
                    raw_obs = self.robot.get_observation()
                    
                    # Optionally filter observation to arm-only (for policies trained without base velocities)
                    filtered_obs = raw_obs
                    if self.arm_only is True:
                        filtered_obs = {
                            k: v for k, v in raw_obs.items() if k not in ["x.vel", "y.vel", "theta.vel"]
                        }
                    
                    # Get camera images from subscriptions
                    front_frame = self._get_camera_frame(self.front_cam_sub, "front")
                    wrist_frame = self._get_camera_frame(self.wrist_cam_sub, "wrist")
                    
                    if front_frame is None or wrist_frame is None:
                        logger.warning("Missing camera frame, skipping inference step")
                        continue
                    
                    # Add camera images to observation
                    filtered_obs["front"] = front_frame
                    filtered_obs["wrist"] = wrist_frame

                    # Build observation frame exactly like `lerobot_record.py` does (numpy dict keyed by
                    # "observation.state" / "observation.images.*").
                    observation_frame = build_dataset_frame(self.dataset_features, filtered_obs, prefix=OBS_STR)

                    # Predict action using LeRobot's standard inference helper (handles:
                    # - image conversion to float32 [0,1] and channel-first
                    # - adding batch dim
                    # - adding task/robot_type
                    # - preprocessor + postprocessor pipelines)
                    action = predict_action(
                        observation=observation_frame,
                        policy=self.policy,
                        device=self.device,
                        preprocessor=self.preprocess,
                        postprocessor=self.postprocess,
                        use_amp=bool(getattr(self.policy.config, "use_amp", False)),
                        task=self._task_description,
                        robot_type=self._robot_type,
                    )
                    
                    # Convert to robot action format
                    action_dict = make_robot_action(action, self.dataset_features)
                    
                    # Execute actions on robot.
                    # `lerobot_record` uses `robot.send_action(...)` which can include safety clamping and
                    # base kinematics conversion. Prefer that path when possible.
                    if hasattr(self.robot, "send_action"):
                        # If policy is arm-only, LeKiwi.send_action requires base velocities; fill zeros.
                        if "x.vel" not in action_dict:
                            action_dict["x.vel"] = 0.0
                        if "y.vel" not in action_dict:
                            action_dict["y.vel"] = 0.0
                        if "theta.vel" not in action_dict:
                            action_dict["theta.vel"] = 0.0
                        self.robot.send_action(action_dict)
                    else:
                        # Fallback for minimal robot interfaces
                        self.robot.send_arm_action(action_dict)

                    # Check if administration is complete
                    if self._is_epipen_administration_complete():
                        completion_time = time.time() - start_time
                        success_msg = f"Epipen administered successfully in {completion_time:.1f} seconds"
                        logger.info(success_msg)
                        print(success_msg)
                        return success_msg

                except Exception as e:
                    error_msg = f"VLA inference/execution error: {str(e)}"
                    logger.error(error_msg)
                    import traceback

                    traceback.print_exc()
                    return error_msg

            # Timeout reached
            timeout_msg = f"Epipen administration timed out after {self.max_administration_time:.1f} seconds"
            logger.warning(timeout_msg)
            print(timeout_msg)
            return timeout_msg

        except Exception as e:
            error_msg = f"Epipen administration failed: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
            return error_msg

    def _is_epipen_administration_complete(self) -> bool:
        """
        Check if epipen administration is complete.

        TODO: Implement actual completion detection based on:
        - Visual confirmation (epipen administered)
        - Force/torque feedback
        - Position/movement confirmation
        - Success indicators from VLA model

        For now, this is a placeholder that returns False.
        """
        # Placeholder implementation
        # In practice, this would check:
        # 1. Visual feedback from camera
        # 2. Force sensors (if available)
        # 3. Position confirmation
        # 4. VLA model confidence/completion signals

        return False  # Never complete for safety during development

    def _get_camera_frame(self, cam_sub, cam_name: str) -> Optional[np.ndarray]:
        """
        Get the latest camera frame from a subscription.

        Args:
            cam_sub: Camera subscription object with pull() method
            cam_name: Name of the camera for logging

        Returns:
            numpy array of shape (H, W, 3) in BGR format, or None if unavailable
        """
        if cam_sub is None:
            logger.warning(f"No subscription for {cam_name} camera")
            return None

        try:
            import cv2

            pulled = cam_sub.pull(timeout=0.1)
            if pulled is None:
                return None
            ts, frame = pulled

            # CameraHub frames are OpenCV-native (BGR). LeRobot's inference pipeline typically
            # expects RGB images, so convert here to match `lerobot_record` behavior.
            if self.output_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply configured rotation (to match how LeRobot's OpenCVCamera does it).
            rotate_code = None
            if cam_name == "front":
                rotate_code = self._front_rotate_code
            elif cam_name == "wrist":
                rotate_code = self._wrist_rotate_code
            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)
            
            # Ensure frame is the right shape (resize if needed)
            if (
                frame.shape[0] != self._image_height
                or frame.shape[1] != self._image_width
            ):
                frame = cv2.resize(frame, (self._image_width, self._image_height))
            return frame
        except Exception as e:
            logger.warning(f"Failed to get {cam_name} camera frame: {e}")
            return None

    def _rebuild_dataset_features(self, robot: LeKiwi) -> None:
        """
        Build dataset features for inference in a way that matches what the policy expects.

        The most common mismatch (and the one you hit) is a policy trained with 6-D arm-only state,
        while the LeKiwi robot exposes 9 state dims (arm + base velocities). If `arm_only` is None,
        we try to infer it from the policy config and automatically filter base velocities.
        """
        if not LEROBOT_AVAILABLE:
            return

        # Decide arm_only if auto
        if self.arm_only is None:
            inferred = self._infer_arm_only_from_policy(robot)
            self.arm_only = inferred
            logger.info(f"EpipenService arm_only auto-detected as {self.arm_only}")

        action_features_hw = robot.action_features
        obs_features_hw = robot.observation_features
        if self.arm_only is True:
            action_features_hw = {
                k: v for k, v in action_features_hw.items() if k not in ["x.vel", "y.vel", "theta.vel"]
            }
            obs_features_hw = {k: v for k, v in obs_features_hw.items() if k not in ["x.vel", "y.vel", "theta.vel"]}

        camera_obs_features = {
            "front": (self._image_height, self._image_width, 3),
            "wrist": (self._image_height, self._image_width, 3),
        }

        action_features = hw_to_dataset_features(action_features_hw, "action")
        obs_features = hw_to_dataset_features({**obs_features_hw, **camera_obs_features}, "observation")
        self.dataset_features = {**action_features, **obs_features}

    def _infer_arm_only_from_policy(self, robot: LeKiwi) -> bool:
        """
        Infer whether the policy expects arm-only state/action sizes.

        Falls back to `True` if we can't infer reliably, because most ACT policies trained for
        manipulation on LeKiwi are arm-only and will crash if provided 9-D state.
        """
        try:
            cfg = getattr(self.policy, "config", None)
            input_features = getattr(cfg, "input_features", None)
            output_features = getattr(cfg, "output_features", None)

            # Expected state dimension from policy config (most reliable signal)
            expected_state_dim = None
            if isinstance(input_features, dict):
                ft = input_features.get("observation.state")
                shape = getattr(ft, "shape", None)
                if isinstance(shape, (tuple, list)) and len(shape) == 1:
                    expected_state_dim = int(shape[0])

            # If policy expects fewer state dims than robot exposes, it's arm-only.
            robot_state_dim = len(robot._state_ft) if hasattr(robot, "_state_ft") else len(
                [k for k in robot.observation_features if k.endswith(".pos") or k.endswith(".vel")]
            )
            if expected_state_dim is not None:
                return expected_state_dim < robot_state_dim

            # Expected action dim from policy config (secondary signal)
            expected_action_dim = None
            if isinstance(output_features, dict):
                aft = output_features.get("action")
                ashape = getattr(aft, "shape", None)
                if isinstance(ashape, (tuple, list)) and len(ashape) == 1:
                    expected_action_dim = int(ashape[0])
            if expected_action_dim is not None:
                robot_action_dim = len(robot.action_features)
                return expected_action_dim < robot_action_dim

        except Exception as e:
            logger.warning(f"Could not infer arm_only from policy config: {e}")

        # Safe default: arm-only
        return True

    def is_ready(self) -> bool:
        """
        Check if the epipen service is ready for operation.

        Returns:
            bool: True if service is ready, False otherwise
        """
        return self.policy is not None
