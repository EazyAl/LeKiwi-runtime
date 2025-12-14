"""
Epipen administration service using ACT policy for precise medical procedure execution.
Uses lerobot's official preprocessor/postprocessor pipeline for proper normalization.
"""

import time
import logging
from typing import Optional
from lekiwi.robot.lekiwi import LeKiwi

# lerobot imports for policy inference with proper normalization
try:
    from lerobot.policies.act.modeling_act import ACTPolicy

    ACT_AVAILABLE = True
except ImportError:
    ACT_AVAILABLE = False
    logging.warning("ACT policy not available. Install with: pip install -e '.[act]'")

logger = logging.getLogger(__name__)


class EpipenService:
    """
    Synchronous epipen administration service using ACT policy.

    Handles the complete epipen administration sequence with precise control.
    Uses lerobot's official preprocessor/postprocessor for proper normalization.
    Designed to be called synchronously from workflow tools.
    """

    def __init__(
        self, robot: LeKiwi, policy_path: str = "CRPlab/lekiwi_test_act_policy_2"
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

        if not ACT_AVAILABLE:
            logger.error("ACT policy not available. Cannot initialize EpipenService.")
            self.policy = None
        else:
            try:
                logger.info(f"Loading ACT policy from {policy_path}")
                self.policy = ACTPolicy.from_pretrained(policy_path)
                logger.info("ACT policy loaded successfully")
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
                    observation = self.robot.get_observation()

                    # Predict actions using ACT policy
                    with torch.inference_mode():
                        action = self.policy.select_action(obs)

                    # Postprocess action (unnormalizes outputs)
                    action = self.postprocess(action)

                    # Convert to robot action format
                    action_dict = make_robot_action(action, self.dataset_features)

                    # Execute actions on robot (arm only since policy was trained on arm)
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

            # Wrist camera is mounted upside down - rotate 180 degrees
            if cam_name == "wrist":
                frame = cv2.rotate(frame, cv2.ROTATE_180)

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

    def is_ready(self) -> bool:
        """
        Check if the epipen service is ready for operation.

        Returns:
            bool: True if service is ready, False otherwise
        """
        return self.policy is not None
