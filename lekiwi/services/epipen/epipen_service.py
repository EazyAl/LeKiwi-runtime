"""
Epipen administration service using ACT policy for precise medical procedure execution.
"""

import time
import logging
from typing import Optional
from lekiwi.robot.lekiwi import LeKiwi

# ACT policy imports (based on user's policy path)
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
    Designed to be called synchronously from workflow tools.
    """

    def __init__(
        self, robot: LeKiwi, policy_path: str = "CRPlab/lekiwi_test_act_policy_2"
    ):
        """
        Initialize epipen service with robot and ACT policy.

        Args:
            robot: Connected LeKiwi robot instance
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

        # Task description for ACT model
        self._task_description = (
            "Administer epipen to person in medical emergency. "
            "Carefully approach the person, locate the epipen, "
            "and administer it safely to the thigh area."
        )

        # Control parameters
        self.max_administration_time = 60  # Maximum 60 seconds for safety
        self.inference_fps = 10  # Adjust based on hardware

    def administer_epipen(self) -> str:
        """
        Execute complete epipen administration sequence using VLA control.

        This method BLOCKS until administration is complete, fails, or times out.
        Designed for synchronous workflow execution.

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
                    action_tensor = self.policy.select_action(observation)

                    # Convert tensor to dict format expected by LeKiwi robot
                    action_dict = self._tensor_to_action_dict(action_tensor)

                    # Execute actions on robot
                    self.robot.send_action(action_dict)

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

    def _tensor_to_action_dict(self, action_tensor) -> dict:
        """
        Convert ACT policy tensor output to LeKiwi robot action dict format.

        Args:
            action_tensor: Tensor output from ACT policy

        Returns:
            dict: Action dict in LeKiwi format
        """
        # Convert tensor to numpy and flatten
        action_values = action_tensor.detach().cpu().numpy().flatten()

        # LeKiwi action features in expected order
        action_keys = [
            "arm_shoulder_pan.pos",
            "arm_shoulder_lift.pos",
            "arm_elbow_flex.pos",
            "arm_wrist_flex.pos",
            "arm_wrist_roll.pos",
            "arm_gripper.pos",
            "x.vel",
            "y.vel",
            "theta.vel",
        ]

        # Create action dict
        action_dict = {}
        for i, key in enumerate(action_keys):
            if i < len(action_values):
                action_dict[key] = float(action_values[i])

        return action_dict

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

    def is_ready(self) -> bool:
        """
        Check if the epipen service is ready for operation.

        Returns:
            bool: True if service is ready, False otherwise
        """
        return self.policy is not None
