"""
Epipen administration service using PI0.5 policy for precise medical procedure execution.
Uses lerobot's official preprocessor/postprocessor pipeline for proper normalization.
"""

import time
import logging
from typing import Optional
import torch
from lekiwi.robot.lekiwi import LeKiwi

# lerobot imports for policy inference with proper normalization
try:
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.utils import build_inference_frame, make_robot_action
    from lerobot.datasets.utils import hw_to_dataset_features

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
        self, robot: LeKiwi, policy_path: str = "CRPlab/lekiwi_full_pi05_policy_1"
    ):
        """
        Initialize epipen service with robot and ACT policy.

        Args:
            robot: Connected LeKiwi robot instance
            policy_path: Path to ACT policy (default: CRPlab/lekiwi_test_act_policy_2)
        """
        self.robot = robot
        self.policy = None
        self.preprocess = None
        self.postprocess = None
        self.dataset_features = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

        if not LEROBOT_AVAILABLE:
            logger.error("lerobot not available. Cannot initialize EpipenService.")
        else:
            try:
                logger.info(f"Loading PI0 policy from {policy_path}")
                self.policy = PI0Policy.from_pretrained(policy_path)
                self.policy.eval()
                
                # Create preprocessor and postprocessor for proper normalization
                self.preprocess, self.postprocess = make_pre_post_processors(
                    self.policy.config,
                    policy_path,
                    preprocessor_overrides={"device_processor": {"device": str(self.device)}},
                )
                
                # Build dataset features from robot's observation/action features
                # Filter out base velocity keys - policy was trained with arm only (6 state values)
                arm_action_features = {k: v for k, v in robot.action_features.items() 
                                       if k not in ["x.vel", "y.vel", "theta.vel"]}
                arm_obs_features = {k: v for k, v in robot.observation_features.items() 
                                    if k not in ["x.vel", "y.vel", "theta.vel"]}
                
                action_features = hw_to_dataset_features(arm_action_features, "action")
                obs_features = hw_to_dataset_features(arm_obs_features, "observation")
                self.dataset_features = {**action_features, **obs_features}
                
                logger.info(f"PI0 policy loaded successfully on device: {self.device}")
            except Exception as e:
                logger.error(f"Failed to load PI0 policy: {e}")
                import traceback
                traceback.print_exc()

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
                    raw_obs = self.robot.get_observation()
                    
                    # Filter observation to only include arm state (policy trained with 6 arm values, not 9)
                    # Remove base velocity keys that aren't in the training data
                    filtered_obs = {k: v for k, v in raw_obs.items() 
                                   if k not in ["x.vel", "y.vel", "theta.vel"]}
                    
                    # Build inference frame using lerobot's utility
                    # Signature: build_inference_frame(observation, device, ds_features, task, robot_type)
                    obs_frame = build_inference_frame(
                        filtered_obs, 
                        self.device,
                        self.dataset_features, 
                        task=self._task_description,
                        robot_type=self._robot_type,
                    )
                    
                    # Preprocess observation (normalizes inputs)
                    obs = self.preprocess(obs_frame)

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

    def is_ready(self) -> bool:
        """
        Check if the epipen service is ready for operation.

        Returns:
            bool: True if service is ready, False otherwise
        """
        return self.policy is not None
