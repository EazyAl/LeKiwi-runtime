from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal


KeyNamespace = Literal["lekiwi", "so101"]


@dataclass(frozen=True)
class _KeyMaps:
    # Map between LeKiwi arm joint keys and SO101Follower keys (both include ".pos").
    lekiwi_to_so101: Dict[str, str]
    so101_to_lekiwi: Dict[str, str]


def _build_keymaps() -> _KeyMaps:
    lekiwi_to_so101 = {
        "arm_shoulder_pan.pos": "shoulder_pan.pos",
        "arm_shoulder_lift.pos": "shoulder_lift.pos",
        "arm_elbow_flex.pos": "elbow_flex.pos",
        "arm_wrist_flex.pos": "wrist_flex.pos",
        "arm_wrist_roll.pos": "wrist_roll.pos",
        "arm_gripper.pos": "gripper.pos",
    }
    so101_to_lekiwi = {v: k for k, v in lekiwi_to_so101.items()}
    return _KeyMaps(lekiwi_to_so101=lekiwi_to_so101, so101_to_lekiwi=so101_to_lekiwi)


_KEYMAPS = _build_keymaps()


class ArmOnlyAdapter:
    """
    Arm-only Robot adapter around `lekiwi.robot.lekiwi.LeKiwi`.

    Why this exists:
    - You can't instantiate a separate LeRobot `SO101Follower` on the same serial port while the base
      is also running (it would fight over the bus).
    - But many policies are trained on a 6-DoF arm-only embodiment. This adapter exposes only the arm
      features (6 joints) and forwards actions via `LeKiwi.send_arm_action`.

    `namespace` controls the joint key naming:
    - "lekiwi": uses keys like `arm_shoulder_pan.pos` (default; matches existing LeKiwi datasets/policies)
    - "so101": uses keys like `shoulder_pan.pos` (matches LeRobot `SO101Follower` datasets/policies)
    """

    # LeRobot uses `name` as the "robot_type" identifier in some flows.
    # We expose the SO101 type when using SO101 keys, otherwise keep "lekiwi".
    def __init__(self, lekiwi_robot: Any, *, namespace: KeyNamespace = "lekiwi") -> None:
        self._robot = lekiwi_robot
        self.namespace: KeyNamespace = namespace

    @property
    def name(self) -> str:
        return "so101_follower" if self.namespace == "so101" else getattr(self._robot, "name", "lekiwi")

    @property
    def robot_type(self) -> str:
        # Some LeRobot utilities use `robot.robot_type`; others use `robot.name`.
        return self.name

    @property
    def is_connected(self) -> bool:
        return bool(getattr(self._robot, "is_connected", False))

    def connect(self, calibrate: bool = True) -> None:
        # Delegate (no-op if already connected).
        return self._robot.connect(calibrate=calibrate)

    def disconnect(self) -> None:
        return self._robot.disconnect()

    @property
    def observation_features(self) -> Dict[str, type]:
        # Only 6 joint positions, no base velocities, no cameras (cameras are supplied externally via CameraHub).
        if self.namespace == "so101":
            return {v: float for v in _KEYMAPS.lekiwi_to_so101.values()}
        return {k: float for k in _KEYMAPS.lekiwi_to_so101.keys()}

    @property
    def action_features(self) -> Dict[str, type]:
        # Same as observation features for arm-only position control.
        return dict(self.observation_features)

    def get_observation(self) -> Dict[str, Any]:
        raw = self._robot.get_observation()
        out: Dict[str, Any] = {}
        for lekiwi_key, so101_key in _KEYMAPS.lekiwi_to_so101.items():
            if lekiwi_key not in raw:
                continue
            out_key = so101_key if self.namespace == "so101" else lekiwi_key
            out[out_key] = raw[lekiwi_key]
        return out

    def send_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send an arm-only position action.

        Accepts either namespace depending on `self.namespace`.
        """
        if self.namespace == "so101":
            lekiwi_action = { _KEYMAPS.so101_to_lekiwi[k]: v for k, v in action.items() if k in _KEYMAPS.so101_to_lekiwi }
            sent = self._robot.send_arm_action(lekiwi_action)
            # Map back to so101 keys for callers expecting that API.
            return { _KEYMAPS.lekiwi_to_so101[k]: v for k, v in sent.items() if k in _KEYMAPS.lekiwi_to_so101 }

        # lekiwi namespace
        lekiwi_action = {k: v for k, v in action.items() if k in _KEYMAPS.lekiwi_to_so101}
        return self._robot.send_arm_action(lekiwi_action)


