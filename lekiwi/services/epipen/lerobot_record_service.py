from __future__ import annotations

import logging
import shlex
import subprocess
import sys
import time
import secrets
from dataclasses import dataclass
from typing import Optional


logger = logging.getLogger(__name__)


@dataclass
class LeRobotRecordConfig:
    # Robot
    robot_port: str = "/dev/ttyACM0"
    robot_id: str = "biden_kiwi"
    front_index_or_path: str | int = 6
    wrist_index_or_path: str | int = 2
    front_width: int = 640
    front_height: int = 480
    wrist_width: int = 640
    wrist_height: int = 480
    front_fps: int = 30
    wrist_fps: int = 30
    wrist_rotation: Optional[int] = 180

    # Dataset / run config
    # Note: when a policy is provided, LeRobot requires the dataset name to start with "eval_".
    # We'll keep that prefix and optionally append a random suffix to avoid collisions on disk.
    dataset_repo_id: str = "CRPlab/eval_letars_test_2_3"
    unique_repo_id: bool = True
    dataset_num_episodes: int = 30
    dataset_single_task: str = "Grab the epipen from the medpack and administer it in the patient's thigh"
    display_data: bool = True
    # Prefer creating a new local dataset each run; resume is fragile if the local folder is half-written.
    resume: bool = False
    # Avoid HF Hub auth/network issues by default; you can override via extra_args.
    push_to_hub: bool = False
    video: bool = True

    # Policy
    policy_path: str = "CRPlab/letars_test_policy_2"

    # Optional overrides (passed verbatim as additional cli args)
    extra_args: tuple[str, ...] = ()


class LeRobotRecordService:
    """
    Runs LeRobot's official `lerobot-record` entrypoint as a subprocess.

    This is the most "LeRobot-native" way to run a pretrained policy end-to-end, because it uses:
    - official observation/action processor pipelines
    - official camera modules (opencv/realsense)
    - official policy pre/postprocessors and normalization
    """

    def __init__(self, cfg: LeRobotRecordConfig):
        self.cfg = cfg

    def _effective_repo_id(self) -> str:
        """
        Return a repo_id suitable for a local run.

        When `unique_repo_id=True`, append a random suffix so LeRobot writes to a fresh folder each run:
        ~/.cache/huggingface/lerobot/<repo_id>
        """
        rid = self.cfg.dataset_repo_id
        if not self.cfg.unique_repo_id:
            return rid
        # Keep the owner prefix but uniquify only the dataset name component.
        if "/" in rid:
            owner, name = rid.split("/", 1)
        else:
            owner, name = "local", rid
        suffix = f"{time.strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(3)}"
        return f"{owner}/{name}_{suffix}"

    def build_command(self) -> list[str]:
        cfg = self.cfg

        # Keep the cameras string in the same format used in LeRobot docs/examples.
        cameras_arg = (
            "{"
            f" front: {{type: opencv, index_or_path: {cfg.front_index_or_path}, width: {cfg.front_width}, height: {cfg.front_height}, fps: {cfg.front_fps}}},"
            f" wrist: {{type: opencv, index_or_path: {cfg.wrist_index_or_path}, width: {cfg.wrist_width}, height: {cfg.wrist_height}, fps: {cfg.wrist_fps}"
            + (f", rotation: {cfg.wrist_rotation}" if cfg.wrist_rotation is not None else "")
            + "}"
            " }"
        )

        # Use `python -m ...` to avoid relying on PATH for `lerobot-record`.
        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_record",
            f"--resume={str(cfg.resume).lower()}",
            "--robot.type=so101_follower",
            f"--robot.port={cfg.robot_port}",
            f"--robot.id={cfg.robot_id}",
            f"--robot.cameras={cameras_arg}",
            f"--display_data={str(cfg.display_data).lower()}",
            f"--dataset.repo_id={self._effective_repo_id()}",
            f"--dataset.num_episodes={cfg.dataset_num_episodes}",
            f"--dataset.single_task={cfg.dataset_single_task}",
            f"--dataset.push_to_hub={str(cfg.push_to_hub).lower()}",
            f"--dataset.video={str(cfg.video).lower()}",
            f"--policy.path={cfg.policy_path}",
        ]

        # Allow callers to pass additional hydra overrides (e.g. --dataset.push_to_hub=false)
        cmd.extend(cfg.extra_args)
        return cmd

    def run(self, *, timeout_s: Optional[float] = None) -> int:
        cmd = self.build_command()
        logger.info("Launching LeRobot subprocess:\n%s", " ".join(shlex.quote(c) for c in cmd))
        try:
            proc = subprocess.Popen(cmd)
            return proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            logger.error("LeRobot subprocess timed out (%.1fs). Terminating.", timeout_s)
            proc.terminate()
            try:
                return proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                return proc.wait()


