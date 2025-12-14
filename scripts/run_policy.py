#!/usr/bin/env python3
"""Run the Epipen (PI0.5) policy directly.

This is a small CLI entrypoint (similar spirit to `nav/nav.py`) that:
- Connects to the LeKiwi robot
- Loads `EpipenService`
- Runs `administer_epipen()` once

Example:
  python run_policy.py --port /dev/ttyACM0 --id biden_kiwi
"""

from __future__ import annotations

import argparse
import logging
import sys

from lekiwi.robot import LeKiwi
from lekiwi.services.epipen import EpipenService


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _build_robot(args: argparse.Namespace) -> LeKiwi:
    # Import here so the script can still show a useful error if lerobot
    # extras aren't installed.
    from lerobot.robots.lekiwi.config_lekiwi import LeKiwiConfig

    cameras = {}
    if not args.no_cameras:
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        cameras = {
            "front": OpenCVCameraConfig(
                index_or_path=args.front_cam,
                width=args.cam_width,
                height=args.cam_height,
                fps=args.cam_fps,
            ),
            "wrist": OpenCVCameraConfig(
                index_or_path=args.wrist_cam,
                width=args.cam_width,
                height=args.cam_height,
                fps=args.cam_fps,
            ),
        }

    config = LeKiwiConfig(port=args.port, id=args.id, cameras=cameras)
    robot = LeKiwi(config)
    robot.connect(calibrate=bool(args.calibrate))
    return robot


def main() -> int:
    _configure_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run the LeKiwi epipen policy")
    parser.add_argument("--id", type=str, default="biden_kiwi", help="Robot ID")
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyACM0",
        help="Serial port for the robot (direct USB)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run motor calibration on connect (default: off)",
    )

    parser.add_argument(
        "--policy_path",
        type=str,
        default="CRPlab/lekiwi_full_pi05_policy_1",
        help="HF model id / local path for PI0 policy",
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=60.0,
        help="Maximum time (seconds) to run the policy for safety",
    )
    parser.add_argument(
        "--inference_fps",
        type=float,
        default=30.0,
        help="Inference loop FPS target",
    )

    parser.add_argument(
        "--no_cameras",
        action="store_true",
        help="Disable robot cameras (state-only observations)",
    )
    parser.add_argument(
        "--front_cam",
        type=int,
        default=0,
        help="OpenCV index for the front camera (if cameras enabled)",
    )
    parser.add_argument(
        "--wrist_cam",
        type=int,
        default=1,
        help="OpenCV index for the wrist camera (if cameras enabled)",
    )
    parser.add_argument(
        "--cam_width",
        type=int,
        default=1280,
        help="Camera width (if cameras enabled)",
    )
    parser.add_argument(
        "--cam_height",
        type=int,
        default=720,
        help="Camera height (if cameras enabled)",
    )
    parser.add_argument(
        "--cam_fps",
        type=int,
        default=30,
        help="Camera FPS (if cameras enabled)",
    )

    args = parser.parse_args()

    robot = None
    try:
        logger.info("Connecting to robot (%s on %s)...", args.id, args.port)
        robot = _build_robot(args)
        logger.info("Robot connected")

        service = EpipenService(robot=robot, policy_path=args.policy_path)
        service.max_administration_time = float(args.max_time)
        service.inference_fps = float(args.inference_fps)

        if not service.is_ready():
            print("EpipenService is not ready (policy failed to load).")
            return 2

        # Warm-up one read to surface camera/serial issues before we start the loop.
        _ = robot.get_observation()

        logger.info("Starting policy run...")
        result = service.administer_epipen()
        print(result)

        # Treat timeouts as a normal outcome for now (completion detection is a placeholder).
        if isinstance(result, str) and result.lower().startswith("error"):
            return 2
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130

    except Exception as e:
        logger.exception("run_policy failed: %s", e)
        return 1

    finally:
        if robot is not None and getattr(robot, "is_connected", False):
            try:
                robot.stop_base()
            except Exception:
                pass
            try:
                robot.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
