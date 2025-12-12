"""
View live data stream from autonomous LeKiwi using Rerun.
Requires autonomous mode running: python main.py --stream

NOTE: Remote IP: 172.20.10.2
NOTE: Stream port: 5556

Displays:
- Camera feeds with pose overlay
- MediaPipe skeleton
- Fall detection status
- Motor states
"""

import argparse
import base64
import numbers

import cv2
import numpy as np
import rerun as rr
import zmq


def _is_scalar(x):
    """Check if value is a scalar."""
    return isinstance(x, (float | numbers.Real | np.integer | np.floating)) or (
        isinstance(x, np.ndarray) and x.ndim == 0
    )


class LeKiwiObserver:
    def __init__(self, host_ip: str = "172.20.10.2", port: int = 5556):
        # ZMQ subscriber
        context = zmq.Context()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect(f"tcp://{host_ip}:{port}")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"Connected to LeKiwi stream at {host_ip}:{port}")

        # Initialize Rerun
        rr.init("lekiwi_autopilot_observer")
        rr.spawn(memory_limit="25%")
        print("Rerun viewer initialized")

    def _log_data(self, prefix: str, data: dict):
        """Log dictionary data to rerun, handling scalars, arrays, and images."""
        for key, value in data.items():
            if value is None:
                continue

            entity_path = f"{prefix}/{key}"

            if _is_scalar(value):
                rr.log(entity_path, rr.Scalars(float(value)))
            elif isinstance(value, str):
                # Handle base64 encoded images (from camera streams)
                if key.endswith("_camera") or "image" in key.lower():
                    try:
                        img_bytes = base64.b64decode(value)
                        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            # Convert BGR to RGB for rerun
                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            rr.log(entity_path, rr.Image(img_rgb))
                    except Exception as e:
                        print(f"Failed to decode image for {key}: {e}")
                else:
                    # Log as text annotation
                    rr.log(entity_path, rr.TextLog(value))
            elif isinstance(value, np.ndarray):
                arr = value
                # Convert CHW -> HWC for images
                if (
                    arr.ndim == 3
                    and arr.shape[0] in (1, 3, 4)
                    and arr.shape[-1] not in (1, 3, 4)
                ):
                    arr = np.transpose(arr, (1, 2, 0))

                if arr.ndim == 1:
                    # Log 1D arrays as individual scalars
                    for i, vi in enumerate(arr):
                        rr.log(f"{entity_path}_{i}", rr.Scalars(float(vi)))
                elif arr.ndim in (2, 3):
                    # Log as image
                    rr.log(entity_path, rr.Image(arr))
                else:
                    # Flatten and log higher-dimensional arrays
                    flat = arr.flatten()
                    for i, vi in enumerate(flat):
                        rr.log(f"{entity_path}_{i}", rr.Scalars(float(vi)))
            elif isinstance(value, dict):
                # Recursively log nested dictionaries
                self._log_data(entity_path, value)
            elif isinstance(value, (list, tuple)):
                # Log sequences as scalars
                for i, vi in enumerate(value):
                    if _is_scalar(vi):
                        rr.log(f"{entity_path}_{i}", rr.Scalars(float(vi)))

    def _render_pose(self, data: dict):
        """Render pose detection data."""
        self._log_data("pose", data)

        # Log specific pose events as annotations
        if "status" in data:
            rr.log("pose/status", rr.TextLog(str(data["status"])))

        # If landmarks are present, visualize them
        if "landmarks" in data and isinstance(data["landmarks"], (list, np.ndarray)):
            # MediaPipe pose landmarks are typically 33 points with x, y, z, visibility
            landmarks = np.array(data["landmarks"])
            if landmarks.size > 0:
                rr.log("pose/landmarks", rr.Points3D(landmarks))

        # Face-related stats (optional)
        face_stats = {
            "awake": data.get("awake_likelihood"),
            "blinks_per_min": data.get("blinks_per_min"),
            "perclos": data.get("perclos"),
            "eyes_open_prob": data.get("eyes_open_prob"),
            "hr_bpm": data.get("hr_bpm"),
            "hr_quality": data.get("hr_quality"),
            "hr_method": data.get("hr_method"),
            "face_status": data.get("face_status"),
        }
        self._log_data("pose/face_stats", face_stats)

        face_box = data.get("face_box")
        if isinstance(face_box, (list, tuple)) and len(face_box) == 4:
            x0, y0, x1, y1 = face_box
            rr.log("pose/face_box", rr.Boxes2D(mins=[[x0, y0]], sizes=[[x1 - x0, y1 - y0]], colors=[[0, 255, 255]]))

        face_landmarks = data.get("face_landmarks")
        if isinstance(face_landmarks, (list, np.ndarray)):
            pts = np.array(face_landmarks)
            if pts.ndim == 2 and pts.shape[0] > 0:
                if pts.shape[1] == 2:
                    rr.log("pose/face_landmarks", rr.Points2D(pts, colors=[[255, 255, 0]]))
                elif pts.shape[1] >= 3:
                    rr.log("pose/face_landmarks", rr.Points3D(pts[:, :3], colors=[[255, 255, 0]]))

    def _render_camera(self, data: dict):
        """Render camera data."""
        self._log_data("camera", data)

    def _render_motors(self, data: dict):
        """Render motor state data."""
        self._log_data("motors", data)

    def run(self):
        """Main visualization loop."""
        print("Starting observation loop...")
        try:
            while True:
                try:
                    # Non-blocking receive with timeout
                    message = self.socket.recv_json(flags=zmq.NOBLOCK)
                    data_type = message.get("type", "unknown")
                    timestamp = message.get("timestamp", 0)
                    data = message.get("data", {})

                    # Set the recording time for rerun
                    rr.set_time_seconds("timestamp", timestamp)

                    # Route to appropriate renderer
                    if data_type == "pose":
                        self._render_pose(data)
                    elif data_type == "camera":
                        self._render_camera(data)
                    elif data_type == "motors":
                        self._render_motors(data)
                    else:
                        # Log unknown data types generically
                        self._log_data(data_type, data)

                except zmq.Again:
                    # No message available, continue
                    continue
                except Exception as e:
                    print(f"Error processing message: {e}")
                    continue

        except KeyboardInterrupt:
            print("\nShutting down observer...")
        finally:
            self.socket.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Observe LeKiwi autonomous mode data")
    parser.add_argument(
        "--ip",
        type=str,
        default="172.20.10.2",
        help="Remote IP of the LeKiwi robot (default: 172.20.10.2)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="ZMQ stream port (default: 5556)",
    )
    args = parser.parse_args()

    observer = LeKiwiObserver(host_ip=args.ip, port=args.port)
    observer.run()
