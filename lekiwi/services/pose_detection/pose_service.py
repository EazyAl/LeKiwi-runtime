"""Modular fall-detection components for reuse by an orchestrator."""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass
from typing import Callable, Optional, Any, Dict, Tuple

import cv2
import mediapipe as mp
import numpy as np

# Assuming these are available from your common service files
from ..base import ServiceBase
from ...vision import FrameRingBuffer, compute_quality_metrics
from lekiwi.viz.rerun_viz import NullViz

_WARNED_OPENCV_GUI = False


@dataclass
class FallEvent:
    is_fall: bool
    score: float
    ratio: float
    timestamp: float


class CameraStream:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> None:
        # Prefer V4L2 on Linux to avoid OpenCV selecting an unexpected backend.
        try:
            self.cap = cv2.VideoCapture(int(self.index), cv2.CAP_V4L2)
        except Exception:
            self.cap = cv2.VideoCapture(self.index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.index}")

    def read(self):
        if self.cap is None:
            raise RuntimeError("Camera not started")
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("Frame grab failed")
        return frame

    def stop(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None


class PoseEstimator:
    def __init__(self) -> None:
        self.pose = mp.solutions.pose.Pose(
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def infer(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb)

    def close(self) -> None:
        # MediaPipe raises if close() is called twice (internal graph becomes None).
        # Make this idempotent so orchestrators can safely stop services multiple times.
        if self.pose is None:
            return
        try:
            self.pose.close()
        except ValueError:
            # "Closing SolutionBase._graph which is already None"
            pass
        finally:
            self.pose = None


class FallDetector:
    def __init__(
        self, ratio_thresh: float = 0.8, window: int = 12, min_conf: float = 0.5
    ) -> None:
        self.ratio_thresh = ratio_thresh
        self.window = collections.deque(maxlen=window)
        self.min_conf = min_conf

    @staticmethod
    def _torso_ratio(landmarks) -> float:
        shx = (landmarks[11].x + landmarks[12].x) * 0.5
        shy = (landmarks[11].y + landmarks[12].y) * 0.5
        hpx = (landmarks[23].x + landmarks[24].x) * 0.5
        hpy = (landmarks[23].y + landmarks[24].y) * 0.5
        vert = abs(shy - hpy)
        horiz = abs(shx - hpx)
        return vert / (horiz + 1e-4)

    def detect(self, landmarks) -> Optional[FallEvent]:
        confs = [landmarks[i].visibility for i in (11, 12, 23, 24)]
        if min(confs) < self.min_conf:
            self.window.append(False)
            return None

        ratio = self._torso_ratio(landmarks)
        is_fall = ratio < self.ratio_thresh
        self.window.append(is_fall)
        score = sum(self.window) / len(self.window)
        event = FallEvent(
            is_fall=score > 0.6, score=score, ratio=ratio, timestamp=time.time()
        )
        return event


# --- Service Interface Types ---
VisualizerFn = Callable[[any, any, Optional[FallEvent], bool, float], bool]
# The event handler now matches the style used in LeKiwi: (event_type: str, details: Dict)
EventHandler = Callable[[str, Dict[str, Any]], None]


class PoseDetectionService(ServiceBase):
    """
    Runs the fall detection loop in a worker thread and emits events
    back to the orchestrator via a callback.
    """

    def __init__(
        self,
        status_callback: EventHandler,
        camera: Optional[CameraStream] = None,
        pose: Optional[PoseEstimator] = None,
        detector: Optional[FallDetector] = None,
        visualizer: Optional[VisualizerFn] = None,
        target_width: Optional[int] = None,
        frame_skip: int = 1,
        viz=None,
        frame_subscription: Optional[Any] = None,
    ) -> None:
        super().__init__("pose_detection")

        # Core components - create defaults if not provided
        self.camera = camera if camera is not None else CameraStream()
        self.pose = pose if pose is not None else PoseEstimator()
        self.detector = detector if detector is not None else FallDetector()
        self.status_callback = status_callback  # The function to call back to LeKiwi
        self.visualizer = visualizer
        self.ring_buffer = FrameRingBuffer(max_seconds=10.0, maxlen=300)
        self._prev_gray = None
        self.viz = viz if viz is not None else NullViz()
        # Optional external frame source (e.g., CameraHub subscription)
        self.frame_subscription = frame_subscription
        self._use_external_frames = frame_subscription is not None

        # State and configuration
        self.prev_is_fall_state = False  # Replaces self.prev_state
        self.target_width = target_width
        self.frame_skip = max(0, frame_skip)
        self._fall_freeze_until = 0.0

        # Variables for event loop (initialized in _event_loop)
        self._frame_idx = 0
        self._prev_time = time.time()
        self._last_fall_event: Optional[FallEvent] = None
        self._last_is_fall = False

    def _resize_for_infer(self, frame):
        if not self.target_width:
            return frame
        h, w, _ = frame.shape
        if w <= self.target_width:
            return frame
        scale = self.target_width / float(w)
        new_h = int(h * scale)
        return cv2.resize(
            frame, (self.target_width, new_h), interpolation=cv2.INTER_AREA
        )

    def _resize_for_buffer(self, frame, target_width: int = 320):
        h, w, _ = frame.shape
        if w <= target_width:
            return frame
        scale = target_width / float(w)
        new_h = int(h * scale)
        return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)

    # 1. Override start() to initialize resources
    def start(self):
        """Starts camera and worker thread."""
        if not self._use_external_frames:
            try:
                self.camera.start()
            except RuntimeError as e:
                self.logger.error(f"Failed to start camera: {e}")
                return

        super().start()
        self.logger.info("Pose Detection Service started")

    # 2. Override stop() to clean up resources
    def stop(self, timeout: float = 5.0):
        """Stops worker thread, camera, and pose model."""
        super().stop(timeout)  # Stop the worker thread first
        if not self._use_external_frames:
            self.camera.stop()
        self.pose.close()
        cv2.destroyAllWindows()
        self.logger.info("Pose Detection Service stopped")

    # 3. Implement the continuous detection loop
    def _event_loop(self):
        """Runs the continuous pose detection and event emission."""

        # Initialize internal loop variables upon starting the thread
        self._prev_time = time.time()
        self._frame_idx = 0
        self._last_fall_event: Optional[FallEvent] = None
        self._last_is_fall = False
        self.ring_buffer.clear()
        self._prev_gray = None

        while self._running.is_set():
            # --- Check for Inbound Control Events (using ServiceBase logic) ---
            # We don't want to block, so we check for inbound events with a timeout of 0.
            # We can still receive control events like "change_camera"
            if self._event_available.wait(timeout=0):
                with self._event_lock:
                    if self._current_event:
                        service_event = self._current_event
                    else:
                        continue

                try:
                    self.handle_event(service_event.event_type, service_event.payload)
                except Exception as e:
                    self.logger.error(
                        f"Error handling inbound event {service_event.event_type}: {e}"
                    )
                finally:
                    with self._event_lock:
                        self._current_event = None
                        self._event_available.clear()
            # ------------------------------------------------------------------

            frame_ts = time.time()
            try:
                if self._use_external_frames and self.frame_subscription:
                    pulled: Optional[Tuple[float, np.ndarray]] = (
                        self.frame_subscription.pull(timeout=0.1)
                    )
                    if pulled is None:
                        continue
                    frame_ts, frame = pulled
                else:
                    frame = self.camera.read()
                    frame_ts = time.time()
            except RuntimeError:
                self.logger.error("Camera frame read failed. Stopping service.")
                self.stop()
                break

            # Core detection logic (moved from original run method)
            process_this = self._frame_idx % (self.frame_skip + 1) == 0
            self._frame_idx += 1
            result = None
            event: Optional[FallEvent] = None
            is_fall = self._last_is_fall

            if process_this:
                infer_frame = self._resize_for_infer(frame)
                result = self.pose.infer(infer_frame)
                landmarks = (
                    result.pose_landmarks.landmark
                    if result and result.pose_landmarks
                    else None
                )
                quality = compute_quality_metrics(
                    frame,
                    self._prev_gray,
                    landmarks,
                    downscale_width=self.target_width or 320,
                )
                gray = quality.pop("gray", None)
                self._prev_gray = gray

                if landmarks:
                    event = self.detector.detect(landmarks)

                # Determine raw detection result
                detected_fall = False
                if event:
                    detected_fall = event.is_fall

                # Apply freeze logic
                if time.time() < self._fall_freeze_until:
                    is_fall = True
                else:
                    is_fall = detected_fall
                    if is_fall:
                        self._fall_freeze_until = time.time() + 3.0

                if is_fall != self.prev_is_fall_state:
                    # Prepare event data
                    if event:
                        event_data = {
                            "score": event.score,
                            "ratio": event.ratio,
                            "timestamp": event.timestamp,
                            "quality": quality,
                        }
                    else:
                        # Fallback if frozen but no current event
                        evt = self._last_fall_event
                        event_data = {
                            "score": evt.score if evt else 1.0,
                            "ratio": evt.ratio if evt else 0.0,
                            "timestamp": time.time(),
                            "quality": quality,
                        }

                    event_type = "PERSON_FALLEN" if is_fall else "PERSON_STABLE"
                    self.status_callback(event_type, event_data)

                self.prev_is_fall_state = is_fall
                self._last_fall_event = event or self._last_fall_event
                self._last_is_fall = is_fall

                # Record in ring buffer (downscaled to save memory)
                try:
                    buf_frame = self._resize_for_buffer(frame)
                    self.ring_buffer.add(
                        buf_frame,
                        landmarks,
                        quality,
                        ts=event.timestamp if event else frame_ts,
                    )
                except Exception as e:
                    self.logger.debug(f"Ring buffer add failed: {e}")

            now = time.time()
            fps = 1.0 / max(now - self._prev_time, 1e-6)
            self._prev_time = now

            # Emit pose landmarks to viz (thread-safe via event queue)
            if self.viz:
                try:
                    self.viz.log_front_rgb(frame, frame_ts)
                except Exception as e:
                    self.logger.debug(f"Viz front log failed: {e}")

            if result and result.pose_landmarks and self.viz:
                h, w = frame.shape[:2]
                pts = np.array(
                    [(lm.x * w, lm.y * h) for lm in result.pose_landmarks.landmark],
                    dtype=np.float32,
                )
                vis = np.array(
                    [
                        getattr(lm, "visibility", 0.0)
                        for lm in result.pose_landmarks.landmark
                    ],
                    dtype=np.float32,
                )
                edges = list(mp.solutions.pose.POSE_CONNECTIONS)
                try:
                    self.viz.log_pose(pts, edges=edges, vis=vis, ts=frame_ts)
                except Exception as e:
                    self.logger.debug(f"Viz pose log failed: {e}")

            # Visualization (runs on every frame read, regardless of frame_skip)
            event_for_vis = event or self._last_fall_event
            is_fall_for_vis = is_fall
            if self.visualizer:
                # The visualizer should stop the service if 'q' is pressed
                should_stop = self.visualizer(
                    frame, result, event_for_vis, is_fall_for_vis, fps
                )
                if should_stop:
                    self._stop_event.set()
            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._stop_event.set()

            # Time to check the stop event and ensure FPS timing (if no visualizer)
            if self._stop_event.is_set():
                break

            # Since the loop runs as fast as the camera read/detection allows,
            # we rely on frame_skip and the continuous camera read to manage speed.
            # If you want a max FPS *rate*, you'd add a sleep here, but for detection,
            # continuous reading is usually better.

    # 4. Implement the required abstract method for inbound events
    def handle_event(self, event_type: str, payload: Any):
        """Handle control events dispatched from the orchestrator."""
        if event_type == "change_camera":
            # Example: Handle a request to change the camera index
            self.logger.info(f"Received request to change camera index to {payload}")
            # Implementation here would involve calling stop() and then start()
            # with the new configuration, or more complex resource swapping.
            pass
        elif event_type == "set_visualizer":
            self.visualizer = payload
            self.logger.info("Visualizer updated.")
        else:
            self.logger.warning(f"Unknown control event type: {event_type}")


def default_visualizer(
    frame, result, event: Optional[FallEvent], is_fall: bool, fps: float
) -> bool:
    label = "FALL" if is_fall else "OK"
    color = (0, 0, 255) if is_fall else (0, 200, 0)
    ratio_txt = f"{event.ratio:.2f}" if event else "--"

    if result and result.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            result.pose_landmarks,
            mp.solutions.pose.POSE_CONNECTIONS,
        )

    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(
        frame,
        f"torso ratio {ratio_txt}",
        (10, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"FPS {fps:.1f}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    try:
        cv2.imshow("Fall Detection (MediaPipe Pose)", frame)
        return bool(cv2.waitKey(1) & 0xFF == ord("q"))
    except cv2.error as e:
        # OpenCV GUI not available (headless mode, worker thread on macOS, etc.)
        # We warn once so users don't think it's "just stuck".
        global _WARNED_OPENCV_GUI
        if not _WARNED_OPENCV_GUI:
            _WARNED_OPENCV_GUI = True
            print(
                "[pose_service] OpenCV window could not be opened. "
                "This commonly happens on macOS when cv2.imshow() is called from a background thread "
                "(PoseDetectionService runs in a worker thread). "
                "To verify the camera feed, run `uv run scripts/test_pose_detection.py` "
                "and set POSE_CAMERA_INDEX=...; then use the same POSE_CAMERA_INDEX for main_workflows."
            )
        return False
