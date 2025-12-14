"""
CameraHub: single-owner for cv2 cameras with fan-out subscriptions.

Each camera index is opened once, frames are duplicated to subscribers
using small bounded queues (latest frame wins). This avoids contention
and keeps timestamps consistent across consumers (viz, pose, etc.).
"""

from __future__ import annotations

import threading
import queue
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import os
from pathlib import Path


@dataclass
class FramePacket:
    ts: float
    frame: np.ndarray


class CameraSubscription:
    def __init__(self, q: "queue.Queue[FramePacket]") -> None:
        self._q = q

    def pull(self, timeout: float = 0.1) -> Optional[Tuple[float, np.ndarray]]:
        """Return (ts, frame) or None if no frame is available within timeout."""
        try:
            pkt = self._q.get(timeout=timeout)
            return pkt.ts, pkt.frame
        except queue.Empty:
            return None


class CameraWorker:
    def __init__(self, name: str, index: int, fps: int = 30) -> None:
        self.name = name
        self.index = index
        self.fps = fps
        self.subscribers: list["queue.Queue[FramePacket]"] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{name}", daemon=True
        )

        # Best-effort robust open on Linux where OpenCV may pick an unexpected backend
        # (e.g. OBSensor) when using integer indices.
        self._cap = self._open_capture(index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {name} (index {index})")

        # Optional tuning knobs via env (keeps call sites stable).
        # These match main_sentry's env knobs.
        fourcc = os.getenv("LEKIWI_CAMERA_FOURCC", "").strip()
        if len(fourcc) == 4:
            try:
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            except Exception:
                pass
        try:
            self._cap.set(cv2.CAP_PROP_FPS, fps)
        except Exception:
            pass
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Optional size override (useful to keep CPU down).
        w = os.getenv("LEKIWI_CAMERA_WIDTH", "").strip()
        h = os.getenv("LEKIWI_CAMERA_HEIGHT", "").strip()
        if w.isdigit() and h.isdigit():
            try:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(int(w)))
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(int(h)))
            except Exception:
                pass

    @staticmethod
    def _open_capture(index: int) -> cv2.VideoCapture:
        """
        Try opening /dev/videoN first (more stable), then fall back to index.
        Prefer V4L2 backend on Linux when possible.
        """
        dev_path = Path("/dev") / f"video{index}"

        # 1) Try device path without forcing backend (works on more OpenCV builds)
        if dev_path.exists():
            cap = cv2.VideoCapture(str(dev_path))
            if cap is not None and cap.isOpened():
                return cap
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

            # 2) Try device path with V4L2 backend
            cap = cv2.VideoCapture(str(dev_path), cv2.CAP_V4L2)
            if cap is not None and cap.isOpened():
                return cap
            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

        # 3) Try integer index with V4L2 backend
        cap = cv2.VideoCapture(int(index), cv2.CAP_V4L2)
        if cap is not None and cap.isOpened():
            return cap
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass

        # 4) Last resort: integer index default backend
        return cv2.VideoCapture(int(index))

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._cap.release()

    def subscribe(self, max_queue: int = 1) -> CameraSubscription:
        q: "queue.Queue[FramePacket]" = queue.Queue(maxsize=max_queue)
        self.subscribers.append(q)
        return CameraSubscription(q)

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            ts = time.time()
            for sub in list(self.subscribers):
                try:
                    pkt = FramePacket(ts=ts, frame=frame.copy())
                    sub.put_nowait(pkt)
                except queue.Full:
                    # Drop the oldest to keep latest
                    try:
                        sub.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        sub.put_nowait(pkt)
                    except queue.Full:
                        pass


class CameraHub:
    def __init__(
        self, front_index: int = 0, wrist_index: int = 2, top_index: int = 4, fps: int = 30
    ) -> None:
        self.front_worker = CameraWorker("front", front_index, fps=fps)
        self.wrist_worker = CameraWorker("wrist", wrist_index, fps=fps)
        self.top_worker = CameraWorker("top", top_index, fps=fps)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.front_worker.start()
        self.wrist_worker.start()
        self.top_worker.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self.front_worker.stop()
        self.wrist_worker.stop()
        self.top_worker.stop()
        self._started = False

    def subscribe_front(self, max_queue: int = 1) -> CameraSubscription:
        return self.front_worker.subscribe(max_queue=max_queue)

    def subscribe_wrist(self, max_queue: int = 1) -> CameraSubscription:
        return self.wrist_worker.subscribe(max_queue=max_queue)

    def subscribe_top(self, max_queue: int = 1) -> CameraSubscription:
        return self.top_worker.subscribe(max_queue=max_queue)
