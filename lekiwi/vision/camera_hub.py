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
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {name} (index {index})")

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
        self, front_index: int = 0, wrist_index: int = 2, fps: int = 30
    ) -> None:
        self.front_worker = CameraWorker("front", front_index, fps=fps)
        self.wrist_worker = CameraWorker("wrist", wrist_index, fps=fps)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self.front_worker.start()
        self.wrist_worker.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self.front_worker.stop()
        self.wrist_worker.stop()
        self._started = False

    def subscribe_front(self, max_queue: int = 1) -> CameraSubscription:
        return self.front_worker.subscribe(max_queue=max_queue)

    def subscribe_wrist(self, max_queue: int = 1) -> CameraSubscription:
        return self.wrist_worker.subscribe(max_queue=max_queue)
