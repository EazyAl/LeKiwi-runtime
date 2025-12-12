from __future__ import annotations

import collections
import dataclasses
import time
from typing import Deque, List, Optional, Any


@dataclasses.dataclass
class FrameRecord:
    timestamp: float
    frame_small: Any  # typically a downscaled numpy array
    landmarks: Optional[Any]  # mediapipe landmarks or None
    metrics: dict


class FrameRingBuffer:
    """
    Time-based ring buffer to retain the last N seconds of frames/landmarks/metrics.
    Stores downscaled frames to avoid memory blowups.
    """

    def __init__(self, max_seconds: float = 10.0, maxlen: int = 300):
        self.max_seconds = max_seconds
        self._records: Deque[FrameRecord] = collections.deque(maxlen=maxlen)

    def add(
        self, frame_small: Any, landmarks: Optional[Any], metrics: dict, ts: Optional[float] = None
    ) -> None:
        now = ts or time.time()
        self._records.append(FrameRecord(now, frame_small, landmarks, metrics))
        self._trim(now)

    def _trim(self, now: float) -> None:
        cutoff = now - self.max_seconds
        while self._records and self._records[0].timestamp < cutoff:
            self._records.popleft()

    def get_recent(self, seconds: float) -> List[FrameRecord]:
        now = time.time()
        cutoff = now - seconds
        return [r for r in self._records if r.timestamp >= cutoff]

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()
