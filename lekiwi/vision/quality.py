from __future__ import annotations

import cv2
import numpy as np
from typing import Any, Dict, Optional


def _to_gray_small(frame, width: int = 320):
    h, w = frame.shape[:2]
    if w > width:
        scale = width / float(w)
        frame = cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray


def compute_quality_metrics(
    frame_bgr: np.ndarray,
    prev_gray: Optional[np.ndarray],
    landmarks: Optional[Any],
    downscale_width: int = 320,
) -> Dict[str, float]:
    """
    Lightweight quality metrics:
    - brightness: mean pixel value (0-255)
    - blur: variance of Laplacian (higher = sharper)
    - motion: mean absolute difference vs previous gray frame
    - visibility_min / visibility_mean: derived from pose landmarks if available
    """
    gray = _to_gray_small(frame_bgr, width=downscale_width)
    brightness = float(np.mean(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    motion = 0.0
    if prev_gray is not None and prev_gray.shape == gray.shape:
        motion = float(np.mean(cv2.absdiff(gray, prev_gray)))

    visibility_min = -1.0
    visibility_mean = -1.0
    if landmarks:
        vis = [lm.visibility for lm in landmarks]
        if vis:
            visibility_min = float(min(vis))
            visibility_mean = float(sum(vis) / len(vis))

    return {
        "brightness": brightness,
        "blur": blur,
        "motion": motion,
        "visibility_min": visibility_min,
        "visibility_mean": visibility_mean,
        "gray": gray,  # returned for caller to reuse as prev_gray (not serialized)
    }
