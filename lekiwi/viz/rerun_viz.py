"""
Thread-safe Rerun visualization helper.

Producers (any thread) enqueue events; a dedicated viz thread is the only
place that talks to the rerun SDK. A NullViz fallback keeps call-sites simple
when visualization is disabled.
"""

from __future__ import annotations

import threading
import queue
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb

    _RR_AVAILABLE = True
except Exception:
    _RR_AVAILABLE = False

# Optional: cv2 only for depth colorization; defer import.
try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


def _now_ts() -> float:
    return time.time()


@dataclass
class VizEvent:
    kind: str
    payload: Any
    ts: float


class NullViz:
    """No-op stand-in when visualization is disabled or rerun is unavailable."""

    enabled: bool = False

    def log_front_rgb(self, *_args, **_kwargs) -> None:
        return

    def log_wrist_rgb(self, *_args, **_kwargs) -> None:
        return

    def log_pose(self, *_args, **_kwargs) -> None:
        return

    def log_depth(self, *_args, **_kwargs) -> None:
        return

    def set_status(self, *_args, **_kwargs) -> None:
        return

    def log_tool_call(self, *_args, **_kwargs) -> None:
        return

    def close(self) -> None:
        return


class RerunViz:
    """
    Thread-safe Rerun logger with a fixed blueprint/layout.

    Usage:
        viz = RerunViz()
        viz.log_front_rgb(frame_bgr, ts)
        viz.log_pose(pts, edges, vis, ts)
    """

    def __init__(self, app_id: str = "lekiwi_viz") -> None:
        if not _RR_AVAILABLE:
            raise RuntimeError("rerun SDK not available; install rerun-sdk")

        self.app_id = app_id
        self.enabled: bool = True
        self._queue: "queue.Queue[VizEvent]" = queue.Queue(maxsize=512)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rerun-viz", daemon=True)
        self._init_viewer()
        self._thread.start()

    # --- Public logging helpers -------------------------------------------------
    def log_front_rgb(self, frame_bgr: np.ndarray, ts: Optional[float] = None) -> None:
        self._enqueue("front_rgb", frame_bgr, ts)

    def log_wrist_rgb(self, frame_bgr: np.ndarray, ts: Optional[float] = None) -> None:
        self._enqueue("wrist_rgb", frame_bgr, ts)

    def log_pose(
        self,
        pts: np.ndarray,
        edges: Iterable[tuple[int, int]],
        vis: Optional[np.ndarray],
        ts: Optional[float] = None,
    ) -> None:
        payload = {"pts": pts, "edges": list(edges), "vis": vis}
        self._enqueue("pose", payload, ts)

    def log_depth(self, depth_img: np.ndarray, ts: Optional[float] = None) -> None:
        """Accepts either a depth map (float/uint16) or a colorized BGR/RGB image."""
        self._enqueue("depth", depth_img, ts)

    def set_status(self, status: str, ts: Optional[float] = None) -> None:
        self._enqueue("status", status, ts)

    def log_tool_call(
        self,
        tool_name: str,
        message: str,
        level: str = "info",
        emoji: str = "🛠️",
        ts: Optional[float] = None,
    ) -> None:
        payload = {
            "tool": tool_name,
            "message": message,
            "level": level,
            "emoji": emoji,
        }
        self._enqueue("tool_log", payload, ts)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.5)

    # --- Internal ---------------------------------------------------------------
    def _enqueue(self, kind: str, payload: Any, ts: Optional[float]) -> None:
        event = VizEvent(
            kind=kind, payload=payload, ts=ts if ts is not None else _now_ts()
        )
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop oldest to make room for new events
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(event)

    def _init_viewer(self) -> None:
        rr.init(self.app_id)
        rr.spawn(memory_limit="25%")
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial2DView(origin="/ui/status", name="Status"),
                    rrb.TextLogView(origin="/ui/logs", name="Logs"),
                    row_shares=[1, 3],
                ),
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.Spatial2DView(origin="/cameras/front", name="Front + pose"),
                        rrb.Spatial2DView(
                            origin="/cameras/front_depth", name="Front depth"
                        ),
                        column_shares=[1, 1],
                    ),
                    rrb.Spatial2DView(origin="/cameras/wrist", name="Wrist"),
                    row_shares=[1, 1],
                ),
                column_shares=[1, 3],
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event: VizEvent = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue

            try:
                rr.set_time_seconds("time", event.ts)
                self._handle_event(event)
            except Exception:
                # Swallow viz errors to avoid crashing the producer threads
                continue

    def _handle_event(self, event: VizEvent) -> None:
        kind = event.kind
        payload = event.payload

        if kind == "front_rgb":
            rr.log("cameras/front/rgb", rr.Image(_to_rgb(payload)))
        elif kind == "wrist_rgb":
            rr.log("cameras/wrist/rgb", rr.Image(_to_rgb(payload)))
        elif kind == "pose":
            self._log_pose_payload(payload)
        elif kind == "depth":
            self._log_depth_payload(payload)
        elif kind == "status":
            self._log_status(payload)
        elif kind == "tool_log":
            self._log_tool(payload)

    def _log_pose_payload(self, payload: dict) -> None:
        pts: np.ndarray = payload["pts"]
        vis: Optional[np.ndarray] = payload.get("vis")
        edges: list[tuple[int, int]] = payload.get("edges", [])

        if pts is None or pts.size == 0:
            return

        colors = None
        if vis is not None and vis.shape[0] == pts.shape[0]:
            colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
            good = vis >= 0.5
            colors[good] = np.array([0, 255, 0], dtype=np.uint8)
            colors[~good] = np.array([255, 0, 0], dtype=np.uint8)

        rr.log(
            "cameras/front/pose/landmarks", rr.Points2D(pts, colors=colors, radii=3.0)
        )

        if edges:
            segments: list[np.ndarray] = []
            for a, b in edges:
                if a < len(pts) and b < len(pts):
                    segments.append(np.array([pts[a], pts[b]], dtype=np.float32))
            if segments:
                rr.log(
                    "cameras/front/pose/skeleton",
                    rr.LineStrips2D(segments, colors=[0, 200, 255], radii=1.5),
                )

    def _log_depth_payload(self, depth_img: np.ndarray) -> None:
        img = depth_img
        if depth_img is None:
            return

        # If already 3-channel, just log it.
        if depth_img.ndim == 3 and depth_img.shape[2] in (3, 4):
            img = _to_rgb(depth_img)
        else:
            if cv2 is None:
                return
            depth_norm = cv2.normalize(
                depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
            )
            img = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rr.log("cameras/front_depth/image", rr.Image(img))

    def _log_status(self, status: str) -> None:
        status_lower = (status or "").lower()
        color = {
            "normal": (0, 255, 0),
            "concerned": (255, 255, 0),
            "emergency": (255, 0, 0),
        }.get(status_lower, (180, 180, 180))

        # Bigger swatch so it's visible in the panel
        swatch = np.zeros((60, 120, 3), dtype=np.uint8)
        swatch[:, :] = np.array(color, dtype=np.uint8)
        rr.log("ui/status/image", rr.Image(swatch))

        # Text with level-based coloring in the TextLog view
        level = (
            rr.TextLogLevel.INFO
            if status_lower == "normal"
            else (
                rr.TextLogLevel.WARN
                if status_lower == "concerned"
                else rr.TextLogLevel.ERROR
            )
        )
        rr.log(
            "ui/status/text", rr.TextLog(f"Status: {status.capitalize()}", level=level)
        )

    def _log_tool(self, payload: dict) -> None:
        level = payload.get("level", "info").lower()
        emoji = payload.get("emoji", "🛠️")
        tool = payload.get("tool", "tool")
        message = payload.get("message", "")

        text = f"{emoji} {tool}: {message}"
        rr.log("ui/logs", rr.TextLog(text, level=_to_rr_level(level)))


def _to_rr_level(level: str):
    level_lower = (level or "").lower()
    if level_lower.startswith("error") or level_lower == "err":
        return rr.TextLogLevel.ERROR
    if level_lower.startswith("warn"):
        return rr.TextLogLevel.WARN
    return rr.TextLogLevel.INFO


def _to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr is None:
        return frame_bgr
    if frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
        return frame_bgr[:, :, ::-1]  # BGR -> RGB
    return frame_bgr


def create_viz(enable: bool, app_id: str = "lekiwi_viz"):
    """
    Helper to create a viz instance respecting a boolean flag and rerun availability.

    Returns:
        RerunViz if enabled and rerun is installed, otherwise NullViz.
    """
    if not enable:
        return NullViz()
    if not _RR_AVAILABLE:
        print("[viz] rerun-sdk not available; running without visualization.")
        return NullViz()
    try:
        return RerunViz(app_id=app_id)
    except Exception as e:  # pragma: no cover - defensive fallback
        print(f"[viz] Failed to start rerun: {e}. Continuing without visualization.")
        return NullViz()
