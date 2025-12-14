#!/usr/bin/env python3
"""
List /dev/video* devices and (optionally) preview them to identify which camera index is which.

Examples:
  python scripts/list_cameras.py
  python scripts/list_cameras.py --preview
  python scripts/list_cameras.py --preview --seconds 2
  python scripts/list_cameras.py --snapshot-dir outputs/captured_images
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VideoDevice:
    index: int
    devnode: Path
    name: str = ""
    sysfs_path: Optional[Path] = None
    usb_id: str = ""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def _guess_usb_id(sysfs_video: Path) -> str:
    """
    Try to find a stable-ish identifier (vendor/product/serial) by walking up sysfs.
    Works for many USB UVC cameras on Linux; best-effort only.
    """
    try:
        # /sys/class/video4linux/videoN/device -> .../usbX/.../video4linux/videoN
        dev = (sysfs_video / "device").resolve()
    except Exception:
        return ""

    cur = dev
    for _ in range(12):
        id_vendor = cur / "idVendor"
        id_product = cur / "idProduct"
        serial = cur / "serial"
        if id_vendor.exists() and id_product.exists():
            v = _read_text(id_vendor)
            p = _read_text(id_product)
            s = _read_text(serial)
            if s:
                return f"usb:{v}:{p}:{s}"
            return f"usb:{v}:{p}"
        if cur.parent == cur:
            break
        cur = cur.parent
    return ""


def list_video_devices() -> list[VideoDevice]:
    devs: list[VideoDevice] = []
    by_dev = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
    for devnode in by_dev:
        # Skip non-numeric nodes like /dev/videobuf?
        suffix = devnode.name.replace("video", "", 1)
        if not suffix.isdigit():
            continue
        idx = int(suffix)
        sysfs_video = Path("/sys/class/video4linux") / f"video{idx}"
        name = _read_text(sysfs_video / "name") if sysfs_video.exists() else ""
        usb_id = _guess_usb_id(sysfs_video) if sysfs_video.exists() else ""
        devs.append(
            VideoDevice(index=idx, devnode=devnode, name=name, sysfs_path=sysfs_video if sysfs_video.exists() else None, usb_id=usb_id)
        )
    return devs


def _open_capture(idx: int):
    try:
        import cv2  # lazy import
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "OpenCV (cv2) is not available in this Python environment.\n"
            f"You're running: {sys.executable}\n"
            "Tip: use your project venv interpreter, e.g.:\n"
            "  .venv/bin/python scripts/list_cameras.py --preview\n"
        ) from e

    # Prefer V4L2 backend on Linux to avoid weird auto-backend behavior.
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    return cap, cv2


def _fourcc_to_str(code_float: float) -> str:
    try:
        code_int = int(code_float)
        return "".join([chr((code_int >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        return ""


def _try_read_one_frame(cap, cv2, warmup_seconds: float = 0.25):
    t0 = time.time()
    last = None
    while time.time() - t0 < warmup_seconds:
        ok, frame = cap.read()
        if ok and frame is not None:
            last = frame
    if last is not None:
        return True, last
    ok, frame = cap.read()
    return ok, frame


def probe_device(
    idx: int,
    *,
    width: int,
    height: int,
    fps: float,
    fourccs: list[str | None],
    measure_seconds: float,
) -> None:
    cap, cv2 = _open_capture(idx)
    if not cap.isOpened():
        print(f"  probe index={idx}: FAILED to open")
        return

    for fourcc in fourccs:
        # New capture per FOURCC tends to be more reliable.
        cap.release()
        cap, cv2 = _open_capture(idx)
        if not cap.isOpened():
            print(f"  probe index={idx} fourcc={fourcc}: FAILED to open")
            continue

        if fourcc is not None:
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            except Exception:
                pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        actual_fourcc = _fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC) or 0.0)

        measured = 0.0
        if measure_seconds > 0:
            t0 = time.time()
            frames = 0
            # Warm up a bit
            _try_read_one_frame(cap, cv2, warmup_seconds=0.2)
            while True:
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames += 1
                if time.time() - t0 >= measure_seconds:
                    break
            dt = time.time() - t0
            measured = (frames / dt) if dt > 0 else 0.0

        label = fourcc if fourcc is not None else "AUTO"
        print(
            f"  probe index={idx:>2} fourcc={label:<4} -> "
            f"{actual_w}x{actual_h} reported_fps={actual_fps:.1f} actual_fourcc={actual_fourcc!r}"
            + (f" measured_fps~{measured:.1f}" if measure_seconds > 0 else "")
        )

    cap.release()


def preview_devices(devs: list[VideoDevice], seconds: float, snapshot_dir: Optional[Path]) -> int:
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError:
        print(
            "OpenCV (cv2) is not available in this Python environment.\n"
            f"You're running: {sys.executable}\n"
            "Tip: use your project venv interpreter, e.g.:\n"
            "  .venv/bin/python scripts/list_cameras.py --preview\n"
        )
        return 2

    if not devs:
        print("No /dev/video* devices found.")
        return 2

    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    print("Preview mode: press 'q' to quit, any key to skip to next camera.")
    for d in devs:
        cap, cv2 = _open_capture(d.index)
        if not cap.isOpened():
            print(f"[{d.index}] {d.devnode}  (FAILED to open)  name='{d.name}' usb='{d.usb_id}'")
            continue

        # Best-effort readback. (Some drivers lie about FPS; that's fine.)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        print(f"[{d.index}] {d.devnode}  name='{d.name}' usb='{d.usb_id}'  ({w}x{h} @ reported_fps={fps:.1f})")

        start = time.time()
        last_frame = None
        while time.time() - start < seconds:
            ok, frame = cap.read()
            if not ok or frame is None:
                ok, frame = _try_read_one_frame(cap, cv2, warmup_seconds=0.15)
                if not ok or frame is None:
                    continue
            last_frame = frame
            overlay = frame.copy()
            label = f"index={d.index}  {d.devnode.name}  {d.name}".strip()
            cv2.putText(
                overlay,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("camera preview (q quits, any key next)", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return 0
            if key != 255:
                break

        if snapshot_dir is not None and last_frame is not None:
            out = snapshot_dir / f"camera_index_{d.index}.png"
            try:
                cv2.imwrite(str(out), last_frame)
                print(f"  saved snapshot -> {out}")
            except Exception as e:
                print(f"  failed to save snapshot: {e}")

        cap.release()
        cv2.destroyAllWindows()

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="List and preview Linux V4L2 cameras (/dev/video*).")
    ap.add_argument("--preview", action="store_true", help="Show a short live preview for each camera index.")
    ap.add_argument("--seconds", type=float, default=2.0, help="Seconds to preview each camera (default: 2).")
    ap.add_argument(
        "--probe",
        action="store_true",
        help="Probe whether each index can do the requested width/height/fps under different FOURCC formats.",
    )
    ap.add_argument("--probe-width", type=int, default=1280, help="Probe width (default: 1280).")
    ap.add_argument("--probe-height", type=int, default=720, help="Probe height (default: 720).")
    ap.add_argument("--probe-fps", type=float, default=30.0, help="Probe fps request (default: 30).")
    ap.add_argument(
        "--probe-fourccs",
        type=str,
        default="MJPG,YUYV",
        help="Comma-separated FOURCCs to try (default: MJPG,YUYV). Use empty to try only AUTO.",
    )
    ap.add_argument(
        "--measure-seconds",
        type=float,
        default=1.0,
        help="If >0, roughly measure fps by reading frames for N seconds (default: 1.0).",
    )
    ap.add_argument(
        "--snapshot-dir",
        type=str,
        default="",
        help="If set, save one snapshot per camera to this directory (e.g. outputs/captured_images).",
    )
    args = ap.parse_args()

    devs = list_video_devices()
    if not devs:
        print("No /dev/video* devices found.")
        return 2

    print("Detected cameras:")
    for d in devs:
        extra = []
        if d.name:
            extra.append(f"name='{d.name}'")
        if d.usb_id:
            extra.append(f"id='{d.usb_id}'")
        print(f"  - index={d.index:>2}  dev={d.devnode}  " + ("  ".join(extra) if extra else ""))

    if args.probe:
        raw = str(args.probe_fourccs).strip()
        fourccs: list[str | None] = []
        if raw:
            for tok in raw.split(","):
                tok = tok.strip()
                if tok:
                    fourccs.append(tok)
        if not fourccs:
            fourccs = [None]
        # Always include AUTO first to see baseline
        if None not in fourccs:
            fourccs = [None] + fourccs
        print(
            f"\nProbing: {args.probe_width}x{args.probe_height} @ {args.probe_fps}fps, fourccs="
            + ",".join([f if f is not None else "AUTO" for f in fourccs])
        )
        for d in devs:
            try:
                probe_device(
                    d.index,
                    width=int(args.probe_width),
                    height=int(args.probe_height),
                    fps=float(args.probe_fps),
                    fourccs=fourccs,
                    measure_seconds=max(0.0, float(args.measure_seconds)),
                )
            except RuntimeError as e:
                print(str(e).rstrip())
                return 2

    if args.preview:
        snap = Path(args.snapshot_dir) if args.snapshot_dir else None
        return preview_devices(devs, seconds=max(0.25, float(args.seconds)), snapshot_dir=snap)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

