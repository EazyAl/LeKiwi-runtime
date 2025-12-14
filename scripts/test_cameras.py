#!/usr/bin/env python3
"""
Capture one frame from each connected V4L2 camera and print an ID for each.

Writes PNG snapshots into outputs/captured_images/ by default.

Examples:
  .venv/bin/python scripts/test_cameras.py
  .venv/bin/python scripts/test_cameras.py --out-dir outputs/captured_images/cam_check
  .venv/bin/python scripts/test_cameras.py --width 1280 --height 720 --fps 30 --fourccs MJPG,YUYV
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VideoDevice:
    index: int
    devnode: Path
    name: str
    usb_id: str


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception:
        return ""


def _guess_usb_id(video_sysfs: Path) -> str:
    """Best-effort USB identifier (vendor/product[/serial]) by walking up sysfs."""
    try:
        dev = (video_sysfs / "device").resolve()
    except Exception:
        return ""

    cur = dev
    for _ in range(14):
        id_vendor = cur / "idVendor"
        id_product = cur / "idProduct"
        serial = cur / "serial"
        if id_vendor.exists() and id_product.exists():
            v = _read_text(id_vendor)
            p = _read_text(id_product)
            s = _read_text(serial)
            return f"usb:{v}:{p}:{s}" if s else f"usb:{v}:{p}"
        if cur.parent == cur:
            break
        cur = cur.parent
    return ""


def list_video_devices() -> list[VideoDevice]:
    devs: list[VideoDevice] = []
    for devnode in sorted(Path("/dev").glob("video*"), key=lambda p: p.name):
        suffix = devnode.name.replace("video", "", 1)
        if not suffix.isdigit():
            continue
        idx = int(suffix)
        sysfs = Path("/sys/class/video4linux") / f"video{idx}"
        name = _read_text(sysfs / "name") if sysfs.exists() else ""
        usb_id = _guess_usb_id(sysfs) if sysfs.exists() else ""
        devs.append(VideoDevice(index=idx, devnode=devnode, name=name, usb_id=usb_id))
    return devs


def _import_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "OpenCV (cv2) is not installed in this Python environment.\n"
            f"You're running: {sys.executable}\n"
            "Tip: use your project venv interpreter, e.g.:\n"
            "  .venv/bin/python scripts/test_cameras.py\n"
        ) from e
    return cv2


def _fourcc_to_str(code_float: float) -> str:
    try:
        code_int = int(code_float)
        return "".join([chr((code_int >> 8 * i) & 0xFF) for i in range(4)])
    except Exception:
        return ""


def _sanitize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "", s)
    return s[:80] if s else "unknown"


def _open_capture(cv2, dev: VideoDevice):
    """
    Open by device path first (more reliable than index on some systems),
    fall back to integer index.
    """
    # On some OpenCV builds, passing CAP_V4L2 with a device path prints warnings and fails.
    # Try without forcing backend for path-based open.
    cap = cv2.VideoCapture(str(dev.devnode))
    if cap is not None and cap.isOpened():
        return cap, "path"
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
    cap = cv2.VideoCapture(int(dev.index), cv2.CAP_V4L2)
    return cap, "index"


def capture_one(
    dev: VideoDevice,
    *,
    out_dir: Path,
    width: int,
    height: int,
    fps: float,
    fourccs: list[str | None],
    warmup_s: float,
) -> tuple[bool, str]:
    """
    Returns (success, message_or_path).
    On success, message_or_path is the saved PNG path.
    """
    cv2 = _import_cv2()

    cap, opened_as = _open_capture(cv2, dev)
    if cap is None or not cap.isOpened():
        return False, "failed to open (busy? permissions? not a capture node?)"

    best_err: Optional[str] = None
    last_report = ""

    for fourcc in fourccs:
        # Re-open for each FOURCC to reduce driver weirdness.
        try:
            cap.release()
        except Exception:
            pass
        cap, opened_as = _open_capture(cv2, dev)
        if cap is None or not cap.isOpened():
            best_err = "failed to open"
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
        last_report = f"{actual_w}x{actual_h} reported_fps={actual_fps:.1f} fourcc={actual_fourcc!r} opened_as={opened_as}"

        # Warmup + snapshot read loop (helps with slow-starting UVC devices).
        frame = None
        deadline = time.time() + max(0.5, float(warmup_s))
        while time.time() < deadline:
            ok, fr = cap.read()
            if ok and fr is not None:
                frame = fr

        # If still no frame, keep trying a bit longer (common when device is initializing).
        if frame is None:
            deadline = time.time() + 1.5
            while time.time() < deadline and frame is None:
                ok, fr = cap.read()
                if ok and fr is not None:
                    frame = fr

        if frame is None:
            best_err = f"no frames ({last_report}) — device may be busy/in-use"
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        name = _sanitize(dev.name)
        uid = _sanitize(dev.usb_id) if dev.usb_id else "noid"
        four = _sanitize(actual_fourcc) if actual_fourcc else "AUTO"
        out_path = out_dir / f"camera_index_{dev.index}_{name}_{uid}_{four}.png"
        try:
            cv2.imwrite(str(out_path), frame)
        except Exception as e:
            best_err = f"failed to write png: {e}"
            continue
        finally:
            try:
                cap.release()
            except Exception:
                pass

        return True, str(out_path)

    try:
        cap.release()
    except Exception:
        pass
    return False, best_err or f"unknown failure ({last_report})"


def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot all connected /dev/video* cameras.")
    ap.add_argument(
        "--out-dir",
        type=str,
        default="",
        help="Output directory for PNGs (default: outputs/captured_images/test_cameras_<timestamp>).",
    )
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--fourccs",
        type=str,
        default="MJPG,YUYV",
        help="Comma-separated FOURCCs to try (default: MJPG,YUYV). AUTO is always tried first.",
    )
    ap.add_argument("--warmup-s", type=float, default=0.3, help="Warmup read time before snapshot (default: 0.3s).")
    args = ap.parse_args()

    devs = list_video_devices()
    if not devs:
        print("No /dev/video* devices found.")
        return 2

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = Path("outputs") / "captured_images" / f"test_cameras_{ts}"

    raw = str(args.fourccs).strip()
    # Prefer MJPG first (commonly required for 30fps at 720p on UVC),
    # keep AUTO as a last resort.
    fourccs: list[str | None] = []
    if raw:
        for tok in raw.split(","):
            tok = tok.strip()
            if tok:
                if len(tok) != 4:
                    print(f"warning: ignoring FOURCC {tok!r} (must be 4 chars)")
                    continue
                if tok not in fourccs:
                    fourccs.append(tok)
    # Always include AUTO last
    fourccs.append(None)

    print(f"Request: {args.width}x{args.height} @ {args.fps}fps, fourccs=" + ",".join([f or "AUTO" for f in fourccs]))
    print(f"Output:  {out_dir}")
    print("Note: if many devices FAIL to open, stop anything currently using cameras (e.g. main_sentry).")
    print("")

    ok_count = 0
    for d in devs:
        label = f"index={d.index:>2} dev={d.devnode} name={d.name!r} id={d.usb_id!r}"
        success, msg = capture_one(
            d,
            out_dir=out_dir,
            width=int(args.width),
            height=int(args.height),
            fps=float(args.fps),
            fourccs=fourccs,
            warmup_s=float(args.warmup_s),
        )
        if success:
            ok_count += 1
            print(f"[OK]   {label}\n       -> {msg}")
        else:
            print(f"[FAIL] {label}\n       -> {msg}")

    print("")
    print(f"Done. {ok_count}/{len(devs)} devices produced a snapshot.")
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

