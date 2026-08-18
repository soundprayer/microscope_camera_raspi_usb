#!/usr/bin/env python3
"""Fullscreen USB camera preview for Raspberry Pi 4B.

The camera frame is always scaled to the real display size, so the
window fills HDMI regardless of camera resolution or monitor.

Keys:
  q / Esc  quit
  s        save snapshot (original camera frame)
  f        toggle fullscreen
  r        rotate 90 degrees clockwise
  c        cycle fit: cover / contain / stretch
  m        next camera
  [ / ]    lower / raise capture resolution
  h / ?    help
  l        camera list (secret)
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ.setdefault("GDK_BACKEND", "x11")
os.environ.setdefault("SDL_VIDEODRIVER", "x11")
os.environ.setdefault("DISPLAY", os.environ.get("DISPLAY", ":0"))

import cv2
import numpy as np

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

SKIP_DEVICE_NAME = re.compile(r"bcm2835|codec|isp|rpivid|pisp|hevc", re.I)

PRESETS = [
    (1920, 1080),
    (1280, 720),
    (1024, 768),
    (800, 600),
    (640, 480),
]
FIT_MODES = ("cover", "contain", "stretch")
HELP_LINES = [
    "q Esc  wyjscie",
    "s      zdjecie",
    "f      pelny ekran",
    "r      obrot 90",
    "c      cover/contain/stretch",
    "m      nastepna kamera",
    "[ ]    rozdzielczosc",
    "l      lista kamer",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fullscreen USB camera preview")
    parser.add_argument(
        "--device",
        default=os.environ.get("CAMERA_DEVICE", ""),
        help="Camera path or index, e.g. /dev/video0 or 0",
    )
    parser.add_argument("--width", type=int, default=int(os.environ.get("CAMERA_WIDTH", "1280")))
    parser.add_argument("--height", type=int, default=int(os.environ.get("CAMERA_HEIGHT", "720")))
    parser.add_argument("--fps", type=int, default=int(os.environ.get("CAMERA_FPS", "30")))
    parser.add_argument("--window", default="Microscope")
    parser.add_argument("--windowed", action="store_true")
    parser.add_argument("--no-mjpeg", action="store_true")
    parser.add_argument(
        "--fit",
        choices=FIT_MODES,
        default=os.environ.get("CAMERA_FIT", "cover"),
        help="cover=fill screen (crop), contain=whole image (bars), stretch=ignore aspect",
    )
    parser.add_argument(
        "--snapshots",
        default=os.path.expanduser("~/Pictures/microscope"),
    )
    return parser.parse_args()


def detect_screen_size() -> tuple[int, int]:
    env = os.environ.copy()
    env.setdefault("DISPLAY", ":0")
    try:
        out = subprocess.check_output(
            ["xrandr", "--current"],
            env=env,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            match = re.search(r"^\s+(\d+)x(\d+)\s+[\d.]+.*\*", line)
            if match:
                return int(match.group(1)), int(match.group(2))
        match = re.search(r" connected(?: primary)? (\d+)x(\d+)\+", out)
        if match:
            return int(match.group(1)), int(match.group(2))
    except (OSError, subprocess.CalledProcessError):
        pass

    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        size = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
        root.destroy()
        if size[0] > 0 and size[1] > 0:
            return size
    except Exception:
        pass

    try:
        text = Path("/sys/class/graphics/fb0/virtual_size").read_text().strip()
        width, height = text.split(",")
        return int(width), int(height)
    except (OSError, ValueError):
        pass

    return 1920, 1080


def fit_to_screen(frame: np.ndarray, screen_w: int, screen_h: int, mode: str) -> np.ndarray:
    """Scale any camera frame onto the full display."""
    frame_h, frame_w = frame.shape[:2]
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    if frame_w < 1 or frame_h < 1 or screen_w < 1 or screen_h < 1:
        return canvas

    if mode == "stretch":
        interpolation = cv2.INTER_AREA if (screen_w < frame_w or screen_h < frame_h) else cv2.INTER_LINEAR
        return cv2.resize(frame, (screen_w, screen_h), interpolation=interpolation)

    if mode == "cover":
        scale = max(screen_w / frame_w, screen_h / frame_h)
    else:
        scale = min(screen_w / frame_w, screen_h / frame_h)

    new_w = max(1, int(round(frame_w * scale)))
    new_h = max(1, int(round(frame_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)

    if mode == "cover":
        x0 = max(0, (new_w - screen_w) // 2)
        y0 = max(0, (new_h - screen_h) // 2)
        crop = resized[y0 : y0 + screen_h, x0 : x0 + screen_w]
        ch, cw = crop.shape[:2]
        canvas[0:ch, 0:cw] = crop
        return canvas

    x = max(0, (screen_w - new_w) // 2)
    y = max(0, (screen_h - new_h) // 2)
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas


def video_nodes() -> list[str]:
    paths = []
    for path in glob.glob("/dev/video*"):
        if re.fullmatch(r"/dev/video\d+", path):
            paths.append(path)
    return sorted(paths, key=lambda p: int(p.replace("/dev/video", "")))


def parse_device_index(device: str) -> int | None:
    device = device.strip()
    if device.isdigit():
        return int(device)
    match = re.fullmatch(r"/dev/video(\d+)", device)
    if match:
        return int(match.group(1))
    return None


def has_capture_formats(path: str) -> bool:
    if not os.path.exists(path):
        return False
    if not shutil.which("v4l2-ctl"):
        return parse_device_index(path) is not None
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "-d", path, "--list-formats"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=1.5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    if "Video Capture" not in out:
        return False
    return re.search(r"\[\d+\]:\s+'[A-Z0-9 ]{4}'", out) is not None


def list_video_devices() -> list[dict]:
    items: list[dict] = []
    grouped: list[tuple[str, str]] = []
    if shutil.which("v4l2-ctl"):
        try:
            out = subprocess.check_output(
                ["v4l2-ctl", "--list-devices"],
                text=True,
                stderr=subprocess.STDOUT,
            )
            name = ""
            for line in out.splitlines():
                if not line.strip():
                    name = ""
                    continue
                if not line.startswith(("\t", " ")) and line.rstrip().endswith(":"):
                    name = line.rstrip()[:-1].strip()
                elif "/dev/video" in line:
                    path = line.strip().split()[0]
                    grouped.append((path, name or path))
        except (OSError, subprocess.CalledProcessError):
            grouped = []
    if not grouped:
        grouped = [(path, path) for path in video_nodes()]

    seen = set()
    for path, name in grouped:
        if path in seen or parse_device_index(path) is None:
            continue
        seen.add(path)
        items.append(
            {
                "path": path,
                "name": name,
                "capture": has_capture_formats(path),
                "usb": "usb" in name.lower() or "uvc" in name.lower(),
                "skip": bool(SKIP_DEVICE_NAME.search(name)),
            }
        )
    return items


def permission_hint(paths: Iterable[str]) -> str:
    for path in paths:
        if os.path.exists(path) and not os.access(path, os.R_OK | os.W_OK):
            return (
                f" No access to {path}. Run: sudo usermod -aG video $USER"
                " then log out and back in."
            )
    return ""


def candidate_devices(explicit: str) -> list[str]:
    if explicit != "":
        index = parse_device_index(explicit)
        return [f"/dev/video{index}" if index is not None else explicit]
    items = list_video_devices()
    capture = [
        item["path"]
        for item in items
        if item["capture"] and not item["skip"]
    ]
    usb = [item["path"] for item in items if item["usb"] and item["path"] in capture]
    rest = [path for path in capture if path not in usb]
    paths = usb + rest
    if not paths:
        paths = video_nodes() or ["0"]
    return paths


def configure_capture(
    cap: cv2.VideoCapture,
    width: int,
    height: int,
    fps: int,
    use_mjpeg: bool,
    set_size: bool,
) -> None:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if use_mjpeg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if set_size:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)


def open_capture(
    device: str,
    width: int,
    height: int,
    fps: int,
    use_mjpeg: bool,
) -> cv2.VideoCapture | None:
    # OpenCV V4L2 on Raspberry Pi cannot open by path name, only by index.
    index = parse_device_index(device)
    if index is None:
        return None
    path = f"/dev/video{index}"
    if os.path.exists(path) and not has_capture_formats(path):
        return None

    attempts = []
    if use_mjpeg:
        attempts.append((True, True))
    attempts.append((False, True))
    attempts.append((False, False))

    for mjpeg, set_size in attempts:
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue
        configure_capture(cap, width, height, fps, mjpeg, set_size)
        ok, frame = cap.read()
        if ok and frame is not None:
            return cap
        cap.release()
    return None


def find_camera(
    devices: Iterable[str],
    width: int,
    height: int,
    fps: int,
    use_mjpeg: bool,
) -> tuple[cv2.VideoCapture, str]:
    tried = []
    for device in devices:
        print(f"Trying {device} as V4L2 index {parse_device_index(device)} ...")
        cap = open_capture(device, width, height, fps, use_mjpeg)
        if cap is not None:
            return cap, device
        tried.append(device)
    hint = permission_hint(tried)
    raise RuntimeError(
        "No working camera found. Tried: "
        + ", ".join(tried)
        + hint
        + " Check with: v4l2-ctl --list-devices"
    )


def apply_window(name: str, fullscreen: bool, screen_w: int, screen_h: int) -> None:
    flags = cv2.WINDOW_NORMAL
    if hasattr(cv2, "WINDOW_FREERATIO"):
        flags |= cv2.WINDOW_FREERATIO
    if hasattr(cv2, "WINDOW_GUI_NORMAL"):
        flags |= cv2.WINDOW_GUI_NORMAL
    cv2.namedWindow(name, flags)
    cv2.setWindowProperty(
        name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL,
    )
    if hasattr(cv2, "WND_PROP_ASPECT_RATIO") and hasattr(cv2, "WINDOW_FREERATIO"):
        cv2.setWindowProperty(name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
    if fullscreen:
        cv2.resizeWindow(name, screen_w, screen_h)
        cv2.moveWindow(name, 0, 0)
        if hasattr(cv2, "WND_PROP_TOPMOST"):
            cv2.setWindowProperty(name, cv2.WND_PROP_TOPMOST, 1)
    else:
        cv2.resizeWindow(name, min(1280, screen_w), min(720, screen_h))


def rotate_frame(frame, turns: int):
    turns = turns % 4
    if turns == 1:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if turns == 2:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if turns == 3:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def save_snapshot(frame, directory: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, datetime.now().strftime("%Y%m%d_%H%M%S.jpg"))
    cv2.imwrite(path, frame)
    return path


def next_preset(width: int, height: int, step: int) -> tuple[int, int]:
    current = (width, height)
    if current in PRESETS:
        idx = (PRESETS.index(current) + step) % len(PRESETS)
        return PRESETS[idx]
    return PRESETS[0]


def draw_text_block(img: np.ndarray, lines: list[str], title: str = "") -> None:
    height, width = img.shape[:2]
    pad = max(24, width // 50)
    line_h = max(28, height // 28)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.6, width / 1920 * 0.9)
    thickness = 2
    rows = len(lines) + (1 if title else 0)
    panel_h = pad * 2 + rows * line_h
    panel_w = min(width - pad * 2, max(width // 2, 640))
    overlay = img.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + panel_w, pad + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    y = pad + line_h
    if title:
        cv2.putText(img, title, (pad + 16, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)
        y += line_h
    for line in lines:
        cv2.putText(img, line, (pad + 16, y), font, scale * 0.85, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_h


def main() -> int:
    args = parse_args()
    devices = candidate_devices(args.device)
    use_mjpeg = not args.no_mjpeg
    width, height, fps = args.width, args.height, args.fps
    fit_mode = args.fit
    screen_w, screen_h = detect_screen_size()

    try:
        cap, device = find_camera(devices, width, height, fps, use_mjpeg)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Display {screen_w}x{screen_h}  camera {device} {cam_w}x{cam_h}  fit={fit_mode}")

    fullscreen = not args.windowed
    apply_window(args.window, fullscreen, screen_w, screen_h)
    rotation = 0
    device_index = devices.index(device) if device in devices else 0
    message = ""
    message_until = 0.0
    show_help = False
    show_cameras = False

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.05)
            continue

        original = rotate_frame(frame, rotation)
        target_w, target_h = (screen_w, screen_h) if fullscreen else (min(1280, screen_w), min(720, screen_h))
        display = fit_to_screen(original, target_w, target_h, fit_mode)

        if show_cameras:
            lines = []
            for item in list_video_devices():
                if item["skip"] and not item["capture"]:
                    continue
                mark = ">" if item["path"] == device else " "
                kind = "USB" if item["usb"] else ("CAP" if item["capture"] else "meta")
                name = item["name"][:40]
                lines.append(f"{mark} {item['path']}  {kind}  {name}")
            if not lines:
                lines = ["brak kamer USB", "v4l2-ctl --list-devices"]
            lines.append("m = nastepna   l = schowaj")
            draw_text_block(display, lines, "Kamery")
        elif show_help:
            draw_text_block(display, HELP_LINES, "Skroty")
        elif time.time() < message_until and message:
            draw_text_block(display, [message])

        cv2.imshow(args.window, display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("f"):
            fullscreen = not fullscreen
            screen_w, screen_h = detect_screen_size()
            apply_window(args.window, fullscreen, screen_w, screen_h)
        if key == ord("r"):
            rotation = (rotation + 1) % 4
            message, message_until = f"obrot {rotation * 90}", time.time() + 1.5
        if key == ord("c"):
            fit_mode = FIT_MODES[(FIT_MODES.index(fit_mode) + 1) % len(FIT_MODES)]
            message, message_until = f"fit {fit_mode}", time.time() + 1.5
        if key == ord("s"):
            path = save_snapshot(original, args.snapshots)
            message, message_until = f"zapis {path}", time.time() + 2.0
        if key in (ord("h"), ord("?")):
            show_help = not show_help
            show_cameras = False
        if key == ord("l"):
            show_cameras = not show_cameras
            show_help = False
        if key in (ord("["), ord("]")):
            width, height = next_preset(width, height, -1 if key == ord("[") else 1)
            cap.release()
            cap, device = find_camera([device], width, height, fps, use_mjpeg)
            message, message_until = f"{width}x{height}", time.time() + 1.5
        if key == ord("m"):
            devices = candidate_devices("")
            if not devices:
                message, message_until = "brak kamer", time.time() + 1.5
                continue
            cap.release()
            device_index = (device_index + 1) % len(devices)
            nxt = devices[device_index]
            try:
                cap, device = find_camera([nxt], width, height, fps, use_mjpeg)
                message, message_until = f"{device}", time.time() + 1.5
            except RuntimeError:
                try:
                    cap, device = find_camera(devices, width, height, fps, use_mjpeg)
                    message, message_until = f"{device}", time.time() + 1.5
                except RuntimeError as exc:
                    print(exc, file=sys.stderr)
                    return 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
