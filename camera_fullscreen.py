#!/usr/bin/env python3
"""Fullscreen USB camera preview for Raspberry Pi 4B.

Capture runs on a side thread (latest frame only). Display reuses buffers,
uses MJPEG, and syncs to the monitor so 720p/1080p HDMI stays smooth.

Keys:
  q / Esc  quit
  s        save snapshot (original camera frame)
  f        toggle fullscreen
  r        rotate 90 degrees clockwise
  c        cycle fit: cover / contain / stretch
  m        next camera
  o        attach / cycle / off overlay camera
  b        overlay blend mode
  - / +    overlay opacity
  [ / ]    lower / raise capture resolution
  h        help
  l        camera list
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")
os.environ.setdefault("DISPLAY", os.environ.get("DISPLAY", ":0"))

import cv2
import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(2)
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
BLEND_MODES = ("alpha", "screen", "multiply", "difference", "lighten", "pip")
HELP_LINES = [
    "Esc / q   wyjscie",
    "Alt+F4    wyjscie",
    "s         zdjecie",
    "f / F11   pelny ekran",
    "r         obrot 90",
    "c         cover/contain/stretch",
    "m         nastepna kamera",
    "o         overlay kamera",
    "b         tryb mix",
    "- +       sila mix (wszystkie tryby)",
    "[ ]       rozdzielczosc",
    "l         lista kamer",
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
    check_formats: bool = True,
) -> cv2.VideoCapture | None:
    index = parse_device_index(device)
    if index is None:
        return None
    path = f"/dev/video{index}"
    if check_formats and os.path.exists(path) and not has_capture_formats(path):
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
        try:
            ok, frame = cap.read()
        except Exception:
            ok, frame = False, None
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


class FrameGrabber:
    """Read USB frames off the UI thread; keep only the latest one."""

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self.seq = 0
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera-grab", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            try:
                ok, frame = self.cap.read()
            except Exception:
                time.sleep(0.02)
                continue
            if not ok or frame is None:
                time.sleep(0.005)
                continue
            with self._lock:
                if self._frame is None or self._frame.shape != frame.shape:
                    self._frame = frame.copy()
                else:
                    np.copyto(self._frame, frame)
                self.seq += 1

    def get_if_new(self, last_seq: int, dest: np.ndarray | None = None) -> tuple[int, np.ndarray | None]:
        with self._lock:
            if self._frame is None or self.seq == last_seq:
                return last_seq, None
            if dest is not None and dest.shape == self._frame.shape:
                np.copyto(dest, self._frame)
                return self.seq, dest
            return self.seq, self._frame.copy()

    def stop(self, timeout: float = 3.0) -> bool:
        self._running = False
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()


class FrameScaler:
    """Scale camera frames onto a reused HDMI-sized canvas."""

    def __init__(self) -> None:
        self.canvas: np.ndarray | None = None
        self.resized: np.ndarray | None = None

    def _canvas(self, height: int, width: int) -> np.ndarray:
        if self.canvas is None or self.canvas.shape[0] != height or self.canvas.shape[1] != width:
            self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        return self.canvas

    def apply(self, frame: np.ndarray, screen_w: int, screen_h: int, mode: str, fast: bool = False) -> np.ndarray:
        canvas = self._canvas(screen_h, screen_w)
        frame_h, frame_w = frame.shape[:2]
        if frame_w < 1 or frame_h < 1:
            canvas.fill(0)
            return canvas

        downscale = screen_w < frame_w or screen_h < frame_h
        interpolation = cv2.INTER_LINEAR if fast else (
            cv2.INTER_AREA if downscale else cv2.INTER_LINEAR
        )

        if mode == "stretch" or (frame_w == screen_w and frame_h == screen_h):
            if frame_w == screen_w and frame_h == screen_h:
                np.copyto(canvas, frame)
                return canvas
            cv2.resize(frame, (screen_w, screen_h), dst=canvas, interpolation=interpolation)
            return canvas

        if mode == "cover":
            scale = max(screen_w / frame_w, screen_h / frame_h)
        else:
            scale = min(screen_w / frame_w, screen_h / frame_h)

        new_w = max(1, int(round(frame_w * scale)))
        new_h = max(1, int(round(frame_h * scale)))
        if not fast:
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        if self.resized is None or self.resized.shape[0] != new_h or self.resized.shape[1] != new_w:
            self.resized = np.empty((new_h, new_w, 3), dtype=np.uint8)
        cv2.resize(frame, (new_w, new_h), dst=self.resized, interpolation=interpolation)

        if mode == "cover":
            x0 = max(0, (new_w - screen_w) // 2)
            y0 = max(0, (new_h - screen_h) // 2)
            crop = self.resized[y0 : y0 + screen_h, x0 : x0 + screen_w]
            ch, cw = crop.shape[:2]
            if ch != screen_h or cw != screen_w:
                canvas.fill(0)
            canvas[0:ch, 0:cw] = crop
            return canvas

        canvas.fill(0)
        x = max(0, (screen_w - new_w) // 2)
        y = max(0, (screen_h - new_h) // 2)
        canvas[y : y + new_h, x : x + new_w] = self.resized
        return canvas


class OverlayMixer:
    """Blend a second camera onto the main canvas. Reuses buffers."""

    def __init__(self) -> None:
        self._not_a: np.ndarray | None = None
        self._not_b: np.ndarray | None = None
        self._effect: np.ndarray | None = None
        self._out: np.ndarray | None = None
        self._pip: np.ndarray | None = None

    def _bufs(self, shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._effect is None or self._effect.shape != shape:
            self._not_a = np.empty(shape, dtype=np.uint8)
            self._not_b = np.empty(shape, dtype=np.uint8)
            self._effect = np.empty(shape, dtype=np.uint8)
        assert self._not_a is not None and self._not_b is not None and self._effect is not None
        return self._not_a, self._not_b, self._effect

    def output(self, shape: tuple[int, ...]) -> np.ndarray:
        if self._out is None or self._out.shape != shape:
            self._out = np.empty(shape, dtype=np.uint8)
        return self._out

    def blend(self, base: np.ndarray, over: np.ndarray, mode: str, alpha: float) -> None:
        if over.shape != base.shape:
            over = cv2.resize(over, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha = min(0.9, max(0.1, float(alpha)))
        if mode == "alpha":
            cv2.addWeighted(base, 1.0 - alpha, over, alpha, 0, dst=base)
            return
        if mode == "pip":
            self.pip(base, over, alpha)
            return

        not_a, not_b, effect = self._bufs(base.shape)
        if mode == "screen":
            cv2.bitwise_not(base, dst=not_a)
            cv2.bitwise_not(over, dst=not_b)
            cv2.multiply(not_a, not_b, dst=effect, scale=1.0 / 255.0)
            cv2.bitwise_not(effect, dst=effect)
        elif mode == "multiply":
            cv2.multiply(base, over, dst=effect, scale=1.0 / 255.0)
        elif mode == "difference":
            cv2.absdiff(base, over, dst=effect)
        elif mode == "lighten":
            np.maximum(base, over, out=effect)
        else:
            np.copyto(effect, over)
        cv2.addWeighted(base, 1.0 - alpha, effect, alpha, 0, dst=base)

    def pip(self, base: np.ndarray, over: np.ndarray, alpha: float = 0.9) -> None:
        height, width = base.shape[:2]
        pip_w = max(160, width // 3)
        pip_h = max(90, height // 3)
        x = max(0, width - pip_w - 24)
        y = max(0, height - pip_h - 24)
        if self._pip is None or self._pip.shape[0] != pip_h or self._pip.shape[1] != pip_w:
            self._pip = np.empty((pip_h, pip_w, 3), dtype=np.uint8)
        cv2.resize(over, (pip_w, pip_h), dst=self._pip, interpolation=cv2.INTER_LINEAR)
        roi = base[y : y + pip_h, x : x + pip_w]
        cv2.addWeighted(roi, 1.0 - alpha, self._pip, alpha, 0, dst=roi)
        cv2.rectangle(base, (x - 2, y - 2), (x + pip_w + 1, y + pip_h + 1), (0, 220, 0), 2)


class Screen:
    """Exclusive fullscreen via SDL/pygame: no title bar, vsync when available."""

    def __init__(self, title: str, fullscreen: bool) -> None:
        pygame.init()
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        pygame.display.set_caption(title)
        try:
            pygame.display.set_allow_screensaver(False)
        except Exception:
            pass
        self.title = title
        self.fullscreen = fullscreen
        self._rgb: np.ndarray | None = None
        self.surface = self._create()

    def _set_mode(self, size: tuple[int, int], flags: int):
        try:
            return pygame.display.set_mode(size, flags, vsync=1)
        except TypeError:
            return pygame.display.set_mode(size, flags)

    def _create(self):
        flags = pygame.DOUBLEBUF
        if self.fullscreen:
            flags |= pygame.FULLSCREEN | pygame.NOFRAME
            try:
                surface = self._set_mode((0, 0), flags)
            except pygame.error:
                width, height = detect_screen_size()
                surface = self._set_mode((width, height), flags)
        else:
            surface = self._set_mode((1280, 720), flags)
        pygame.mouse.set_visible(not self.fullscreen)
        self._rgb = None
        return surface

    @property
    def size(self) -> tuple[int, int]:
        return self.surface.get_size()

    def set_fullscreen(self, fullscreen: bool) -> None:
        self.fullscreen = fullscreen
        self.surface = self._create()

    def show(self, bgr: np.ndarray) -> None:
        width, height = self.size
        if bgr.shape[1] != width or bgr.shape[0] != height:
            bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
        if self._rgb is None or self._rgb.shape[0] != height or self._rgb.shape[1] != width:
            self._rgb = np.empty((height, width, 3), dtype=np.uint8)
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=self._rgb)
        frame = pygame.image.frombuffer(self._rgb, (width, height), "RGB")
        self.surface.blit(frame, (0, 0))
        pygame.display.flip()

    def close(self) -> None:
        pygame.mouse.set_visible(True)
        pygame.display.quit()
        pygame.quit()


def poll_actions() -> list[str]:
    actions = []
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            actions.append("quit")
        elif event.type == pygame.KEYDOWN:
            mods = pygame.key.get_mods()
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                actions.append("quit")
            elif event.key == pygame.K_F4 and (mods & pygame.KMOD_ALT):
                actions.append("quit")
            elif event.key in (pygame.K_f, pygame.K_F11):
                actions.append("fullscreen")
            elif event.key == pygame.K_s:
                actions.append("snap")
            elif event.key == pygame.K_r:
                actions.append("rotate")
            elif event.key == pygame.K_c:
                actions.append("fit")
            elif event.key == pygame.K_h:
                actions.append("help")
            elif event.key == pygame.K_l:
                actions.append("list")
            elif event.key == pygame.K_m:
                actions.append("next")
            elif event.key == pygame.K_o:
                actions.append("overlay")
            elif event.key == pygame.K_b:
                actions.append("blend")
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_UNDERSCORE):
                actions.append("alpha_down")
            elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                actions.append("alpha_up")
            elif event.key == pygame.K_LEFTBRACKET:
                actions.append("res_down")
            elif event.key == pygame.K_RIGHTBRACKET:
                actions.append("res_up")
    return actions


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


def overlay_camera_list(
    display: np.ndarray,
    items: list[dict],
    current: str,
    overlay: str = "",
) -> None:
    lines = []
    for item in items:
        if item["skip"] and not item["capture"]:
            continue
        if item["path"] == current:
            mark = ">"
        elif item["path"] == overlay:
            mark = "O"
        else:
            mark = " "
        kind = "USB" if item["usb"] else ("CAP" if item["capture"] else "meta")
        name = item["name"][:40]
        lines.append(f"{mark} {item['path']}  {kind}  {name}")
    if not lines:
        lines = ["brak kamer USB", "v4l2-ctl --list-devices"]
    lines.append("m = nastepna   o = overlay   l = schowaj")
    draw_text_block(display, lines, "Kamery")


def main() -> int:
    if pygame is None:
        print("Brak pygame. Na Pi wpisz:  sudo apt-get install -y python3-pygame", file=sys.stderr)
        return 1

    args = parse_args()
    devices = candidate_devices(args.device)
    use_mjpeg = not args.no_mjpeg
    width, height, fps = args.width, args.height, args.fps
    fit_mode = args.fit

    try:
        cap, device = find_camera(devices, width, height, fps, use_mjpeg)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    fullscreen = not args.windowed
    screen = Screen(args.window, fullscreen)
    scaler = FrameScaler()
    overlay_scaler = FrameScaler()
    mixer = OverlayMixer()
    grabber = FrameGrabber(cap)
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    print(f"Display {screen.size[0]}x{screen.size[1]}  camera {device} {cam_w}x{cam_h}  fit={fit_mode}")
    print("Wyjscie: Esc albo q  (albo Alt+F4)")
    print("Overlay: o  mix: b  przezroczystosc: - +")
    print("Dla plynnosci wtykaj kamere w niebieski port USB 3.0 (Pi 4).")

    rotation = 0
    device_index = devices.index(device) if device in devices else 0
    message = "Esc / q = wyjscie"
    message_until = time.time() + 2.5
    show_help = False
    show_cameras = False
    camera_list_cache: list[dict] = []
    running = True
    last_seq = -1
    last_raw: np.ndarray | None = None
    overlay_cap: cv2.VideoCapture | None = None
    overlay_grabber: FrameGrabber | None = None
    overlay_device = ""
    overlay_seq = -1
    overlay_raw: np.ndarray | None = None
    main_cache_seq = -1
    over_cache_seq = -1
    main_cache_key = None
    over_cache_key = None
    blend_mode = "alpha"
    blend_alpha = 0.45
    dirty = True
    overlay_was = True
    clock = pygame.time.Clock()
    tick_hz = 60 if cam_fps >= 50 else 30

    def attach(new_cap: cv2.VideoCapture, new_device: str) -> None:
        nonlocal cap, device, grabber, last_seq, last_raw, dirty, cam_w, cam_h, width, height
        cap = new_cap
        device = new_device
        grabber = FrameGrabber(cap)
        last_seq = -1
        last_raw = None
        dirty = True
        cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        if cam_w >= 16 and cam_h >= 16:
            width, height = cam_w, cam_h

    def release_current() -> None:
        grabber.stop()
        cap.release()
        time.sleep(0.2)

    def recover(old_device: str, old_w: int, old_h: int) -> bool:
        seen = []
        for candidate in [old_device, *candidate_devices("")]:
            if candidate in seen or (overlay_device and candidate == overlay_device):
                continue
            seen.append(candidate)
            new_cap = open_capture(candidate, old_w, old_h, fps, use_mjpeg, check_formats=False)
            if new_cap is None:
                new_cap = open_capture(candidate, old_w, old_h, fps, False, check_formats=False)
            if new_cap is not None:
                attach(new_cap, candidate)
                return True
        return False

    def change_resolution(step: int) -> tuple[bool, str]:
        old_device, old_w, old_h = device, width, height
        wanted: list[tuple[int, int]] = []
        w, h = old_w, old_h
        for _ in range(len(PRESETS)):
            w, h = next_preset(w, h, step)
            if (w, h) not in wanted and (w, h) != (old_w, old_h):
                wanted.append((w, h))
        if not wanted:
            wanted = [PRESETS[0]]
        release_current()
        try:
            for w, h in wanted:
                new_cap = open_capture(old_device, w, h, fps, use_mjpeg, check_formats=False)
                if new_cap is not None:
                    attach(new_cap, old_device)
                    return True, f"{cam_w}x{cam_h}"
            if recover(old_device, old_w, old_h):
                return True, f"brak {wanted[0][0]}x{wanted[0][1]}"
            return False, "kamera utracona"
        except Exception as exc:
            if recover(old_device, old_w, old_h):
                return True, "blad rozdzielczosci"
            return False, str(exc)

    def switch_device(nxt: str, all_devices: list[str]) -> tuple[bool, str]:
        old_device, old_w, old_h = device, width, height
        release_current()
        try:
            order = [nxt] + [item for item in all_devices if item != nxt]
            for candidate in order:
                if overlay_device and candidate == overlay_device:
                    continue
                new_cap = open_capture(candidate, old_w, old_h, fps, use_mjpeg, check_formats=False)
                if new_cap is not None:
                    attach(new_cap, candidate)
                    return True, candidate
            if recover(old_device, old_w, old_h):
                return True, f"nie mozna {nxt}"
            return False, "kamera utracona"
        except Exception as exc:
            if recover(old_device, old_w, old_h):
                return True, "blad kamery"
            return False, str(exc)

    def stop_overlay() -> None:
        nonlocal overlay_cap, overlay_grabber, overlay_device, overlay_raw, overlay_seq, dirty
        if overlay_grabber is not None:
            try:
                overlay_grabber.stop()
            except Exception:
                pass
        if overlay_cap is not None:
            try:
                overlay_cap.release()
            except Exception:
                pass
        overlay_cap = None
        overlay_grabber = None
        overlay_device = ""
        overlay_raw = None
        overlay_seq = -1
        dirty = True
        time.sleep(0.15)

    def attach_overlay(new_cap: cv2.VideoCapture, new_device: str) -> None:
        nonlocal overlay_cap, overlay_grabber, overlay_device, overlay_raw, overlay_seq, dirty
        overlay_cap = new_cap
        overlay_device = new_device
        overlay_grabber = FrameGrabber(new_cap)
        overlay_raw = None
        overlay_seq = -1
        dirty = True

    def cycle_overlay() -> str:
        others = [item for item in candidate_devices("") if item != device]
        if not others:
            stop_overlay()
            return "brak drugiej kamery"
        if overlay_device:
            idx = others.index(overlay_device) if overlay_device in others else -1
            rest = others[idx + 1 :]
            stop_overlay()
            if not rest:
                return "overlay off"
            to_try = rest
        else:
            to_try = others
        sizes = [(1280, 720), (640, 480), (width, height)]
        for candidate in to_try:
            for ow, oh in sizes:
                new_cap = open_capture(candidate, ow, oh, fps, use_mjpeg, check_formats=False)
                if new_cap is not None:
                    attach_overlay(new_cap, candidate)
                    got_w = int(new_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or ow
                    got_h = int(new_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or oh
                    return f"overlay {candidate} {got_w}x{got_h}"
        return "nie mozna overlay"

    try:
        while running:
            for action in poll_actions():
                if action == "quit":
                    running = False
                    break
                if action == "fullscreen":
                    fullscreen = not fullscreen
                    screen.set_fullscreen(fullscreen)
                    dirty = True
                elif action == "rotate":
                    rotation = (rotation + 1) % 4
                    message, message_until = f"obrot {rotation * 90}", time.time() + 1.5
                    dirty = True
                elif action == "fit":
                    fit_mode = FIT_MODES[(FIT_MODES.index(fit_mode) + 1) % len(FIT_MODES)]
                    message, message_until = f"fit {fit_mode}", time.time() + 1.5
                    dirty = True
                elif action == "snap":
                    if last_raw is not None:
                        path = save_snapshot(rotate_frame(last_raw, rotation), args.snapshots)
                        message, message_until = f"zapis {path}", time.time() + 2.0
                        dirty = True
                elif action == "help":
                    show_help = not show_help
                    show_cameras = False
                    dirty = True
                elif action == "list":
                    show_cameras = not show_cameras
                    show_help = False
                    if show_cameras:
                        camera_list_cache = list_video_devices()
                    dirty = True
                elif action in ("res_down", "res_up"):
                    step = -1 if action == "res_down" else 1
                    ok, note = change_resolution(step)
                    message, message_until = note, time.time() + 1.8
                    dirty = True
                    if not ok:
                        print(note, file=sys.stderr)
                elif action == "next":
                    devices = candidate_devices("")
                    mains = [item for item in devices if item != overlay_device] or devices
                    if not mains:
                        message, message_until = "brak kamer", time.time() + 1.5
                        dirty = True
                        continue
                    if device in mains:
                        device_index = (mains.index(device) + 1) % len(mains)
                    else:
                        device_index = (device_index + 1) % len(mains)
                    nxt = mains[device_index]
                    ok, note = switch_device(nxt, mains)
                    message, message_until = note, time.time() + 1.8
                    dirty = True
                    if not ok:
                        print(note, file=sys.stderr)
                elif action == "overlay":
                    note = cycle_overlay()
                    message, message_until = note, time.time() + 1.8
                    dirty = True
                elif action == "blend":
                    if not overlay_device:
                        message, message_until = "najpierw o = overlay", time.time() + 1.5
                    else:
                        blend_mode = BLEND_MODES[(BLEND_MODES.index(blend_mode) + 1) % len(BLEND_MODES)]
                        extra = f" {int(round(blend_alpha * 100))}%"
                        message, message_until = f"mix {blend_mode}{extra}", time.time() + 1.5
                    dirty = True
                elif action in ("alpha_down", "alpha_up"):
                    if not overlay_device:
                        message, message_until = "najpierw o = overlay", time.time() + 1.5
                    else:
                        step = -0.1 if action == "alpha_down" else 0.1
                        blend_alpha = min(0.9, max(0.1, round(blend_alpha + step, 2)))
                        message, message_until = f"mix {blend_mode} {int(round(blend_alpha * 100))}%", time.time() + 1.5
                    dirty = True

            try:
                seq, frame = grabber.get_if_new(last_seq, last_raw)
            except Exception:
                seq, frame = last_seq, None
            if frame is not None:
                last_seq = seq
                last_raw = frame
                dirty = True

            if overlay_grabber is not None:
                try:
                    oseq, oframe = overlay_grabber.get_if_new(overlay_seq, overlay_raw)
                except Exception:
                    oseq, oframe = overlay_seq, None
                if oframe is not None:
                    overlay_seq = oseq
                    overlay_raw = oframe
                    dirty = True

            overlay_now = show_cameras or show_help or (time.time() < message_until and bool(message))
            if overlay_now != overlay_was:
                dirty = True
                overlay_was = overlay_now

            if dirty and last_raw is not None:
                try:
                    screen_w, screen_h = screen.size
                    main_key = (screen_w, screen_h, fit_mode, rotation)
                    if main_cache_seq != last_seq or main_cache_key != main_key:
                        original = rotate_frame(last_raw, rotation)
                        scaler.apply(original, screen_w, screen_h, fit_mode)
                        main_cache_seq = last_seq
                        main_cache_key = main_key
                    display = scaler.canvas
                    if display is not None and overlay_raw is not None and overlay_device:
                        if blend_mode == "pip":
                            out = mixer.output(display.shape)
                            np.copyto(out, display)
                            mixer.pip(out, overlay_raw, blend_alpha)
                            display = out
                        else:
                            over_key = (screen_w, screen_h, fit_mode)
                            if over_cache_seq != overlay_seq or over_cache_key != over_key:
                                overlay_scaler.apply(
                                    overlay_raw, screen_w, screen_h, fit_mode, fast=True
                                )
                                over_cache_seq = overlay_seq
                                over_cache_key = over_key
                            if overlay_scaler.canvas is not None:
                                out = mixer.output(display.shape)
                                np.copyto(out, display)
                                mixer.blend(out, overlay_scaler.canvas, blend_mode, blend_alpha)
                                display = out
                    if display is not None:
                        if show_cameras:
                            overlay_camera_list(display, camera_list_cache, device, overlay_device)
                        elif show_help:
                            draw_text_block(display, HELP_LINES, "Skroty")
                        elif overlay_now and message:
                            draw_text_block(display, [message])
                        screen.show(display)
                except Exception as exc:
                    print(f"display: {exc}", file=sys.stderr)
                dirty = False

            clock.tick(tick_hz)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            stop_overlay()
        except Exception:
            pass
        try:
            grabber.stop()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        try:
            screen.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
