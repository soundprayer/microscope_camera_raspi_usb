#!/usr/bin/env python3
"""Launch fullscreen USB camera preview. Safe to copy from Windows (CRLF is fine)."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def wait_until(paths: list[Path], timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if any(path.exists() for path in paths):
            return True
        time.sleep(0.4)
    return any(path.exists() for path in paths)


def prepare_session() -> None:
    uid = os.getuid()
    runtime = Path(f"/run/user/{uid}")
    os.environ.setdefault("XDG_RUNTIME_DIR", str(runtime))
    os.environ.setdefault("HOME", str(Path.home()))

    wait_until(
        [
            Path("/tmp/.X11-unix/X0"),
            runtime / "wayland-0",
            runtime / "wayland-1",
        ],
        timeout=90,
    )
    if (runtime / "wayland-0").exists():
        os.environ.setdefault("WAYLAND_DISPLAY", "wayland-0")
    elif (runtime / "wayland-1").exists():
        os.environ.setdefault("WAYLAND_DISPLAY", "wayland-1")
    os.environ.setdefault("DISPLAY", ":0")

    deadline = time.time() + 45
    while time.time() < deadline and not glob.glob("/dev/video*"):
        time.sleep(0.4)

    if shutil.which("xset") and os.environ.get("DISPLAY"):
        for args in (["s", "off"], ["-dpms"], ["s", "noblank"]):
            subprocess.run(["xset", *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if shutil.which("unclutter"):
        subprocess.Popen(
            ["unclutter", "-idle", "0.3", "-root"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


prepare_session()
script = Path(__file__).resolve().parent / "camera_fullscreen.py"
os.execv(sys.executable, [sys.executable, str(script), *sys.argv[1:]])
