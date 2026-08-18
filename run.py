#!/usr/bin/env python3
"""Launch fullscreen USB camera preview. Safe to copy from Windows (CRLF is fine)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("DISPLAY", ":0")

if shutil.which("xset") and os.environ.get("DISPLAY"):
    for args in (["s", "off"], ["-dpms"], ["s", "noblank"]):
        subprocess.run(["xset", *args], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

script = Path(__file__).resolve().parent / "camera_fullscreen.py"
os.execv(sys.executable, [sys.executable, str(script), *sys.argv[1:]])
