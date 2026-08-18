#!/usr/bin/env python3
"""Install fullscreen USB camera preview on Raspberry Pi OS.

Safe to copy from Windows: Python accepts CRLF. Also converts .sh files to Unix LF.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

PACKAGES = [
    "python3-opencv",
    "python3-numpy",
    "python3-pygame",
    "v4l-utils",
    "ffmpeg",
    "unclutter",
]


def to_unix(path: Path) -> None:
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    path.write_bytes(data)
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def main() -> int:
    root = Path(__file__).resolve().parent
    autostart = "--autostart" in sys.argv

    for path in list(root.glob("*.sh")) + list(root.glob("*.py")):
        to_unix(path)

    if os.geteuid() != 0:
        print(f"Run with:  sudo python3 {Path(__file__).name} [--autostart]")
        return 1

    subprocess.check_call(["apt-get", "update"])
    subprocess.check_call(["apt-get", "install", "-y", *PACKAGES])

    target_user = os.environ.get("SUDO_USER") or "pi"
    passwd = subprocess.check_output(["getent", "passwd", target_user], text=True)
    target_home = Path(passwd.split(":")[5])
    groups = subprocess.check_output(["id", "-nG", target_user], text=True).split()
    if "video" not in groups:
        subprocess.check_call(["usermod", "-aG", "video", target_user])
        print(f"Added {target_user} to the video group. Log out and back in once.")
    else:
        print(f"User {target_user} is already in the video group.")

    if autostart:
        desktop_dir = target_home / ".config" / "autostart"
        desktop_dir.mkdir(parents=True, exist_ok=True)
        desktop = desktop_dir / "microscope-camera.desktop"
        exec_line = (
            "env QT_QPA_PLATFORM=xcb GDK_BACKEND=x11 "
            f"{sys.executable} {root / 'camera_fullscreen.py'}"
        )
        desktop.write_text(
            "[Desktop Entry]\n"
            "Type=Application\n"
            "Name=Microscope Camera\n"
            f"Exec={exec_line}\n"
            "X-GNOME-Autostart-enabled=true\n",
            encoding="utf-8",
            newline="\n",
        )
        subprocess.check_call(
            ["chown", "-R", f"{target_user}:{target_user}", str(target_home / ".config" / "autostart")]
        )
        print(f"Autostart enabled for {target_user} (starts after graphical login).")

    print()
    print("Done. Plug in the USB camera, then run:")
    print(f"  python3 {root / 'camera_fullscreen.py'}")
    print("List cameras with:  v4l2-ctl --list-devices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
