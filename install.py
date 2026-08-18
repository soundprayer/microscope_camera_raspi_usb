#!/usr/bin/env python3
"""Install fullscreen USB camera preview on Raspberry Pi OS.

Safe to copy from Windows: Python accepts CRLF. Also converts .sh files to Unix LF.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

SERVICE_NAME = "microscope-camera.service"
SERVICE_PATH = Path("/etc/systemd/system") / SERVICE_NAME
DESKTOP_NAME = "microscope-camera.desktop"

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


def python_bin() -> str:
    return shutil.which("python3") or "/usr/bin/python3"


def enable_autostart(root: Path, target_user: str, target_home: Path) -> None:
    uid = subprocess.check_output(["id", "-u", target_user], text=True).strip()
    launcher = root / "run.py"
    unit = f"""[Unit]
Description=Microscope USB camera fullscreen preview
After=graphical.target
Wants=graphical.target

[Service]
Type=simple
User={target_user}
Group={target_user}
SupplementaryGroups=video
WorkingDirectory={root}
Environment=HOME={target_home}
Environment=DISPLAY=:0
Environment=WAYLAND_DISPLAY=wayland-0
Environment=XDG_RUNTIME_DIR=/run/user/{uid}
Environment=XAUTHORITY={target_home}/.Xauthority
Environment=PYGAME_HIDE_SUPPORT_PROMPT=1
ExecStart={python_bin()} {launcher}
Restart=always
RestartSec=4

[Install]
WantedBy=graphical.target
"""
    SERVICE_PATH.write_text(unit, encoding="utf-8")
    subprocess.check_call(["systemctl", "daemon-reload"])
    subprocess.check_call(["systemctl", "enable", SERVICE_NAME])
    subprocess.run(["systemctl", "restart", SERVICE_NAME], check=False)

    desktop = target_home / ".config" / "autostart" / DESKTOP_NAME
    if desktop.exists():
        desktop.unlink()

    print(f"Autostart enabled: {SERVICE_NAME} (starts after graphical desktop).")
    print("Reboot once if the preview does not appear:  sudo reboot")
    print("Disable later with:  sudo python3 install.py --disable-autostart")


def disable_autostart(target_home: Path) -> None:
    subprocess.run(["systemctl", "disable", "--now", SERVICE_NAME], check=False)
    if SERVICE_PATH.exists():
        SERVICE_PATH.unlink()
        subprocess.check_call(["systemctl", "daemon-reload"])
    desktop = target_home / ".config" / "autostart" / DESKTOP_NAME
    if desktop.exists():
        desktop.unlink()
    print("Autostart disabled.")


def main() -> int:
    root = Path(__file__).resolve().parent
    autostart = "--autostart" in sys.argv

    for path in list(root.glob("*.sh")) + list(root.glob("*.py")):
        to_unix(path)

    if os.geteuid() != 0:
        print(f"Run with:  sudo python3 {Path(__file__).name} [--autostart|--disable-autostart]")
        return 1

    target_user = os.environ.get("SUDO_USER") or "pi"
    passwd = subprocess.check_output(["getent", "passwd", target_user], text=True)
    target_home = Path(passwd.split(":")[5])

    if "--disable-autostart" in sys.argv:
        disable_autostart(target_home)
        return 0

    subprocess.check_call(["apt-get", "update"])
    subprocess.check_call(["apt-get", "install", "-y", *PACKAGES])

    groups = subprocess.check_output(["id", "-nG", target_user], text=True).split()
    if "video" not in groups:
        subprocess.check_call(["usermod", "-aG", "video", target_user])
        print(f"Added {target_user} to the video group. Log out and back in once.")
    else:
        print(f"User {target_user} is already in the video group.")

    if autostart:
        enable_autostart(root, target_user, target_home)

    print()
    print("Done. Plug in the USB camera, then run:")
    print(f"  python3 {root / 'run.py'}")
    print("List cameras with:  v4l2-ctl --list-devices")
    if autostart:
        print("Status:  systemctl status microscope-camera.service")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
