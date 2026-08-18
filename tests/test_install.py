from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import install


class TestToUnix:
    def test_converts_crlf(self, tmp_path):
        path = tmp_path / "run.sh"
        path.write_bytes(b"#!/bin/sh\r\necho hi\r\n")
        install.to_unix(path)
        assert path.read_bytes() == b"#!/bin/sh\necho hi\n"

    def test_converts_cr_only(self, tmp_path):
        path = tmp_path / "a.py"
        path.write_bytes(b"print(1)\rprint(2)\r")
        install.to_unix(path)
        assert b"\r" not in path.read_bytes()

    def test_leaves_lf_untouched(self, tmp_path):
        path = tmp_path / "ok.py"
        original = b"print('lf')\n"
        path.write_bytes(original)
        before = path.stat().st_mtime_ns
        install.to_unix(path)
        assert path.read_bytes() == original
        assert path.stat().st_mtime_ns == before


class TestPythonBin:
    def test_which(self, monkeypatch):
        monkeypatch.setattr(install.shutil, "which", lambda name: "/opt/bin/python3")
        assert install.python_bin() == "/opt/bin/python3"

    def test_fallback(self, monkeypatch):
        monkeypatch.setattr(install.shutil, "which", lambda name: None)
        assert install.python_bin() == "/usr/bin/python3"


class TestAutostart:
    def test_enable_writes_unit_and_enables(self, monkeypatch, tmp_path):
        service = tmp_path / "microscope-camera.service"
        monkeypatch.setattr(install, "SERVICE_PATH", service)
        monkeypatch.setattr(install, "python_bin", lambda: "/usr/bin/python3")
        calls = []
        monkeypatch.setattr(
            install.subprocess,
            "check_output",
            lambda *a, **k: "1000\n",
        )
        monkeypatch.setattr(
            install.subprocess,
            "check_call",
            lambda cmd: calls.append(cmd),
        )
        monkeypatch.setattr(
            install.subprocess,
            "run",
            lambda cmd, check=False: calls.append(cmd),
        )
        desktop_dir = tmp_path / ".config" / "autostart"
        desktop_dir.mkdir(parents=True)
        stale = desktop_dir / install.DESKTOP_NAME
        stale.write_text("old")
        install.enable_autostart(tmp_path / "app", "audioraspi", tmp_path)
        text = service.read_text(encoding="utf-8")
        assert "User=audioraspi" in text
        assert "WantedBy=graphical.target" in text
        assert str(tmp_path / "app" / "run.py") in text
        assert ["systemctl", "enable", install.SERVICE_NAME] in calls
        assert not stale.exists()

    def test_disable_removes_unit_and_desktop(self, monkeypatch, tmp_path):
        service = tmp_path / "microscope-camera.service"
        service.write_text("unit")
        monkeypatch.setattr(install, "SERVICE_PATH", service)
        monkeypatch.setattr(install.subprocess, "run", lambda *a, **k: None)
        monkeypatch.setattr(install.subprocess, "check_call", lambda *a, **k: None)
        desktop = tmp_path / ".config" / "autostart" / install.DESKTOP_NAME
        desktop.parent.mkdir(parents=True)
        desktop.write_text("old")
        install.disable_autostart(tmp_path)
        assert not service.exists()
        assert not desktop.exists()


class TestInstallMain:
    def test_requires_root(self, monkeypatch):
        monkeypatch.setattr(install.os, "geteuid", lambda: 1000, raising=False)
        monkeypatch.setattr(install, "to_unix", lambda path: None)
        assert install.main() == 1

    def test_disable_autostart_flag(self, monkeypatch):
        monkeypatch.setattr(install.os, "geteuid", lambda: 0, raising=False)
        monkeypatch.setattr(install, "to_unix", lambda path: None)
        monkeypatch.setattr("sys.argv", ["install.py", "--disable-autostart"])
        monkeypatch.setenv("SUDO_USER", "audioraspi")
        monkeypatch.setattr(
            install.subprocess,
            "check_output",
            lambda *a, **k: "audioraspi:x:1000:1000::/home/audioraspi:/bin/bash\n",
        )
        called = {}
        monkeypatch.setattr(install, "disable_autostart", lambda home: called.setdefault("home", home))
        assert install.main() == 0
        assert called["home"] == Path("/home/audioraspi")

    def test_installs_packages_and_autostart(self, monkeypatch, tmp_path):
        monkeypatch.setattr(install.os, "geteuid", lambda: 0, raising=False)
        monkeypatch.setattr(install, "to_unix", lambda path: None)
        monkeypatch.setattr("sys.argv", ["install.py", "--autostart"])
        monkeypatch.setenv("SUDO_USER", "pi")
        outputs = {
            "getent": f"pi:x:1000:1000::/home/pi:/bin/bash\n",
            "id": "pi adm video\n",
        }

        def check_output(cmd, text=True):
            if cmd[0] == "getent":
                return outputs["getent"]
            if cmd[0] == "id":
                return outputs["id"]
            raise AssertionError(cmd)

        calls = []
        monkeypatch.setattr(install.subprocess, "check_output", check_output)
        monkeypatch.setattr(install.subprocess, "check_call", lambda cmd: calls.append(cmd))
        monkeypatch.setattr(install, "enable_autostart", lambda *a: calls.append(("autostart", a)))
        assert install.main() == 0
        assert ["apt-get", "update"] in calls
        assert any(cmd[0] == "apt-get" and "install" in cmd for cmd in calls if isinstance(cmd, list))
        assert any(cmd[0] == "autostart" for cmd in calls)

    def test_adds_video_group(self, monkeypatch):
        monkeypatch.setattr(install.os, "geteuid", lambda: 0, raising=False)
        monkeypatch.setattr(install, "to_unix", lambda path: None)
        monkeypatch.setattr("sys.argv", ["install.py"])
        monkeypatch.setenv("SUDO_USER", "pi")

        def check_output(cmd, text=True):
            if cmd[0] == "getent":
                return "pi:x:1000:1000::/home/pi:/bin/bash\n"
            if cmd[0] == "id":
                return "pi adm\n"
            raise AssertionError(cmd)

        calls = []
        monkeypatch.setattr(install.subprocess, "check_output", check_output)
        monkeypatch.setattr(install.subprocess, "check_call", lambda cmd: calls.append(cmd))
        assert install.main() == 0
        assert ["usermod", "-aG", "video", "pi"] in calls
