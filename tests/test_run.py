from __future__ import annotations

import os
from unittest.mock import MagicMock

import run


class TestWaitUntil:
    def test_true_when_present(self, tmp_path):
        path = tmp_path / "ready"
        path.write_text("ok")
        assert run.wait_until([path], timeout=0) is True

    def test_false_when_missing(self, tmp_path):
        assert run.wait_until([tmp_path / "missing"], timeout=0) is False

    def test_appears_during_wait(self, tmp_path, monkeypatch):
        path = tmp_path / "later"
        calls = {"n": 0}

        def fake_sleep(_seconds):
            calls["n"] += 1
            path.write_text("now")

        monkeypatch.setattr(run.time, "sleep", fake_sleep)
        assert run.wait_until([path], timeout=1) is True
        assert calls["n"] >= 1


class FakePath:
    def __init__(self, *parts):
        self._s = "/".join(str(p).replace("\\", "/").strip("/") for p in parts if str(p))
        if parts and str(parts[0]).startswith("/"):
            self._s = "/" + self._s

    def __truediv__(self, other):
        return FakePath(self._s, str(other))

    def exists(self):
        return self._s.endswith("wayland-0")

    def __str__(self):
        return self._s

    def __fspath__(self):
        return self._s

    @staticmethod
    def home():
        return FakePath("/home/pi")


class TestPrepareSession:
    def test_sets_wayland_and_display(self, monkeypatch):
        monkeypatch.setattr(run.os, "getuid", lambda: 1000, raising=False)
        monkeypatch.setattr(run, "wait_until", lambda *a, **k: True)
        monkeypatch.setattr(run.glob, "glob", lambda pattern: ["/dev/video0"])
        monkeypatch.setattr(run.shutil, "which", lambda name: None)
        monkeypatch.setattr(run, "Path", FakePath)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
        run.prepare_session()
        assert os.environ["WAYLAND_DISPLAY"] == "wayland-0"
        assert os.environ["DISPLAY"] == ":0"
        assert os.environ["XDG_RUNTIME_DIR"].endswith("/run/user/1000")

    def test_starts_xset_and_unclutter(self, monkeypatch):
        monkeypatch.setattr(run.os, "getuid", lambda: 1000, raising=False)
        monkeypatch.setattr(run, "wait_until", lambda *a, **k: True)
        monkeypatch.setattr(run.glob, "glob", lambda pattern: ["/dev/video0"])
        monkeypatch.setattr(run.shutil, "which", lambda name: "/usr/bin/" + name)
        monkeypatch.setattr(run, "Path", FakePath)
        runs = MagicMock()
        pops = MagicMock()
        monkeypatch.setattr(run.subprocess, "run", runs)
        monkeypatch.setattr(run.subprocess, "Popen", pops)
        os.environ["DISPLAY"] = ":0"
        run.prepare_session()
        assert runs.call_count == 3
        pops.assert_called_once()


class TestRunMain:
    def test_execs_camera_script(self, monkeypatch):
        monkeypatch.setattr(run, "prepare_session", lambda: None)
        execv = MagicMock()
        monkeypatch.setattr(run.os, "execv", execv)
        run.main(["--windowed"])
        args = execv.call_args[0]
        assert args[1][1].endswith("camera_fullscreen.py")
        assert args[1][2] == "--windowed"
