from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pygame
import pytest

import camera_fullscreen as cam


def bgr(h: int, w: int, color=(0, 0, 255)) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = color
    return frame


class FakeCapture:
    def __init__(self, width=64, height=48, opened=True, frame=None, fail_reads=0) -> None:
        self.width = width
        self.height = height
        self.opened = opened
        self.released = False
        self.sets: list[tuple] = []
        self.reads = 0
        self.fail_reads = fail_reads
        self.frame = frame if frame is not None else bgr(height, width)

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def read(self):
        self.reads += 1
        if self.released or self.reads <= self.fail_reads:
            return False, None
        return True, self.frame.copy()

    def release(self) -> None:
        self.released = True

    def get(self, prop: int) -> float:
        mapping = {
            cv2.CAP_PROP_FRAME_WIDTH: float(self.width),
            cv2.CAP_PROP_FRAME_HEIGHT: float(self.height),
            cv2.CAP_PROP_FPS: 30.0,
        }
        return mapping.get(prop, 0.0)

    def set(self, prop: int, value) -> bool:
        self.sets.append((prop, value))
        return True


class TestParseArgs:
    def test_defaults(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["cam"])
        monkeypatch.delenv("CAMERA_DEVICE", raising=False)
        monkeypatch.delenv("CAMERA_WIDTH", raising=False)
        monkeypatch.delenv("CAMERA_HEIGHT", raising=False)
        monkeypatch.delenv("CAMERA_FPS", raising=False)
        monkeypatch.delenv("CAMERA_FIT", raising=False)
        args = cam.parse_args()
        assert args.device == ""
        assert args.width == 1280
        assert args.height == 720
        assert args.fps == 30
        assert args.fit == "cover"
        assert args.windowed is False
        assert args.no_mjpeg is False

    def test_flags_and_env(self, monkeypatch):
        monkeypatch.setenv("CAMERA_DEVICE", "/dev/video2")
        monkeypatch.setenv("CAMERA_WIDTH", "640")
        monkeypatch.setenv("CAMERA_HEIGHT", "480")
        monkeypatch.setenv("CAMERA_FPS", "15")
        monkeypatch.setenv("CAMERA_FIT", "contain")
        monkeypatch.setattr("sys.argv", ["cam", "--windowed", "--no-mjpeg", "--window", "Test"])
        args = cam.parse_args()
        assert args.device == "/dev/video2"
        assert args.width == 640
        assert args.height == 480
        assert args.fps == 15
        assert args.fit == "contain"
        assert args.windowed is True
        assert args.no_mjpeg is True
        assert args.window == "Test"


class TestDetectScreenSize:
    def test_xrandr_star_mode(self, monkeypatch):
        monkeypatch.setattr(
            cam.subprocess,
            "check_output",
            lambda *a, **k: "HDMI-1 connected primary 1280x720+0+0\n   1920x1080  60.00\n   1280x720  60.00*\n",
        )
        assert cam.detect_screen_size() == (1280, 720)

    def test_xrandr_connected_size(self, monkeypatch):
        monkeypatch.setattr(
            cam.subprocess,
            "check_output",
            lambda *a, **k: "HDMI-1 connected primary 1024x768+0+0 (normal)\n",
        )
        assert cam.detect_screen_size() == (1024, 768)

    def test_sysfs_fallback(self, monkeypatch):
        monkeypatch.setattr(
            cam.subprocess, "check_output", MagicMock(side_effect=OSError("no xrandr"))
        )
        fake_tk = MagicMock()
        fake_tk.Tk.side_effect = RuntimeError("no tk")
        monkeypatch.setitem(sys.modules, "tkinter", fake_tk)
        monkeypatch.setattr(
            Path,
            "read_text",
            lambda self, *a, **k: "800,600" if "virtual_size" in str(self) else (_ for _ in ()).throw(OSError()),
        )
        assert cam.detect_screen_size() == (800, 600)

    def test_default_when_all_fail(self, monkeypatch):
        monkeypatch.setattr(
            cam.subprocess, "check_output", MagicMock(side_effect=OSError("no xrandr"))
        )
        fake_tk = MagicMock()
        fake_tk.Tk.side_effect = RuntimeError("no tk")
        monkeypatch.setitem(sys.modules, "tkinter", fake_tk)
        monkeypatch.setattr(Path, "read_text", MagicMock(side_effect=OSError("no fb")))
        assert cam.detect_screen_size() == (1920, 1080)


class TestDeviceHelpers:
    def test_parse_device_index(self):
        assert cam.parse_device_index("0") == 0
        assert cam.parse_device_index(" 12 ") == 12
        assert cam.parse_device_index("/dev/video3") == 3
        assert cam.parse_device_index("/dev/video") is None
        assert cam.parse_device_index("cam") is None

    def test_video_nodes(self, monkeypatch):
        monkeypatch.setattr(
            cam.glob, "glob", lambda pattern: ["/dev/video2", "/dev/video10", "/dev/video0", "/dev/video-x"]
        )
        assert cam.video_nodes() == ["/dev/video0", "/dev/video2", "/dev/video10"]

    def test_has_capture_formats_missing_path(self, tmp_path):
        assert cam.has_capture_formats(str(tmp_path / "nope")) is False

    def test_has_capture_formats_without_v4l2(self, monkeypatch, tmp_path):
        node = tmp_path / "video0"
        node.write_text("")
        monkeypatch.setattr(cam.shutil, "which", lambda name: None)
        monkeypatch.setattr(cam.os.path, "exists", lambda p: True)
        assert cam.has_capture_formats("/dev/video0") is True
        assert cam.has_capture_formats("/dev/notvideo") is False

    def test_has_capture_formats_parses_v4l2(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cam.os.path, "exists", lambda p: True)
        monkeypatch.setattr(cam.shutil, "which", lambda name: "/usr/bin/v4l2-ctl")
        monkeypatch.setattr(
            cam.subprocess,
            "check_output",
            lambda *a, **k: "ioctl: VIDIOC_ENUM_FMT\nType: Video Capture\n\t[0]: 'MJPG'\n",
        )
        assert cam.has_capture_formats("/dev/video0") is True
        monkeypatch.setattr(
            cam.subprocess,
            "check_output",
            lambda *a, **k: "Type: Video Output\n",
        )
        assert cam.has_capture_formats("/dev/video0") is False
        monkeypatch.setattr(
            cam.subprocess,
            "check_output",
            MagicMock(side_effect=subprocess.TimeoutExpired(cmd="v4l2-ctl", timeout=1.5)),
        )
        assert cam.has_capture_formats("/dev/video0") is False

    def test_list_video_devices(self, monkeypatch):
        listing = (
            "USB Camera: USB Camera (usb-0000:01:00.0-2):\n"
            "\t/dev/video0\n"
            "\t/dev/video1\n"
            "\n"
            "bcm2835-codec-decode (platform):\n"
            "\t/dev/video31\n"
        )
        monkeypatch.setattr(cam.shutil, "which", lambda name: "/usr/bin/v4l2-ctl")
        monkeypatch.setattr(cam.subprocess, "check_output", lambda *a, **k: listing)
        monkeypatch.setattr(cam, "has_capture_formats", lambda path: path == "/dev/video0")
        items = cam.list_video_devices()
        by_path = {item["path"]: item for item in items}
        assert by_path["/dev/video0"]["usb"] is True
        assert by_path["/dev/video0"]["capture"] is True
        assert by_path["/dev/video31"]["skip"] is True

    def test_list_video_devices_fallback_nodes(self, monkeypatch):
        monkeypatch.setattr(cam.shutil, "which", lambda name: None)
        monkeypatch.setattr(cam, "video_nodes", lambda: ["/dev/video0"])
        monkeypatch.setattr(cam, "has_capture_formats", lambda path: True)
        items = cam.list_video_devices()
        assert items[0]["path"] == "/dev/video0"
        assert items[0]["usb"] is False

    def test_permission_hint(self, monkeypatch):
        monkeypatch.setattr(cam.os.path, "exists", lambda p: p == "/dev/video0")
        monkeypatch.setattr(cam.os, "access", lambda p, mode: False)
        hint = cam.permission_hint(["/dev/video0"])
        assert "No access to /dev/video0" in hint
        monkeypatch.setattr(cam.os.path, "exists", lambda p: False)
        assert cam.permission_hint(["/dev/video0"]) == ""

    def test_candidate_devices_explicit(self):
        assert cam.candidate_devices("2") == ["/dev/video2"]
        assert cam.candidate_devices("/dev/video4") == ["/dev/video4"]
        assert cam.candidate_devices("other") == ["other"]

    def test_candidate_devices_prefers_usb(self, monkeypatch):
        monkeypatch.setattr(
            cam,
            "list_video_devices",
            lambda: [
                {"path": "/dev/video2", "capture": True, "skip": False, "usb": False, "name": "plat"},
                {"path": "/dev/video0", "capture": True, "skip": False, "usb": True, "name": "USB"},
                {"path": "/dev/video31", "capture": False, "skip": True, "usb": False, "name": "codec"},
            ],
        )
        assert cam.candidate_devices("") == ["/dev/video0", "/dev/video2"]

    def test_candidate_devices_fallback(self, monkeypatch):
        monkeypatch.setattr(cam, "list_video_devices", lambda: [])
        monkeypatch.setattr(cam, "video_nodes", lambda: [])
        assert cam.candidate_devices("") == ["0"]


class TestCapture:
    def test_configure_capture_mjpeg_and_size(self):
        cap = FakeCapture()
        cam.configure_capture(cap, 1280, 720, 30, True, True)
        props = {prop for prop, _value in cap.sets}
        assert cv2.CAP_PROP_BUFFERSIZE in props
        assert cv2.CAP_PROP_FOURCC in props
        assert cv2.CAP_PROP_FRAME_WIDTH in props
        assert cv2.CAP_PROP_FPS in props

    def test_configure_capture_skip_size(self):
        cap = FakeCapture()
        cam.configure_capture(cap, 1280, 720, 30, False, False)
        props = {prop for prop, _value in cap.sets}
        assert cv2.CAP_PROP_FOURCC not in props
        assert cv2.CAP_PROP_FRAME_WIDTH not in props

    def test_open_capture_invalid_device(self):
        assert cam.open_capture("nope", 640, 480, 30, True) is None

    def test_open_capture_rejects_non_capture(self, monkeypatch):
        monkeypatch.setattr(cam.os.path, "exists", lambda p: True)
        monkeypatch.setattr(cam, "has_capture_formats", lambda p: False)
        assert cam.open_capture("/dev/video0", 640, 480, 30, True) is None

    def test_open_capture_success(self, monkeypatch):
        fake = FakeCapture()

        def factory(index, api):
            assert index == 0
            assert api == cv2.CAP_V4L2
            return fake

        monkeypatch.setattr(cam.os.path, "exists", lambda p: False)
        monkeypatch.setattr(cam.cv2, "VideoCapture", factory)
        assert cam.open_capture("/dev/video0", 64, 48, 30, True, check_formats=False) is fake

    def test_open_capture_retries_then_none(self, monkeypatch):
        class Dead:
            def isOpened(self):
                return False

            def release(self):
                pass

        monkeypatch.setattr(cam.os.path, "exists", lambda p: False)
        monkeypatch.setattr(cam.cv2, "VideoCapture", lambda *a, **k: Dead())
        assert cam.open_capture("1", 64, 48, 30, True, check_formats=False) is None

    def test_find_camera_returns_first(self, monkeypatch):
        cap = FakeCapture()
        monkeypatch.setattr(cam, "open_capture", lambda *a, **k: cap)
        found, device = cam.find_camera(["/dev/video2"], 64, 48, 30, True)
        assert found is cap
        assert device == "/dev/video2"

    def test_find_camera_raises(self, monkeypatch):
        monkeypatch.setattr(cam, "open_capture", lambda *a, **k: None)
        monkeypatch.setattr(cam, "permission_hint", lambda paths: "")
        with pytest.raises(RuntimeError, match="No working camera"):
            cam.find_camera(["/dev/video0"], 64, 48, 30, True)


class TestFrameGrabber:
    def test_pulls_new_frames_and_stop(self):
        grabber = cam.FrameGrabber(FakeCapture(32, 24))
        seq, frame = (0, None)
        for _ in range(50):
            seq, frame = grabber.get_if_new(seq)
            if frame is not None:
                break
            __import__("time").sleep(0.01)
        assert frame is not None
        assert frame.shape == (24, 32, 3)
        same_seq, none = grabber.get_if_new(seq)
        assert none is None
        dest = np.zeros_like(frame)
        newer, reused = grabber.get_if_new(-1, dest)
        assert reused is dest
        assert grabber.stop() is True


class TestFrameScaler:
    def test_stretch_and_same_size(self):
        scaler = cam.FrameScaler()
        src = bgr(40, 80, (10, 20, 30))
        out = scaler.apply(src, 80, 40, "stretch")
        assert out.shape == (40, 80, 3)
        np.testing.assert_array_equal(out, src)

    def test_cover_fills_screen(self):
        scaler = cam.FrameScaler()
        src = bgr(20, 40, (1, 2, 3))
        out = scaler.apply(src, 30, 30, "cover")
        assert out.shape == (30, 30, 3)
        assert np.any(out)

    def test_contain_letterbox(self):
        scaler = cam.FrameScaler()
        src = bgr(10, 40, (255, 0, 0))
        out = scaler.apply(src, 40, 40, "contain")
        assert out.shape == (40, 40, 3)
        assert np.all(out[0] == 0) or np.all(out[-1] == 0)

    def test_empty_frame_and_fast(self):
        scaler = cam.FrameScaler()
        empty = np.zeros((0, 10, 3), dtype=np.uint8)
        out = scaler.apply(empty, 16, 16, "cover")
        assert out.shape == (16, 16, 3)
        src = bgr(80, 80)
        fast = scaler.apply(src, 20, 20, "stretch", fast=True)
        assert fast.shape == (20, 20, 3)


class TestOverlayMixer:
    @pytest.mark.parametrize("mode", list(cam.BLEND_MODES))
    def test_all_modes(self, mode):
        mixer = cam.OverlayMixer()
        base = bgr(270, 480, (10, 10, 10))
        over = bgr(270, 480, (200, 40, 40))
        before = base.copy()
        mixer.blend(base, over, mode, 0.5)
        assert base.shape == before.shape
        assert not np.array_equal(base, before)

    def test_alpha_clamped_and_resize(self):
        mixer = cam.OverlayMixer()
        base = bgr(40, 40, (0, 0, 0))
        over = bgr(10, 10, (255, 255, 255))
        mixer.blend(base, over, "alpha", 5.0)
        assert base.mean() > 0

    def test_output_reuses_buffer(self):
        mixer = cam.OverlayMixer()
        a = mixer.output((8, 8, 3))
        b = mixer.output((8, 8, 3))
        assert a is b
        c = mixer.output((12, 12, 3))
        assert c.shape == (12, 12, 3)


class TestScreen:
    def test_windowed_show_and_close(self):
        screen = cam.Screen("t", fullscreen=False)
        assert screen.size[0] > 0
        screen.show(bgr(screen.size[1], screen.size[0]))
        screen.set_fullscreen(False)
        screen.close()


class TestPollActions:
    def _run(self, events, mods=0, monkeypatch=None):
        monkeypatch.setattr(cam.pygame.event, "get", lambda: events)
        monkeypatch.setattr(cam.pygame.key, "get_mods", lambda: mods)
        return cam.poll_actions()

    def test_quit_events(self, monkeypatch):
        assert self._run([pygame.event.Event(pygame.QUIT)], monkeypatch=monkeypatch) == ["quit"]
        assert self._run([pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)], monkeypatch=monkeypatch) == ["quit"]
        assert self._run(
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_F4)],
            mods=pygame.KMOD_ALT,
            monkeypatch=monkeypatch,
        ) == ["quit"]

    @pytest.mark.parametrize(
        "key_name,expected",
        [
            ("K_f", "fullscreen"),
            ("K_F11", "fullscreen"),
            ("K_s", "snap"),
            ("K_r", "rotate"),
            ("K_c", "fit"),
            ("K_h", "help"),
            ("K_l", "list"),
            ("K_m", "next"),
            ("K_o", "overlay"),
            ("K_x", "overlay_off"),
            ("K_b", "blend"),
            ("K_MINUS", "alpha_down"),
            ("K_EQUALS", "alpha_up"),
            ("K_LEFTBRACKET", "res_down"),
            ("K_RIGHTBRACKET", "res_up"),
        ],
    )
    def test_keys(self, monkeypatch, key_name, expected):
        event = pygame.event.Event(pygame.KEYDOWN, key=getattr(pygame, key_name))
        assert self._run([event], monkeypatch=monkeypatch) == [expected]

    def test_shift_o_turns_overlay_off(self, monkeypatch):
        event = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_o)
        assert self._run([event], mods=pygame.KMOD_SHIFT, monkeypatch=monkeypatch) == ["overlay_off"]


class TestImageHelpers:
    def test_rotate_frame(self):
        src = np.zeros((4, 8, 3), dtype=np.uint8)
        src[0, 0] = (1, 2, 3)
        assert cam.rotate_frame(src, 0).shape == (4, 8, 3)
        assert cam.rotate_frame(src, 1).shape == (8, 4, 3)
        assert cam.rotate_frame(src, 2).shape == (4, 8, 3)
        assert cam.rotate_frame(src, 3).shape == (8, 4, 3)
        assert cam.rotate_frame(src, 4).shape == (4, 8, 3)

    def test_save_snapshot(self, tmp_path):
        path = cam.save_snapshot(bgr(8, 8), str(tmp_path))
        saved = Path(path)
        assert saved.exists()
        assert saved.suffix == ".jpg"
        assert saved.parent == tmp_path

    def test_next_preset(self):
        assert cam.next_preset(1280, 720, 1) == (1024, 768)
        assert cam.next_preset(1280, 720, -1) == (1920, 1080)
        assert cam.next_preset(640, 480, 1) == (1920, 1080)
        assert cam.next_preset(123, 45, 1) == cam.PRESETS[0]

    def test_draw_text_and_camera_list(self):
        img = bgr(240, 320, (8, 8, 8))
        cam.draw_text_block(img, ["hello"], "Title")
        assert img.mean() != 8
        blank = bgr(240, 320, (8, 8, 8))
        cam.overlay_camera_list(
            blank,
            [
                {"path": "/dev/video0", "name": "USB Cam", "usb": True, "capture": True, "skip": False},
                {"path": "/dev/video1", "name": "USB Cam", "usb": True, "capture": True, "skip": False},
                {"path": "/dev/video31", "name": "bcm2835", "usb": False, "capture": False, "skip": True},
            ],
            "/dev/video0",
            "/dev/video1",
        )
        cam.overlay_camera_list(bgr(240, 320), [], "")


class TestMain:
    def test_missing_pygame(self, monkeypatch):
        monkeypatch.setattr(cam, "pygame", None)
        assert cam.main() == 1

    def test_no_camera(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["cam", "--windowed"])
        monkeypatch.setattr(cam, "candidate_devices", lambda explicit: ["/dev/video0"])
        monkeypatch.setattr(
            cam, "find_camera", MagicMock(side_effect=RuntimeError("No working camera found."))
        )
        assert cam.main() == 1

    def test_quit_loop(self, monkeypatch):
        cap = FakeCapture(64, 48)
        monkeypatch.setattr("sys.argv", ["cam", "--windowed"])
        monkeypatch.setattr(cam, "candidate_devices", lambda explicit: ["/dev/video0"])
        monkeypatch.setattr(cam, "find_camera", lambda *a, **k: (cap, "/dev/video0"))
        monkeypatch.setattr(cam, "poll_actions", lambda: ["quit"])
        assert cam.main() == 0
        assert cap.released is True
