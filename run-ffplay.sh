#!/usr/bin/env bash
# Lightweight fullscreen preview. If "pipefail" appears, run:
#   sed -i 's/\r$//' *.sh *.py
DEVICE="${CAMERA_DEVICE:-/dev/video0}"
WIDTH="${CAMERA_WIDTH:-1280}"
HEIGHT="${CAMERA_HEIGHT:-720}"
FPS="${CAMERA_FPS:-30}"
export DISPLAY="${DISPLAY:-:0}"
ffplay -fs -noborder -alwaysontop \
  -fflags nobuffer -flags low_delay -framedrop \
  -f v4l2 -input_format mjpeg -video_size "${WIDTH}x${HEIGHT}" -framerate "$FPS" \
  -i "$DEVICE"
