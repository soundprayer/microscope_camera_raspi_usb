#!/usr/bin/env bash
# Prefer: python3 camera_fullscreen.py
# If you see "pipefail", run:  sed -i 's/\r$//' *.sh *.py
python3 "$(dirname "$0")/run.py" "$@"
