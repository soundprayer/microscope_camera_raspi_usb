#!/usr/bin/env bash
# Prefer: sudo python3 install.py
# This file is only a wrapper. If you see "pipefail", the copy still has Windows CRLF.
# Fix with:  sed -i 's/\r$//' *.sh *.py
python3 "$(dirname "$0")/install.py" "$@"
