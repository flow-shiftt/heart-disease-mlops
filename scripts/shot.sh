#!/usr/bin/env bash
# Headless-Chrome screenshot helper: scripts/shot.sh <url> <out.png> [width] [height] [wait_ms]
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --no-first-run \
  --user-data-dir="${TMPDIR:-/tmp}/cardiorisk-chrome" \
  --window-size="${3:-1600},${4:-1000}" --timeout="${5:-8000}" \
  --screenshot="$2" "$1" >/dev/null 2>&1
echo "saved $2"
