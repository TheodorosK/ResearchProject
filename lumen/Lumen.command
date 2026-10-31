#!/bin/bash
# Double-click this file in Finder to start Lumen on macOS.
# The first run sets everything up (a few minutes); after that it starts in seconds.
cd "$(dirname "$0")" || exit 1

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Lumen needs Python 3.10 or newer (the one built into macOS is too old)."
  echo "Install it from python.org (opening now) or with Homebrew: brew install python@3.12"
  echo "Then double-click Lumen.command again."
  open "https://www.python.org/downloads/macos/"
  read -n 1 -s -r -p "Press any key to close this window."
  exit 1
fi

if [ ! -x .venv/bin/python ]; then
  echo "✨ First run: setting up Lumen. This takes a few minutes…"
  "$PY" -m venv .venv || exit 1
  .venv/bin/python -m pip install --quiet --upgrade pip
fi
# Install or update the libraries whenever requirements.txt changes.
if ! cmp -s requirements.txt .venv/.installed-requirements; then
  if .venv/bin/python -m pip install --quiet -r requirements.txt; then
    cp requirements.txt .venv/.installed-requirements
  else
    echo "Installing Lumen's libraries failed (see above)."
    read -n 1 -s -r -p "Press any key to close this window."
    exit 1
  fi
fi

echo "Close this window or press Ctrl+C to stop Lumen."
exec .venv/bin/python -m lumen "$@"
