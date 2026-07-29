#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "Starting U.S. Infrastructure Stress Monitor..."
if command -v py >/dev/null 2>&1; then
  py server.py
elif command -v python >/dev/null 2>&1; then
  python server.py
elif command -v python3 >/dev/null 2>&1; then
  python3 server.py
else
  echo "Python was not found."
  exit 1
fi
