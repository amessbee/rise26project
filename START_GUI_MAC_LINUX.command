#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/GUI"

echo
echo "U.S. INFRASTRUCTURE STRESS MONITOR"
echo "Starting at http://127.0.0.1:8081"
echo "Press Ctrl+C to stop the server."
echo

if command -v python3 >/dev/null 2>&1; then
    PORT=8081 python3 server.py
elif command -v python >/dev/null 2>&1; then
    PORT=8081 python server.py
else
    echo "Python 3 was not found."
    exit 1
fi
