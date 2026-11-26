#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
if [ ! -f ".venv/bin/python" ]; then ./install.sh; fi
echo "[AnkaLM] Starting Persistent Overnight Training Supervisor..."
.venv/bin/python training_supervisor.py --detach "$@"
sleep 1
.venv/bin/python training_status.py
