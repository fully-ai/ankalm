#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
if [ ! -f ".venv/bin/python" ]; then ./install.sh; fi
.venv/bin/python -m ankalm.cli stop-training
