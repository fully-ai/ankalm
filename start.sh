#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$(pwd)"
if [ ! -f ".venv/bin/python" ]; then ./install.sh; fi
.venv/bin/python -m ankalm.cli serve --host 127.0.0.1 --port 8420 --open-browser
