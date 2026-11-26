#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
echo "[AnkaLM] Preparing local Python environment..."

PY_CMD="python3"
if command -v python3.12 >/dev/null 2>&1; then
    PY_CMD="python3.12"
fi

if [ ! -f ".venv/bin/python" ]; then
    "$PY_CMD" -m venv .venv --system-site-packages || "$PY_CMD" -m venv .venv
fi

export PYTHONPATH="$(pwd)"
.venv/bin/python -m pip install -r requirements/base.txt
.venv/bin/python -m pip install -e . --no-build-isolation --no-deps || true
.venv/bin/python -m ankalm.cli doctor
echo "[AnkaLM] Installation complete!"
