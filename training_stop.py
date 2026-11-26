"""Gracefully stop AnkaLM overnight training supervisor after completing current step and saving checkpoint (Section 14.1 & 14.4)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ankalm.training.process_manager import TrainingProcessManager
from training_supervisor import read_supervisor_state


def stop_overnight_supervisor(timeout_sec: float = 25.0) -> dict:
    pm = TrainingProcessManager()
    pm.request_stop()
    t0 = time.time()
    while (time.time() - t0) < timeout_sec:
        sup = read_supervisor_state()
        live = pm.read_state()
        if not sup.get("supervisor_alive") and not live.get("process_alive"):
            return {
                "stopped_cleanly": True,
                "waited_sec": round(time.time() - t0, 2),
                "final_supervisor_state": sup,
            }
        time.sleep(0.5)
    return {
        "stopped_cleanly": False,
        "waited_sec": round(time.time() - t0, 2),
        "final_supervisor_state": read_supervisor_state(),
    }


if __name__ == "__main__":
    res = stop_overnight_supervisor()
    print(json.dumps(res, indent=2, ensure_ascii=False))
