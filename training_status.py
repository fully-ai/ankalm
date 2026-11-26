"""Inspect live status of AnkaLM overnight training supervisor and checkpoints (Section 14.1)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ankalm.storage.checkpoint import CheckpointManager
from ankalm.training.process_manager import TrainingProcessManager
from training_supervisor import read_supervisor_state


def get_comprehensive_training_status() -> dict:
    sup = read_supervisor_state()
    live = TrainingProcessManager().read_state()
    ckpt_mgr = CheckpointManager()
    ckpts = {}
    for slot in ("small_final", "small_sft", "moe_final", "edge_final", "vision_final", "latest"):
        p = PROJECT_ROOT / "checkpoints" / slot / "metadata.json"
        if p.exists():
            try:
                m = json.loads(p.read_text(encoding="utf-8"))
                ckpts[slot] = {
                    "model_name": m.get("model_name"),
                    "step": m.get("global_step"),
                    "tokens": m.get("processed_tokens"),
                    "train_loss": m.get("train_loss"),
                    "val_loss": m.get("val_loss"),
                    "val_ppl": m.get("val_perplexity"),
                    "maturity": m.get("maturity_level"),
                }
            except Exception:
                pass
    return {
        "overnight_supervisor": sup,
        "live_worker_state": live,
        "verified_checkpoints": ckpts,
    }


if __name__ == "__main__":
    status = get_comprehensive_training_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))
