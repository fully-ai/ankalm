"""Resume AnkaLM training from the latest verified checkpoint via the detached overnight supervisor (Section 14.1 & 14.4)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ankalm.storage.checkpoint import CheckpointManager
from training_supervisor import launch_detached_supervisor, read_supervisor_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume AnkaLM Overnight Training")
    parser.add_argument("--config", default="configs/small.yaml")
    parser.add_argument("--chunk-steps", type=int, default=20)
    parser.add_argument("--max-total-steps", type=int, default=5000)
    args = parser.parse_args()

    ckpt_mgr = CheckpointManager()
    meta, _ = ckpt_mgr.load_checkpoint("small_final" if "small" in args.config else "latest", restore_rng=False)
    pid = launch_detached_supervisor(args.config, args.chunk_steps, args.max_total_steps)
    time.sleep(1.2)
    state = read_supervisor_state()
    print(
        json.dumps(
            {
                "resumed_checkpoint": meta.checkpoint_name,
                "resumed_from_step": meta.global_step,
                "resumed_from_tokens": meta.processed_tokens,
                "supervisor_pid": pid,
                "supervisor_alive": state.get("supervisor_alive", True),
                "log_file": state.get("log_file"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
