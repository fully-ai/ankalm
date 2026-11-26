# Training Recovery & Checkpointing — AnkaLM

## Atomic Checkpoint Architecture
Every checkpoint directory in `checkpoints/` contains:
- `metadata.json`: Global step, epoch, processed tokens, train/val loss, perplexity, initial & current parameter SHA-256 fingerprints, L2 weight delta from random initialization, and maturity level.
- `config.yaml`: Full model, training, and storage configuration.
- `model.safetensors`: Safe, fast model weight serialization.
- `training_state.pt`: Complete PyTorch model state dict, AdamW optimizer state dict, CosineWarmupScheduler state dict, GradScaler state, dataset sampling cursor/order, and Python/NumPy/PyTorch RNG states.

Checkpoints are written to a temporary `.tmp_*` directory, verified for non-zero file integrity, and atomically renamed. `latest` rotates to `previous` before updating, and `best` tracks the lowest validation loss.

## Resuming Training
To resume training after Ctrl+C, `stop_training.bat`, or a system reboot:
```bat
resume.bat
```
Or via CLI:
```bash
ankalm train --config configs/small.yaml --resume latest
```
