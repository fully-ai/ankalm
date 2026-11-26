# Troubleshooting Guide — AnkaLM

## 1. "Another AnkaLM training process is already running"
AnkaLM prevents competing training jobs from corrupting checkpoints. Run `stop_training.bat` (or `ankalm stop-training`) to gracefully stop the active job.

## 2. "AnkaLM refuses to present randomly initialized weights as a trained model"
Checkpoints ending in `_init` (such as `micro_init` or `small_init`) are saved at step 0 (`maturity_level = random_init`) for weight-delta auditing. Select a trained checkpoint (`small_sft`, `small_final`, `latest`, `best`, `tiny_final`, or `micro_final`).

## 3. Port 8420 Already in Use
Pass a custom port when launching the server:
```bash
ankalm serve --host 127.0.0.1 --port 8421
```
