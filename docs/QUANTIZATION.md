# Quantization & Deployment Optimization — AnkaLM

## Supported Modes (`ankalm quantize`)
AnkaLM provides reproducible weight quantization in `ankalm/quantization/quantizer.py` without modifying original full-precision checkpoints:
- **FP32:** `64.03 MB` (`1.00x` compression, `247.5 tok/s`, val perplexity `15.48`)
- **BF16:** `32.01 MB` (`2.00x` compression, `232.5 tok/s`, val perplexity `15.49`)
- **FP16:** `32.01 MB` (`2.00x` compression, **`332.6 tok/s`**, val perplexity `15.48`)
- **INT8 (`Int8WeightOnlyLinear`):** Per-channel symmetric 8-bit quantization — `28.13 MB` (`2.28x` compression, `167.1 tok/s`, val perplexity `15.49`)
- **INT4 (`Int4WeightOnlyLinear`):** Group-wise (`group_size=32`) packed 4-bit nibble quantization — `22.78 MB` (`2.81x` compression, `61.9 tok/s`, val perplexity `15.48`)

All quantized artifacts are saved under `exports/quantized_<mode>/quantized_model.pt` and summarized in `exports/quantization_report.json`.
