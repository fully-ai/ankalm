# Inference Runtime Guide — AnkaLM

## Capabilities (`ankalm/inference/engine.py`)
- **KV Cache Acceleration:** Autoregressive decoding caches key/value states (`past_key_values`) across all GQA layers, avoiding redundant prefix computation.
- **Sampling Controls:** Temperature scaling, Top-K filtering, Top-P (nucleus) sampling, Greedy decoding, Repetition penalty (default `1.15`), and deterministic random seeds.
- **Stop Sequences & EOS Handling:** Automatically stops on `<|eos|>`, `<|eot|>`, `<|user|>`, or user-supplied stop strings.
- **Random-Init Protection Guard:** `AnkaLMInferenceEngine.from_checkpoint` inspects `metadata.json` and raises a `RuntimeError` if `maturity_level == "random_init"` or `global_step == 0`.
