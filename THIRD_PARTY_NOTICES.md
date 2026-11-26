# Third-Party Notices — AnkaLM

AnkaLM is an independently implemented and trained language model. It uses the following open-source software libraries for tensor computation, tokenization data structures, HTTP serving, and testing:

1. **PyTorch (`torch`)** — BSD-3-Clause License (Tensor computation, autograd, and neural network primitives)
2. **HuggingFace Tokenizers (`tokenizers`)** — Apache-2.0 License (Rust-backed BPE/Unigram subword training structures)
3. **Safetensors (`safetensors`)** — Apache-2.0 License (Safe tensor serialization)
4. **FastAPI (`fastapi`) & Uvicorn (`uvicorn`)** — MIT / BSD-3-Clause License (Localhost-only HTTP API & Web UI serving)
5. **Typer (`typer`) & Rich (`rich`)** — MIT License (Command-line interface and terminal formatting)
6. **NumPy (`numpy`) & Psutil (`psutil`)** — BSD-3-Clause License (Numerical arrays and hardware process inspection)
7. **PyYAML (`PyYAML`) & Requests (`requests`)** — MIT / Apache-2.0 License (YAML parsing and Wikipedia API retrieval)
8. **PyPDF (`pypdf`)** — BSD-3-Clause License (Local PDF text parsing for file attachments)
9. **Pytest (`pytest`)** — MIT License (Automated test execution)

## Training Data Attribution
- **Turkish Wikipedia Articles (`datasets/raw/tr_wikipedia_cc_by_sa.jsonl`)**: Licensed under Creative Commons Attribution-ShareAlike 4.0 International (`CC-BY-SA-4.0`), retrieved via the MediaWiki Action API (`https://tr.wikipedia.org`).
- **AnkaLM Foundational & Instruction Turkish Corpora**: Licensed under `CC-BY-4.0` and `CC0-1.0`.
