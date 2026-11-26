# Localhost API Reference — AnkaLM

Default address: `http://127.0.0.1:8420` (Strictly localhost-only).

## Core & OpenAI-Compatible Endpoints
- `GET /health` & `GET /api/health` — Server status, loaded model, and inference readiness.
- `GET /v1/models` & `GET /api/models` — Lists all checkpoints with maturity level, step count, perplexity, and `inference_ready` boolean.
- `GET /api/models/{model_id}` — Full checkpoint metadata.
- `POST /api/models/load` & `POST /api/models/unload` — Load/unload a checkpoint with selectable precision (`fp32`, `bf16`, `fp16`, `int8`, `int4`).
- `POST /v1/chat/completions` — OpenAI-compatible chat completion supporting `stream: false` (JSON) and `stream: true` (SSE `text/event-stream`), plus `use_rag: true` for local document retrieval with citations.
- `POST /api/tokenize` & `POST /api/detokenize` — Encode Turkish text to token IDs and decode back.

## Chat Persistence, Local RAG & Training Control Center Endpoints
- `GET /api/conversations`, `POST /api/conversations`, `GET /api/conversations/{cid}`, `PATCH /api/conversations/{cid}`, `DELETE /api/conversations/{cid}`, `GET /api/conversations/search?q=...`
- `POST /api/rag/upload`, `GET /api/rag/documents`, `POST /api/rag/query`, `DELETE /api/rag/documents/{doc_id}`
- `GET /api/training/state`, `GET /api/training/logs`, `GET /api/training/configs`, `POST /api/training/start`, `POST /api/training/stop`, `POST /api/training/pause`, `POST /api/training/resume`, `POST /api/training/save-checkpoint`, `POST /api/training/evaluate`
