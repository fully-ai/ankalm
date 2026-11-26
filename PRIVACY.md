# Privacy Policy — AnkaLM

## 100% Local Data Residency
AnkaLM operates exclusively on your local computer:
- **No Cloud Inference:** All tokenization, embedding, RAG retrieval, training, and text generation run locally on your CPU/GPU.
- **No Telemetry:** AnkaLM contains zero analytics, tracking scripts, or outbound telemetry beacons.
- **No Conversation or Checkpoint Uploads:** All chat conversations are stored locally in `cache/conversations.sqlite3`, local RAG documents in `cache/rag/rag_index.sqlite3`, and model checkpoints in `checkpoints/`.
- **Offline Operation:** After initial dependency setup and dataset acquisition, the entire training, evaluation, and chat ecosystem operates completely offline.
