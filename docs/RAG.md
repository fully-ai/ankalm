# Hybrid RAG & Local Document Collections — AnkaLM

## Architecture (`ankalm/chat/hybrid_rag.py`)
`HybridRAGEngine` combines lexical and learned neural retrieval 100% locally in SQLite (`cache/rag/hybrid_rag.sqlite3`):
1. **Okapi BM25 ($k_1 = 1.5, b = 0.75$):** Includes 5-character morphological prefix stems for Turkish agglutinative root matching.
2. **AnkaLM's Own Trained Bi-Encoder (`AnkaLM-BiEncoder-Embed-v1`):** A 128-dimensional neural bi-encoder trained from scratch via symmetric InfoNCE contrastive loss (`1.9355 -> 0.3121`) on AnkaLM's Turkish corpus (`checkpoints/rag_embedder/embedder.pt`).
3. **Hybrid Scoring Formula:** $\text{Score} = 0.55 \cdot \text{BM25}_{\text{norm}} + 0.35 \cdot \text{Cosine}_{\text{Neural}} + 0.10 \cdot \text{ExactKeywordBoost}$.
4. **Workspaces, Deduplication & Citations:** Supports workspace isolation (`general`, `pdf`, `spreadsheet`, `image`, `research`), SHA-256 duplicate detection, and verified `[Format: Filename | Location]` citations (`Recall@1 = 100%`, `MRR = 1.0`, `Citation Accuracy = 100%` on benchmark queries).
