# Dataset Guide — AnkaLM Turkish Dataset Engine

## Supported Formats
The streaming dataset loader (`stream_raw_dataset_file`) natively ingests:
- `.txt` (Plain text passages)
- `.md` (Markdown technical & programming documentation)
- `.jsonl` (Line-delimited JSON and multi-turn structured conversations)
- `.json` (Structured JSON arrays)
- `.csv` (Question-Answer & educational tables)
- `.parquet` (Columnar Apache Parquet tables)

## Cleaning & Deduplication Pipeline
1. **UTF-8 Validation & NFC Normalization:** Rejects invalid UTF-8 sequences and normalizes combining characters while preserving Turkish `İ`/`ı`.
2. **HTML & Boilerplate Stripping:** Removes HTML tags, cookie banners, copyright footers, and spam patterns while keeping code blocks intact.
3. **Turkish Language Identification:** Scores documents using Turkish character density, stopword frequency, and agglutinative suffix patterns (`-lar/-ler`, `-dan/-den`, `-lık/-lik`, etc.).
4. **Exact & Near-Duplicate Filtering:** `DeduplicationEngine` filters exact SHA-256 duplicates and near-duplicates via 64-bit Blake2b word/character shingle Jaccard similarity (threshold `0.85`).
5. **Cross-Split Leakage Prevention:** Held-out `validation` (`50` docs) and `evaluation` (`50` docs) splits are indexed in a leakage guard before finalizing `train` (`602` docs), guaranteeing `zero_leakage_verified = True`.
