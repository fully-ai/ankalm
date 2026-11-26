# Quickstart Guide — AnkaLM

## 1. Verify Environment & Hardware
```bat
install.bat
doctor.bat
```

## 2. Chat with the Trained Turkish Model (CLI)
```bat
chat.bat --checkpoint small_sft --prompt "Türkiye'nin başkenti neresidir?"
```

## 3. Launch the Local Web Chat Application & Training Control Center
```bat
start.bat
```
Opens `http://127.0.0.1:8420` with:
- **💬 Sohbet:** Local Turkish AI Chat with dark/light themes, SQLite history, streaming, and Local RAG attachments.
- **📊 Eğitim Merkezi:** Live training telemetry, loss curves, and start/stop/pause/resume controls.
- **🧪 Değerlendirme:** 16-category Turkish Evaluation Lab & Hardware report.

## 4. Continue Training from the Latest Checkpoint
```bat
resume.bat
```
