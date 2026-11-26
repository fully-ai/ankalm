# Installation Guide — AnkaLM

## Windows Installation
Run `install.bat` (Command Prompt) or `.\install.ps1` (PowerShell):
1. Detects Python 3.12 (`C:\Users\ubeyt\AppData\Local\Programs\Python\Python312\python.exe` or `py -3.12`).
2. Creates a local `.venv` virtual environment inside `AnkaLM\.venv` without modifying system Python or GPU drivers.
3. Installs the `ankalm` package and dependencies.
4. Runs `ankalm doctor` to verify hardware and compute backends.

## Linux Installation
```bash
chmod +x *.sh
./install.sh
```
Creates `.venv`, installs `requirements/base.txt`, registers `ankalm`, and executes `ankalm doctor`.
