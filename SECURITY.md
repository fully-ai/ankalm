# Security Policy — AnkaLM

## 1. Localhost-Only Binding
AnkaLM binds its HTTP API and Web UI strictly to `127.0.0.1` (`localhost`). Binding to external interfaces (`0.0.0.0`) is blocked by `ankalm.utils.security.verify_localhost_binding`.

## 2. Path & Archive Traversal Protection
- All file operations validate target paths using `validate_safe_path(target_path, PROJECT_ROOT)`, rejecting `..` traversal attempts outside the project root.
- Archive extraction functions (`safe_extract_zip` and `safe_extract_tar`) inspect every member path prior to extraction and block zip-slip, tar-slip, and symlink escapes.

## 3. Secret Redaction & Zero Credentials
- No API keys, tokens, or credentials are stored in source code or configuration files.
- `redact_secrets()` automatically scrubs any pattern resembling an API key or secret before writing to logs.

## 4. Checkpoint Integrity
- Checkpoints are written atomically via temporary directories and verified before replacing existing slots.
- Checkpoints with `maturity_level == "random_init"` are blocked from being loaded as trained inference models.
