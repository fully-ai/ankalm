# Controlled Tool Calling & Python Sandbox — AnkaLM

## Protocol (`ankalm/tools/registry.py`)
Every tool call is validated against a typed JSON schema (`ToolDefinition`), checked for `requires_permission` (`user_authorized=True` required for destructive operations), bounded by wall-clock timeouts, and returned as a structured `ToolCallResponse`.

## Built-In Local Tools
- `calculator`: Safe AST mathematical & statistical evaluator (`sqrt`, `mean`, `median`, `stdev`, `sin`, `cos`, `log`, `Fraction`, quadratic root solver).
- `python_sandbox`: Isolated subprocess (`python -I`) running in a temporary directory with static AST security enforcement blocking `socket`, `requests`, `subprocess`, `os`, `sys`, `open`, `eval`, `exec`, and `__dunder__` escapes, plus strict timeout enforcement.
- `unit_converter`: Deterministic physical and digital unit converter.
- `json_validator`: Validates and auto-repairs markdown-fenced or trailing-comma JSON/YAML against required schema keys.
- `chart_generator`: Renders local PNG bar charts to `exports/charts/`.
- `cross_document_comparator`: Multi-step workflow reading a PDF and an Excel workbook, comparing numeric totals, and generating a cited Markdown report.
- `file_search` & `datetime_tool`.
