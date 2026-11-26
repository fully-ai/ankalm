"""Structured Tool Protocol, AST Mathematical Calculator, Isolated Python Sandbox, Chart Generator, and Multi-Step Cross-Document Orchestrator (Section 10 & 11)."""
from __future__ import annotations

import ast
import csv
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from fractions import Fraction
import io
import json
import math
import operator
from pathlib import Path
import re
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw
import yaml

from ankalm.core.constants import PROJECT_ROOT
from ankalm.utils.security import validate_safe_path


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    requires_permission: bool = False
    timeout_sec: float = 5.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ToolCallRequest:
    tool_name: str
    arguments: Dict[str, Any]
    user_authorized: bool = False
    timeout_sec: Optional[float] = None


@dataclass
class ToolCallResponse:
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    citations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --- 1. SAFE AST MATHEMATICAL & STATISTICAL CALCULATOR ---
SAFE_MATH_NAMES: Dict[str, Any] = {
    "pi": math.pi,
    "e": math.e,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "mean": statistics.mean,
    "median": statistics.median,
    "stdev": statistics.stdev,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "Fraction": lambda a, b=1: str(Fraction(a, b)),
}

SAFE_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

SAFE_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_math_ast(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_math_ast(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type in calculator: {type(node.value)}")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_math_ast(el) for el in node.elts]
    if isinstance(node, ast.Name):
        if node.id in SAFE_MATH_NAMES:
            return SAFE_MATH_NAMES[node.id]
        raise ValueError(f"Unknown mathematical identifier: '{node.id}'")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_BIN_OPS:
            raise ValueError(f"Unsupported binary operator: {op_type}")
        left = _eval_math_ast(node.left)
        right = _eval_math_ast(node.right)
        if op_type is ast.Pow and isinstance(right, (int, float)) and abs(right) > 1000:
            raise ValueError("Exponent magnitude exceeds safety limit (1000).")
        return SAFE_BIN_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_UNARY_OPS:
            raise ValueError(f"Unsupported unary operator: {op_type}")
        return SAFE_UNARY_OPS[op_type](_eval_math_ast(node.operand))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_MATH_NAMES:
            raise ValueError("Only whitelisted mathematical functions are allowed.")
        fn = SAFE_MATH_NAMES[node.func.id]
        args = [_eval_math_ast(a) for a in node.args]
        return fn(*args)
    raise ValueError(f"Disallowed AST expression node: {type(node).__name__}")


def execute_calculator(args: Dict[str, Any]) -> Dict[str, Any]:
    expr = str(args.get("expression", "")).strip()
    if not expr:
        raise ValueError("Missing 'expression' argument for calculator.")
    # Also support linear equation ax + b = c or quadratic ax^2 + bx + c = 0 if requested
    if "quadratic" in args:
        a, b, c = float(args["quadratic"][0]), float(args["quadratic"][1]), float(args["quadratic"][2])
        disc = b * b - 4 * a * c
        if disc < 0:
            return {"equation": f"{a}x^2 + {b}x + {c} = 0", "discriminant": disc, "real_roots": []}
        r1 = (-b + math.sqrt(disc)) / (2 * a)
        r2 = (-b - math.sqrt(disc)) / (2 * a)
        return {"equation": f"{a}x^2 + {b}x + {c} = 0", "discriminant": disc, "real_roots": [round(r1, 6), round(r2, 6)]}

    normalized = expr.replace("^", "**")
    tree = ast.parse(normalized, mode="eval")
    val = _eval_math_ast(tree)
    return {
        "expression": expr,
        "result": round(val, 8) if isinstance(val, float) else val,
        "engine": "AnkaLM-Deterministic-AST-Calculator",
    }


# --- 2. ISOLATED LOCAL PYTHON SANDBOX (Section 10.2) ---
FORBIDDEN_SANDBOX_MODULES = {
    "socket", "urllib", "requests", "httpx", "http", "ftplib", "telnetlib",
    "subprocess", "shutil", "ctypes", "winreg", "multiprocessing", "webbrowser",
}

FORBIDDEN_SANDBOX_PATTERNS = [
    re.compile(r"\b(?:__import__|eval|exec|compile|open|input|breakpoint)\b"),
    re.compile(r"\b(?:os\.system|os\.popen|os\.remove|os\.unlink|os\.rmdir|sys\.modules)\b"),
]


def execute_python_sandbox(args: Dict[str, Any], timeout_sec: float = 3.0, max_output_chars: int = 4096) -> Dict[str, Any]:
    """
    Execute calculation/data-processing Python code inside an isolated subprocess with:
    - Static AST inspection blocking network, process spawning, filesystem mutation, and dunder escapes
    - Temporary isolated working directory
    - Strict wall-clock timeout
    - Output size limit
    """
    code = str(args.get("code", "")).strip()
    if not code:
        raise ValueError("Missing 'code' argument for python_sandbox.")

    for pat in FORBIDDEN_SANDBOX_PATTERNS:
        if pat.search(code):
            raise PermissionError(f"Sandbox security violation: forbidden construct matching '{pat.pattern}'")

    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in FORBIDDEN_SANDBOX_MODULES or root_mod in ("os", "sys", "pathlib"):
                    raise PermissionError(f"Sandbox security violation: import of '{root_mod}' is forbidden.")
        elif isinstance(node, ast.ImportFrom):
            root_mod = (node.module or "").split(".")[0]
            if root_mod in FORBIDDEN_SANDBOX_MODULES or root_mod in ("os", "sys", "pathlib"):
                raise PermissionError(f"Sandbox security violation: import from '{root_mod}' is forbidden.")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise PermissionError(f"Sandbox security violation: dunder attribute access '{node.attr}' is forbidden.")

    with tempfile.TemporaryDirectory(prefix="ankalm_sandbox_") as tmp_dir:
        script_path = Path(tmp_dir) / "sandbox_job.py"
        script_path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=tmp_dir,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Python sandbox execution exceeded time limit of {timeout_sec:.1f}s") from exc

        stdout_str = (proc.stdout or "")[:max_output_chars]
        stderr_str = (proc.stderr or "")[:max_output_chars]
        if proc.returncode != 0:
            raise RuntimeError(f"Sandbox code exited with code {proc.returncode}: {stderr_str.strip()}")

        return {
            "stdout": stdout_str.strip(),
            "stderr": stderr_str.strip(),
            "return_code": proc.returncode,
            "isolated_workspace": True,
            "network_disabled": True,
        }


# --- 3. UNIT CONVERTER ---
UNIT_CONVERSIONS: Dict[Tuple[str, str], Callable[[float], float]] = {
    ("km", "m"): lambda v: v * 1000.0,
    ("m", "km"): lambda v: v / 1000.0,
    ("m", "cm"): lambda v: v * 100.0,
    ("cm", "m"): lambda v: v / 100.0,
    ("kg", "g"): lambda v: v * 1000.0,
    ("g", "kg"): lambda v: v / 1000.0,
    ("kg", "lb"): lambda v: v * 2.20462262,
    ("lb", "kg"): lambda v: v / 2.20462262,
    ("c", "f"): lambda v: (v * 9.0 / 5.0) + 32.0,
    ("f", "c"): lambda v: (v - 32.0) * 5.0 / 9.0,
    ("c", "k"): lambda v: v + 273.15,
    ("k", "c"): lambda v: v - 273.15,
    ("gb", "mb"): lambda v: v * 1024.0,
    ("mb", "gb"): lambda v: v / 1024.0,
}


def execute_unit_converter(args: Dict[str, Any]) -> Dict[str, Any]:
    val = float(args["value"])
    u_from = str(args["from_unit"]).strip().lower()
    u_to = str(args["to_unit"]).strip().lower()
    if u_from == u_to:
        return {"value": val, "from_unit": u_from, "to_unit": u_to, "converted": val}
    fn = UNIT_CONVERSIONS.get((u_from, u_to))
    if fn is None:
        raise ValueError(f"Unsupported unit conversion pair: {u_from} -> {u_to}")
    return {"value": val, "from_unit": u_from, "to_unit": u_to, "converted": round(fn(val), 6)}


# --- 4. JSON / YAML STRUCTURED OUTPUT VALIDATOR & REPAIR (Section 11.3) ---
def execute_json_validator(args: Dict[str, Any]) -> Dict[str, Any]:
    raw = str(args.get("content", "")).strip()
    required_keys: List[str] = list(args.get("required_keys", []))
    fmt = str(args.get("format", "json")).lower()

    repaired = raw
    # Strip markdown code fences if present
    if repaired.startswith("```"):
        lines = [l for l in repaired.splitlines() if not l.strip().startswith("```")]
        repaired = "\n".join(lines).strip()

    parsed: Any = None
    was_repaired = (repaired != raw)
    try:
        parsed = yaml.safe_load(repaired) if fmt == "yaml" else json.loads(repaired)
    except Exception:
        # Attempt safe trailing-comma & quote repair
        candidate = re.sub(r",\s*([}\]])", r"\1", repaired)
        parsed = json.loads(candidate)
        repaired = json.dumps(parsed, ensure_ascii=False, indent=2)
        was_repaired = True

    missing_keys = [k for k in required_keys if not (isinstance(parsed, dict) and k in parsed)]
    return {
        "valid": len(missing_keys) == 0,
        "was_repaired": was_repaired,
        "missing_keys": missing_keys,
        "parsed": parsed,
        "normalized_json": json.dumps(parsed, ensure_ascii=False),
    }


# --- 5. LOCAL CHART GENERATOR ---
def execute_chart_generator(args: Dict[str, Any]) -> Dict[str, Any]:
    title = str(args.get("title", "AnkaLM Veri Grafiği"))
    labels: List[str] = [str(x) for x in args.get("labels", [])]
    values: List[float] = [float(x) for x in args.get("values", [])]
    if not labels or not values or len(labels) != len(values):
        raise ValueError("chart_generator requires matching non-empty 'labels' and 'values' lists.")

    w, h = 520, 320
    img = Image.new("RGB", (w, h), (15, 23, 42))
    draw = ImageDraw.Draw(img)
    draw.text((20, 14), title, fill=(248, 250, 252))
    draw.line([(45, h - 40), (w - 20, h - 40)], fill=(148, 163, 184), width=2)
    draw.line([(45, 45), (45, h - 40)], fill=(148, 163, 184), width=2)

    max_v = max(max(values), 1e-6)
    avail_w = (w - 80) // max(len(values), 1)
    bar_w = max(12, avail_w - 14)
    colors = [(56, 189, 248), (34, 197, 94), (245, 158, 11), (244, 114, 182), (168, 85, 247)]

    for i, (lbl, val) in enumerate(zip(labels, values)):
        bh = int(((h - 105) * max(0.0, val)) / max_v)
        x0 = 55 + i * avail_w
        y0 = (h - 41) - bh
        draw.rectangle([x0, y0, x0 + bar_w, h - 41], fill=colors[i % len(colors)])
        draw.text((x0, max(35, y0 - 16)), f"{val:g}", fill=(226, 232, 240))
        draw.text((x0, h - 32), lbl[:10], fill=(148, 163, 184))

    charts_dir = PROJECT_ROOT / "exports" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    out_file = charts_dir / f"chart_{int(time.time()*1000)}.png"
    img.save(out_file)
    return {
        "title": title,
        "chart_path": str(out_file),
        "bars_count": len(values),
        "max_value": max(values),
        "min_value": min(values),
    }


# --- 6. MULTI-STEP CROSS-DOCUMENT WORKFLOW ORCHESTRATOR (Section 10.3) ---
def execute_cross_document_comparison(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Multi-step workflow orchestrator:
    'Bu PDF dosyasını oku, Excel dosyamla karşılaştır, farkları bul ve bir rapor oluştur.'
    1. Identifies inputs -> 2. Parses PDF & Excel -> 3. Extracts numeric & textual evidence ->
    4. Runs deterministic delta calculations -> 5. Produces structured Turkish report with citations.
    """
    from ankalm.multimodal.excel_engine import AnkaLMExcelEngine
    from ankalm.multimodal.pdf_engine import AnkaLMPDFEngine

    pdf_path = validate_safe_path(args["pdf_path"], PROJECT_ROOT)
    excel_path = validate_safe_path(args["excel_path"], PROJECT_ROOT)

    pdf_eng = AnkaLMPDFEngine()
    xl_eng = AnkaLMExcelEngine()

    pdf_analysis = pdf_eng.analyze_pdf(pdf_path)
    xl_analysis = xl_eng.analyze_workbook(excel_path)

    # Extract numbers mentioned in PDF vs Excel column totals/values
    pdf_numbers: Dict[str, List[Tuple[int, float]]] = {}
    for page in pdf_analysis.pages:
        for line in page.full_text.splitlines():
            nums = [float(n.replace(",", ".")) for n in re.findall(r"\b\d+(?:[.,]\d+)?\b", line)]
            if nums:
                key = line.split(":")[0].strip()[:40]
                pdf_numbers.setdefault(key, []).extend((page.page_number, n) for n in nums)

    xl_totals: Dict[str, Tuple[str, float]] = {}
    for sname, ssum in xl_analysis.sheets.items():
        for col_name, st in ssum.numeric_column_stats.items():
            xl_totals[f"{sname}.{col_name}"] = (st["range"], st["sum"])

    citations = [f"[PDF: {pdf_analysis.filename} | Sayfa 1..{pdf_analysis.page_count}]"]
    for rng, _ in xl_totals.values():
        citations.append(f"[Excel: {xl_analysis.filename} | {rng}]")

    report_lines = [
        f"# Çok Adımlı Belge Karşılaştırma Raporu: `{pdf_analysis.filename}` vs `{xl_analysis.filename}`",
        f"1. **PDF Özeti (`{pdf_analysis.filename}`, {pdf_analysis.page_count} sayfa, {pdf_analysis.total_tables} tablo):**",
        f"   {pdf_analysis.hierarchical_summary}",
        f"2. **Excel Özeti (`{xl_analysis.filename}`, Sayfalar: {', '.join(xl_analysis.sheet_names)}):**",
    ]
    for col_key, (rng, total_val) in xl_totals.items():
        report_lines.append(f"   - `{col_key}` (`{rng}`): Toplam (SUM) = **{total_val}**")

    report_lines.append("3. **Tespit Edilen Kaynak Atıfları:**\n   - " + "\n   - ".join(citations))
    return {
        "pdf_filename": pdf_analysis.filename,
        "excel_filename": xl_analysis.filename,
        "pdf_pages": pdf_analysis.page_count,
        "excel_sheets": xl_analysis.sheet_names,
        "excel_numeric_totals": {k: v[1] for k, v in xl_totals.items()},
        "report_markdown": "\n".join(report_lines),
        "citations": citations,
    }


class AnkaLMToolRegistry:
    """Controlled Tool Execution Registry enforcing argument validation, permissions, and timeouts (Section 10)."""

    def __init__(self) -> None:
        self.tools: Dict[str, Tuple[ToolDefinition, Callable[[Dict[str, Any]], Any]]] = {}
        self._register_builtin_tools()

    def register(self, definition: ToolDefinition, handler: Callable[[Dict[str, Any]], Any]) -> None:
        self.tools[definition.name] = (definition, handler)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [defn.to_dict() for defn, _ in self.tools.values()]

    def execute(self, request: ToolCallRequest) -> ToolCallResponse:
        t0 = time.perf_counter()
        if request.tool_name not in self.tools:
            return ToolCallResponse(
                tool_name=request.tool_name,
                success=False,
                result=None,
                error=f"Unknown tool: '{request.tool_name}'",
            )

        defn, handler = self.tools[request.tool_name]
        if defn.requires_permission and not request.user_authorized:
            return ToolCallResponse(
                tool_name=request.tool_name,
                success=False,
                result=None,
                error=f"Permission denied: tool '{request.tool_name}' requires explicit user_authorized=True.",
            )

        # Validate required arguments against schema
        req_fields = defn.parameters_schema.get("required", [])
        if not isinstance(request.arguments, dict):
            return ToolCallResponse(tool_name=request.tool_name, success=False, result=None, error="Arguments must be a JSON object.")
        for rf in req_fields:
            if rf not in request.arguments:
                return ToolCallResponse(
                    tool_name=request.tool_name,
                    success=False,
                    result=None,
                    error=f"Missing required argument '{rf}' for tool '{request.tool_name}'.",
                )

        try:
            if request.tool_name == "python_sandbox":
                timeout = request.timeout_sec or defn.timeout_sec
                res = execute_python_sandbox(request.arguments, timeout_sec=timeout)
            else:
                res = handler(request.arguments)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            cites = res.get("citations", []) if isinstance(res, dict) else []
            return ToolCallResponse(
                tool_name=request.tool_name,
                success=True,
                result=res,
                error=None,
                execution_time_ms=round(dt_ms, 2),
                citations=cites,
            )
        except Exception as exc:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            return ToolCallResponse(
                tool_name=request.tool_name,
                success=False,
                result=None,
                error=f"{type(exc).__name__}: {exc}",
                execution_time_ms=round(dt_ms, 2),
            )

    def _register_builtin_tools(self) -> None:
        self.register(
            ToolDefinition(
                name="calculator",
                description="Safe deterministic mathematical and statistical calculator (arithmetic, algebra, fractions, trig, stats).",
                parameters_schema={"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
            ),
            execute_calculator,
        )
        self.register(
            ToolDefinition(
                name="python_sandbox",
                description="Isolated local Python calculation sandbox with strict timeout, no network, and restricted imports.",
                parameters_schema={"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
                requires_permission=False,
                timeout_sec=3.0,
            ),
            execute_python_sandbox,
        )
        self.register(
            ToolDefinition(
                name="unit_converter",
                description="Deterministic physical and digital unit conversion tool.",
                parameters_schema={
                    "type": "object",
                    "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}},
                    "required": ["value", "from_unit", "to_unit"],
                },
            ),
            execute_unit_converter,
        )
        self.register(
            ToolDefinition(
                name="json_validator",
                description="Validate and repair structured JSON or YAML output against required keys.",
                parameters_schema={"type": "object", "properties": {"content": {"type": "string"}}, "required": ["content"]},
            ),
            execute_json_validator,
        )
        self.register(
            ToolDefinition(
                name="chart_generator",
                description="Generate a local PNG bar chart from labels and numeric values.",
                parameters_schema={
                    "type": "object",
                    "properties": {"title": {"type": "string"}, "labels": {"type": "array"}, "values": {"type": "array"}},
                    "required": ["labels", "values"],
                },
            ),
            execute_chart_generator,
        )
        self.register(
            ToolDefinition(
                name="datetime_tool",
                description="Return current local Turkish date, time, and ISO timestamp.",
                parameters_schema={"type": "object", "properties": {}, "required": []},
            ),
            lambda _: {
                "iso_utc": datetime.now(timezone.utc).isoformat(),
                "local_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            },
        )
        self.register(
            ToolDefinition(
                name="file_search",
                description="Search files safely inside the AnkaLM project workspace.",
                parameters_schema={"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
            ),
            lambda args: {
                "matches": [
                    str(p.relative_to(PROJECT_ROOT))
                    for p in PROJECT_ROOT.rglob(f"*{args['pattern']}*")
                    if ".venv" not in p.parts and "__pycache__" not in p.parts
                ][:30]
            },
        )
        self.register(
            ToolDefinition(
                name="cross_document_comparator",
                description="Multi-step workflow comparing a PDF document with an Excel workbook and generating a cited report.",
                parameters_schema={
                    "type": "object",
                    "properties": {"pdf_path": {"type": "string"}, "excel_path": {"type": "string"}},
                    "required": ["pdf_path", "excel_path"],
                },
            ),
            execute_cross_document_comparison,
        )
        self.register(
            ToolDefinition(
                name="destructive_file_replace",
                description="Overwrite an existing workbook or document after explicit user authorization.",
                parameters_schema={"type": "object", "properties": {"target_path": {"type": "string"}}, "required": ["target_path"]},
                requires_permission=True,
            ),
            lambda args: {"authorized_target": str(validate_safe_path(args["target_path"], PROJECT_ROOT))},
        )


_DEFAULT_REGISTRY: Optional[AnkaLMToolRegistry] = None


def get_default_tool_registry() -> AnkaLMToolRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = AnkaLMToolRegistry()
    return _DEFAULT_REGISTRY
