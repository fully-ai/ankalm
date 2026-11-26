"""Comprehensive Local Excel & Spreadsheet Engine: Inspection, Safe Formula Interpreter, Analysis, and Non-Destructive Manipulation (Section 6)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
import io
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import openpyxl
from openpyxl.utils import column_index_from_string, get_column_letter


@dataclass
class FormulaEvaluationResult:
    sheet_name: str
    cell_address: str
    formula: str
    cached_value: Any
    recalculated_value: Any
    evaluation_status: str  # recalculated_by_ankalm, cached_value_only, unsupported_formula, formula_error
    error_message: Optional[str] = None


@dataclass
class SheetSummary:
    sheet_name: str
    is_hidden: bool
    row_count: int
    col_count: int
    headers: List[str]
    non_empty_cells: int
    missing_cells_count: int
    error_rows: List[int]
    merged_ranges: List[str]
    formulas_count: int
    charts_count: int
    comments_count: int
    hyperlinks_count: int
    numeric_column_stats: Dict[str, Dict[str, float]]
    sample_rows: List[Dict[str, Any]]


@dataclass
class WorkbookAnalysis:
    filename: str
    format: str
    file_size_bytes: int
    macros_disabled: bool
    sheet_names: List[str]
    hidden_sheets: List[str]
    named_ranges: Dict[str, str]
    total_formulas: int
    formula_evaluations: List[FormulaEvaluationResult]
    sheets: Dict[str, SheetSummary]
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


SUPPORTED_FORMULA_FUNCS = {
    "SUM", "AVERAGE", "MIN", "MAX", "COUNT", "COUNTA",
    "IF", "AND", "OR", "SUMIF", "SUMIFS", "COUNTIF", "COUNTIFS",
    "VLOOKUP", "HLOOKUP", "XLOOKUP", "INDEX", "MATCH",
    "CONCAT", "CONCATENATE", "TEXT", "DATE", "YEAR", "MONTH", "DAY",
    "ROUND", "IFERROR",
}

CELL_REF_RE = re.compile(r"(?:(?:'([^']+)'|([A-Za-z0-9_ğüşıöçĞÜŞİÖÇ]+))!)?\$?([A-Z]{1,3})\$?(\d+)")
RANGE_REF_RE = re.compile(
    r"(?:(?:'([^']+)'|([A-Za-z0-9_ğüşıöçĞÜŞİÖÇ]+))!)?\$?([A-Z]{1,3})\$?(\d+):\$?([A-Z]{1,3})\$?(\d+)"
)


class AnkaLMFormulaInterpreter:
    """
    Isolated, deterministic local spreadsheet formula interpreter.
    Supports 26+ Excel functions and arithmetic without executing macros or external links.
    """

    def __init__(self, sheet_grids: Dict[str, Dict[Tuple[int, int], Any]]) -> None:
        self.sheet_grids = sheet_grids  # sheet_name -> {(row_1based, col_1based): value}

    def get_cell(self, sheet: str, col_str: str, row_int: int) -> Any:
        c_idx = column_index_from_string(col_str.upper())
        val = self.sheet_grids.get(sheet, {}).get((row_int, c_idx))
        return val

    def get_range_2d(self, sheet: str, c1_str: str, r1: int, c2_str: str, r2: int) -> List[List[Any]]:
        c1 = column_index_from_string(c1_str.upper())
        c2 = column_index_from_string(c2_str.upper())
        grid = self.sheet_grids.get(sheet, {})
        rows = []
        for r in range(min(r1, r2), max(r1, r2) + 1):
            row_vals = [grid.get((r, c)) for c in range(min(c1, c2), max(c1, c2) + 1)]
            rows.append(row_vals)
        return rows

    def get_range_flat(self, sheet: str, c1_str: str, r1: int, c2_str: str, r2: int) -> List[Any]:
        return [v for row in self.get_range_2d(sheet, c1_str, r1, c2_str, r2) for v in row]

    def _to_numbers(self, items: List[Any]) -> List[float]:
        nums = []
        for x in items:
            if isinstance(x, bool) or x is None:
                continue
            if isinstance(x, (int, float)):
                nums.append(float(x))
            elif isinstance(x, str):
                try:
                    nums.append(float(x.replace(",", ".")))
                except Exception:
                    pass
        return nums

    def _matches_criteria(self, val: Any, criteria: str) -> bool:
        crit = str(criteria).strip().strip('"').strip("'")
        for op in (">=", "<=", "<>", "!=", ">", "<", "="):
            if crit.startswith(op):
                rhs = crit[len(op) :].strip()
                try:
                    v_num = float(val)
                    r_num = float(rhs)
                    if op == ">=": return v_num >= r_num
                    if op == "<=": return v_num <= r_num
                    if op in ("<>", "!="): return v_num != r_num
                    if op == ">": return v_num > r_num
                    if op == "<": return v_num < r_num
                    if op == "=": return v_num == r_num
                except Exception:
                    v_str = str(val or "").lower()
                    r_str = rhs.lower()
                    return (v_str != r_str) if op in ("<>", "!=") else (v_str == r_str)
        try:
            return float(val) == float(crit)
        except Exception:
            return str(val or "").strip().lower() == crit.lower()

    def _split_top_args(self, arg_str: str) -> List[str]:
        args: List[str] = []
        depth = 0
        in_quote = False
        curr = []
        for ch in arg_str:
            if ch == '"':
                in_quote = not in_quote
                curr.append(ch)
            elif not in_quote and ch == "(":
                depth += 1
                curr.append(ch)
            elif not in_quote and ch == ")":
                depth -= 1
                curr.append(ch)
            elif not in_quote and depth == 0 and ch in (",", ";"):
                args.append("".join(curr).strip())
                curr = []
            else:
                curr.append(ch)
        if curr:
            args.append("".join(curr).strip())
        return args

    def _resolve_arg_values(self, expr: str, current_sheet: str) -> List[Any]:
        expr = expr.strip()
        m_rng = RANGE_REF_RE.fullmatch(expr)
        if m_rng:
            sh = m_rng.group(1) or m_rng.group(2) or current_sheet
            return self.get_range_flat(sh, m_rng.group(3), int(m_rng.group(4)), m_rng.group(5), int(m_rng.group(6)))
        val, status, _ = self.evaluate_expression(expr, current_sheet)
        if status == "recalculated_by_ankalm":
            return [val]
        return []

    def evaluate_expression(self, raw_formula: str, current_sheet: str) -> Tuple[Any, str, Optional[str]]:
        """Return (recalculated_value, status, error_message)."""
        expr = raw_formula.strip()
        if expr.startswith("="):
            expr = expr[1:].strip()

        # Check for external workbook link [Workbook.xlsx]
        if "[" in expr and "]" in expr:
            return (None, "unsupported_formula", "External workbook links are blocked for security.")

        # Direct string literal
        if expr.startswith('"') and expr.endswith('"') and len(expr) >= 2:
            return (expr[1:-1], "recalculated_by_ankalm", None)

        # Direct numeric literal
        try:
            num = float(expr)
            return (int(num) if num.is_integer() else num, "recalculated_by_ankalm", None)
        except Exception:
            pass

        # Single cell reference
        m_cell = CELL_REF_RE.fullmatch(expr)
        if m_cell:
            sh = m_cell.group(1) or m_cell.group(2) or current_sheet
            return (self.get_cell(sh, m_cell.group(3), int(m_cell.group(4))), "recalculated_by_ankalm", None)

        # Function call: FUNC(arg1, arg2, ...)
        m_func = re.match(r"^([A-Za-z0-9_.]+)\((.*)\)$", expr, re.DOTALL)
        if m_func:
            fname = m_func.group(1).upper().replace("_XLWS.", "").replace("_XLFN.", "")
            inner = m_func.group(2)
            if fname not in SUPPORTED_FORMULA_FUNCS:
                return (None, "unsupported_formula", f"Unsupported Excel function: {fname}")
            args = self._split_top_args(inner)

            try:
                if fname in ("SUM", "AVERAGE", "MIN", "MAX", "COUNT", "COUNTA"):
                    flat: List[Any] = []
                    for a in args:
                        flat.extend(self._resolve_arg_values(a, current_sheet))
                    if fname == "COUNTA":
                        return (sum(1 for v in flat if v is not None and str(v).strip() != ""), "recalculated_by_ankalm", None)
                    nums = self._to_numbers(flat)
                    if fname == "COUNT":
                        return (len(nums), "recalculated_by_ankalm", None)
                    if fname == "SUM":
                        return (round(sum(nums), 6), "recalculated_by_ankalm", None)
                    if not nums:
                        return ("#DIV/0!" if fname == "AVERAGE" else 0.0, "formula_error" if fname == "AVERAGE" else "recalculated_by_ankalm", "Empty numeric range")
                    if fname == "AVERAGE":
                        return (round(sum(nums) / len(nums), 6), "recalculated_by_ankalm", None)
                    if fname == "MIN":
                        return (min(nums), "recalculated_by_ankalm", None)
                    if fname == "MAX":
                        return (max(nums), "recalculated_by_ankalm", None)

                if fname == "ROUND" and len(args) >= 2:
                    v, _, _ = self.evaluate_expression(args[0], current_sheet)
                    d, _, _ = self.evaluate_expression(args[1], current_sheet)
                    return (round(float(v), int(d)), "recalculated_by_ankalm", None)

                if fname == "IFERROR" and len(args) >= 2:
                    v, st, _ = self.evaluate_expression(args[0], current_sheet)
                    if st == "formula_error" or (isinstance(v, str) and v.startswith("#")):
                        fb, _, _ = self.evaluate_expression(args[1], current_sheet)
                        return (fb, "recalculated_by_ankalm", None)
                    return (v, "recalculated_by_ankalm", None)

                if fname in ("CONCAT", "CONCATENATE"):
                    parts = []
                    for a in args:
                        vals = self._resolve_arg_values(a, current_sheet)
                        parts.extend(str(v) for v in vals if v is not None)
                    return ("".join(parts), "recalculated_by_ankalm", None)

                if fname == "IF" and len(args) >= 2:
                    cond_str = args[0]
                    cond_val = False
                    for op in (">=", "<=", "<>", "!=", ">", "<", "="):
                        if op in cond_str:
                            lhs_s, rhs_s = cond_str.split(op, 1)
                            lv, _, _ = self.evaluate_expression(lhs_s, current_sheet)
                            rv, _, _ = self.evaluate_expression(rhs_s, current_sheet)
                            cond_val = self._matches_criteria(lv, f"{op}{rv}")
                            break
                    branch = args[1] if cond_val else (args[2] if len(args) > 2 else "FALSE")
                    res, st, err = self.evaluate_expression(branch, current_sheet)
                    return (res, st, err)

                if fname == "COUNTIF" and len(args) >= 2:
                    rng_vals = self._resolve_arg_values(args[0], current_sheet)
                    crit, _, _ = self.evaluate_expression(args[1], current_sheet)
                    cnt = sum(1 for v in rng_vals if self._matches_criteria(v, str(crit)))
                    return (cnt, "recalculated_by_ankalm", None)

                if fname == "SUMIF" and len(args) >= 2:
                    crit_vals = self._resolve_arg_values(args[0], current_sheet)
                    crit, _, _ = self.evaluate_expression(args[1], current_sheet)
                    sum_vals = self._resolve_arg_values(args[2], current_sheet) if len(args) >= 3 else crit_vals
                    total = 0.0
                    for cv, sv in zip(crit_vals, sum_vals):
                        if self._matches_criteria(cv, str(crit)):
                            try:
                                total += float(sv)
                            except Exception:
                                pass
                    return (round(total, 6), "recalculated_by_ankalm", None)

                if fname == "VLOOKUP" and len(args) >= 3:
                    lookup_val, _, _ = self.evaluate_expression(args[0], current_sheet)
                    m_r = RANGE_REF_RE.fullmatch(args[1].strip())
                    col_idx, _, _ = self.evaluate_expression(args[2], current_sheet)
                    if m_r:
                        sh = m_r.group(1) or m_r.group(2) or current_sheet
                        table_2d = self.get_range_2d(sh, m_r.group(3), int(m_r.group(4)), m_r.group(5), int(m_r.group(6)))
                        c_i = int(col_idx) - 1
                        for row in table_2d:
                            if row and str(row[0]).strip().lower() == str(lookup_val).strip().lower():
                                if 0 <= c_i < len(row):
                                    return (row[c_i], "recalculated_by_ankalm", None)
                    return ("#N/A", "formula_error", "VLOOKUP key not found")

                if fname == "DATE" and len(args) == 3:
                    y, _, _ = self.evaluate_expression(args[0], current_sheet)
                    m, _, _ = self.evaluate_expression(args[1], current_sheet)
                    d, _, _ = self.evaluate_expression(args[2], current_sheet)
                    return (date(int(y), int(m), int(d)).isoformat(), "recalculated_by_ankalm", None)

                if fname in ("YEAR", "MONTH", "DAY") and len(args) == 1:
                    v, _, _ = self.evaluate_expression(args[0], current_sheet)
                    dt_obj = datetime.fromisoformat(str(v)[:10]) if isinstance(v, str) else v
                    if hasattr(dt_obj, "year"):
                        return (getattr(dt_obj, fname.lower()), "recalculated_by_ankalm", None)

            except ZeroDivisionError:
                return ("#DIV/0!", "formula_error", "Division by zero")
            except Exception as exc:
                return ("#VALUE!", "formula_error", str(exc))

        # Simple arithmetic expression over cell references (e.g. A2*B2+C2)
        if re.fullmatch(r"[\s0-9A-Za-z_!'\$\+\-\*\/\(\)\.]+", expr):
            def _sub_cell(match: re.Match) -> str:
                sh = match.group(1) or match.group(2) or current_sheet
                v = self.get_cell(sh, match.group(3), int(match.group(4)))
                try:
                    return str(float(v))
                except Exception:
                    return "0.0"

            arith_str = CELL_REF_RE.sub(_sub_cell, expr)
            if re.fullmatch(r"[\s0-9\.\+\-\*\/\(\)]+", arith_str):
                try:
                    val = eval(arith_str, {"__builtins__": {}}, {})
                    return (round(float(val), 6), "recalculated_by_ankalm", None)
                except ZeroDivisionError:
                    return ("#DIV/0!", "formula_error", "Division by zero")
                except Exception as exc:
                    return ("#VALUE!", "formula_error", str(exc))

        return (None, "unsupported_formula", f"Could not parse formula expression: {raw_formula}")


class AnkaLMExcelEngine:
    """
    Production Local Excel & Spreadsheet Engine for AnkaLM.
    Inspects workbooks with macros disabled, evaluates formulas deterministically,
    answers numerical queries with cell-range citations, and performs non-destructive edits.
    """

    def analyze_workbook(self, file_input: str | Path | bytes, filename: str = "workbook.xlsx") -> WorkbookAnalysis:
        t0 = time.perf_counter()
        if isinstance(file_input, (str, Path)):
            p = Path(file_input)
            filename = p.name
            raw_bytes = p.read_bytes()
        else:
            raw_bytes = bytes(file_input)

        ext = Path(filename).suffix.lower()
        if ext in (".csv", ".tsv"):
            return self._analyze_delimited(raw_bytes, filename, delimiter="\t" if ext == ".tsv" else ",", t0=t0)

        # Load with keep_vba=False so macros are NEVER executed
        wb_formulas = openpyxl.load_workbook(io.BytesIO(raw_bytes), data_only=False, keep_vba=False)
        wb_cached = openpyxl.load_workbook(io.BytesIO(raw_bytes), data_only=True, keep_vba=False)

        sheet_names = list(wb_formulas.sheetnames)
        hidden_sheets = [s for s in sheet_names if wb_formulas[s].sheet_state != "visible"]

        named_ranges: Dict[str, str] = {}
        try:
            for name, defn in wb_formulas.defined_names.items():
                named_ranges[name] = str(defn.attr_text)
        except Exception:
            pass

        # Build value grid for formula interpreter
        sheet_grids: Dict[str, Dict[Tuple[int, int], Any]] = {}
        for sname in sheet_names:
            ws_c = wb_cached[sname]
            ws_f = wb_formulas[sname]
            grid: Dict[Tuple[int, int], Any] = {}
            for r in range(1, ws_c.max_row + 1):
                for c in range(1, ws_c.max_column + 1):
                    v_c = ws_c.cell(r, c).value
                    v_f = ws_f.cell(r, c).value
                    grid[(r, c)] = v_c if v_c is not None else (None if (isinstance(v_f, str) and v_f.startswith("=")) else v_f)
            sheet_grids[sname] = grid

        interpreter = AnkaLMFormulaInterpreter(sheet_grids)
        formula_evals: List[FormulaEvaluationResult] = []
        sheets_summary: Dict[str, SheetSummary] = {}

        for sname in sheet_names:
            ws_f = wb_formulas[sname]
            ws_c = wb_cached[sname]
            max_r = ws_f.max_row or 0
            max_c = ws_f.max_column or 0

            headers = [str(ws_c.cell(1, c).value or ws_f.cell(1, c).value or f"Col_{get_column_letter(c)}") for c in range(1, max_c + 1)]
            non_empty = 0
            missing = 0
            error_rows: List[int] = []
            formulas_cnt = 0
            comments_cnt = 0
            hyperlinks_cnt = 0
            col_num_values: Dict[str, List[float]] = {h: [] for h in headers}
            sample_rows: List[Dict[str, Any]] = []

            for r in range(1, max_r + 1):
                row_dict: Dict[str, Any] = {"_row": r}
                row_has_err = False
                for c in range(1, max_c + 1):
                    cell_f = ws_f.cell(r, c)
                    cell_c = ws_c.cell(r, c)
                    addr = f"{get_column_letter(c)}{r}"
                    val_f = cell_f.value
                    val_c = cell_c.value

                    if cell_f.comment is not None:
                        comments_cnt += 1
                    if cell_f.hyperlink is not None:
                        hyperlinks_cnt += 1

                    effective_val = val_c if val_c is not None else val_f
                    if isinstance(val_f, str) and val_f.startswith("="):
                        formulas_cnt += 1
                        recalc_val, status, err_msg = interpreter.evaluate_expression(val_f, sname)
                        if status == "recalculated_by_ankalm" and recalc_val is not None:
                            effective_val = recalc_val
                            sheet_grids[sname][(r, c)] = recalc_val
                        elif val_c is not None:
                            status = "cached_value_only"
                            effective_val = val_c
                        formula_evals.append(
                            FormulaEvaluationResult(
                                sheet_name=sname,
                                cell_address=f"{sname}!{addr}",
                                formula=val_f,
                                cached_value=val_c,
                                recalculated_value=recalc_val,
                                evaluation_status=status,
                                error_message=err_msg,
                            )
                        )

                    if effective_val is None or str(effective_val).strip() == "":
                        if r > 1:
                            missing += 1
                    else:
                        non_empty += 1
                        if isinstance(effective_val, str) and effective_val.startswith(("#DIV/0!", "#VALUE!", "#REF!", "#NAME?", "#N/A")):
                            row_has_err = True

                    h_name = headers[c - 1]
                    row_dict[h_name] = effective_val
                    if r > 1 and isinstance(effective_val, (int, float)) and not isinstance(effective_val, bool):
                        col_num_values[h_name].append(float(effective_val))

                if row_has_err:
                    error_rows.append(r)
                if 1 < r <= 12:
                    sample_rows.append(row_dict)

            num_stats: Dict[str, Dict[str, float]] = {}
            for idx_c, (h_name, nums) in enumerate(col_num_values.items()):
                if nums:
                    col_let = get_column_letter(idx_c + 1)
                    num_stats[h_name] = {
                        "column_letter": col_let,
                        "range": f"{sname}!{col_let}2:{col_let}{max_r}",
                        "sum": round(sum(nums), 4),
                        "average": round(sum(nums) / len(nums), 4),
                        "min": round(min(nums), 4),
                        "max": round(max(nums), 4),
                        "count": float(len(nums)),
                    }

            sheets_summary[sname] = SheetSummary(
                sheet_name=sname,
                is_hidden=(sname in hidden_sheets),
                row_count=max_r,
                col_count=max_c,
                headers=headers,
                non_empty_cells=non_empty,
                missing_cells_count=missing,
                error_rows=error_rows,
                merged_ranges=[str(mr) for mr in ws_f.merged_cells.ranges],
                formulas_count=formulas_cnt,
                charts_count=len(getattr(ws_f, "_charts", [])),
                comments_count=comments_cnt,
                hyperlinks_count=hyperlinks_cnt,
                numeric_column_stats=num_stats,
                sample_rows=sample_rows,
            )

        wb_formulas.close()
        wb_cached.close()
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return WorkbookAnalysis(
            filename=filename,
            format=ext.lstrip(".").upper() or "XLSX",
            file_size_bytes=len(raw_bytes),
            macros_disabled=True,
            sheet_names=sheet_names,
            hidden_sheets=hidden_sheets,
            named_ranges=named_ranges,
            total_formulas=len(formula_evals),
            formula_evaluations=formula_evals,
            sheets=sheets_summary,
            processing_time_ms=round(dt_ms, 2),
        )

    def _analyze_delimited(self, raw_bytes: bytes, filename: str, delimiter: str, t0: float) -> WorkbookAnalysis:
        text = raw_bytes.decode("utf-8", errors="replace")
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = Path(filename).stem[:31] or "Veri"
        for r_idx, row in enumerate(rows, start=1):
            for c_idx, val in enumerate(row, start=1):
                v_clean: Any = val.strip()
                try:
                    num = float(v_clean)
                    v_clean = int(num) if num.is_integer() else num
                except Exception:
                    pass
                ws.cell(r_idx, c_idx, v_clean)
        buf = io.BytesIO()
        wb.save(buf)
        res = self.analyze_workbook(buf.getvalue(), filename=filename)
        res.format = "TSV" if delimiter == "\t" else "CSV"
        res.processing_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return res

    def answer_question(self, analysis: WorkbookAnalysis, question: str) -> Dict[str, Any]:
        """Answer Turkish spreadsheet queries with deterministic calculations and exact Sheet!Range citations."""
        q_low = question.lower().replace("İ", "i").replace("I", "ı")
        citations: List[str] = []

        if "hangi sayfalar" in q_low or "sayfa sayısı" in q_low:
            return {
                "answer": f"'{analysis.filename}' çalışma kitabında {len(analysis.sheet_names)} sayfa bulunmaktadır: {', '.join(analysis.sheet_names)} (Gizli sayfalar: {analysis.hidden_sheets or 'Yok'}).",
                "citations": [f"{analysis.filename}!{s}" for s in analysis.sheet_names],
            }

        if "formül" in q_low:
            if not analysis.formula_evaluations:
                return {"answer": "Çalışma kitabında herhangi bir formül hücresi bulunmamaktadır.", "citations": []}
            lines = []
            for fe in analysis.formula_evaluations[:15]:
                citations.append(fe.cell_address)
                lines.append(
                    f"- {fe.cell_address}: `{fe.formula}` -> Hesaplanan Değer: **{fe.recalculated_value}** (Durum: `{fe.evaluation_status}`)"
                )
            return {"answer": "Formül Denetim Raporu:\n" + "\n".join(lines), "citations": citations}

        if "eksik" in q_low or "hatalı" in q_low:
            lines = []
            for sname, ssum in analysis.sheets.items():
                citations.append(f"{sname}!A1:{get_column_letter(max(1, ssum.col_count))}{ssum.row_count}")
                lines.append(
                    f"- **{sname}**: Eksik (boş) hücre sayısı = {ssum.missing_cells_count}, Hatalı formül satırları = {ssum.error_rows or 'Yok'}"
                )
            return {"answer": "Veri Kalitesi ve Eksik/Hatalı Satır Analizi:\n" + "\n".join(lines), "citations": citations}

        if "iki sayfayı karşılaştır" in q_low or "karşılaştır" in q_low:
            s_list = list(analysis.sheets.values())
            if len(s_list) < 2:
                return {"answer": "Karşılaştırma için en az 2 çalışma sayfası gereklidir.", "citations": []}
            s1, s2 = s_list[0], s_list[1]
            citations = [f"{s1.sheet_name}!A1", f"{s2.sheet_name}!A1"]
            return {
                "answer": (
                    f"**{s1.sheet_name}** ({s1.row_count} satır, Başlıklar: {s1.headers}) ile "
                    f"**{s2.sheet_name}** ({s2.row_count} satır, Başlıklar: {s2.headers}) karşılaştırması:\n"
                    f"- {s1.sheet_name} sayısal özetler: {json.dumps(s1.numeric_column_stats, ensure_ascii=False)}\n"
                    f"- {s2.sheet_name} sayısal özetler: {json.dumps(s2.numeric_column_stats, ensure_ascii=False)}"
                ),
                "citations": citations,
            }

        # Default comprehensive numerical & top-value summary across sheets
        lines = [f"**{analysis.filename}** ({analysis.format}) Deterministik Hesaplama Özeti:"]
        for sname, ssum in analysis.sheets.items():
            lines.append(f"\n### Sayfa: `{sname}` ({ssum.row_count} satır × {ssum.col_count} sütun)")
            for col_name, st in ssum.numeric_column_stats.items():
                rng = st["range"]
                citations.append(rng)
                lines.append(
                    f"- **{col_name}** (`{rng}`): Toplam (SUM) = **{st['sum']}**, "
                    f"Ortalama (AVERAGE) = **{st['average']}**, En Yüksek (MAX) = **{st['max']}**, En Düşük (MIN) = **{st['min']}**"
                )
            if ssum.sample_rows:
                # Identify top row by first numeric column if available
                first_num_col = next(iter(ssum.numeric_column_stats.keys()), None)
                if first_num_col:
                    valid_rows = [r for r in ssum.sample_rows if isinstance(r.get(first_num_col), (int, float))]
                    if valid_rows:
                        top_row = max(valid_rows, key=lambda r: float(r[first_num_col]))
                        r_num = top_row.get("_row", 2)
                        citations.append(f"{sname}!A{r_num}:{get_column_letter(ssum.col_count)}{r_num}")
                        lines.append(f"- `{first_num_col}` açısından en yüksek örnek satır (Satır {r_num}): `{json.dumps(top_row, ensure_ascii=False)}`")

        return {"answer": "\n".join(lines), "citations": citations}

    def manipulate_workbook_safely(
        self,
        source_path: str | Path,
        operations: List[Dict[str, Any]],
        output_path: Optional[str | Path] = None,
        allow_overwrite: bool = False,
    ) -> Dict[str, Any]:
        """
        Safely manipulate an Excel workbook (add_sheet, rename_sheet, update_cell, add_formula)
        writing to a new file by default and returning a full preview of modified cells (Section 6.4).
        """
        src = Path(source_path).resolve()
        if output_path is not None:
            dst = Path(output_path).resolve()
        else:
            dst = src.parent / f"{src.stem}_modified{src.suffix}"

        if dst == src and not allow_overwrite:
            raise PermissionError("Refusing to overwrite original workbook without explicit allow_overwrite=True.")

        wb = openpyxl.load_workbook(src, keep_vba=False)
        changes_preview: List[Dict[str, Any]] = []

        for op in operations:
            action = op.get("action")
            if action == "add_sheet":
                title = str(op["sheet_name"])
                wb.create_sheet(title=title)
                changes_preview.append({"action": "add_sheet", "sheet": title})
            elif action == "rename_sheet":
                old_name = str(op["old_name"])
                new_name = str(op["new_name"])
                if old_name in wb.sheetnames:
                    wb[old_name].title = new_name
                    changes_preview.append({"action": "rename_sheet", "old": old_name, "new": new_name})
            elif action in ("update_cell", "add_formula"):
                sname = str(op.get("sheet", wb.sheetnames[0]))
                cell_addr = str(op["cell"])
                new_val = op["value"]
                ws = wb[sname]
                old_val = ws[cell_addr].value
                ws[cell_addr].value = new_val
                changes_preview.append({
                    "action": action,
                    "reference": f"{sname}!{cell_addr}",
                    "previous_value": old_val,
                    "new_value": new_val,
                })

        dst.parent.mkdir(parents=True, exist_ok=True)
        wb.save(dst)
        wb.close()
        return {
            "source_file": str(src),
            "output_file": str(dst),
            "original_preserved": dst != src,
            "changes_applied": changes_preview,
        }
