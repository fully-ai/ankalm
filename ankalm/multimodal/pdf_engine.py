"""Comprehensive Local PDF Understanding Pipeline: Multi-page text, OCR fallback, structured tables, metadata, and Turkish Q&A (Section 5)."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field, asdict
import io
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pymupdf
from PIL import Image

from ankalm.multimodal.ocr_engine import AnkaLMLocalOCR, OCRPageResult


@dataclass
class PDFTableExtracted:
    table_id: str
    page_number: int
    bbox: List[float]
    row_count: int
    col_count: int
    headers: List[str]
    rows: List[List[str]]
    confidence: float
    uncertain: bool

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        if self.headers:
            writer.writerow(self.headers)
        for r in self.rows:
            writer.writerow(r)
        return buf.getvalue()

    def to_json(self) -> str:
        records = []
        hdrs = self.headers or [f"col_{i+1}" for i in range(self.col_count)]
        for r in self.rows:
            records.append({hdrs[i] if i < len(hdrs) else f"col_{i+1}": val for i, val in enumerate(r)})
        return json.dumps(records, indent=2, ensure_ascii=False)


@dataclass
class PDFPageExtract:
    page_number: int
    width_pt: float
    height_pt: float
    extraction_method: str  # native_text, local_ocr, hybrid_text_and_ocr
    is_scanned_or_image_page: bool
    is_multi_column: bool
    headings: List[str]
    paragraphs: List[Dict[str, Any]]
    full_text: str
    links: List[Dict[str, str]]
    annotations: List[Dict[str, str]]
    embedded_images_count: int
    embedded_images_info: List[Dict[str, Any]]
    tables: List[PDFTableExtracted]
    ocr_result: Optional[Dict[str, Any]] = None


@dataclass
class PDFDocumentAnalysis:
    filename: str
    page_count: int
    file_size_bytes: int
    pdf_type: str  # text_based, scanned_image_only, mixed_text_and_images
    metadata: Dict[str, Any]
    total_characters: int
    total_tables: int
    total_embedded_images: int
    total_links: int
    all_headings: List[Dict[str, Any]]
    hierarchical_summary: str
    processing_time_ms: float
    pages: List[PDFPageExtract] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DATE_REGEX = re.compile(
    r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{4}[./-]\d{1,2}[./-]\d{1,2}|\d{1,2}\s+(?:Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{4})\b",
    re.IGNORECASE,
)
PRICE_REGEX = re.compile(
    r"(?:(?:\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)\s*(?:TL|₺|USD|\$|EUR|€|Türk Lirası)|(?:₺|\$|€)\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{1,2})?)",
    re.IGNORECASE,
)


class AnkaLMPDFEngine:
    """
    Production Local PDF Intelligence Engine for AnkaLM.
    Preserves page order, multi-column structure, headings, tables (with CSV/JSON export),
    embedded images, annotations, and runs automatic local OCR on scanned/image-only pages.
    """

    def __init__(self, ocr_engine: Optional[AnkaLMLocalOCR] = None) -> None:
        self.ocr = ocr_engine or AnkaLMLocalOCR()

    def _extract_tables_from_page(self, page: pymupdf.Page, page_num: int, page_text: str) -> List[PDFTableExtracted]:
        tables_out: List[PDFTableExtracted] = []
        try:
            finder = page.find_tables()
            for idx, tab in enumerate(finder.tables):
                extracted = tab.extract()
                if not extracted or len(extracted) < 2:
                    continue
                cleaned_rows = [
                    [str(cell or "").replace("\n", " ").strip() for cell in row]
                    for row in extracted
                    if any(str(c or "").strip() for c in row)
                ]
                if len(cleaned_rows) >= 2:
                    headers = cleaned_rows[0]
                    body = cleaned_rows[1:]
                    tables_out.append(
                        PDFTableExtracted(
                            table_id=f"p{page_num}_t{idx+1}",
                            page_number=page_num,
                            bbox=[round(float(v), 2) for v in tab.bbox],
                            row_count=len(body),
                            col_count=len(headers),
                            headers=headers,
                            rows=body,
                            confidence=0.95,
                            uncertain=False,
                        )
                    )
        except Exception:
            pass

        # Fallback pipe/tab-delimited table detection in page text
        if not tables_out and ("|" in page_text or "\t" in page_text):
            candidate_lines = [
                line.strip()
                for line in page_text.splitlines()
                if ("|" in line and line.count("|") >= 2) or ("\t" in line and line.count("\t") >= 2)
            ]
            if len(candidate_lines) >= 2:
                delim = "|" if "|" in candidate_lines[0] else "\t"
                parsed_rows = []
                for cl in candidate_lines:
                    cells = [c.strip() for c in cl.strip("|").split(delim)]
                    # Skip markdown separator rows like ---|---
                    if all(set(c) <= {"-", ":"} and len(c) >= 2 for c in cells if c):
                        continue
                    if len(cells) >= 2:
                        parsed_rows.append(cells)
                if len(parsed_rows) >= 2:
                    tables_out.append(
                        PDFTableExtracted(
                            table_id=f"p{page_num}_t1",
                            page_number=page_num,
                            bbox=[0.0, 0.0, float(page.rect.width), float(page.rect.height)],
                            row_count=len(parsed_rows) - 1,
                            col_count=len(parsed_rows[0]),
                            headers=parsed_rows[0],
                            rows=parsed_rows[1:],
                            confidence=0.86,
                            uncertain=False,
                        )
                    )
        return tables_out

    def analyze_pdf(self, pdf_input: str | Path | bytes, filename: str = "document.pdf") -> PDFDocumentAnalysis:
        t0 = time.perf_counter()
        if isinstance(pdf_input, (str, Path)):
            p = Path(pdf_input)
            filename = p.name
            raw_bytes = p.read_bytes()
        else:
            raw_bytes = bytes(pdf_input)

        doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        meta = dict(doc.metadata or {})
        pages_out: List[PDFPageExtract] = []
        all_headings: List[Dict[str, Any]] = []
        scanned_pages_cnt = 0
        native_pages_cnt = 0

        for idx in range(len(doc)):
            page_num = idx + 1
            page = doc[idx]
            rect = page.rect
            w_pt, h_pt = round(float(rect.width), 2), round(float(rect.height), 2)

            # Extract structured text blocks and font sizes
            text_dict = page.get_text("dict")
            blocks = text_dict.get("blocks", [])
            text_blocks = [b for b in blocks if b.get("type") == 0]

            # Detect multi-column layout if distinct text blocks sit side-by-side horizontally
            left_col_blocks = 0
            right_col_blocks = 0
            font_sizes: List[float] = []
            for b in text_blocks:
                bx0, _, bx1, _ = b.get("bbox", (0, 0, 0, 0))
                if bx1 < w_pt * 0.52 and (bx1 - bx0) > 60:
                    left_col_blocks += 1
                elif bx0 > w_pt * 0.48 and (bx1 - bx0) > 60:
                    right_col_blocks += 1
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            font_sizes.append(float(span.get("size", 11.0)))

            is_multi_col = (left_col_blocks >= 2) and (right_col_blocks >= 2)
            median_font = float(sorted(font_sizes)[len(font_sizes) // 2]) if font_sizes else 11.0

            # Sort blocks in reading order: column-first if multi-column, else top-to-bottom
            if is_multi_col:
                ordered_blocks = sorted(
                    text_blocks,
                    key=lambda b: (0 if b.get("bbox", [0])[0] < w_pt * 0.48 else 1, b.get("bbox", [0, 0])[1]),
                )
            else:
                ordered_blocks = sorted(text_blocks, key=lambda b: (round(b.get("bbox", [0, 0])[1] / 12.0), b.get("bbox", [0])[0]))

            page_headings: List[str] = []
            paragraphs: List[Dict[str, Any]] = []
            for b_idx, b in enumerate(ordered_blocks):
                b_lines = []
                max_span_size = 0.0
                is_bold = False
                for line in b.get("lines", []):
                    l_str = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
                    if l_str:
                        b_lines.append(l_str)
                    for s in line.get("spans", []):
                        sz = float(s.get("size", 11.0))
                        if sz > max_span_size:
                            max_span_size = sz
                        if "bold" in str(s.get("font", "")).lower():
                            is_bold = True
                para_text = "\n".join(b_lines).strip()
                if not para_text:
                    continue
                is_heading = (max_span_size >= median_font * 1.18 or (is_bold and len(para_text) < 90)) and len(para_text) < 140 and ("\n" not in para_text)
                if is_heading:
                    page_headings.append(para_text)
                    all_headings.append({"page": page_num, "heading": para_text, "font_size": round(max_span_size, 1)})
                paragraphs.append({
                    "block_id": b_idx + 1,
                    "page": page_num,
                    "bbox": [round(float(v), 2) for v in b.get("bbox", (0, 0, 0, 0))],
                    "is_heading": is_heading,
                    "text": para_text,
                    "citation": f"[PDF: {filename} | Sayfa {page_num} | Paragraf #{b_idx + 1}]",
                })

            native_text = "\n\n".join(p["text"] for p in paragraphs).strip()

            # Embedded images
            img_list = page.get_images(full=True)
            img_infos: List[Dict[str, Any]] = []
            for img_i, img_meta in enumerate(img_list):
                xref = img_meta[0]
                img_infos.append({
                    "image_index": img_i + 1,
                    "xref": xref,
                    "width": img_meta[2],
                    "height": img_meta[3],
                    "bpc": img_meta[4],
                    "colorspace": img_meta[5],
                })

            # Links & Annotations
            links = [
                {"uri": lnk.get("uri", ""), "from_rect": [round(float(v), 1) for v in lnk.get("from", [])]}
                for lnk in page.get_links()
                if lnk.get("uri")
            ]
            annots = []
            if page.annots():
                for an in page.annots():
                    info = an.info or {}
                    annots.append({"type": an.type[1] if isinstance(an.type, tuple) else str(an.type), "content": info.get("content", "")})

            # Check if scanned / image-only page requiring Local OCR
            is_scanned = (len(native_text) < 25) and (len(img_list) > 0 or len(blocks) > 0)
            ocr_dict: Optional[Dict[str, Any]] = None
            extraction_method = "native_text"

            if is_scanned or len(native_text) < 15:
                scanned_pages_cnt += 1
                pix = page.get_pixmap(dpi=150)
                pil_page = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_res = self.ocr.process_image(pil_page, page_number=page_num)
                ocr_dict = ocr_res.to_dict()
                if ocr_res.full_text.strip():
                    extraction_method = "local_ocr" if not native_text else "hybrid_text_and_ocr"
                    native_text = (native_text + "\n" + ocr_res.full_text).strip()
                    paragraphs.append({
                        "block_id": len(paragraphs) + 1,
                        "page": page_num,
                        "bbox": [0.0, 0.0, w_pt, h_pt],
                        "is_heading": False,
                        "text": ocr_res.full_text.strip(),
                        "citation": f"[PDF OCR: {filename} | Sayfa {page_num} | Güven: {ocr_res.average_confidence:.2f}]",
                    })
            else:
                native_pages_cnt += 1

            tables = self._extract_tables_from_page(page, page_num, native_text)

            pages_out.append(
                PDFPageExtract(
                    page_number=page_num,
                    width_pt=w_pt,
                    height_pt=h_pt,
                    extraction_method=extraction_method,
                    is_scanned_or_image_page=is_scanned,
                    is_multi_column=is_multi_col,
                    headings=page_headings,
                    paragraphs=paragraphs,
                    full_text=native_text,
                    links=links,
                    annotations=annots,
                    embedded_images_count=len(img_list),
                    embedded_images_info=img_infos,
                    tables=tables,
                    ocr_result=ocr_dict,
                )
            )

        doc.close()

        if scanned_pages_cnt > 0 and native_pages_cnt == 0:
            pdf_type = "scanned_image_only"
        elif scanned_pages_cnt > 0 and native_pages_cnt > 0:
            pdf_type = "mixed_text_and_images"
        else:
            pdf_type = "text_based"

        # Build Map-Reduce Hierarchical Summary across all pages
        page_summaries = []
        for p in pages_out:
            first_para = p.paragraphs[0]["text"][:180] if p.paragraphs else "(Metin yok)"
            h_str = f"Başlıklar: {', '.join(p.headings[:3])}. " if p.headings else ""
            t_str = f"[{len(p.tables)} Tablo] " if p.tables else ""
            page_summaries.append(f"- Sayfa {p.page_number}: {t_str}{h_str}Özet: {first_para}")

        hier_summary = (
            f"PDF Belge Özeti ({filename}, {len(pages_out)} sayfa, Tür: {pdf_type}):\n"
            + "\n".join(page_summaries[:25])
        )
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return PDFDocumentAnalysis(
            filename=filename,
            page_count=len(pages_out),
            file_size_bytes=len(raw_bytes),
            pdf_type=pdf_type,
            metadata=meta,
            total_characters=sum(len(p.full_text) for p in pages_out),
            total_tables=sum(len(p.tables) for p in pages_out),
            total_embedded_images=sum(p.embedded_images_count for p in pages_out),
            total_links=sum(len(p.links) for p in pages_out),
            all_headings=all_headings,
            hierarchical_summary=hier_summary,
            processing_time_ms=round(dt_ms, 2),
            pages=pages_out,
        )

    def answer_question(self, analysis: PDFDocumentAnalysis, question: str) -> Dict[str, Any]:
        """
        Answer Turkish PDF questions with exact page references, table lookups, date/price extraction,
        keyword page search, section comparison, and clear uncertainty handling (Section 5.5).
        """
        q_low = question.lower().replace("İ", "i").replace("I", "ı")
        citations: List[str] = []

        # 1. Specific page table query: e.g. "12. sayfadaki tabloyu açıkla" / "2. sayfadaki tablo"
        page_match = re.search(r"(\d+)\s*\.\s*sayfa", q_low)
        if "tablo" in q_low:
            target_pages = [int(page_match.group(1))] if page_match else [p.page_number for p in analysis.pages]
            found_tables: List[PDFTableExtracted] = []
            for p in analysis.pages:
                if p.page_number in target_pages:
                    found_tables.extend(p.tables)
            if not found_tables:
                return {
                    "answer": f"Belirtilen sayfada ({target_pages}) yapılandırılmış bir tablo tespit edilemedi (Belirsizlik/Veri Yok).",
                    "citations": [f"{analysis.filename} | Sayfa {tp}" for tp in target_pages if tp <= analysis.page_count],
                    "tables_csv": [],
                }
            lines = []
            csvs = []
            for t in found_tables:
                citations.append(f"[PDF Tablo: {analysis.filename} | Sayfa {t.page_number} | Tablo {t.table_id}]")
                lines.append(
                    f"Sayfa {t.page_number} üzerindeki Tablo ({t.row_count} satır × {t.col_count} sütun) — "
                    f"Başlıklar: {', '.join(t.headers)} | İlk Satırlar: {t.rows[:3]}"
                )
                csvs.append(t.to_csv())
            return {"answer": "\n".join(lines), "citations": citations, "tables_csv": csvs}

        # 2. Dates extraction: "Bu belgede geçen tarihleri listele"
        if "tarih" in q_low:
            dates_found = []
            for p in analysis.pages:
                matches = DATE_REGEX.findall(p.full_text)
                if matches:
                    citations.append(f"[PDF: {analysis.filename} | Sayfa {p.page_number}]")
                    for m in matches:
                        dates_found.append(f"Sayfa {p.page_number}: {m}")
            if not dates_found:
                return {"answer": "Bu PDF belgesinde standart formatta bir tarih ifadesi bulunamadı.", "citations": []}
            return {"answer": "Belgede tespit edilen tarihler:\n- " + "\n- ".join(dates_found), "citations": citations}

        # 3. Prices / financial figures extraction: "Belgedeki tüm fiyatları bul"
        if "fiyat" in q_low or "tutar" in q_low or "ücret" in q_low or "maliyet" in q_low:
            prices_found = []
            for p in analysis.pages:
                matches = PRICE_REGEX.findall(p.full_text)
                if matches:
                    citations.append(f"[PDF: {analysis.filename} | Sayfa {p.page_number}]")
                    for m in matches:
                        prices_found.append(f"Sayfa {p.page_number}: {m.strip()}")
            if not prices_found:
                return {"answer": "Bu PDF belgesinde fiyat veya para birimi (TL, ₺, USD, EUR) içeren bir değer bulunamadı.", "citations": []}
            return {"answer": "Belgede tespit edilen fiyatlar/tutarlar:\n- " + "\n- ".join(prices_found), "citations": citations}

        # 4. Section-by-section or general summary: "Bu PDF ne anlatıyor?", "Önemli noktaları özetle", "Bana bölüm bölüm anlat"
        if any(k in q_low for k in ("ne anlatıyor", "özet", "bölüm bölüm", "önemli nokta")):
            citations = [f"[PDF: {analysis.filename} | Sayfa {p.page_number}]" for p in analysis.pages]
            return {"answer": analysis.hierarchical_summary, "citations": citations}

        # 5. Keyword search across pages
        words = [w.strip(".,!?;:\"'()[]") for w in q_low.split() if len(w.strip(".,!?;:\"'()[]")) >= 3]
        stop = {"geçtiği", "sayfaları", "bul", "belirli", "bir", "kelimenin", "nedir", "bu", "belgede", "rapordaki", "arasındaki", "fark"}
        keywords = [w for w in words if w not in stop]

        matched_snippets: List[str] = []
        for p in analysis.pages:
            p_low = p.full_text.lower().replace("İ", "i").replace("I", "ı")
            hits = [kw for kw in keywords if kw in p_low]
            if hits:
                citations.append(f"[PDF: {analysis.filename} | Sayfa {p.page_number}]")
                for para in p.paragraphs:
                    para_low = para["text"].lower().replace("İ", "i").replace("I", "ı")
                    if any(kw in para_low for kw in hits):
                        matched_snippets.append(f"Sayfa {p.page_number} ({para['citation']}): \"{para['text'][:240]}\"")

        if matched_snippets:
            return {
                "answer": "PDF içerisinden eşleşen kaynak bölümler:\n" + "\n".join(matched_snippets[:6]),
                "citations": citations,
            }

        return {
            "answer": (
                f"Sorgunuzla doğrudan eşleşen bir ifade '{analysis.filename}' ({analysis.page_count} sayfa) içinde bulunamadı. "
                f"Belge genel özeti:\n{analysis.hierarchical_summary}"
            ),
            "citations": [f"[PDF: {analysis.filename} | Sayfa 1..{analysis.page_count}]"],
        }
