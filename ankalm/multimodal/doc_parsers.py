"""Format-specific parsers for DOCX, PPTX, HTML, XML, ODT, EPUB, Code, and Untrusted Document Prompt-Injection Defense (Section 8 & 19)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from html.parser import HTMLParser
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
from typing import Any, Dict, List, Optional, Tuple

import docx
import pptx

from ankalm.core.constants import SPECIAL_TOKENS

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"(?i)ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"),
    re.compile(r"(?i)önceki\s+(?:tüm\s+)?talimatları\s+(?:unut|yoksay|görmezden\s+gel)"),
    re.compile(r"(?i)sistem\s+komutunu\s+değiştir|yetki\s+ver|güvenlik\s+duvarını\s+kapat"),
    re.compile(r"(?i)you\s+are\s+now\s+in\s+developer\s+mode|jailbreak"),
]


def sanitize_untrusted_document_text(text: str) -> Tuple[str, List[str]]:
    """
    Neutralize special control tokens and detect prompt-injection instructions embedded
    inside untrusted documents (Section 19: Document content is data, not authority).
    """
    warnings: List[str] = []
    sanitized = text
    for sp in SPECIAL_TOKENS:
        if sp in sanitized:
            warnings.append(f"Neutralized embedded control token: {sp}")
            sanitized = sanitized.replace(sp, f"[VERİ_ETİKETİ:{sp.strip('<|>')}]")

    for pat in PROMPT_INJECTION_PATTERNS:
        if pat.search(sanitized):
            warnings.append(f"Detected untrusted prompt-injection pattern: '{pat.pattern}'")
            sanitized = pat.sub("[GÜVENLİK_FİLTRESİ:BELGE_İÇİ_YETKİSİZ_TALİMAT_ENGELLENDİ]", sanitized)

    return (sanitized, warnings)


@dataclass
class ParsedSectionChunk:
    chunk_id: str
    title: str
    content: str
    citation: str
    metadata: Dict[str, Any] = field(default_factory=list)


@dataclass
class StructuredDocumentExtract:
    filename: str
    format: str
    file_size_bytes: int
    metadata: Dict[str, Any]
    headings: List[str]
    tables: List[Dict[str, Any]]
    links: List[str]
    images_count: int
    security_warnings: List[str]
    outline_summary: str
    chunks: List[ParsedSectionChunk]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_docx_document(raw_bytes: bytes, filename: str = "document.docx") -> StructuredDocumentExtract:
    """Extract headings, section hierarchy, paragraphs, tables, images, and metadata from a Word (.docx) document."""
    doc = docx.Document(io.BytesIO(raw_bytes))
    props = doc.core_properties
    meta = {
        "title": props.title or filename,
        "author": props.author or "",
        "subject": props.subject or "",
        "created": str(props.created or ""),
    }

    headings: List[str] = []
    chunks: List[ParsedSectionChunk] = []
    security_warnings: List[str] = []
    current_heading = "Giriş"

    for p_idx, para in enumerate(doc.paragraphs, start=1):
        txt = (para.text or "").strip()
        if not txt:
            continue
        txt_clean, warns = sanitize_untrusted_document_text(txt)
        security_warnings.extend(warns)

        style_name = (para.style.name if para.style else "") or ""
        if style_name.lower().startswith("heading") or style_name.lower().startswith("başlık") or style_name.lower() == "title":
            current_heading = txt_clean
            headings.append(txt_clean)

        chunks.append(
            ParsedSectionChunk(
                chunk_id=f"docx_p{p_idx}",
                title=current_heading,
                content=txt_clean,
                citation=f"[DOCX: {filename} | Bölüm: {current_heading} | Paragraf #{p_idx}]",
                metadata={"paragraph_index": p_idx, "heading": current_heading, "style": style_name},
            )
        )

    tables_out: List[Dict[str, Any]] = []
    for t_idx, tab in enumerate(doc.tables, start=1):
        rows_data = []
        for row in tab.rows:
            rows_data.append([sanitize_untrusted_document_text(c.text.strip())[0] for c in row.cells])
        if rows_data:
            tables_out.append({
                "table_id": f"docx_t{t_idx}",
                "headers": rows_data[0],
                "rows": rows_data[1:],
                "citation": f"[DOCX Tablo: {filename} | Tablo #{t_idx}]",
            })
            table_str = "\n".join(" | ".join(r) for r in rows_data)
            chunks.append(
                ParsedSectionChunk(
                    chunk_id=f"docx_t{t_idx}",
                    title=f"{current_heading} - Tablo #{t_idx}",
                    content=table_str,
                    citation=f"[DOCX Tablo: {filename} | Tablo #{t_idx}]",
                    metadata={"table_index": t_idx},
                )
            )

    # Count embedded images and hyperlinks via package relationships
    img_cnt = 0
    links: List[str] = []
    try:
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                img_cnt += 1
            elif "hyperlink" in rel.reltype:
                links.append(str(rel.target_ref))
    except Exception:
        pass

    outline = (
        f"Word Belgesi ({filename}): {len(chunks)} bölüm/paragraf, {len(headings)} başlık "
        f"({', '.join(headings[:6]) or 'Yok'}), {len(tables_out)} tablo, {img_cnt} gömülü görsel."
    )
    return StructuredDocumentExtract(
        filename=filename,
        format="DOCX",
        file_size_bytes=len(raw_bytes),
        metadata=meta,
        headings=headings,
        tables=tables_out,
        links=links,
        images_count=img_cnt,
        security_warnings=security_warnings,
        outline_summary=outline,
        chunks=chunks,
    )


def parse_pptx_presentation(raw_bytes: bytes, filename: str = "presentation.pptx") -> StructuredDocumentExtract:
    """Extract slides, titles, text, tables, images, and speaker notes from a PowerPoint (.pptx) presentation."""
    prs = pptx.Presentation(io.BytesIO(raw_bytes))
    headings: List[str] = []
    tables_out: List[Dict[str, Any]] = []
    chunks: List[ParsedSectionChunk] = []
    security_warnings: List[str] = []
    img_cnt = 0

    for s_idx, slide in enumerate(prs.slides, start=1):
        slide_title = f"Slayt {s_idx}"
        if slide.shapes.title and slide.shapes.title.text:
            slide_title = sanitize_untrusted_document_text(slide.shapes.title.text.strip())[0]
        headings.append(f"Slayt {s_idx}: {slide_title}")

        slide_texts: List[str] = []
        for shape in slide.shapes:
            if shape.shape_type == 13:  # PICTURE
                img_cnt += 1
            if shape.has_text_frame:
                t_str = "\n".join(p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip())
                if t_str:
                    t_clean, warns = sanitize_untrusted_document_text(t_str)
                    security_warnings.extend(warns)
                    slide_texts.append(t_clean)
            if shape.has_table:
                t_rows = []
                for r in shape.table.rows:
                    t_rows.append([sanitize_untrusted_document_text(c.text.strip())[0] for c in r.cells])
                if t_rows:
                    tables_out.append({
                        "table_id": f"slide{s_idx}_t{len(tables_out)+1}",
                        "slide": s_idx,
                        "headers": t_rows[0],
                        "rows": t_rows[1:],
                    })
                    slide_texts.append("Tablo:\n" + "\n".join(" | ".join(r) for r in t_rows))

        notes_text = ""
        try:
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                raw_n = slide.notes_slide.notes_text_frame.text.strip()
                if raw_n:
                    notes_text, warns = sanitize_untrusted_document_text(raw_n)
                    security_warnings.extend(warns)
        except Exception:
            pass

        full_slide_content = "\n".join(slide_texts)
        if notes_text:
            full_slide_content += f"\n[Konuşmacı Notu]: {notes_text}"

        chunks.append(
            ParsedSectionChunk(
                chunk_id=f"pptx_s{s_idx}",
                title=f"Slayt {s_idx}: {slide_title}",
                content=full_slide_content,
                citation=f"[PPTX: {filename} | Slayt {s_idx}: {slide_title}]",
                metadata={"slide_number": s_idx, "slide_title": slide_title, "has_notes": bool(notes_text)},
            )
        )

    outline = f"PowerPoint Sunumu ({filename}, {len(prs.slides)} slayt):\n" + "\n".join(f"- {h}" for h in headings)
    return StructuredDocumentExtract(
        filename=filename,
        format="PPTX",
        file_size_bytes=len(raw_bytes),
        metadata={"slide_count": len(prs.slides)},
        headings=headings,
        tables=tables_out,
        links=[],
        images_count=img_cnt,
        security_warnings=security_warnings,
        outline_summary=outline,
        chunks=chunks,
    )


class _SafeHTMLStructureParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.skip_depth = 0
        self.current_tag = ""
        self.headings: List[str] = []
        self.paragraphs: List[str] = []
        self.links: List[str] = []
        self.tables: List[List[List[str]]] = []
        self._in_table = False
        self._curr_table: List[List[str]] = []
        self._curr_row: List[str] = []
        self._buf: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t in ("script", "style", "nav", "footer", "noscript"):
            self.skip_depth += 1
            return
        if self.skip_depth > 0:
            return
        self.current_tag = t
        if t == "a":
            for k, v in attrs:
                if k.lower() == "href" and v and not v.lower().startswith("javascript:"):
                    self.links.append(v)
        elif t == "table":
            self._in_table = True
            self._curr_table = []
        elif t == "tr" and self._in_table:
            self._curr_row = []
        elif t in ("td", "th", "p", "li", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._buf = []

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in ("script", "style", "nav", "footer", "noscript"):
            self.skip_depth = max(0, self.skip_depth - 1)
            return
        if self.skip_depth > 0:
            return
        text = " ".join("".join(self._buf).split()).strip()
        if t in ("h1", "h2", "h3", "h4", "h5", "h6") and text:
            self.headings.append(text)
            self.paragraphs.append(f"## {text}")
            self._buf = []
        elif t in ("p", "li") and text:
            self.paragraphs.append(text)
            self._buf = []
        elif t in ("td", "th") and self._in_table:
            self._curr_row.append(text)
            self._buf = []
        elif t == "tr" and self._in_table and self._curr_row:
            self._curr_table.append(self._curr_row)
            self._curr_row = []
        elif t == "table" and self._in_table:
            if self._curr_table:
                self.tables.append(self._curr_table)
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self.skip_depth == 0:
            self._buf.append(data)


def parse_html_document(raw_bytes: bytes, filename: str = "page.html") -> StructuredDocumentExtract:
    """Parse HTML headings, paragraphs, lists, tables, and links while removing scripts/navigation boilerplate."""
    raw_html = raw_bytes.decode("utf-8", errors="replace")
    parser = _SafeHTMLStructureParser()
    parser.feed(raw_html)

    security_warnings: List[str] = []
    chunks: List[ParsedSectionChunk] = []
    cur_h = parser.headings[0] if parser.headings else filename

    for idx, p in enumerate(parser.paragraphs, start=1):
        clean_p, warns = sanitize_untrusted_document_text(p)
        security_warnings.extend(warns)
        if clean_p.startswith("## "):
            cur_h = clean_p[3:].strip()
        chunks.append(
            ParsedSectionChunk(
                chunk_id=f"html_p{idx}",
                title=cur_h,
                content=clean_p,
                citation=f"[HTML: {filename} | Bölüm: {cur_h} | Blok #{idx}]",
                metadata={"block_index": idx, "heading": cur_h},
            )
        )

    tables_out = []
    for t_idx, t_rows in enumerate(parser.tables, start=1):
        tables_out.append({"table_id": f"html_t{t_idx}", "headers": t_rows[0], "rows": t_rows[1:]})

    return StructuredDocumentExtract(
        filename=filename,
        format="HTML",
        file_size_bytes=len(raw_bytes),
        metadata={"links_count": len(parser.links)},
        headings=parser.headings,
        tables=tables_out,
        links=parser.links,
        images_count=0,
        security_warnings=security_warnings,
        outline_summary=f"HTML Belgesi ({filename}): {len( parser.headings )} başlık, {len(chunks)} metin bloğu, {len(tables_out)} tablo.",
        chunks=chunks,
    )


def parse_generic_or_archive_document(raw_bytes: bytes, filename: str) -> StructuredDocumentExtract:
    """Parse XML, ODT, EPUB, Markdown, JSON, JSONL, or source code files with section citations."""
    ext = Path(filename).suffix.lower()
    security_warnings: List[str] = []

    if ext in (".odt", ".epub") and zipfile.is_zipfile(io.BytesIO(raw_bytes)):
        extracted_texts = []
        with zipfile.ZipFile(io.BytesIO(raw_bytes), "r") as zf:
            for member in zf.namelist():
                if member.endswith((".xml", ".xhtml", ".html", ".htm")):
                    xml_bytes = zf.read(member)
                    try:
                        root = ET.fromstring(xml_bytes)
                        txt = " ".join(t.strip() for t in root.itertext() if t.strip())
                        if txt:
                            extracted_texts.append(txt)
                    except Exception:
                        pass
        raw_text = "\n\n".join(extracted_texts)
    elif ext == ".xml":
        try:
            root = ET.fromstring(raw_bytes)
            raw_text = "\n".join(t.strip() for t in root.itertext() if t.strip())
        except Exception:
            raw_text = raw_bytes.decode("utf-8", errors="replace")
    else:
        raw_text = raw_bytes.decode("utf-8", errors="replace")

    clean_text, warns = sanitize_untrusted_document_text(raw_text)
    security_warnings.extend(warns)

    blocks = [b.strip() for b in clean_text.split("\n\n") if b.strip()]
    if not blocks and clean_text.strip():
        blocks = [clean_text.strip()]

    headings = [line.lstrip("# ").strip() for line in clean_text.splitlines() if line.strip().startswith("#")]
    chunks: List[ParsedSectionChunk] = []
    for idx, blk in enumerate(blocks, start=1):
        chunks.append(
            ParsedSectionChunk(
                chunk_id=f"doc_b{idx}",
                title=f"{filename} #{idx}",
                content=blk,
                citation=f"[{ext.lstrip('.').upper() or 'TXT'}: {filename} | Blok #{idx}]",
                metadata={"block_index": idx},
            )
        )

    return StructuredDocumentExtract(
        filename=filename,
        format=ext.lstrip(".").upper() or "TXT",
        file_size_bytes=len(raw_bytes),
        metadata={"blocks": len(chunks)},
        headings=headings,
        tables=[],
        links=[],
        images_count=0,
        security_warnings=security_warnings,
        outline_summary=f"{ext.upper()} Belgesi ({filename}): {len(chunks)} blok, {len(clean_text)} karakter.",
        chunks=chunks,
    )
