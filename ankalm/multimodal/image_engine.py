"""Comprehensive Image Understanding Pipeline: High-Res Tiling, OCR, AnkaLM-Vision Encoding, and Turkish Visual Q&A (Section 7)."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import io
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

from ankalm.model.vision import (
    VISION_CATEGORY_DESCRIPTIONS_TR,
    VISION_SCENE_CATEGORIES,
    load_trained_vision_model,
    pil_to_vision_tensor,
)
from ankalm.multimodal.ocr_engine import AnkaLMLocalOCR, OCRPageResult

# Decompression bomb protection (Section 19)
Image.MAX_IMAGE_PIXELS = 36_000_000


@dataclass
class ImageTileInfo:
    tile_id: int
    grid_row: int
    grid_col: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    mean_brightness: float
    edge_density: float
    predicted_category: str
    category_confidence: float


@dataclass
class ImageAnalysisResult:
    filename: str
    format: str
    width: int
    height: int
    aspect_ratio: float
    color_mode: str
    frame_count: int
    file_size_bytes: int
    exif_metadata: Dict[str, str]
    dominant_rgb: Tuple[int, int, int]
    high_res_tiled: bool
    tiles_count: int
    tiles: List[ImageTileInfo]
    spatial_quadrant_activity: Dict[str, float]
    vision_predicted_category: str
    vision_category_description_tr: str
    vision_confidence: float
    vision_model_used: str
    ocr_result: Dict[str, Any]
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnkaLMImageEngine:
    """
    True Multimodal Image Understanding Pipeline for AnkaLM.
    Combines:
    1. Resolution, orientation, metadata, and high-resolution tiling (never crushing detail into a tiny thumbnail).
    2. Local Offline OCR (`AnkaLMLocalOCR`) with spatial bounding boxes and table grid detection.
    3. Independent From-Scratch Trained `AnkaLM-Vision` encoder (`checkpoints/vision_final`) for visual scene classification
       and projected multimodal tokens, clearly distinguishing learned vision features from OCR text.
    """

    def __init__(self, ocr_engine: Optional[AnkaLMLocalOCR] = None) -> None:
        self.ocr = ocr_engine or AnkaLMLocalOCR()
        self._vision_model = None

    def _get_vision_model(self):
        if self._vision_model is None:
            self._vision_model = load_trained_vision_model()
        return self._vision_model

    def extract_high_res_tiles(self, img: Image.Image, tile_size: int = 256, overlap: int = 32) -> List[Tuple[int, int, Tuple[int, int, int, int], Image.Image]]:
        """Slice images into overlapping tiles to preserve high-resolution detail."""
        w, h = img.size
        if w <= tile_size and h <= tile_size:
            return [(0, 0, (0, 0, w, h), img)]

        step = max(64, tile_size - overlap)
        tiles = []
        r_idx = 0
        for y in range(0, h, step):
            c_idx = 0
            y1 = min(h, y + tile_size)
            y0 = max(0, y1 - tile_size)
            for x in range(0, w, step):
                x1 = min(w, x + tile_size)
                x0 = max(0, x1 - tile_size)
                crop = img.crop((x0, y0, x1, y1))
                tiles.append((r_idx, c_idx, (x0, y0, x1 - x0, y1 - y0), crop))
                c_idx += 1
                if x1 >= w:
                    break
            r_idx += 1
            if y1 >= h:
                break
        return tiles[:16]

    def analyze_image(self, image_input: str | Path | bytes, filename: str = "image.png") -> ImageAnalysisResult:
        t0 = time.perf_counter()
        if isinstance(image_input, (str, Path)):
            p = Path(image_input)
            filename = p.name
            raw_bytes = p.read_bytes()
        else:
            raw_bytes = bytes(image_input)

        try:
            pil_img = Image.open(io.BytesIO(raw_bytes))
            pil_img.verify()
            pil_img = Image.open(io.BytesIO(raw_bytes))
        except Exception as exc:
            raise ValueError(f"Invalid or corrupt image file '{filename}': {exc}") from exc

        fmt = (pil_img.format or Path(filename).suffix.lstrip(".") or "PNG").upper()
        frame_count = int(getattr(pil_img, "n_frames", 1))
        orig_info = dict(pil_img.info) if hasattr(pil_img, "info") else {}

        try:
            pil_img = ImageOps.exif_transpose(pil_img)
        except Exception:
            pass
        pil_img.info.update(orig_info)

        w, h = pil_img.size
        aspect = round(w / max(h, 1), 3)
        mode = pil_img.mode

        exif_meta: Dict[str, str] = {}
        for k, v in orig_info.items():
            if isinstance(k, str) and isinstance(v, (str, int, float)):
                exif_meta[k] = str(v)[:120]

        rgb_img = pil_img.convert("RGB")
        rgb_arr = np.asarray(rgb_img, dtype=np.float32)
        mean_rgb = tuple(int(round(v)) for v in rgb_arr.mean(axis=(0, 1)))

        # Compute spatial quadrant visual activity (edge gradient density in 4 quadrants + center)
        gray_arr = rgb_arr.mean(axis=2)
        gy, gx = np.gradient(gray_arr) if (h >= 4 and w >= 4) else (np.zeros_like(gray_arr), np.zeros_like(gray_arr))
        grad_mag = np.hypot(gx, gy)
        mh, mw = max(1, h // 2), max(1, w // 2)
        quadrants = {
            "sol_ust (top-left)": round(float(grad_mag[:mh, :mw].mean()), 2),
            "sag_ust (top-right)": round(float(grad_mag[:mh, mw:].mean()), 2),
            "sol_alt (bottom-left)": round(float(grad_mag[mh:, :mw].mean()), 2),
            "sag_alt (bottom-right)": round(float(grad_mag[mh:, mw:].mean()), 2),
            "merkez (center)": round(float(grad_mag[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean()) if (h >= 8 and w >= 8) else float(grad_mag.mean()), 2),
        }

        # Run independent trained AnkaLM-Vision model on full image and high-res tiles
        v_model = self._get_vision_model()
        with torch.no_grad():
            full_tensor = pil_to_vision_tensor(rgb_img, v_model.config.image_size).unsqueeze(0)
            logits, _, _ = v_model(full_tensor)
            probs = F.softmax(logits[0], dim=-1)
            top_idx = int(torch.argmax(probs).item())
            top_cat = VISION_SCENE_CATEGORIES[top_idx]
            top_conf = round(float(probs[top_idx].item()), 4)

        raw_tiles = self.extract_high_res_tiles(rgb_img, tile_size=256, overlap=32)
        tiles_info: List[ImageTileInfo] = []
        with torch.no_grad():
            for t_id, (gr, gc, bbox, crop) in enumerate(raw_tiles, start=1):
                c_arr = np.asarray(crop, dtype=np.float32)
                c_gray = c_arr.mean(axis=2)
                c_gy, c_gx = np.gradient(c_gray) if (crop.height >= 4 and crop.width >= 4) else (np.zeros_like(c_gray), np.zeros_like(c_gray))
                t_tensor = pil_to_vision_tensor(crop, v_model.config.image_size).unsqueeze(0)
                t_logits, _, _ = v_model(t_tensor)
                t_probs = F.softmax(t_logits[0], dim=-1)
                t_top = int(torch.argmax(t_probs).item())
                tiles_info.append(
                    ImageTileInfo(
                        tile_id=t_id,
                        grid_row=gr,
                        grid_col=gc,
                        bbox=bbox,
                        mean_brightness=round(float(c_gray.mean()), 1),
                        edge_density=round(float(np.hypot(c_gx, c_gy).mean()), 2),
                        predicted_category=VISION_SCENE_CATEGORIES[t_top],
                        category_confidence=round(float(t_probs[t_top].item()), 4),
                    )
                )

        # Run Local Offline OCR
        ocr_res = self.ocr.process_image(pil_img, page_number=1)

        # Refine category when high-confidence OCR or table grids provide structural certainty
        if ocr_res.detected_table_grids and top_cat not in ("table_grid", "bar_chart"):
            top_cat = "table_grid"

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return ImageAnalysisResult(
            filename=filename,
            format=fmt,
            width=w,
            height=h,
            aspect_ratio=aspect,
            color_mode=mode,
            frame_count=frame_count,
            file_size_bytes=len(raw_bytes),
            exif_metadata=exif_meta,
            dominant_rgb=mean_rgb,
            high_res_tiled=len(tiles_info) > 1,
            tiles_count=len(tiles_info),
            tiles=tiles_info,
            spatial_quadrant_activity=quadrants,
            vision_predicted_category=top_cat,
            vision_category_description_tr=VISION_CATEGORY_DESCRIPTIONS_TR.get(top_cat, top_cat),
            vision_confidence=top_conf,
            vision_model_used=f"{v_model.config.name} (From-Scratch Trained ViT, checkpoints/vision_final)",
            ocr_result=ocr_res.to_dict(),
            processing_time_ms=round(dt_ms, 2),
        )

    def answer_question(self, analysis: ImageAnalysisResult, question: str) -> Dict[str, Any]:
        """
        Answer Turkish visual & OCR questions while honestly distinguishing between
        AnkaLM-Vision's learned visual representation and AnkaLMLocalOCR's text extraction (Section 7.3).
        """
        q_low = question.lower().replace("İ", "i").replace("I", "ı")
        ocr_text = analysis.ocr_result.get("full_text", "").strip()
        ocr_regions = analysis.ocr_result.get("regions", [])
        table_grids = analysis.ocr_result.get("detected_table_grids", [])

        citations = [
            f"[AnkaLM-Vision: {analysis.filename} ({analysis.width}×{analysis.height}) | Kategori: {analysis.vision_predicted_category}]"
        ]
        for r in ocr_regions[:5]:
            citations.append(f"[Görüntü OCR: {analysis.filename} | Bölge {r['bbox']} | Güven: {r['confidence']}]")

        if "kaç tablo" in q_low or "tablo" in q_low:
            nums = [float(x.replace(",", ".")) for x in ocr_text.replace("|", " ").split() if x.replace(",", ".").replace(".", "", 1).isdigit()]
            max_str = f" Tespit edilen en yüksek sayısal değer: **{max(nums)}**." if nums else ""
            return {
                "answer": (
                    f"**Tablo ve Izgara Analizi ({analysis.filename}):**\n"
                    f"- Tespit edilen tablo ızgarası sayısı: **{len(table_grids)}** {table_grids}\n"
                    f"- OCR ile okunan tablo/metin içeriği: `{ocr_text or 'Metin yok'}`.{max_str}\n"
                    f"- *Bileşen Ayrımı:* Izgara ve metin koordinatları `AnkaLMLocalOCR` ile, görsel sınıf (`{analysis.vision_predicted_category}`) `AnkaLM-Vision` ile tespit edilmiştir."
                ),
                "citations": citations,
            }

        if "hata" in q_low or "yazı" in q_low or "oku" in q_low:
            return {
                "answer": (
                    f"**Yerel OCR Metin Okuma Sonucu ({analysis.filename}):**\n"
                    f"- Okunan Metin:\n```\n{ocr_text or '(Görselde okunabilir metin tespit edilemedi)'}\n```\n"
                    f"- Ortalama OCR Güveni: **{analysis.ocr_result.get('average_confidence', 0.0)}** "
                    f"(Belirsiz bölge sayısı: {analysis.ocr_result.get('uncertain_regions_count', 0)})\n"
                    f"- Bölge Koordinatları `[x, y, w, h]`: {[r['bbox'] for r in ocr_regions]}"
                ),
                "citations": citations,
            }

        if "konum" in q_low or "nerede" in q_low:
            return {
                "answer": (
                    f"**Uzamsal Bölge ve Nesne Konum Analizi ({analysis.filename}, {analysis.width}×{analysis.height}):**\n"
                    f"- Kadran Kenar/Nesne Yoğunlukları: `{analysis.spatial_quadrant_activity}`\n"
                    f"- Yüksek Çözünürlüklü Parça (Tile) Sayısı: **{analysis.tiles_count}**\n"
                    f"- OCR Metin Bölgeleri: {[(r['text'], r['bbox']) for r in ocr_regions]}"
                ),
                "citations": citations,
            }

        return {
            "answer": (
                f"**Çok Modlu Görüntü Analizi ({analysis.filename} — {analysis.format}, {analysis.width}×{analysis.height} px):**\n"
                f"1. **AnkaLM-Vision Görsel Sınıflandırma (`{analysis.vision_model_used}`):**\n"
                f"   - Tahmin Edilen Görsel Türü: **{analysis.vision_predicted_category}** — {analysis.vision_category_description_tr} (Güven: {analysis.vision_confidence:.2%})\n"
                f"   - Baskın Renk (RGB): `{analysis.dominant_rgb}` | Yüksek Çözünürlüklü Parçalama: `{analysis.tiles_count} parça`\n"
                f"   - Uzamsal Yoğunluk Dağılımı: `{analysis.spatial_quadrant_activity}`\n"
                f"2. **Yerel OCR Metin Çıkarımı (`{analysis.ocr_result.get('engine_used')}`):**\n"
                f"   - Çıkarılan Metin: `{ocr_text or '(Metin yok)'}` (Bölge sayısı: {len(ocr_regions)}, Tablo ızgarası: {len(table_grids)})"
            ),
            "citations": citations,
        }
