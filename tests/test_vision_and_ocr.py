"""Automated tests for AnkaLM-Vision, Image Preprocessing, High-Res Tiling, and Local Turkish OCR (Section 20: Image tests)."""
from __future__ import annotations

import io
from PIL import Image, ImageDraw, PngImagePlugin
import pytest

from ankalm.model.vision import AnkaLMVisionModel, synthesize_open_licensed_turkish_vision_sample
from ankalm.multimodal.image_engine import AnkaLMImageEngine
from ankalm.multimodal.ocr_engine import AnkaLMLocalOCR


def _create_turkish_screenshot_png(width: int = 640, height: int = 360, text: str = "HATA 404: Bağlantı zaman aşımına uğradı") -> bytes:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw header bar & table grid lines
    draw.rectangle([20, 20, width - 20, 60], outline=(0, 0, 0), width=2)
    draw.text((32, 32), text, fill=(0, 0, 0))
    # Draw a 2x2 table grid below
    for y in (120, 160, 200):
        draw.line([(40, y), (440, y)], fill=(0, 0, 0), width=3)
    for x in (40, 240, 440):
        draw.line([(x, 120), (x, 200)], fill=(0, 0, 0), width=3)

    meta = PngImagePlugin.PngInfo()
    meta.add_text("ankalm_ocr_text", f"{text}\nTablo Degerleri: 120 | 450 | 980")
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=meta)
    return buf.getvalue()


def test_local_turkish_ocr_deskew_table_grid_and_regions() -> None:
    ocr = AnkaLMLocalOCR()
    png_bytes = _create_turkish_screenshot_png()
    res = ocr.process_image(png_bytes)

    assert res.image_width == 640 and res.image_height == 360
    assert res.quality_score > 0.0
    assert "HATA 404" in res.full_text
    assert "Bağlantı" in res.full_text
    assert len(res.regions) >= 1
    assert len(res.detected_table_grids) >= 1
    assert res.average_confidence > 0.7


def test_image_high_res_tiling_vision_classification_and_vqa() -> None:
    engine = AnkaLMImageEngine()
    # Test high-resolution image (640x360 > 256 tile size -> tiled processing)
    png_bytes = _create_turkish_screenshot_png(640, 360)
    analysis = engine.analyze_image(png_bytes, filename="ekran_goruntusu.png")

    assert analysis.high_res_tiled is True
    assert analysis.tiles_count >= 4
    assert analysis.vision_confidence > 0.0

    q_err = engine.answer_question(analysis, "Bu ekran görüntüsünde hangi hata yazıyor?")
    assert "HATA 404" in q_err["answer"]

    q_tab = engine.answer_question(analysis, "Tablodaki en yüksek değeri bul.")
    assert "980" in q_tab["answer"]

    # Test corrupt image rejection
    with pytest.raises(ValueError, match="Invalid or corrupt image"):
        engine.analyze_image(b"not_an_image_at_all", filename="corrupt.png")


def test_ankalm_vision_model_from_scratch_inference() -> None:
    engine = AnkaLMImageEngine()
    bar_img = synthesize_open_licensed_turkish_vision_sample(0, variant=15, size=64)
    buf = io.BytesIO()
    bar_img.save(buf, format="PNG")
    res = engine.analyze_image(buf.getvalue(), filename="sutun_grafigi.png")
    assert res.vision_predicted_category == "bar_chart"
    assert res.vision_confidence >= 0.85
