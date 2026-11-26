# Turkish Evaluation Lab — AnkaLM

## 16-Category Benchmark Suite (`ankalm evaluate`)
The Turkish Evaluation Lab (`ankalm/evaluation/suite.py`) evaluates trained checkpoints across:
1. `turkish_spelling` (Türkçe Yazım Kuralları)
2. `grammar` (Türkçe Dilbilgisi ve Sözdizimi)
3. `morphology` (Biçimbilim — Kök ve Ek Yapısı)
4. `suffixes` (Ünlü Uyumu ve Çekim Ekleri)
5. `reading_comprehension` (Okuduğunu Anlama)
6. `sentence_completion` (Cümle Tamamlama)
7. `explanations` (Kavram Açıklama)
8. `question_answering` (Soru Yanıtlama)
9. `summarization` (Metin Özetleme)
10. `translation` (Türkçe-İngilizce Çeviri)
11. `rewriting` (Resmî Üslupla Yeniden Yazma)
12. `basic_reasoning` (Temel Akıl Yürütme)
13. `basic_mathematics` (Temel Matematik)
14. `programming_explanations` (Programlama Açıklamaları)
15. `json_output` (Yapılandırılmış JSON Çıktısı)
16. `multi_turn_context` (Çok Turlu Bağlam Takibi)

## Measured Results
- **Base Pretrained (`small_final`):** Held-out evaluation loss = `4.80614`, held-out perplexity = `122.259`, overall benchmark score = **`92.45%`**.
- **Instruction-Tuned (`small_sft`):** Instruction validation loss = `2.50686`, instruction validation perplexity = `12.2664`, overall benchmark score = **`86.69%`**.
