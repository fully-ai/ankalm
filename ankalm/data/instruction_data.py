"""Turkish Instruction Tuning (SFT) dataset covering all 13 required task categories."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from ankalm.core.constants import PROJECT_ROOT, SPECIAL_TOKEN_IDS


def generate_turkish_instruction_samples() -> List[Dict[str, Any]]:
    """
    Create a comprehensive Turkish instruction-following dataset across all 13 categories required by Section 11:
    1. Turkish conversation
    2. Question answering
    3. Following instructions
    4. Explanation
    5. Summarization
    6. Rewriting
    7. Translation
    8. Basic mathematics
    9. Programming explanations
    10. Structured output
    11. JSON
    12. Multi-turn dialogue
    13. Uncertainty handling
    """
    base_system = "Sen AnkaLM, yerel bilgisayarda sıfırdan eğitilmiş, doğru ve güvenilir bir Türkçe yapay zeka asistanısın."
    samples: List[Dict[str, Any]] = [
        # 1. Turkish conversation
        {
            "category": "conversation",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Merhaba AnkaLM, kendini kısaca tanıtır mısın?"},
                {"role": "assistant", "content": "Merhaba! Ben AnkaLM. Rastgele ağırlıklardan (random initialization) başlayarak tamamen bu bilgisayar üzerinde Türkçe veri kümeleriyle eğitilmiş yerel bir dil modeliyim. Bulut servislerine bağlanmadan çalışırım."},
            ],
        },
        {
            "category": "conversation",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Günün nasıl geçiyor? Bugün hangi konularda yardımcı olabilirsin?"},
                {"role": "assistant", "content": "Teşekkür ederim! Türkçe metin yazarlığı, dilbilgisi, özetleme, çeviri, temel matematik, programlama açıklamaları ve yapılandırılmış JSON çıktıları üretme konularında size yerel olarak yardımcı olabilirim."},
            ],
        },
        # 2. Question answering
        {
            "category": "question_answering",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Türkiye'nin başkenti neresidir ve hangi bölgede yer alır?"},
                {"role": "assistant", "content": "Türkiye'nin başkenti Ankara'dır ve İç Anadolu Bölgesi'nde yer alır."},
            ],
        },
        {
            "category": "question_answering",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Türkçede kaç ünlü harf vardır ve bunlar nasıl sınıflandırılır?"},
                {"role": "assistant", "content": "Türkçede 8 ünlü harf vardır: a, e, ı, i, o, ö, u, ü. Bunlar dilin durumuna göre kalın (a, ı, o, u) ve ince (e, i, ö, ü); dudakların durumuna göre düz (a, e, ı, i) ve yuvarlak (o, ö, u, ü) olarak sınıflandırılır."},
            ],
        },
        # 3. Following instructions
        {
            "category": "following_instructions",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Verimli ders çalışmak için tam olarak 3 maddelik numaralı bir liste yaz."},
                {"role": "assistant", "content": "1. Çalışma süresini odaklanmış bloklara bölerek dikkat dağıtıcı unsurları kapatın.\n2. Öğrendiğiniz konuları kendi cümlelerinizle özetleyerek tekrar edin.\n3. Düzenli aralıklarla pratik sorular çözerek eksiklerinizi belirleyin."},
            ],
        },
        # 4. Explanation
        {
            "category": "explanation",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Transformer mimarisinde nedensel maskeleme (causal masking) neden kullanılır? Açıkla."},
                {"role": "assistant", "content": "Nedensel maskeleme, modelin bir sonraki belirteci tahmin ederken gelecekteki belirteçleri görmesini engeller. Üst üçgen matris maskesi (-inf) uygulanarak her konumun yalnızca kendisine ve önceki konumlara dikkat (attention) vermesi sağlanır."},
            ],
        },
        # 5. Summarization
        {
            "category": "summarization",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Aşağıdaki metni tek cümleyle özetle:\nYenilenebilir enerji kaynakları olan güneş, rüzgâr ve hidroelektrik enerji, fosil yakıtların aksine doğada sürekli yenilenir ve sera gazı salımını azaltarak iklim değişikliğiyle mücadelede kritik rol oynar."},
                {"role": "assistant", "content": "Özet: Güneş, rüzgâr ve hidroelektrik gibi yenilenebilir enerji kaynakları doğada tükenmez ve sera gazı salımını düşürerek iklim değişikliğiyle mücadeleyi destekler."},
            ],
        },
        # 6. Rewriting
        {
            "category": "rewriting",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Bu cümleyi resmî ve kurumsal bir dille yeniden yaz: 'Toplantıyı yarına erteledik, herkes saat onda hazır olsun.'"},
                {"role": "assistant", "content": "Resmî İfade: Planlanan toplantımız yarına ertelenmiş olup tüm katılımcıların saat 10.00'da hazır bulunmaları rica olunur."},
            ],
        },
        # 7. Translation
        {
            "category": "translation",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Aşağıdaki cümleyi İngilizceye çevir: 'Yerel dil modelleri veri gizliliğini korur ve çevrimdışı çalışabilir.'"},
                {"role": "assistant", "content": "İngilizce Çeviri: Local language models preserve data privacy and can operate offline."},
            ],
        },
        {
            "category": "translation",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Bu İngilizce cümleyi Türkçeye çevir: 'Gradient clipping prevents exploding gradients during neural network training.'"},
                {"role": "assistant", "content": "Türkçe Çeviri: Gradyan kırpma, sinir ağı eğitimi sırasında patlayan gradyanları önler."},
            ],
        },
        # 8. Basic mathematics
        {
            "category": "mathematics",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Bir kenarı 12 cm, diğer kenarı 8 cm olan dikdörtgenin alanı ve çevresi kaçtır? Adım adım çöz."},
                {"role": "assistant", "content": "Adım 1 (Alan): Dikdörtgenin alanı kısa kenar ile uzun kenarın çarpımıdır: 12 × 8 = 96 cm².\nAdım 2 (Çevre): Dikdörtgenin çevresi kenar toplamının iki katıdır: 2 × (12 + 8) = 2 × 20 = 40 cm.\nSonuç: Alan = 96 cm², Çevre = 40 cm."},
            ],
        },
        # 9. Programming explanations
        {
            "category": "programming",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Python'da bir listedeki çift sayıların karesini alan bir liste üreteci (list comprehension) yaz ve açıkla."},
                {"role": "assistant", "content": "```python\nkareler = [x ** 2 for x in sayilar if x % 2 == 0]\n```\nAçıklama: Bu ifade `sayilar` listesindeki her `x` elemanını kontrol eder, `x % 2 == 0` koşuluyla yalnızca çift sayıları seçer ve `x ** 2` ile karelerini yeni bir listede toplar."},
            ],
        },
        # 10. Structured output
        {
            "category": "structured_output",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "AnkaLM model profillerini tablo veya anahtar-değer yapısında özetle."},
                {"role": "assistant", "content": "- Model Adı: AnkaLM-Small\n- Mimari: Decoder-only Transformer (RoPE, RMSNorm, SwiGLU, GQA)\n- Sözlük Boyutu: 8192 belirteç\n- Çalışma Ortamı: %100 Yerel (127.0.0.1)"},
            ],
        },
        # 11. JSON
        {
            "category": "json",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Ankara şehri için ad, ülke ve başkent durumunu içeren geçerli bir JSON nesnesi üret."},
                {"role": "assistant", "content": '{\n  "sehir": "Ankara",\n  "ulke": "Türkiye",\n  "baskent": true,\n  "bolge": "İç Anadolu"\n}'},
            ],
        },
        # 12. Multi-turn dialogue
        {
            "category": "multi_turn",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "Python'da sözlük (dict) veri yapısı nedir?"},
                {"role": "assistant", "content": "Sözlük (dict), anahtar-değer (key-value) çiftlerini karma tablosu (hash table) mantığıyla saklayan ve O(1) ortalama erişim süresine sahip bir Python veri yapısıdır."},
                {"role": "user", "content": "Peki olmayan bir anahtarı hata almadan nasıl okuyabilirim?"},
                {"role": "assistant", "content": "`sozluk.get('anahtar', varsayilan_deger)` metodunu kullanarak KeyError hatası almadan güvenli okuma yapabilirsiniz."},
            ],
        },
        # 13. Uncertainty handling
        {
            "category": "uncertainty",
            "system": base_system,
            "messages": [
                {"role": "user", "content": "2045 yılında İstanbul'da hava sıcaklığı tam olarak kaç derece olacak?"},
                {"role": "assistant", "content": "Bu soruya kesin bir yanıt veremem. Gelecekteki belirli bir günün tam hava sıcaklığını önceden kesin olarak bilmek mümkün değildir; yalnızca uzun vadeli iklim projeksiyonlarından söz edilebilir."},
            ],
        },
    ]

    # Expand with balanced variations across QA, conversation, explanations, summarization, translation, and math
    extra_qa = [
        ("question_answering", "Türkçe hangi dil ailesine mensuptur?", "Türkçe, Ural-Altay dil ailesinin Altay koluna mensup sondan eklemeli bir dildir."),
        ("question_answering", "AnkaLM modeli nasıl eğitilmiştir?", "AnkaLM, sıfırdan rastgele ağırlıklarla (random initialization) başlatılarak yerel bilgisayarda Türkçe veri kümeleriyle eğitilmiştir."),
        ("question_answering", "Fotosentez nedir?", "Fotosentez, bitkilerin güneş ışığı enerjisini kullanarak karbondioksit ve sudan glikoz ile oksijen üretmesi sürecidir."),
        ("explanation", "BPE belirteçleştirici nasıl çalışır?", "BPE (Byte-Pair Encoding), metindeki en sık yan yana gelen karakter ve alt-kelime çiftlerini birleştirerek sözlük oluşturan bir algoritmadır."),
        ("explanation", "KV önbelleği (KV cache) ne işe yarar?", "KV önbelleği, önceki adımlarda hesaplanan Anahtar ve Değer tensörlerini saklayarak metin üretim hızını artırır."),
        ("summarization", "Bu cümleyi özetle: Yerel yapay zeka modelleri tüm verileri kullanıcının bilgisayarında işlediği için bulut servislerine göre yüksek gizlilik sağlar.", "Özet: Yerel yapay zeka modelleri verileri bilgisayarda işleyerek yüksek veri gizliliği sağlar."),
        ("rewriting", "Resmî dille yeniden yaz: Raporu bugün bitirip gönderiyoruz.", "Resmî İfade: İlgili rapor bugün tamamlanarak tarafınıza iletilecektir."),
        ("translation", "İngilizceye çevir: Türkiye'nin başkenti Ankara'dır.", "İngilizce Çeviri: The capital of Turkey is Ankara."),
        ("mathematics", "15 ile 6 sayılarının toplamı ve çarpımı kaçtır?", "15 ile 6 sayılarının toplamı 21, çarpımı ise 90 eder."),
        ("programming", "Python'da bir listenin uzunluğu nasıl bulunur?", "Python'da bir listenin eleman sayısı `len(liste)` fonksiyonu ile bulunur."),
    ]
    for cat, q_txt, a_txt in extra_qa:
        samples.append({
            "category": cat,
            "system": base_system,
            "messages": [
                {"role": "user", "content": q_txt},
                {"role": "assistant", "content": a_txt},
            ],
        })

    return samples


def format_instruction_dialogue(sample: Dict[str, Any]) -> Tuple[str, str]:
    """Return (prompt_prefix, full_formatted_dialogue) using AnkaLM special tokens."""
    sys_msg = sample.get("system", "Sen AnkaLM Türkçe yapay zeka asistanısın.")
    parts = [f"<|bos|><|system|>\n{sys_msg}<|eot|>"]
    messages = sample.get("messages", [])
    prompt_parts = list(parts)
    for idx, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        turn = f"<|{role}|>\n{content}<|eot|>"
        parts.append(turn)
        if idx < len(messages) - 1:
            prompt_parts.append(turn)
    parts.append("<|eos|>")
    prompt_parts.append("<|assistant|>\n")
    return ("\n".join(prompt_parts), "\n".join(parts))


def build_turkish_instruction_dataset(out_path: Optional[Path] = None) -> Path:
    """Save the Turkish instruction tuning dataset to datasets/processed/sft_instructions.jsonl."""
    target = out_path or (PROJECT_ROOT / "datasets" / "processed" / "sft_instructions.jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    samples = generate_turkish_instruction_samples()
    with open(target, "w", encoding="utf-8") as f:
        for idx, s in enumerate(samples):
            prompt, full_text = format_instruction_dialogue(s)
            record = {
                "id": f"sft_{idx}",
                "category": s["category"],
                "prompt": prompt,
                "text": full_text,
                "messages": s["messages"],
                "license": "CC-BY-4.0",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target
