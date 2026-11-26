# Yapay Zeka ve Derin Öğrenme Mimarileri

Derin öğrenme, çok katmanlı yapay sinir ağları kullanarak verilerden hiyerarşik temsiller öğrenen bir makine öğrenimi dalıdır. Modern dil modelleri, dikkat mekanizmasına (attention) dayanan Transformer mimarisini kullanır.

---

# Transformer ve Öz-Dikkat (Self-Attention) Mekanizması

Transformer mimarisinde her belirteç (token), dizideki önceki belirteçlerle olan ilişkisini Sorgu (Query), Anahtar (Key) ve Değer (Value) matris çarpımları aracılığıyla hesaplar. Nedensel maskeleme (causal masking) sayesinde model yalnızca geçmiş belirteçlere bakarak bir sonraki belirteci tahmin eder.

---

# Gruplanmış Sorgu Dikkati (Grouped-Query Attention - GQA)

GQA mimarisinde birden fazla sorgu başlığı (query head) aynı anahtar ve değer (KV) başlığını paylaşır. Bu yöntem, çıkarım sırasında KV önbellek (KV cache) bellek tüketimini önemli ölçüde azaltırken modelin doğruluk oranını korur.

---

# RMSNorm ve SwiGLU Aktivasyon Fonksiyonu

Modern dil modellerinde katman normalizasyonu için RMSNorm (Kök Ortalama Kare Normalizasyonu) tercih edilir. İleri beslemeli katmanlarda kullanılan SwiGLU ise kapılı doğrusal birim (Gated Linear Unit) ile SiLU aktivasyonunu birleştirerek gradyan akışını iyileştirir.

---

# Döner Konum Gömmeleri (Rotary Position Embedding - RoPE)

RoPE, sorgu ve anahtar vektörlerini karmaşık düzlemde konumlarına bağlı açılarla döndürerek göreceli konum bilgisini doğrudan iç çarpıma kodlar. Böylece uzun bağlamlarda konum genellemesi güçlenir.

---

## İkili Arama (Binary Search)

Sıralı bir dizide aranan elemanı O(log n) zaman karmaşıklığı ile bulan temel algoritmadır.

```python
def ikili_arama(dizi, hedef):
    sol, sag = 0, len(dizi) - 1
    while sol <= sag:
        orta = (sol + sag) // 2
        if dizi[orta] == hedef:
            return orta
        elif dizi[orta] < hedef:
            sol = orta + 1
        else:
            sag = orta - 1
    return -1
```

---

## Fibonacci Dizisi ve Dinamik Programlama

Alt problemlerin sonuçlarını bellekte tutarak tekrarlı hesaplamaları önleyen O(n) zaman ve O(1) bellek karmaşıklıklı çözümdür.

```python
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

---

## PyTorch ile Doğrusal Katman ve Gradyan Güncellemesi

PyTorch otomatik türev (autograd) motoru, ileri geçişte hesaplama grafiğini oluşturur ve backward() çağrısıyla ağırlık gradyanlarını hesaplar.

```python
import torch
import torch.nn as nn

katman = nn.Linear(16, 4)
opt = torch.optim.AdamW(katman.parameters(), lr=1e-3)
girdi = torch.randn(8, 16)
cikti = katman(girdi)
kayip = cikti.pow(2).mean()
kayip.backward()
opt.step()
```

---

## SQLite Yerel Veritabanı Sorgulama

SQLite, sunucu gerektirmeyen, yerel dosya tabanlı, güvenilir ve hızlı bir ilişkisel veritabanı motorudur.

```python
import sqlite3

with sqlite3.connect(':memory:') as conn:
    conn.execute('CREATE TABLE belgeler (id INTEGER PRIMARY KEY, baslik TEXT)')
    conn.execute('INSERT INTO belgeler (baslik) VALUES (?)', ('AnkaLM Mimarisi',))
    sonuc = conn.execute('SELECT baslik FROM belgeler').fetchall()
```

---

## Türkçe Harf Duyarlı Küçük Harfe Dönüştürme

Standart ASCII dönüşümleri 'I' harfini yanlışlıkla 'i' yaptığı için Türkçede önce 'İ' -> 'i' ve 'I' -> 'ı' dönüşümü uygulanmalıdır.

```python
def turkce_kucuk_harf(metin: str) -> str:
    return metin.replace('İ', 'i').replace('I', 'ı').lower()
```

---

## Kosinüs Benzerliği Hesaplama

Metin gömmeleri (embeddings) arasındaki anlamsal yakınlığı ölçmek için yerel RAG sistemlerinde yaygın olarak kullanılır.

```python
import math

def kosinus_benzerligi(v1, v2):
    ic_carpim = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return ic_carpim / max(norm1 * norm2, 1e-8)
```
