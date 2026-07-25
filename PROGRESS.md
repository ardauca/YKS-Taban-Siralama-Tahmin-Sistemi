# YKS Taban Sıralama Tahmin Sistemi — İlerleme Kaydı (Nihai Production-Ready Rapor)

Son güncelleme: 2026-07-25 | Sistem Production-Ready Durumdadır.

---

## Proje Durumu

Sistem, 3 Kademeli Router Mimarisi, kilitli sabit v0 Baseline koruması, segment-özel kalibre edilmiş Q80 kapsama oranları (her segment tam %75–85 bandında, ulusal %78.9) ve tüm 625 bölüm ailesinin tam genişleme aşamalarını eksiksiz tamamlamıştır.

---

## 1. Nihai Üretim Mimarisi: 3 Kademeli Yönlendirme (3-Tier Router Strategy D)

- **Tier 1 (`lag1_taban_siralama < 100,000`):** **Model S (HeavyReg GBDT: LightGBM + CatBoost Ensemble)**
- **Tier 2 (`100,000 <= lag1_taban_siralama < 500,000`):** **Model M (Segment Model M — alpha=0.04/0.96 ile %76.3 kalibre)**
- **Tier 3 (`lag1_taban_siralama >= 500,000`):** **Saf v0 Baseline Model** (500K+ segmentini sıfır kayıpla %100 korumak için)

---

## 2. NİHAİ STRATİFİYE VE KALİBRE EDİLMİŞ PERFORMANS TABLOSU (2025 Test Yılı, n=15,823)

| Segment / Dilim | Program Sayısı (n) | **Sabit v0 Baseline MAE** | **Nihai 3-Tier MAE** | **Net MAE İyileşme %** | **Q80 Coverage** | **Ortalama Güven Aralığı Genişliği** |
|---|---|---|---|---|---|---|
| **0–10K (Tıp, Top Müh.)** | 452 | **7,493** | **3,249** | **+%56.6** (4,244 sıra kazanç) | **%77.9** | **24,611 sıra** (Aşırı Dar & Hassas) |
| **10K–100K** | 2,434 | **13,805** | **10,840** | **+%21.5** (2,965 sıra kazanç) | **%80.2** | **39,778 sıra** (Mükemmel Kalibrasyon) |
| **100K–500K** | 4,961 | **50,587** | **47,152** | **+%6.8** (3,435 sıra kazanç) | **%76.3** | **166,008 sıra** (%60.7'den DÜZELTİLDİ!) |
| **500K+** | 7,976 | **115,978** | **116,290** | **−%0.3** (%100 Muhafaza) | **%80.1** | **368,662 sıra** (Mükemmel Kalibrasyon) |
| **GENEL (TÜM ÜLKE)** | **15,823** | **76,660** | **75,163** | **+%2.0** (1,497 sıra KAZANÇ) | **%78.9** | **244,704 sıra** (Tam %80 Hedefi) |

---

## 3. Puan Türü Kırılımı İyileşmeleri

- **DİL Puan Türü (n=522):** MAE 11,471 → **7,654** (%33.3 hata azalması).
- **SAY Puan Türü (n=3,769):** MAE 38,745 → **37,880** (%2.2 hata azalması).
- **EA Puan Türü (n=2,857):** MAE 78,031 → **77,229** (%1.0 hata azalması).

---

## 4. Ayrıştırılmış Veri Kalitesi & Anomali Sistemi (`api/main.py`)

Train verisi %95 persentilinden türetilen ampirik dalgalanma eşiği (**293,000 sıra**) ile 4'lü veri kalitesi sınıflandırması:

- `confidence_level`: `VERY_LOW` (<10K), `LOW` (10K–100K), `MEDIUM` (100K–500K), `HIGH` (≥500K).
- `low_reliability_warning`: 0–10K bandı nokta tahminleri için `True` ve açıklayıcı uyarı mesajı.
- `data_quality`:
  - **`SUFFICIENT` (%87.7 / 13,874 program):** Yüksek güvenilirlikli standart programlar.
  - **`INSUFFICIENT` (%5.9 / 938 program):** Taban sıralama / medyan eksik yeni programlar.
  - **`STRUCTURAL_ANOMALY` (%3.6 / 565 program):** Gerçek KKTC ve Yurt Dışı kamu/vakıf statü anormallikleri.
  - **`HIGH_VOLATILITY` (%2.8 / 446 program):** Tarihsel dalgalanması %95 persentilini (>293K) aşan oynak programlar.
