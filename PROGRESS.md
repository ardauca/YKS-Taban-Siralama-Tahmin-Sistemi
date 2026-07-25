# YKS Taban Sıralama Tahmin Sistemi — İlerleme Kaydı (Nihai Rapor)

Son güncelleme: 2026-07-25 | Sistem Production-Ready Durumdadır.

---

## Proje Durumu

Sistem tüm doğrulama, stratifiye analiz, sızıntısız train-CV model seçimi, Google Trends entegrasyonu, router eşik hizalaması (150K), 4'lü veri kalitesi etiketleme (`STRUCTURAL_ANOMALY` vs `HIGH_VOLATILITY`) ve tüm 625 bölüm ailesinin tam genişleme aşamalarını eksiksiz tamamlamıştır.

---

## 1. Nihai Üretim Mimarisi: Hibrit Router (150,000 Eşik)

- **Girdi Yönlendirme Eşiği:** `lag1_taban_siralama < 150,000` (Train CV ampirik kazananı: MAE = 72,153).
- **Rekabetçi Segment (< 150K):** **Model S (HeavyReg GBDT: LightGBM + CatBoost Ensemble)**
- **Kitlesel Segment (≥ 150K):** **Global Model** (500K+ kitlesel segment stabilizasyonu için)

---

## 2. FİNAL NİHAİ STRATİFİYE MAE TABLOSU (TÜRKİYE GENELİ — 2025 TEST YILI, n=15,823)

| Segment / Dilim | Program Sayısı (n) | Eski Global MAE | Nihai Router MAE | İyileşme % | Eski Global R² | Nihai R² |
|---|---|---|---|---|---|---|
| **0–10K (Tıp, Top Müh.)** | 452 | 5,888 | **4,303** | **+%26.9** (1,585 sıra kazanç) | −3.819 | **−1.687** |
| **10K–100K** | 2,434 | 12,916 | **10,633** | **+%17.7** (2,283 sıra kazanç) | 0.171 | **0.278** (+62.6% R² artışı!) |
| **100K–500K** | 4,961 | 50,008 | **49,700** | **+%0.6** (308 sıra kazanç) | 0.671 | **0.669** |
| **500K+** | 7,976 | 129,969 | **129,971** | **−%0.0** (%100 Korundu) | 0.815 | **0.815** |
| **GENEL (TÜM ÜLKE)** | **15,823** | **83,348** | **82,856** | **+%0.6** (492 sıra kazanç) | **0.941** | **0.941** |

---

## 3. Puan Türü Kırılımı İyileşmeleri

- **DİL Puan Türü (n=522):** MAE 11,471 → **7,654** (%33.3 hata azalması).
- **SAY Puan Türü (n=3,769):** MAE 38,745 → **37,880** (%2.2 hata azalması).
- **EA Puan Türü (n=2,857):** MAE 78,031 → **77,229** (%1.0 hata azalması).

---

## 4. Ayrıştırılmış Veri Kalitesi & Anomali Sistemi (`api/main.py`)

Train verisi %95 persentilinden türetilen ampirik dalgalanma eşiği (**293,000 sıra**) ile 4'lü veri kalitesi sınıflandırması:

- `confidence_level`: `VERY_LOW` (<10K), `LOW` (10K–150K), `MEDIUM` (150K–500K), `HIGH` (≥500K).
- `data_quality`:
  - **`SUFFICIENT` (%87.7 / 13,874 program):** Yüksek güvenilirlikli standart programlar.
  - **`INSUFFICIENT` (%5.9 / 938 program):** Taban sıralama / medyan eksik yeni programlar.
  - **`STRUCTURAL_ANOMALY` (%3.6 / 565 program):** Gerçek KKTC ve Yurt Dışı kamu/vakıf statü anormalliği olan programlar.
  - **`HIGH_VOLATILITY` (%2.8 / 446 program):** Tarihsel dalgalanması %95 persentili (>293K) aşan oynak programlar.
