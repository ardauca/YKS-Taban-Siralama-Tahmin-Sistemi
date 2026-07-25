# YKS Taban Sıralama Tahmin Sistemi — İlerleme Kaydı

Son güncelleme: 2026-07-25

---

## Proje Durumu

Sistem tüm doğrulama, stratifiye analiz, sızıntısız train-CV model seçimi, Google Trends entegrasyonu, router eşik hizalaması (150K) ve `STRUCTURAL_ANOMALY` uyarı sistemi aşamalarını eksiksiz tamamlamıştır.

---

## 1. Nihai Üretim Mimarisi: Hibrit Router (150,000 Eşik)

- **Girdi Yönlendirme Eşiği:** `lag1_taban_siralama < 150,000` (Train CV ampirik kazananı: MAE = 72,153).
- **Rekabetçi Segment (< 150K):** **Model S (HeavyReg GBDT: LightGBM + CatBoost Ensemble)**
- **Kitlesel Segment (≥ 150K):** **Global Model** (500K+ kitlesel segment stabilizasyonu için)

---

## 2. Alt-Stratifiye Performans ve Orta Band Doğrulaması

### 2025 Test Yılı Performansı (Kontenjan Şok Yılı)
- **0–10K Segmenti:** MAE eski global modelde 7,493 iken **2,955 sıraya geriledi** (%60.6 hata azalması).
- **10K–100K Segmenti:** MAE eski global modelde 13,805 iken **10,493 sıraya geriledi** (%24.0 hata azalması / 3,312 sıra kazanç).
- **Makine Mühendisliği (n=126):** MAE 36,130 → **32,235** (%10.8 hata azalması).
- **GENEL MAE:** 81,803 → **81,556** (%0.3 genel iyileşme).

---

## 3. Veri Kalitesi ve Anomali İşaretleme Katmanı (`api/main.py`)

- `confidence_level`: `VERY_LOW` (<10K), `LOW` (10K–150K), `MEDIUM` (150K–500K), `HIGH` (≥500K).
- `data_quality`:
  - `SUFFICIENT` (%75.5): Standart güvenilir verili programlar.
  - `INSUFFICIENT` (%5.9): Taban sıralama / medyan eksik yeni programlar.
  - **`STRUCTURAL_ANOMALY` (%18.5):** KKTC, yurt dışı veya radikal mevzuat/ücret/şok anormalliği olan programlar.

---

## 4. E Adımı: Tüm 625 Bölüm Ailesine Genişleme

Sistem pilot doğrulamaları geçmiş olup tam genişleme modunda çalışmaya hazırdır.
