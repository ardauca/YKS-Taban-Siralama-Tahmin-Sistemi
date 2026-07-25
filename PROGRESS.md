# YKS Taban Sıralama Tahmin Sistemi — İlerleme Kaydı

Son güncelleme: 2026-07-25

---

## Proje Durumu

Sistem tüm doğrulama, stratifiye analiz, sızıntısız train-CV model seçimi, Google Trends entegrasyonu, router eşik hizalaması (150K), 4'lü veri kalitesi etiketleme (`STRUCTURAL_ANOMALY` vs `HIGH_VOLATILITY`) ve tam genişleme doğrulama aşamalarını tamamlamıştır.

---

## 1. Nihai Üretim Mimarisi: Hibrit Router (150,000 Eşik)

- **Girdi Yönlendirme Eşiği:** `lag1_taban_siralama < 150,000` (Train CV ampirik kazananı: MAE = 72,153).
- **Rekabetçi Segment (< 150K):** **Model S (HeavyReg GBDT: LightGBM + CatBoost Ensemble)**
- **Kitlesel Segment (≥ 150K):** **Global Model** (500K+ kitlesel segment stabilizasyonu için)

---

## 2. Alt-Stratifiye Performans ve Puan Türü Doğrulaması

### 2025 Test Yılı Performansı (Kontenjan Şok Yılı)
- **0–10K Segmenti:** MAE eski global modelde 7,493 iken **2,955 sıraya geriledi** (%60.6 hata azalması).
- **10K–100K Segmenti:** MAE eski global modelde 13,805 iken **10,493 sıraya geriledi** (%24.0 hata azalması / 3,312 sıra kazanç).
- **DİL Puan Türü (n=522):** MAE 11,471 → **7,654** (%33.3 hata azalması).
- **SAY Puan Türü (n=3,769):** MAE 38,745 → **37,880** (%2.2 hata azalması).
- **EA Puan Türü (n=2,857):** MAE 78,031 → **77,229** (%1.0 hata azalması).
- **GENEL MAE:** 81,803 → **81,556** (%0.3 genel iyileşme).

---

## 3. Ayrıştırılmış Veri Kalitesi & Anomali Sistemi (`api/main.py`)

Train verisi %95 persentilinden türetilen ampirik dalgalanma eşiği (**293,000 sıra**) ile 4'lü veri kalitesi sınıflandırması kurulmuştur:

- `confidence_level`: `VERY_LOW` (<10K), `LOW` (10K–150K), `MEDIUM` (150K–500K), `HIGH` (≥500K).
- `data_quality`:
  - **`SUFFICIENT` (%87.7 / 13,874 program):** Yüksek güvenilirlikli standart programlar.
  - **`INSUFFICIENT` (%5.9 / 938 program):** Taban sıralama / medyan eksik yeni programlar.
  - **`STRUCTURAL_ANOMALY` (%3.6 / 565 program):** Gerçek KKTC ve Yurt Dışı kamu/vakıf statü anormalliği olan programlar.
  - **`HIGH_VOLATILITY` (%2.8 / 446 program):** Tarihsel dalgalanması %95 persentili (>293K) aşan oynak programlar.

---

## 4. E Adımı: Tam Genişleme

Sistem 625 bölüm ailesinin tamamı üzerinde üretime hazır hale getirilmiştir.
