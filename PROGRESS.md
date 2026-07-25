# YKS Taban Sıralama Tahmin Sistemi — İlerleme Kaydı

Son güncelleme: 2026-07-25

---

## Proje Durumu

Sistem doğrulama, stratifiye analiz, sızıntısız train-CV model seçimi, Google Trends entegrasyonu ve pilot bölüm doğrulaması aşamalarını tamamlamıştır.

---

## 1. Veri ve Metodoloji Doğrulaması

### Temporal Leakage Giderme
- `program_hist_medyan_siralama` ve `univ_hist_medyan_siralama` öznitelikleri global groupby'dan **expanding-window medyan** (`_expanding_window_median()`) metoduna geçirildi.
- Her satır için yalnızca `yil < current_year` verisi kullanılarak temporal sızıntı engellendi.

### Alt-Stratifiye Performans (0–10K ve 10K–100K)
- **0–10K Segmenti:** MAE eski global modelde 7,493 iken segment modelinde **2,955 sıraya geriledi** (%60.6 hata azalması).
- **10K–100K Segmenti:** MAE eski global modelde 13,805 iken **10,493 sıraya geriledi** (%24.0 hata azalması / 3,312 sıra kazanç).
- R² 10K–100K segmentinde 0.111'den **0.202'ye** yükseldi.

---

## 2. Nihai Production Mimarisi: Hibrit Router

### Router Mantığı
- `lag1_taban_siralama < 100_000` → **Model S (HeavyReg GBDT)**
- `lag1_taban_siralama >= 100_000` → **Global Model** (500K+ kitlesel segment stabilizasyonu için)

### Train İç Validasyon 5-Fold CV Seçimi (Sızıntısız)
Model S seçimi SADECE Train setleri üzerinde 5-Fold CV ile yapılmıştır:
- Ridge (alpha=100) Train-CV MAE: `9,072`
- ElasticNet Train-CV MAE: `21,818`
- **HeavyReg-GBDT Train-CV MAE: `7,994`** (5 iç fold'un 5'inde de kazanan)

---

## 3. Pilot Bölüm Ailesi Doğrulaması (Overfitting Kontrolü)

Bilgisayar Mühendisliği dışındaki 3 farklı karakterdeki bölüm ailesinde test edilmiştir (2025 Test Yılı):

| Bölüm Ailesi | Program Sayısı (n) | Eski Global MAE | Hibrit Router MAE | İyileşme % |
|---|---|---|---|---|
| **Tıp (Yüksek Rekabet)** | 184 | 4,029 | **3,126** | **+%22.4** |
| **Hukuk (Orta Rekabet)** | 121 | 12,295 | **12,414** | -%1.0 |
| **Gastronomi (Değişken)** | 128 | 44,248 | **42,388** | **+%4.2** |
| **TÜM 625 BÖLÜM AİLESİ** | 15,823 | 81,803 | **81,556** | **+%0.3** |

---

## 4. API Güvenilirlik Bayrakları (`api/main.py`)

- `confidence_level`: `VERY_LOW` (<10K), `LOW` (10K–100K), `MEDIUM` (100K–500K), `HIGH` (≥500K).
- `data_quality`: Geçmiş medyanı bulunmayan programlar için `INSUFFICIENT`, tam verili programlar için `SUFFICIENT`.
