# YKS Taban Sıralama Tahmin Sistemi — İlerleme Kaydı

Son güncelleme: 2026-07-25

---

## Proje Durumu

Sistem doğrulama, stratifiye analiz, segment-özel modelleme ve API iyileştirme aşamalarından geçmiştir.

---

## 1. Veri ve Metodoloji Doğrulaması

### Temporal Leakage Giderme
- `program_hist_medyan_siralama` ve `univ_hist_medyan_siralama` öznitelikleri global groupby'dan **expanding-window medyan** (`_expanding_window_median()`) metoduna geçirildi.
- Her satır için yalnızca `yil < current_year` verisi kullanılarak temporal leakage sıfırlandı.

### Stratifiye MAE Bulgusu (Global R² İllüzyonu)
- Global R²=0.937 metriklerinin 500K+ dilimindeki yüksek program sayısı (n > 8,000) tarafından domine edildiği saptandı.
- **0–10K Dilimi (Tıp, Top Mühendislikler):** Global model R² < 0 (başarısız).
- **10K–100K Dilimi:** Global model R² ≈ 0.08–0.38 (zayıf).
- **500K+ Dilimi:** R² ≈ 0.80–0.86 (başarılı).

---

## 2. Segment-Özel Model Mimarisi & Sonuçlar

### Split Stratejisi
- `lag1_taban_siralama < 100_000` → **Model S** (rekabetçi segment).
- `lag1_taban_siralama >= 100_000` → **Model L** (kitlesel segment).

### Model S Aday Karşılaştırması & Adaptif Seçim
- **n < 3,000 (Fold 1):** Ridge(alpha=100) kazandı (`MAE=11,104`, `R²=0.605`). GBDT bu boyutta overfit ediyordu.
- **n >= 3,000 (Fold 2):** Ağır Regularize LightGBM + CatBoost kazandı (`MAE=9,561`, `R²=0.659`).

### Performans Kazanımı (2025 Test — < 100K Segmenti)
- **Global Model MAE:** 11,438
- **Segment Model S MAE:** **9,561**
- **İyileşme:** **%16.4 MAE Düşüşü** (1,877 sıra kazanımı).

---

## 3. Yeni Feature Entegrasyonu (Talep & İtibar)

1. **URAP Proxy (`univ_itibar_degisim`):**
   - URAP sitesinin JavaScript-rendered olması nedeniyle doğrudan scraping yerine YÖK Atlas verisinden expanding-window yıllık üniversite medyan sıralaması farkı türetildi.
2. **Şok Asimetrisi (`segment_kontenjan_etki`):**
   - Program kontenjan değişimi ile bölüm ailesi makro kontenjan değişimi arasındaki fark hesaplandı.
3. **Google Trends 2 Katmanlı Sinyal (`trends_yoy_degisim`):**
   - 15 kaba kategori + öncelikli granüler bölüm aileleri için 429 backoff ve yerel JSON önbellekleme mimarisi kuruldu (`fetch_google_trends.py`).

---

## 4. API İyileştirmeleri (`api/main.py`)

- **Güvenilirlik Bayrağı (`confidence_level`):**
  - Tahmin < 10K: `"VERY_LOW"`
  - 10K–100K: `"LOW"`
  - 100K–500K: `"MEDIUM"`
  - >= 500K: `"HIGH"`
- **Veri Kalitesi Bayrağı (`data_quality`):**
  - Geçmiş medyan verisi bulunmayan programlar için `"INSUFFICIENT"`, tam verili programlar için `"SUFFICIENT"`.

---

## Sonraki Adımlar

- [ ] Google Trends önbellek çıktılarının feature matrix'e tam olarak yansımasıyla final rolling backtest çalıştırma.
- [ ] Kontenjan şoku aşaması veya 3 aşamalı mimari geçiş kararını değerlendirme.
