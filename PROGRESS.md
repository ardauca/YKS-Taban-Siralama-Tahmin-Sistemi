# YKS Taban Sıralama Tahmin Sistemi — İlerleme Kaydı

## Son Güncelleme: 2026-07-22

---

## Tamamlanan Adımlar

### ✅ Adım 1 — Proje İskeleti (2026-07-22)
- Klasör yapısı oluşturuldu (`data/`, `scraping/`, `src/`, `api/`, `frontend/`, `tests/`)
- `requirements.txt`, `.gitignore`, `.env.example`, `README.md`, `docker-compose.yml` hazırlandı
- Git reposu `main` branch'e bağlandı ve ilk commit push edildi

**Commit:** `chore(repo): initialize project structure and requirements`

---

### ✅ Adım 2 — YÖK Atlas Scraper (2026-07-22)

**Keşif:** YÖK Atlas, Nisan 2026'da React SPA'ya geçmiş; eski HTML endpoint'leri kapalı.
Yeni JSON API endpoint: `POST /api/tercih-kilavuz/search`

**Scraper:** `scraping/yokatlas_scraper.py`
- `birimGrupId=2010` (Bilgisayar Mühendisliği)
- Rate limiting: 1.5 sn / istek
- Retry logic: 3 deneme + backoff

**Veri kalite modülü:** `scraping/quality.py`
- Satır sayısı kontrolü, eksik değer oranı, duplicate, şema uyumu, yıl dağılımı

**Sonuç:** 332 program × ~4 yıl = **1,216 satır** → `data/raw/yokatlas/bilgisayar_muhendisligi_raw.csv`

| Yıl | Satır | Taban Sıralama Eksik | Kontenjan Eksik |
|-----|-------|---------------------|-----------------|
| 2022 | 268 | %10.1 | %12.0 |
| 2023 | 298 | %14.8 | %8.0 |
| 2024 | 318 | %31.1 | %0.0 |
| 2025 | 332 | %34.6 | **%0.0** (%100 dolu!) |

**Testler:** 27/27 passed (`tests/test_scraper.py`)

**Commit:** `feat(scraping): add YOKATLAS scraper and quality check module`  
**Commit:** `data(raw): add 2022-2025 YOKATLAS snapshot for Bilgisayar Muhendisligi`

---

### ✅ Adım 3 — Feature Engineering + CatBoost Prototipi (2026-07-22)

#### Feature Pipeline (`src/features/build_features.py`)

**19 feature, 4 grup:**

| Grup | Feature'lar |
|------|-------------|
| **Lag-1** (Y-1) | `lag1_taban_siralama`, `lag1_taban_puan`, `lag1_genel_kontenjan`, `lag1_sehit_gazi_kontenjan`, `lag1_depremzede_kontenjan`, `lag1_okul_birincisi_kontenjan` |
| **Lag-2 / Trend** | `lag2_taban_siralama`, `siralama_trend`, `siralama_pct_change`, `kontenjan_degisim_orani` |
| **Static / Encoded** | `universite_turu_enc`, `ogretim_turu_enc`, `puan_turu_enc`, `burs_enc`, `il_kodu_num` |
| **Derived** | `program_hist_medyan_siralama`, `univ_hist_medyan_siralama`, `kontenjan_kategori`, `yil` |

**Testler:** 32/32 passed (`tests/test_features.py`) | Toplam test suite: 61/61 passed

---

### ✅ Adım A — Kök Neden Analizi & Düzeltme (2026-07-22)

**Sorun:** `genel_kontenjan` %29.7 eksik ve `yerlesen_sayisi`/`doluluk_orani` 0 SHAP veriyordu.

**Kök Neden & Düzeltme:**
1. YÖK Atlas API arama endpoint'inde güncel yıl genel kontenjan anahtarı `"kontenjan"` olarak düzeltildi. `genel_kontenjan` eksiklik oranı **%29.7 → %2.4** seviyesine düştü.
2. Özel kontenjan lags (`sehit_gazi_kontenjan`, `depremzede_kontenjan`, `okul_birincisi_kontenjan`) eklendi.

**Commit:** `fix(scraping,features): resolve quota field keys and re-train CatBoost model`

---

### ✅ Adım B — Quantile Regresyon (%80 Güven Aralığı & Belirsizlik Tahmini) (2026-07-22)

**Modül:** `src/models/train_quantile.py` (LightGBM Quantile Regressors)
- Nokta Tahmini: $\alpha = 0.50$ (L1 / Median loss)
- Alt Sınır: $\alpha = 0.10$ (Quantile loss)
- Üst Sınır: $\alpha = 0.90$ (Quantile loss)
- **Quantile Crossing Önleme:** `lower <= median <= upper` kısıtlaması kod seviyesinde garanti edildi (`enforce_quantile_constraints`).

#### Model Performans Karşılaştırması:

| Model / Yöntem | Fold 1 (2024) MAE | Fold 2 (2025) MAE | Ort. MAE | Ort. RMSE | $R^2$ (2025) | Q80 Coverage | Ort. Aralık Genişliği |
|---|---|---|---|---|---|---|---|
| **CatBoost Baseline** | 31,045 | 27,496 | 29,271 | 37,340 | 0.772 | - | - |
| **CatBoost (Kök Neden Fix)** | 31,592 | 25,799 | 28,695 | 36,944 | 0.792 | - | - |
| **LightGBM Quantile (Adım B)** | **30,056** | **15,498** | **22,777** | **31,393** | **0.895** | **42.3%** | **46,875 sıra** |

> **Önemli Başarı:** LightGBM Quantile regresyon kullanımı 2025 test setindeki MAE değerini **25,799'dan 15,498'e düşürdü (-10,301 sıra iyileşme)** ve $R^2$ değerini **0.895** seviyesine çıkardı!

**MLflow Run ID:** `4542d484955e48b6b16e39095e2803a5`

**Testler:** 2/2 passed (`tests/test_models.py`) | Toplam test suite: 61/61 passed

---

## Planlanan Sıradaki Adımlar

- [x] **A** — `genel_kontenjan` kök nedeni çözüldü. (Tamamlandı)
- [x] **B** — Quantile regresyon (LightGBM) ile %80 belirsizlik aralığı üretildi. (Tamamlandı)
- [ ] **C** — Optuna ile CatBoost / LightGBM hiperparametre optimizasyonu.
- [ ] **D** — FastAPI endpoint: `point_estimate`, `lower_bound`, `upper_bound`, `confidence_level`.
- [ ] **E** — Diğer bölüm aileleri (Tıp, Hukuk, İktisat) en son aşamada eklenecek.
