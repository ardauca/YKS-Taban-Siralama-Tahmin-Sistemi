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

| Yıl | Satır | Taban Sıralama Eksik |
|-----|-------|---------------------|
| 2022 | 268 | %10.1 |
| 2023 | 298 | %14.8 |
| 2024 | 318 | %31.1 |
| 2025 | 332 | %34.6 |

> **Not:** 2025 eksikleri beklenen — YKS 2025 yerleştirme süreci henüz tamamlanmamış.

**Testler:** 27/27 passed (`tests/test_scraper.py`)

**Commit:** `feat(scraping): add YOKATLAS scraper and quality check module`  
**Commit:** `data(raw): add 2022-2025 YOKATLAS snapshot for Bilgisayar Muhendisligi`

---

### ✅ Adım 3 — Feature Engineering + CatBoost Prototipi (2026-07-22)

#### Feature Pipeline (`src/features/build_features.py`)

**18 feature, 4 grup:**

| Grup | Feature'lar |
|------|-------------|
| **Lag-1** (Y-1) | `lag1_taban_siralama`, `lag1_taban_puan`, `lag1_genel_kontenjan`, `lag1_yerlesen_sayisi`, `lag1_doluluk_orani` |
| **Lag-2 / Trend** | `lag2_taban_siralama`, `siralama_trend`, `siralama_pct_change`, `kontenjan_degisim_orani` |
| **Static / Encoded** | `universite_turu_enc`, `ogretim_turu_enc`, `puan_turu_enc`, `burs_enc`, `il_kodu_num` |
| **Derived** | `program_hist_medyan_siralama`, `univ_hist_medyan_siralama`, `kontenjan_kategori`, `yil` |

**Testler:** 32/32 passed (`tests/test_features.py`)

#### CatBoost Modeli (`src/models/train_catboost.py`)

**Mimari:** Rolling backtest (Aşama 1 — Talep/Sıralama Tahmini)
- Fold 1: train=2023 → test=2024
- Fold 2: train=2023+2024 → test=2025

**Sonuçlar:**

| Fold | Train | Test | n_test | MAE | RMSE | R² | MAE% |
|------|-------|------|--------|-----|------|----|------|
| 1 | 2023 | 2024 | 198 | **31,045** | 41,053 | 0.706 | 36.5% |
| 2 | 2023+2024 | 2025 | 199 | **27,496** | 33,626 | 0.772 | 31.0% |
| **Ort.** | | | | **29,271** | **37,340** | | |

#### SHAP Feature Önem (Fold 2 — Final Model)

| # | Feature | Ortalama |SHAP| |
|---|---------|----------|---|
| 1 | `program_hist_medyan_siralama` | 22,620 | ██████████████████████████████ |
| 2 | `lag1_taban_puan` | 12,283 | ████████████████ |
| 3 | `lag1_taban_siralama` | 10,518 | █████████████ |
| 4 | `lag2_taban_siralama` | 5,869 | ███████ |
| 5 | `univ_hist_medyan_siralama` | 4,276 | █████ |
| 6 | `yil` | 3,921 | █████ |
| 7 | `universite_turu_enc` | 1,405 | ██ |
| 8 | `siralama_trend` | 1,113 | █ |
| 9 | `kontenjan_degisim_orani` | 976 | █ |
| 10 | `siralama_pct_change` | 791 | █ |

**MLflow:**
- Backend: `sqlite:///mlruns/mlflow.db`
- Experiment: `yks-taban-siralama`
- Run ID: `20f46693df9f4f0b9c3c1aa8c006bae4`
- UI: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`

**Commit:** `feat(features): add feature engineering pipeline and CatBoost baseline model`

---

## Değerlendirme

### Model Makul mu?
**Evet — baseline için kabul edilebilir, iyileştirme gerektiriyor.**

| Metrik | Değer | Yorum |
|--------|-------|-------|
| MAE | ~29,000 sıralama | Medyan ~85K → %34 hata |
| R² | 0.71–0.77 | Varyansın %71-77'sini açıklıyor |
| SHAP #1 | `program_hist_medyan_siralama` | Geçmiş medyan sıralama en belirleyici |

**Sorunlar:**
- Fold sayısı az (2 fold) — daha eski yıl verisi eklenince artacak
- `lag1_yerlesen_sayisi` ve `lag1_doluluk_orani` 2024/2025'te sıfır SHAP → eksik veri sorunu
- MAE %34 yüksek — quantile regresyon + Optuna ile iyileştirilecek

---

## Sıradaki Adımlar

- [ ] **A** — `lag1_yerlesen_sayisi` eksiklik nedeni araştır (API'de var mı?)
- [ ] **B** — Quantile regresyon (LightGBM) ile `lower_bound`/`upper_bound` üret
- [ ] **C** — Optuna ile CatBoost hiperparametre optimizasyonu
- [ ] **D** — FastAPI endpoint: `point_estimate`, `lower_bound`, `upper_bound`, `confidence_level`
- [ ] **E** — Diğer bölüm aileleri (Tıp, Hukuk, İktisat) için scraper genişlet
