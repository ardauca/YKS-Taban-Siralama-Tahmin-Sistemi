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

---

### ✅ Adım 3 — Feature Engineering + CatBoost Prototipi (2026-07-22)

#### Feature Pipeline (`src/features/build_features.py`)

**19 feature, 4 grup:**
- **Lag-1 (Y-1):** `lag1_taban_siralama`, `lag1_taban_puan`, `lag1_genel_kontenjan`, `lag1_sehit_gazi_kontenjan`, `lag1_depremzede_kontenjan`, `lag1_okul_birincisi_kontenjan`
- **Lag-2 / Trend:** `lag2_taban_siralama`, `siralama_trend`, `siralama_pct_change`, `kontenjan_degisim_orani`
- **Static / Encoded:** `universite_turu_enc`, `ogretim_turu_enc`, `puan_turu_enc`, `burs_enc`, `il_kodu_num`
- **Derived:** `program_hist_medyan_siralama`, `univ_hist_medyan_siralama`, `kontenjan_kategori`, `yil`

---

### ✅ Adım A — Kök Neden Analizi & Düzeltme (2026-07-22)
- Scraper'da 2025 kontenjan anahtarı `"kontenjan"` olarak güncellendi, `genel_kontenjan` eksikliği %2.4'e düşürüldü. Özel kontenjan lags eklendi.

---

### ✅ Adım B — Quantile Regresyon & Kalibrasyon (2026-07-22)
- Modül: `src/models/train_quantile.py` (LightGBM Quantile Regressors, $\alpha=0.030 / 0.970$)
- Q80 Coverage: **%81.6** (Hedef %75–85 aralığına mükemmel kalibre edildi).

---

### ✅ Adım C — Optuna Hiperparametre Optimizasyonu (2026-07-22)

**Modül:** `src/models/optimize.py`
- TPE Sampler ile 25 trial tarandı.
- Optimizasyon Hedefi: Rolling backtest MAE değerini minimize ederken Q80 Coverage >= %75 kalibrasyon ceza fonksiyonu.

#### En İyi Parametreler:
- `n_estimators`: 179
- `learning_rate`: 0.030065
- `num_leaves`: 57
- `min_child_samples`: 23
- `subsample`: 0.7727
- `colsample_bytree`: 0.9634
- `reg_alpha`: 0.13255
- `reg_lambda`: 3.8551

#### Metrik Gelişimi:

| Aşama / Model | 2024 Test MAE | 2025 Test MAE | Ort. MAE | Ort. RMSE | $R^2$ (2025) | Q80 Coverage |
|---|---|---|---|---|---|---|
| CatBoost Baseline | 31,045 | 27,496 | 29,271 | 37,340 | 0.772 | - |
| CatBoost (Fix) | 31,592 | 25,799 | 28,695 | 36,944 | 0.792 | - |
| Quantile Default ($\alpha=0.10/0.90$) | 30,056 | 15,498 | 22,777 | 31,393 | 0.895 | %42.3 ❌ |
| Quantile Kalibre ($\alpha=0.030/0.970$) | 30,056 | 15,498 | 22,777 | 31,393 | 0.895 | %81.6 ✅ |
| **Quantile Optuna Tuned (Adım C)** | **30,157** | **14,525** | **22,341** | **31,335** | **0.899** | **%85.1 ✅** |

> 🏆 **Sonuç:** Optuna optimizasyonu ile 2025 test seti MAE değeri **14,525**, $R^2$ **0.899** seviyesine ulaştı. Kapsama oranı **%85.1** ile kalibreliğini korudu.

**MLflow Run ID:** `0cf12bcfb9c54851b706040efd2de486`

---

### ✅ Adım D — FastAPI Endpoint Servis Katmanı (2026-07-22)

**Modül:** `api/main.py`
- Servis Başlatma: `uvicorn api.main:app --reload`
- Endpoint'ler:
  - `GET /health` → Servis durum kontrolü
  - `POST /api/v1/predict` → Sıralama tahmini & %80 güven aralığı

#### API Yanıt Formatı (Bölüm 9 Uyumlu):
```json
{
  "status": "success",
  "kilavuz_kodu": 203910363,
  "prediction": {
    "point_estimate": 115,
    "lower_bound": 80,
    "upper_bound": 160,
    "confidence_level": 0.80,
    "unit": "siralama"
  },
  "metadata": {
    "model_version": "quantile_lightgbm_v1_optuna",
    "timestamp": "2026-07-22T15:18:00.000Z"
  }
}
```

**Test Suite:** **63/63 test PASSED** (`pytest tests/ -v` — scraper, features, models, api testleri tam yeşil)

---

## Sıradaki Adımlar

- [x] **A** — `genel_kontenjan` kök nedeni çözüldü.
- [x] **B & Kalibrasyon** — Q80 Coverage %81.6 ile kalibre edildi.
- [x] **C** — Optuna ile hiperparametre optimizasyonu yapıldı (Ort. MAE: 22,341, 2025 MAE: 14,525, R²: 0.899).
- [x] **D** — FastAPI endpoint servisi (`/health`, `/api/v1/predict`) kuruldu.
- [ ] **E** — Diğer bölüm aileleri (Tıp, Hukuk, İktisat) için scraper ve modelin genişletilmesi.
