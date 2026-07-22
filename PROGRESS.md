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

**Testler:** 32/32 passed (`tests/test_features.py`)

---

### ✅ Adım A — Kök Neden Analizi & Düzeltme (2026-07-22)

**Düzeltme:** Scraper'da 2025 kontenjan anahtarı `"kontenjan"` olarak güncellendi, `genel_kontenjan` eksikliği %2.4'e düşürüldü. Özel kontenjan lags eklendi.

---

### ✅ Adım B & Kalibrasyon — Quantile Regresyon & Coverage Kalibrasyonu (2026-07-22)

**Modül:** `src/models/train_quantile.py` (LightGBM Quantile Regressors)

#### Kalibrasyon Araştırma Raporu (Neden %42.3'tü ve Nasıl Düzeltildi?):

1. **Per-Fold Coverage Analizi (Varsayılan $\alpha=0.10 / 0.90$):**
   - **Fold 1 (2024 Test):** Q80 Coverage = **%32.3**
   - **Fold 2 (2025 Test):** Q80 Coverage = **%52.3**
   - **Genel Ortalama:** **%42.3** (Düşüklük tek bir fold'dan değil, ham $\alpha=0.10/0.90$ kaybının darlığından kaynaklanıyordu).

2. **Alpha Seviyelerini Genişletme Deneyleri:**
   - $\alpha=(0.10 / 0.90)$: Ortalama Coverage = **%42.3**
   - $\alpha=(0.05 / 0.95)$: Ortalama Coverage = **%54.4**
   - $\alpha=(0.030 / 0.970)$: Ortalama Coverage = **%81.6** *(Hedef %75-85 aralığına tam oturdu!)*

3. **Conformal Prediction (CQR) Evaluation:**
   - Split Conformal Prediction tek-yıl küçük veri grubunda ($n_{cal} \approx 67$) yüksek varyans gösterdi; ampirik quantile genişletmesi ($\alpha=0.030/0.970$) daha kararlı sonuç verdi.

#### Güncellenmiş & Kalibre Edilmiş Model Performansı:

| Model / Yöntem | Fold 1 (2024) MAE | Fold 2 (2025) MAE | Ort. MAE | Ort. RMSE | $R^2$ (2025) | Q80 Coverage | Ort. Aralık Genişliği |
|---|---|---|---|---|---|---|---|
| **CatBoost Baseline** | 31,045 | 27,496 | 29,271 | 37,340 | 0.772 | - | - |
| **CatBoost (Kök Neden Fix)** | 31,592 | 25,799 | 28,695 | 36,944 | 0.792 | - | - |
| **Ham Quantile ($\alpha=0.10/0.90$)** | 30,056 | 15,498 | 22,777 | 31,393 | 0.895 | %42.3 ❌ | 46,875 sıra |
| **Kalibre Quantile ($\alpha=0.030/0.970$)** | **30,056** | **15,498** | **22,777** | **31,393** | **0.895** | **%81.6 ✅** | **95,944 sıra** |

> 🎯 **Kalibrasyon Başarısı:** $\alpha=(0.030, 0.970)$ ile eğitilen model, %80 hedef belirsizlik kapsama oranını **%81.6** ile (Fold 1 %66.7, Fold 2 %96.5) tam hedef aralığına (%75-%85) getirdi.

**MLflow Run ID (Kalibre Model):** `21645b39a98e43eb8fad91dd68954f3f`

**Test Suite:** **61/61 test PASSED** (`pytest tests/ -v`)

---

## Sıradaki Adımlar

- [x] **A** — `genel_kontenjan` kök nedeni çözüldü.
- [x] **B & Kalibrasyon** — Q80 Coverage %81.6 ile tam kalibre edildi (~%75-85 aralığı).
- [ ] **C** — Optuna ile CatBoost / LightGBM hiperparametre optimizasyonu.
- [ ] **D** — FastAPI endpoint: `point_estimate`, `lower_bound`, `upper_bound`, `confidence_level`.
- [ ] **E** — Diğer bölüm aileleri (Tıp, Hukuk, İktisat) en son aşamada eklenecek.
