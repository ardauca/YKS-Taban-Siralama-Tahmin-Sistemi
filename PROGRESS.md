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

**Testler:** 32/32 passed (`tests/test_features.py`) | Toplam test suite: 59/59 passed

#### CatBoost Modeli (`src/models/train_catboost.py`)

**Mimari:** Rolling backtest (Aşama 1 — Talep/Sıralama Tahmini)
- Fold 1: train=2023 → test=2024
- Fold 2: train=2023+2024 → test=2025

---

### ✅ Adım A — Kök Neden Analizi & Düzeltme (2026-07-22)

**Sorun:** `genel_kontenjan` %29.7 eksik ve `yerlesen_sayisi`/`doluluk_orani` 0 SHAP veriyordu.

**Kök Neden:**
1. YÖK Atlas API arama endpoint'inde güncel yıl genel kontenjan anahtarı `"gk"` değil `"kontenjan"` olarak adlandırılmış! Bu nedenle 2025 yılı kontenjanları 100% NaN geliyordu.
2. `sgy1` (Şehit/Gazi Yakını kontenjanı) ve `dprm1` (Depremzede kontenjanı) alanları genel yerleşen/doluluk sanılarak yanlış adlandırılmıştı. Genel yerleşen sayısı bu API endpoint'inde yer almamaktadır (genel yerleşim kontenjan kadar gerçekleşmektedir).

**Düzeltme & İyileştirme:**
- Scraper'da 2025 kontenjan anahtarı `"kontenjan"` olarak düzeltildi. `genel_kontenjan` eksiklik oranı **%29.7 → %2.4** seviyesine düştü.
- Yanıltıcı alanlar kaldırılarak yerine özel kontenjan lags (`sehit_gazi_kontenjan`, `depremzede_kontenjan`, `okul_birincisi_kontenjan`) eklendi.
- Model yeniden eğitildi.

#### Güncellenmiş Rolling Backtest Sonuçları:

| Fold | Train | Test | n_test | MAE | RMSE | R² | MAE% | Değişim (Önceki vs Şimdi) |
|------|-------|------|--------|-----|------|----|------|---------------------------|
| 1 | 2023 | 2024 | 198 | 31,592 | 41,790 | 0.696 | %37.2 | MAE +547 |
| 2 | 2023+2024 | 2025 | 199 | **25,799** | **32,098** | **0.792** | **%29.1** | **MAE -1,697 (İyileşme!)** |
| **Ort.** | | | | **28,695** | **36,944** | **0.744** | **%33.1** | **MAE -576 (Net İyileşme)** |

#### Güncellenmiş SHAP Feature Önem (Fold 2 — Final Model)

| # | Feature | Ortalama |SHAP| Değişim / Not |
|---|---------|----------|---|
| 1 | `program_hist_medyan_siralama` | 20,365.6 | En belirleyici feature |
| 2 | `lag1_taban_siralama` | 14,439.0 | Geçen yıl sıralaması 3. sıradan 2. sıraya yükseldi |
| 3 | `lag1_taban_puan` | 11,277.1 | Geçen yıl taban puanı |
| 4 | `yil` | 4,848.0 | Genel zaman trendi |
| 5 | `lag2_taban_siralama` | 3,899.3 | 2 yıl önceki sıralama |
| 6 | `univ_hist_medyan_siralama` | 2,831.9 | Üniversite geneli medyan |
| 7 | `universite_turu_enc` | 1,418.3 | Devlet / Vakıf farkı |
| 8 | `kontenjan_degisim_orani` | 1,098.9 | **Düzeltme sonrası aktif katkı veriyor!** |
| 9 | `siralama_pct_change` | 1,084.0 | Yüzde değişim trendi |
| 10 | `il_kodu_num` | 862.0 | Şehir lokasyon etkisi |

**MLflow Run ID:** `83466f000d7b4377bcddeff5b397bd30`

---

## Planlanan Sıradaki Adımlar

- [x] **A** — `lag1_yerlesen_sayisi` ve `genel_kontenjan` eksiklik kök nedeni çözüldü, model yeniden eğitildi. (Tamamlandı)
- [ ] **B** — Quantile regresyon (LightGBM) ile `lower_bound` / `upper_bound` (%80 güven aralığı) üretimi ve coverage testi.
- [ ] **C** — Optuna ile CatBoost hiperparametre optimizasyonu.
- [ ] **D** — FastAPI endpoint: `point_estimate`, `lower_bound`, `upper_bound`, `confidence_level`.
- [ ] **E** — Diğer bölüm aileleri (Tıp, Hukuk, İktisat) en son aşamada eklenecek.
