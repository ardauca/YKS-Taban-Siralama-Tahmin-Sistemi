# YKS Taban Sıralama Tahmin Sistemi — İlerleme Kaydı

## Son Güncelleme: 2026-07-22

---

## Tamamlanan Adımlar

### ✅ Adım 1 — Proje İskeleti (2026-07-22)
- Klasör yapısı oluşturuldu (`data/`, `scraping/`, `src/`, `api/`, `frontend/`, `tests/`)
- `requirements.txt`, `.gitignore`, `.env.example`, `README.md`, `docker-compose.yml` hazırlandı
- Git reposu `main` branch'e bağlandı ve ilk commit push edildi

---

### ✅ Adım 2 — YÖK Atlas Scraper (2026-07-22)
- Yeni YÖK Atlas JSON API (`POST /api/tercih-kilavuz/search`) entegrasyonu sağlandı.
- Rate-limiting (1.5 sn) ve retry mantığı eklendi.
- Veri kalite modülü (`scraping/quality.py`) ile kontroller otomatize edildi.

---

### ✅ Adım 3 — Feature Engineering + CatBoost Prototipi (2026-07-22)
- 19 öznitelikli (lag-1, lag-2/trend, statik, türetilmiş medyanlar) feature pipeline yazıldı.

---

### ✅ Adım A — Kök Neden Analizi & Düzeltme (2026-07-22)
- Scraper'da 2025 kontenjan anahtarı `"kontenjan"` olarak düzeltilerek `genel_kontenjan` eksiklik oranı **%29.7 → %1.3** seviyesine indirildi.

---

### ✅ Adım B — Quantile Regresyon & Kalibrasyon (2026-07-22)
- LightGBM Quantile Regressors ile %80 güven aralığı ($\alpha=0.030 / 0.970$) kalibre edildi. Q80 Coverage: **%81.6**.

---

### ✅ Adım C — Optuna Hiperparametre Optimizasyonu (2026-07-22)
- TPE Sampler ile 25 trial tarandı. Bilgisayar Mühendisliği test seti MAE değeri **14,525**, $R^2$ **0.899** seviyesine ulaştı.

---

### ✅ Adım D — FastAPI Endpoint Servis Katmanı (2026-07-22)
- Servis başlatma: `uvicorn api.main:app --reload`
- Endpoint'ler: `/health` ve `/api/v1/predict` (nokta tahmini, alt sınır, üst sınır, confidence_level: 0.80, unit: "siralama").

---

### ✅ Adım E — Diğer Bölüm Aileleri (10 Ana Bölüm Ailesi) (2026-07-22)

**Çekilen & Konsolide Edilen Bölüm Aileleri (`data/raw/yokatlas/yokatlas_all_departments_raw.csv`):**

| Bölüm Ailesi | YÖK Atlas ID | Puan Türü | Toplam Satır (2022–2025) | Program Sayısı (2025) |
|---|---|---|---|---|
| **Bilgisayar Mühendisliği** | 2010 | SAY | 1,216 | 332 |
| **İşletme** | 3549 | EA | 1,385 | 382 |
| **Psikoloji** | 4679 | EA | 1,309 | 352 |
| **Elektrik-Elektronik Mühendisliği** | 2644 | SAY | 990 | 270 |
| **Tıp** | 5370 | SAY | 901 | 242 |
| **Endüstri Mühendisliği** | 2704 | SAY | 763 | 206 |
| **Makine Mühendisliği** | 3987 | SAY | 750 | 206 |
| **Yönetim Bilişim Sistemleri (YBS)** | 5874 | EA | 729 | 219 |
| **İktisat** | 3353 | EA | 702 | 193 |
| **Hukuk** | 3309 | EA | 657 | 172 |
| **TOPLAM** | - | - | **9,402 Satır** | **2,574 Program** |

#### Konsolide Çoklu-Bölüm Model Sonuçları:
- **Toplam Veri Hacmi:** 9,402 satır × 19 feature (7,577 eğitilebilir veri noktası)
- **Model Açıklayıcılık ($R^2$ Score):** **0.935** (2024 Test) | **0.844** (2025 Test)
- **Q80 Coverage:** **87.4%** (Fold 2 2025 Test: **80.6%** — tam %80 kalibrasyon hedefinde!)
- **MLflow Run ID:** `511a7e973bd841baac4015b593525bbb`

---

## 📈 Proje Özeti & Genel Durum

- **Tüm Ana Görevler (A, B, C, D, E) Tamamlandı!**
- **Test Suite:** **63/63 test PASSED** (`pytest tests/ -v`)
- **Tüm Kodlar & Ham Veriler:** GitHub `main` branch'inde versiyonlandı ve push edildi.
