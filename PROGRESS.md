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
- Rate-limiting ve retry mantığı eklendi.
- Veri kalite modülü (`scraping/quality.py`) ile kontroller otomatize edildi.

---

### ✅ Adım 3 — Feature Engineering + CatBoost Prototipi (2026-07-22)
- 19 öznitelikli (lag-1, lag-2/trend, statik, türetilmiş medyanlar) feature pipeline yazıldı.

---

### ✅ Adım A — Kök Neden Analizi & Düzeltme (2026-07-22)
- Scraper'da 2025 kontenjan anahtarı `"kontenjan"` olarak düzeltilerek `genel_kontenjan` eksiklik oranı **%29.7 → %1.3** seviyesine indirildi.

---

### ✅ Adım B — Quantile Regresyon & Kalibrasyon (2026-07-22)
- LightGBM Quantile Regressors ile %80 güven aralığı ($\alpha=0.030 / 0.970$) kalibre edildi. Q80 Coverage: **%81.6 / %86.9**.

---

### ✅ Adım C — Optuna Hiperparametre Optimizasyonu (2026-07-22)
- TPE Sampler ile 25 trial tarandı. Bilgisayar Mühendisliği test seti MAE değeri **14,525**, $R^2$ **0.899** seviyesine ulaştı.

---

### ✅ Adım D — FastAPI Endpoint Servis Katmanı (2026-07-22)
- Servis başlatma: `uvicorn api.main:app --reload`
- Endpoint'ler: `/health` ve `/api/v1/predict` (nokta tahmini, alt sınır, üst sınır, confidence_level: 0.80, unit: "siralama").

---

### ✅ ÖSYM 2026-2027 Ön Kontenjan Kılavuzu (Tablo-4 PDF) Entegrasyonu (2026-07-22)
- **Kılavuz PDF:** `kontkilavuz_yktd21072026.pdf` (793 Sayfa)
- **Parser Modülü:** `scraping/parse_osym_pdf.py` (PyMuPDF ultra-fast text-stream engine ile 4 saniyede tüm PDF parse edildi).
- **Çıkarılan Veri Seti:** [`data/raw/osym/kontenjan_kilavuzu_2026.csv`](file:///C:/Users/ARDA/.gemini/antigravity/scratch/yks-tahmin/data/raw/osym/kontenjan_kilavuzu_2026.csv) (**11,676 Lisans Programı**)

---

### ✅ 24 Ana Bölüm Ailesi Veri Genişletmesi (15,321 Satır) (2026-07-22)
- **24 Ana Bölüm Ailesi (`yokatlas_all_departments_raw.csv`):** **15,321 Satır** (3,117 Lisans Programı)

---

### ✅ Zamana Duyarlı Kör (Blind Walk-Forward) Backtest Doğrulaması (2026-07-22)
- **Kullanıcı Önerisi Entegrasyonu:** Model hiç ilgili yılın puan/sıralama verisini görmeden, **sadece o yılın yeni kontenjanları** verilerek tamamen "kör" (blind) şekilde geleceği tahmin etme testine sokuldu (`src/models/backtest_walkforward.py`):
  - **2024 Kör Simülasyonu ($\le 2023$ ile eğitim):** MAE **38,895** | $R^2$ **%93.1** | Güven Kapsama: **%88.0**
  - **2025 Kör Simülasyonu ($\le 2024$ ile eğitim):** MAE **52,073** | $R^2$ **%87.6** | Güven Kapsama: **%78.0**

---

## 📈 Proje Özeti & Genel Durum

- **Tüm Gelişmiş Öznitelikler & 24 Bölüm Ailesi Verisi & Kör Backtest Motoru Tamamlandı!**
- **Test Suite:** **63/63 test PASSED** (`pytest tests/ -v`)
- **Tüm Kodlar & Ham Veriler:** GitHub `main` branch'inde versiyonlandı ve push edildi.
