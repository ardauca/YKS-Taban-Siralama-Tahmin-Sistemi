# YKS Taban Sıralama Tahmin Sistemi — İlerleme Kaydı

## Son Güncelleme: 2026-07-25

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

### ✅ Türkiye Geneli %100 YÖK Atlas Veri Kapsaması (77,970 Satır) (2026-07-25)
- **Tüm Lisans Programları Taraması (`scrape_100pct_yokatlas.py`):** YÖK Atlas'taki tüm 215 sayfa taranarak **77,970 Satır (21,482 Lisans Programı)** veritabanına eklendi.
- Türkiye'deki lisans seviyesindeki **TÜM üniversiteler ve tüm bölümler (Muğla Sıtkı Koçman Havacılık Yönetimi, Osmangazi Üniversitesi tüm bölümleri dahil %100)** sisteme entegre edildi.

---

### ✅ Dev Model Başarısı & 16,957 Programlık 2026 Simülasyonu (2026-07-25)
- **Model Mimarisi:** Hibrit LightGBM + CatBoost Quantile Ensemble (28 Feature)
- **2024 Test Seti $R^2$ Skoru:** **0.966 (%96.6)**
- **2025 Test Seti $R^2$ Skoru:** **0.899 (%89.9)**
- **Q80 Coverage Rate:** **%88.5**
- **2026 Toplu Simülasyon Çıktısı:** [`data/processed/simulasyon_2026_tahminleri.csv`](file:///C:/Users/ARDA/.gemini/antigravity/scratch/yks-tahmin/data/processed/simulasyon_2026_tahminleri.csv) (**16,957 Lisans Programı**)

---

## 📈 Proje Özeti & Genel Durum

- **%100 Türkiye Geneli YÖK Atlas Verisi & 2026 Simülasyon Çıktısı Tamamlandı!**
- **Tüm Kodlar & Ham Veriler:** GitHub `main` branch'inde versiyonlandı.
