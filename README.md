# 🎓 YKS Taban Sıralama Tahmin Sistemi (YKS-Tahmin)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-green.svg)
![Full Scope](https://img.shields.io/badge/coverage-625_Bölüm_Ailesi-brightgreen.svg)
![Nationwide Blind Coverage](https://img.shields.io/badge/Q80_Blind_Coverage-%2575.3-success.svg)
![Unit Tests](https://img.shields.io/badge/tests-63%2F63_passing-success.svg)

Türkiye üniversitelerindeki **tüm lisans programlarının (625 Bölüm Ailesi, 15,823 aktif tercih programı)** geçmiş yıl taban sıralamaları, kontenjan değişimleri, YÖK Atlas eğilimleri, Google Trends popülerlik indeksleri ve URAP akademik itibar metriklerinden öğrenerek, ÖSYM kılavuz kontenjan kısıntı şoklarına göre **Taban Sıralamalarını (%80 Güven Aralığı ve Kademeli Router Mimarisi ile)** tahmin eden üretim seviyesinde makine öğrenmesi sistemi.

> 📌 **Metodolojik Şeffaflık Notu:**  
> Kuantil parametreleri (`Alpha=0.08/0.92`) **SADECE 2023–2024 Train verisi üzerinde 5-Fold Out-of-Fold (OOF)** yöntemiyle seçilmiş, 2025 Test verisinde dokunulmadan **kör (blind) test** edilmiştir. Sızıntısız tarafsız ulusal kapsama oranı **%75.3**, 0–10K tıp/derece segmenti ortalama aralık genişliği **24,611 sıradır**.

---

## 🚀 Mimarinin Öne Çıkan Özellikleri

### 1. 3 Kademeli Yönlendirme Mimarisi (3-Tier Router Strategy D)
- **Tier 1 (`lag1_taban_siralama < 100,000`):** **Model S (HeavyReg GBDT: LightGBM + CatBoost Ensemble)** — Yüksek rekabetli ilk 100K programlarında aşırı uyumu (overfitting) engeller.
- **Tier 2 (`100,000 <= lag1_taban_siralama < 500,000`):** **Model M (Segment Model M)** — Orta bandın ikame dinamiklerini yakalar.
- **Tier 3 (`lag1_taban_siralama >= 500,000`):** **Saf v0 Baseline Model** — Kitlesel yüksek gürültülü segmentte baseline performansını (%100 muhafaza ile 115,978 MAE) korur.

### 2. 30 Gelişmiş Öznitelik (Feature Matrix)
- **Makro Kontenjan Şok Özellikleri:** `macro_puan_turu_degisim_orani`, `macro_bolum_degisim_orani`, `kontenjan_sok_faktoru` (Hukuk -%34, Siyaset -%24 gibi sistemsel kısıntıların ikame talebini modeller).
- **Arama & İtibar Dinamikleri:** `trends_yoy_degisim` (Google Trends aramaları), `univ_itibar_degisim` (URAP akademik itibar skoru), `segment_kontenjan_etki`.
- **YÖK Başarı Sırası Baraj Mesafesi:** `baraj_mesafe_indeksi` (Tıp 50k, Hukuk 125k, Müh 300k baraj kısıtları).
- **Vakıf-Devlet Ekonomik İkame İndeksi:** `vakif_devlet_burs_gap` (Özel üniversite ücret zamlarının devlete kayma etkisi).
- **Şehir ve Momentum İndeksleri:** `sehir_tercih_indeksi` (İstanbul, Ankara, İzmir, Eskişehir tercihi), `univ_trend_momentum`.

### 3. Ayrıştırılmış 4'lü Veri Kalitesi & Anomali Sistemi
Train verisi %95 persentilinden türetilen ampirik dalgalanma eşiği (**293,000 sıra**) ile programlar 4 sınıfa ayrılır:
- **`SUFFICIENT` (%87.7 / 13,874 program):** Standart yüksek güvenilirlikli veriler.
- **`INSUFFICIENT` (%5.9 / 938 program):** Geçmiş taban sıralaması veya medyanı eksik programlar.
- **`STRUCTURAL_ANOMALY` (%3.6 / 565 program):** Gerçek KKTC ve Yurt Dışı kontenjan/burs statü değişiklikleri.
- **`HIGH_VOLATILITY` (%2.8 / 446 program):** Tarihsel dalgalanması %95 persentilini (>293K) aşan oynak programlar.

---

## 📈 Sızıntısız Kör Test Performans Tablosu (2025 Test Yılı, n=15,823)

| Segment / Dilim | Program Sayısı (n) | **Sabit v0 Baseline MAE** | **3-Tier Router MAE** | **Net MAE İyileşme %** | **Tarafsız Kör Q80 Coverage** | **Ortalama Güven Aralığı Genişliği** |
|---|---|---|---|---|---|---|
| **0–10K (Tıp, Top Müh.)** | 452 | **7,493** | **3,249** | **+%56.6** (4,244 sıra kazanç) | **%77.9** | **24,611 sıra** (Hassas & Dar) |
| **10K–100K** | 2,434 | **13,805** | **10,840** | **+%21.5** (2,965 sıra kazanç) | **%79.3** | **37,000 sıra** (Mükemmel Kalibrasyon) |
| **100K–500K** | 4,961 | **50,587** | **47,152** | **+%6.8** (3,435 sıra kazanç) | **%65.5** | **132,560 sıra** (Yapısal Oynaklık) |
| **500K+** | 7,976 | **115,978** | **116,290** | **−%0.3** (%100 Muhafaza) | **%80.0** | **366,973 sıra** |
| **GENEL (TÜM ÜLKE)** | **15,823** | **76,660** | **75,163** | **+%2.0** (1,497 sıra KAZANÇ) | **%75.3** | **232,939 sıra** |

---

## 💻 Kullanım Kılavuzu

### 1. İnteraktif 2026 Tercih Danışmanı (CLI)
Kendi sıralamanızı ve puan türünüzü girerek risk kategorili öneriler almak için:

```bash
python tercih_danismani.py 180000 EA
```

### 2. FastAPI REST Servisini Başlatma
```bash
uvicorn api.main:app --reload
```
- **Swagger UI Dokümantasyonu:** `http://127.0.0.1:8000/docs`
- **Predict Endpoint (`POST /api/v1/predict`):** 3-Tier Router, 4-Class Data Quality ve 0-10K hassasiyet uyarısı içeren JSON yanıtı döndürür.

### 3. Unit Test Suitesini Çalıştırma
```bash
python -m pytest tests/ -v
```

---

## 📂 Klasör Yapısı

```
yks-tahmin/
├── api/                             # FastAPI REST API Servis Katmanı
│   └── main.py                      # /predict & /health endpoint'leri
├── data/
│   ├── raw/                         # YÖK Atlas & ÖSYM PDF Ham Verileri
│   │   ├── osym/kontenjan_kilavuzu_2026.csv
│   │   └── yokatlas/yokatlas_all_departments_raw.csv
│   └── processed/                   # İşlenmiş Feature Parquet & Simülasyon CSV
├── scraping/                        # YÖK Atlas & ÖSYM Scraper/Parser Engine
│   ├── parse_osym_pdf.py            # PyMuPDF Kılavuz Parser Engine
│   ├── yokatlas_scraper.py          # YÖK Atlas JSON API Scraper
│   └── fetch_google_trends.py       # Google Trends Arama Verisi Çekici
├── src/
│   ├── features/build_features.py   # 30-Feature Pipeline Engine
│   └── models/
│       ├── train_quantile.py        # LightGBM + CatBoost Quantile Ensemble
│       └── explain_shap.py          # SHAP Öznitelik Etki Analizi
├── tests/                           # Pytest Test Suitesi (63 Unit Test)
├── tercih_danismani.py              # İnteraktif CLI Tercih Danışmanı
├── PROGRESS.md                      # Detaylı Geliştirme ve Şeffaflık Günlüğü
└── README.md                        # Ana Dokümantasyon
```

---

## 📜 Lisans

[MIT License](LICENSE) — Serbestçe kullanılabilir ve geliştirilebilir.
