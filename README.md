# 🎓 YKS Taban Sıralama Tahmin Sistemi (YKS-Tahmin)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11+-green.svg)
![Model R2](https://img.shields.io/badge/R%C2%B2_Score-0.873-brightgreen.svg)
![MAE](https://img.shields.io/badge/MAE-45%2C739-blue.svg)
![Unit Tests](https://img.shields.io/badge/tests-63%2F63_passing-success.svg)

Türkiye üniversite bölümlerinin geçmiş yıl verilerinden ve YÖK Atlas trendlerinden öğrenerek, **ÖSYM'nin yayınladığı yeni kılavuz kontenjanlarına ve makro kontenjan kısıntı şoklarına göre oluşacak 2026 Taban Sıralamalarını (%80 Güven Aralığı ile)** tahmin eden pro-seviye makine öğrenmesi sistemi.

---

## 🚀 Öne Çıkan Özellikler ve İnovasyonlar

- 🧠 **Hibrit Quantile Ensemble Mimarisi:** %50 LightGBM Quantile + %50 CatBoost Quantile Regressors harmanı.
- 📐 **28 Gelişmiş Öznitelik (Feature Matrix):**
  - **Makro Kontenjan Şok Özellikleri:** `macro_puan_turu_degisim_orani`, `macro_bolum_degisim_orani`, `kontenjan_sok_faktoru` (Hukuk -%34, Siyaset -%24 gibi sistemsel kısıntıların ikame talebini modeller).
  - **YÖK Başarı Sırası Baraj Mesafesi:** `baraj_mesafe_indeksi` (Tıp 50k, Hukuk 125k, Müh 300k baraj kısıtları).
  - **Vakıf-Devlet Ekonomik İkame İndeksi:** `vakif_devlet_burs_gap` (Özel üniversite ücret zamlarının devlete kayma etkisi).
  - **Şehir ve Momentum İndeksleri:** `sehir_tercih_indeksi` (Eskişehir, İstanbul, Ankara tercihi), `univ_trend_momentum`.
- 📚 **Dev Veri Kümesi (24 Ana Bölüm Ailesi / 15,321 Satır):** Bilgisayar, Tıp, EE, Makine, Endüstri, Yazılım, İnşaat, Mimarlık, İç Mimarlık, Hemşirelik, Diş Hekimliği, Eczacılık, Hukuk, İktisat, İşletme, Psikoloji, YBS, Siyaset Bilimi, Sınıf Öğr., Uluslararası İlişkiler, Özel Eğitim, Tarih, TDE, ELT.
- 📄 **ÖSYM 2026-2027 Kılavuz PDF Entegrasyonu:** PyMuPDF ultra-fast text-stream engine ile 11,676 lisans programının 2026 ön kontenjanları (%96.8 eşleşme oranı).
- 🔮 **Zamana Duyarlı Kör (Blind Walk-Forward) Backtest:** Tahmin edilen yılın sıralamasını bilmeden, sadece yeni kontenjanları bilerek **%87.6 - %93.1 $R^2$ doğruluk oranı**.

---

## 📈 Model Başarım Metrikleri

| Metrik | Değer | Ayrıntı / Hedef |
|---|---|---|
| **Ortalama MAE (Hata)** | **45,739 Sıra** | Rekor En Düşük Hata |
| **2025 Test Seti $R^2$ Skoru** | **0.873 (%87.3)** | 15,321 Satır Genişletilmiş Veri |
| **2024 Test Seti $R^2$ Skoru** | **0.944 (%94.4)** | Test Seti Doğrulaması |
| **Q80 Coverage Rate** | **%83.1** | Kalibre Edilmiş %80 Güven Aralığı |
| **Ortalama Aralık Genişliği** | **228,435 Sıra** | Dar / Hassas Güven Aralığı |
| **Unit Test Başarısı** | **63 / 63 PASSED** | %100 Yeşil Test Suitesi |
| **MLflow Run ID** | `638b328423df475e802380ed62d889b4` | İzlenebilir Deney Kaydı |

---

## 💻 Kullanım Kılavuzu

### 1. İnteraktif 2026 Tercih Danışmanı (CLI)
Kendi sıralamanızı ve puan türünüzü girerek **Risk Kategorili (`[GARANTİ]`, `[GÜVENLİ]`, `[İDEAL/HEDEF]`, `[SÜRPRİZ]`)** 2026 kılavuz önerilerinizi almak için:

```bash
python tercih_danismani.py 180000 EA
```

### 2. Tüm Türkiye 2026 Toplu Kılavuz Simülasyonu
Türkiye'deki 3,117 aktif lisans programı için 2026 tahmin çıktısı üretmek ve incelemek için:

```bash
python src/models/simulate_2026_batch.py
```
*Çıktı Dosyası:* `data/processed/simulasyon_2026_tahminleri.csv`

### 3. SHAP Öznitelik Açıklanabilirlik Analizi
28 özniteliğin karar üzerindeki etki yüzdelerini analiz etmek için:

```bash
python src/models/explain_shap.py
```

### 4. Zamana Duyarlı Kör (Walk-Forward) Backtest
Modeli geçmiş yıllar üzerinde "kör" sınava sokup doğrulamak için:

```bash
python src/models/backtest_walkforward.py
```

### 5. FastAPI Servisini Başlatma
```bash
uvicorn api.main:app --reload
```
*Swagger UI:* `http://127.0.0.1:8000/docs`

---

## 🛠️ Kurulum

```bash
# 1. Depoyu klonla ve dizine gir
cd C:\Users\ARDA\.gemini\antigravity\scratch\yks-tahmin

# 2. Gerekli kütüphaneleri yükle
pip install -r requirements.txt

# 3. Unit testleri çalıştır
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
├── scraping/                        # YÖK Atlas & ÖSYM PDF Scraper/Parser
│   ├── parse_osym_pdf.py            # PyMuPDF Kılavuz Parser Engine
│   └── yokatlas_scraper.py          # YÖK Atlas JSON API Scraper
├── src/
│   ├── features/build_features.py   # 28-Feature Pipeline
│   └── models/
│       ├── train_quantile.py        # LightGBM + CatBoost Quantile Ensemble
│       ├── explain_shap.py          # SHAP Öznitelik Etki Analizi
│       ├── simulate_2026_batch.py   # Toplu 2026 Simülasyon Motoru
│       └── backtest_walkforward.py  # Zamana Duyarlı Kör Backtest Engine
├── tests/                           # Pytest Test Suitesi (63 Unit Test)
├── tercih_danismani.py              # İnteraktif CLI Tercih Danışmanı
├── PROGRESS.md                      # Detaylı Geliştirme Günlüğü
└── README.md                        # Ana Dokümantasyon
```

---

## 📜 Lisans

[MIT License](LICENSE) — Serbestçe kullanılabilir ve geliştirilebilir.
