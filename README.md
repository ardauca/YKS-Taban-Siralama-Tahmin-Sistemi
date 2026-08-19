# 🎓 YKS 2026 Taban Sıralama Tahmin & Tercih Yönetim Sistemi

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)
![CatBoost](https://img.shields.io/badge/ML-CatBoost%20Model-green?logo=scikitlearn)
![Polars](https://img.shields.io/badge/Data-Polars%20Engine-cyan)
![Textual](https://img.shields.io/badge/UI-Textual%20TUI-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

**YKS 2026 Taban Sıralama Tahmin Sistemi**, üniversite adayları ve tercih danışmanları için geliştirilmiş **production-level** profesyonel bir terminal uygulamasıdır (TUI/CLI). 

CatBoost Makine Öğrenmesi Modeli, 3 kaynaklı Polars veri birleştirme (JOIN) mimarisi ve Türkçe karakter arama motoru ile 21.000'den fazla lisans programının **2022-2026 tarihsel eğilimlerini** ve **2026 tahmini başarı sıralamalarını** sunar.

---

## 🌟 Öne Çıkan Özellikler

- **🤖 CatBoost ML 2026 Tahmin Motoru:** 16,957 lisans programı için nokta tahmini ve **%80 alt/üst güven aralığı**.
- **📊 4 Yıllık Tarihsel Tablo (2022 → 2025 → 2026 Tahmin):** Her program için 4 yıllık taban sıralama, taban puanı, kontenjan değişimi ve kaynak bilgisi.
- **💡 Model Rasyoneli & Tahmin Sebepleri:** Kontenjan şoku etkisi, geçmiş trend ivmesi, puan türü rekabeti ve Google Trends dijital arama popülerliği.
- **🔤 Türkçe Karakter Duyarsız Arama Engine:** `Tip` = `Tıp`, `Eskisehir` = `Eskişehir` kusursuz eşleşme.
- **🎛️ Çoklu-Filtreleme Motoru:** Şehir, Puan Türü (SAY/EA/SÖZ/DİL), Üniversite Türü (Devlet/Vakıf), Öğretim Türü, Burs Oranı ve Sıralama Aralığı filtreleri aynı anda çalışır.
- **🖥️ İki Farklı Çalışma Modu:**
  - **TUI (Textual Terminal Grafik Arayüzü):** Butonlar, dinamik kartlar, tablolar ve renkli zengin terminal ekranları.
  - **CLI (Typer Komut Satırı):** Terminalden tek satır komutla arama, detay, simülasyon ve raporlama.

---

## 🚀 Hızlı Başlangıç (Quick Start)

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/ardauca/YKS-Taban-Siralama-Tahmin-Sistemi.git
cd YKS-Taban-Siralama-Tahmin-Sistemi
```

### 2. Tek Tıkla Kurulum
- **Windows için:** `install.bat` dosyasına çift tıklayın veya CMD'de çalıştırın:
  ```cmd
  install.bat
  ```
- **Linux / macOS için:**
  ```bash
  chmod +x install.sh
  ./install.sh
  ```

### 3. Uygulamayı Başlatın
Sistem otomatik menü başlatıcısı ile çalışmaya hazırdır:
```bash
python baslat.py
```

Veya doğrudan terminal komutu ile:
```bash
python cli/app.py tui
# ya da 'pip install -e .' yaptıysanız:
yks-tahmin
```

---

## 🖥️ TUI Terminal Grafik Arayüzü

Uygulama içerisindeki kısayol tuşları ile ekranlar arası anında geçiş yapabilirsiniz:

| Tuş | Ekran | Açıklama |
|:---:|-------|----------|
| **`D`** | **Dashboard** | Genel istatistik kartları, hızlı arama, favoriler ve son tahminler |
| **`S`** | **Arama Engine** | Etiketli çoklu-filtreleme arama paneli (Şehir, Puan, Tür, Burs, Max Sıra) |
| **`M`** | **ML 2026 Simülasyon** | 16,957 program CatBoost ML tahminleri ve trend filtresi |
| **`U`** | **Üniversite Analizi** | Üniversite bazlı tüm programlar, 2026 ortalama tahmin ve risk dağılımı |
| **`L`** | **Tercih Listem** | Tercih listenizi yönetin, sıra çakışmalarını görün |
| **`T`** | **Trendler** | Türkiye geneli en çok yükselen ve gerileyen bölümler |
| **`V`** | **Favoriler** | Kaydettiğiniz favori programlar |
| **`C`** | **Karşılaştırma** | İki programı yan yana 4 yıllık verileriyle kıyaslayın |
| **`Q`** | **Çıkış** | Uygulamadan güvenle çıkın |

---

## 💻 CLI Komut Satırı Kullanım Rehberi

Komut satırından arama yapmak ve rapor almak son derece hızlıdır:

### 🔎 Hızlı Program & Şehir Araması
```bash
# Eskişehir'deki Devlet Üniversitelerinin EA programları
python cli/app.py search --city "Eskisehir" -p EA --uni-turu DEVLET

# Sıralaması 100.000 altında olan Tıp programları
python cli/app.py search -q "Tıp" -p SAY --max-rank 100000
```

### 📌 Program Detayı (4 Yıllık Tablo & Model Sebepleri)
```bash
python cli/app.py detail 103890170
```

### 🏛️ Üniversite Analizi
```bash
python cli/app.py university "Boğaziçi"
```

### 🎯 2026 ML Simülasyonu
```bash
python cli/app.py simulate -p SAY --trend yukselenler --limit 20
```

### 📊 Türkiye Geneli İpuçları & Özet
```bash
python cli/app.py stats
```

### 📄 Tercih Listesini Dışa Aktar (PDF / Markdown)
```bash
python cli/app.py export --list-id 1 --format pdf --output tercih_listem_2026
```

---

## 🏗️ Sistem Mimarisi & Teknolojiler

```
┌─────────────────────────────────────────────────────────────────┐
│                     TUI (Textual) / CLI (Typer)                 │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│                        Search / Analytics Services              │
│    (Türkçe Character Normalizer + Combined Filter Engine)      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────┐
│               Polars Master DataFrame (Single Source)           │
│   ┌───────────────────┬───────────────────┬─────────────────┐   │
│   │   Feature Matrix  │   YÖK Atlas Raw   │   CatBoost ML   │   │
│   │   (Lag 2022-2025) │   (2022-2025 Raw) │   (2026 Preds)  │   │
│   └───────────────────┴───────────────────┴─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

- **Core:** Python 3.12, Polars (Yüksek Hızlı DataFrame), SQLite & SQLAlchemy
- **Machine Learning:** CatBoost Regressor, Scikit-Learn
- **User Interface:** Textual, Rich, Plotext
- **CLI Engine:** Typer

---

## 📁 Proje Klasör Yapısı

```
yks-tahmin/
├── baslat.py             # Kolay başlatıcı menü (Cross-platform auto-env)
├── install.bat           # Windows otomatik kurulum betiği
├── install.sh            # Linux/macOS otomatik kurulum betiği
├── pyproject.toml        # PIP paketleme ve CLI entrypoint tanımları
├── requirements.txt      # Bağımlılık listesi
├── cli/
│   └── app.py            # Typer CLI komut satırı uygulaması
├── tui/
│   ├── app.py            # Textual TUI ana uygulama ve CSS teması
│   └── screens/          # Dashboard, Search, Detail, University, Simulation vb.
├── services/
│   ├── search_service.py # 3-Kaynak Master Polars JOIN & Arama Motoru
│   ├── analytics_service.py
│   ├── preference_service.py
│   └── chart_service.py  # Rich & Plotext Grafik Servisi
├── data/
│   ├── raw/              # Ham YÖK Atlas & ÖSYM Kılavuz verileri
│   └── processed/        # CatBoost 2026 tahmin sonuçları (CSV)
└── db/                   # SQLite Veritabanı & Repository katmanı
```

---

## 📄 Lisans

Bu proje **MIT Lisansı** altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakabilirsiniz.
