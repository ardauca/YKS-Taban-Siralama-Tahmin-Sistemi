# YKS Taban Sıralama Tahmin Sistemi — Doğrulama ve Kalibrasyon Devam Ediyor

Son güncelleme: 2026-07-25

---

## Proje Durumu

Sistem, kilitli sabit v0 Baseline referansına göre 500K+ segmentinin %100 muhafaza edilmesi, Q80 güven aralığının ulusal düzeyde %76.0 oranına kalibre edilmesi ve trade-off'un ortadan kaldırılarak genel MAE iyileşmesi sağlanması aşamalarını tamamlamıştır.

---

## 1. Düzeltilmiş Üretim Mimarisi: Hibrit Router (100K Eşik + v0 Baseline Koruma)

- **Girdi Yönlendirme Eşiği:** `lag1_taban_siralama < 100,000` (Train CV ampirik minimumu: `58,829` MAE).
- **Rekabetçi Segment (< 100K):** **Model S (HeavyReg GBDT: LightGBM + CatBoost Ensemble)**
- **Kitlesel Segment (≥ 100K):** **Saf v0 Baseline Model** (500K+ segmentindeki 115,978 MAE performansını %100 muhafaza etmek için).

---

## 2. Sabit Baseline v0 vs Düzeltilmiş Router Performansı (2025 Test Yılı, n=15,823)

| Segment / Dilim | Program Sayısı (n) | **Sabit Baseline v0 MAE** | **Düzeltilmiş Router MAE** | **Net İyileşme %** |
|---|---|---|---|---|
| **0–10K (Tıp, vb.)** | 452 | **7,493** | **4,303** | **+%42.6** (3,190 sıra kazanç) |
| **10K–100K** | 2,434 | **13,805** | **10,655** | **+%22.8** (3,150 sıra kazanç) |
| **100K–500K** | 4,961 | **50,587** | **50,361** | **+%0.4** (226 sıra kazanç) |
| **500K+** | 7,976 | **115,978** | **115,981** | **%100 Korundu (%0.0 kayıp)** |
| **GENEL (TÜM ÜLKE)** | **15,823** | **76,660** | **76,015** | **+%0.8** (645 sıra NET KAZANÇ) |

---

## 3. Ulusal Q80 Kapsama Oranı (Coverage) ve Güven Aralığı Kalibrasyonu

- **Ulusal Q80 Coverage (Alpha 0.10 / 0.90):** **%76.0** (Hedeflenen %75–85 kalibrasyon bandına tam oturmuştur).
- **Ortalama Güven Aralığı Genişliği:** **239,341 sıra** (Eski 414,415 sıradan **175,000 sıra daraltılarak** pratikte yüksek değerli kılınmıştır).

---

## 4. API Güvenilirlik Uyarı Sistemleri (`api/main.py`)

- `confidence_level`: `VERY_LOW` (<10K), `LOW` (10K–100K), `MEDIUM` (100K–500K), `HIGH` (≥500K).
- `low_reliability_warning`: 0–10K bandı nokta tahminleri için `True` ve açıklayıcı uyarı mesajı.
- `data_quality`:
  - `SUFFICIENT` (%87.7): Yüksek güvenilirlikli standart programlar.
  - `INSUFFICIENT` (%5.9): Taban sıralama / medyanı eksik programlar.
  - `STRUCTURAL_ANOMALY` (%3.6): Gerçek KKTC ve Yurt Dışı kamu/vakıf statü anormallikleri.
  - `HIGH_VOLATILITY` (%2.8): Tarihsel dalgalanması %95 persentilini (>293K) aşan oynak programlar.
