# YKS Taban Sıralama Tahmin Sistemi — İlerleme Kaydı

Son güncelleme: 2026-07-25

---

## Proje Durumu

Sistem aktif geliştirme aşamasındadır. Aşağıdaki bulgu ve metrikler doğrulama sürecinden geçmiştir.

---

## Gerçekleştirilen Adımlar

### Veri Toplama

- YÖK Atlas REST API: Tüm lisans programları, 2022–2025 yılları. Toplam: **77,970 satır, 21,482 program**.
- ÖSYM 2026–2027 Ön Kontenjan Kılavuzu (PDF): PyMuPDF ile parse edildi. **11,676 programın 2026 kontenjnı** elde edildi.
- Duplicate kayıt: 0 (kilavuz_kodu+yil bazında).

### Veri Kalitesi (Bölüm Ailesi Bazında)

- Toplam bölüm ailesi (birim_grup_adi): 625
- Ortalama taban_siralama eksik oranı: %22.6 (büyük çoğunluğu 2025 yılına ait — yerleştirme kesinleşmedi)
- Eksik ≥ %80 olan bölüm ailesi: 31 (tamamı 2025'te açılmış yeni bölümler, n < 5)
- Eksik ≥ %50 olan bölüm ailesi: 91

Büyük bölüm aileleri (n > 700 satır) için eksik sıralama oranları %5–46 arasında değişmektedir.

### Feature Engineering (28 Öznitelik)

1. **Lag-1:** Y-1 sıralama, puan, kontenjan (shift(1) ile programın kendi tarihinden)
2. **Lag-2:** Y-2 sıralama (trend için)
3. **Delta:** siralama_trend, siralama_pct_change, kontenjan_degisim_orani
4. **Statik:** üniversite türü, il kodu, puan türü, burs oranı
5. **Türetilmiş:** program_hist_medyan_siralama, univ_hist_medyan_siralama, makro kontenjan şoku, baraj mesafe indeksi vb.

---

## Doğrulama: Temporal Leakage Tespiti ve Düzeltmesi

### Tespit (2026-07-25)

`program_hist_medyan_siralama` ve `univ_hist_medyan_siralama` öznitelikleri, `build_features.py`'de **tüm veri seti üzerinde global groupby** ile hesaplanıyordu. Bu, temporal sıra gözetilmeksizin test yılının lag değerlerini medyan hesabına dahil ediyor ve dolaylı leakage yaratıyordu.

### Düzeltme

`_expanding_window_median()` fonksiyonu eklendi. Her satır için yalnızca `yil < current_year` olan satırların değerlerinden medyan hesaplanıyor. 2022 yılındaki satırlar için geçmiş veri olmadığından bu feature NaN kalıyor (doğru davranış); count 70,736 → 32,619'a indi.

### Backtest Sonuçları — Öncesi/Sonrası

| Metrik | Global (Hatalı) | Expanding Window (Düzeltilmiş) |
|---|---|---|
| MAE — 2024 Test | 68,923 | **89,224** |
| R² — 2024 Test | 0.966 | **0.950** |
| MAE — 2025 Test | 112,655 | **86,676** |
| R² — 2025 Test | 0.899 | **0.937** |
| Ortalama MAE | 90,789 | **87,950** |
| Q80 Coverage | 88.5% | **91.8%** |

**Gözlemler:**
- 2024 R²'de beklenen düşüş gerçekleşti: 0.966 → 0.950 (~1.6 puan). Leakage etkisi mevcut ancak tahmin edilenden az.
- 2025 R²'de beklenmedik iyileşme: 0.899 → 0.937. Nedenin global medyanın 2025 kontenjan şokunu yanlış temsil etmesinden kaynaklandığı değerlendirilmektedir.
- Q80 Coverage %91.8 (hedef %80 üzerinde — aralıklar biraz geniş).

**Bu metrikler şu anda kullanılan geçerli referans değerlerdir.**

---

## Mimari: Mevcut Durum

Sistem **tek regresyon modeli** olarak çalışmaktadır:
- LightGBM + CatBoost Hybrid Quantile Ensemble
- Giriş: Y-1 ve Y-2 lag değerleri + statik öznitelikler + makro kontenjan şoku
- Çıkış: taban_siralama nokta tahmini + %80 güven aralığı

Ayrı bir talep tahmini veya yerleştirme simülasyonu katmanı mevcut değildir. Mimari karar (tek regresyon mu kalacak, üstüne düzeltme kuralı mı eklenecek) gerçek metrikler netleştikten sonra alınacak.

---

## Sonraki Adımlar

- [ ] Mimari kararı: Mevcut regresyon + kontenjan şoku düzeltme kuralı mı, yoksa çok aşamalı yapı mı?
- [ ] Q80 Coverage %91.8 → %80 kalibrasyonu için alpha değerlerini ayarla.
- [ ] 2025 test setinin büyük bölümü null olduğundan 2026 simülasyonunun güvenilirliğini nesnel bir ölçütle değerlendir.
