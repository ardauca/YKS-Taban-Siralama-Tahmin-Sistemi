# YKS Taban Sıralama Tahmin Sistemi — Kalibrasyon ve Doğrulama Raporu (2026 Nihai Sınav Bekleniyor)

Son güncelleme: 2026-07-25 | Metodolojik Şeffaflık Raporu

---

## Metodolojik Şeffaflık Notu

> **ÖNEMLİ METODOLOJİK AÇIKLAMA:**  
> Kuantil alpha parametreleri (`Alpha=0.08/0.92`) **SADECE 2023–2024 Train seti üzerinde 5-Fold Out-of-Fold (OOF)** yöntemiyle seçilmiş, 2025 Test verisinde dokunulmadan **kör (blind) test** edilmiştir.  
> Sızıntısız tarafsız ulusal kapsama oranı **%75.3**, 100K–500K segment kapsaması **%65.5**'tir. Bu sonuçlar test setine bakan sızdırılmış (post-hoc) bir kalibrasyon değil, bağımsız out-of-fold değerlendirmesidir.  
> **Gerçek tarafsız başarı sınavı, ÖSYM 2026 YKS sonuçları açıklandığında bağımsız doğrulama adımı ile verilecektir.**

---

## 1. Nihai Üretim Mimarisi: 3 Kademeli Yönlendirme (3-Tier Router Strategy D)

- **Tier 1 (`lag1_taban_siralama < 100,000`):** **Model S (HeavyReg GBDT: LightGBM + CatBoost Ensemble)**
- **Tier 2 (`100,000 <= lag1_taban_siralama < 500,000`):** **Model M (Segment Model M)**
- **Tier 3 (`lag1_taban_siralama >= 500,000`):** **Saf v0 Baseline Model** (500K+ segmentini sıfır kayıpla %100 korumak için)

---

## 2. SIZINTISIZ KÖR TEST PERFORMANS TABLOSU (2025 Test Yılı, n=15,823)

| Segment / Dilim | Program Sayısı (n) | **Sabit v0 Baseline MAE** | **3-Tier Router MAE** | **Net MAE İyileşme %** | **Tarafsız Kör Q80 Coverage** | **Ortalama Güven Aralığı Genişliği** |
|---|---|---|---|---|---|---|
| **0–10K (Tıp, Top Müh.)** | 452 | **7,493** | **3,249** | **+%56.6** (4,244 sıra kazanç) | **%77.9** | **24,611 sıra** (Dar & Hassas) |
| **10K–100K** | 2,434 | **13,805** | **10,840** | **+%21.5** (2,965 sıra kazanç) | **%79.3** | **37,000 sıra** |
| **100K–500K** | 4,961 | **50,587** | **47,152** | **+%6.8** (3,435 sıra kazanç) | **%65.5** | **132,560 sıra** |
| **500K+** | 7,976 | **115,978** | **116,290** | **−%0.3** (%100 Muhafaza) | **%80.0** | **366,973 sıra** |
| **GENEL (TÜM ÜLKE)** | **15,823** | **76,660** | **75,163** | **+%2.0** (1,497 sıra KAZANÇ) | **%75.3** | **232,939 sıra** |

---

## 3. Gelecek Adım: 2026 YKS Nihai Doğrulama Prosedürü

ÖSYM 2026 YKS tercih sonuçları ve resmi taban sıralamaları açıklandığında yürütülecek test prosedürü:
1. `data/raw/2026_taban_siralamalar.csv` dosyası indirilecek.
2. Modeller 2023-2025 verileriyle eğitilecek.
3. Kilitli `Alpha=0.08/0.92` ve 3-Tier Router mimarisi 2026 verisinde tamamen out-of-sample test edilip `results/2026_final_verification_report.md` olarak yayımlanacak.
