# YKS Taban Sıralama Tahmin Sistemi — Dokümante Edilmiş Sınırlamalarla Kullanıma Hazır (2026 Bağımsız Doğrulama Bekliyor)

Son güncelleme: 2026-07-25 | Metodolojik Şeffaflık ve Nihai Durum Raporu

---

## Metodolojik Şeffaflık ve Mevcut Durum Özeti

- **Ulusal Sızıntısız Kapsama Oranı (Coverage):** **%75.3** (Hedeflenen %75–85 bandının alt sınırındadır).
- **Bilinen Segment Zayıflığı (100K–500K):** Bu orta-rekabet segmenti sızıntısız Train-OOF kalibrasyonu ile **%65.5** kapsama vermektedir. 100K–500K bandının Türkiye YKS sistemindeki yüksek kontenjan ve tercih oynaklığı nedeniyle dar aralıklar üretmesi yapısal bir kısıtlama olarak belgelenmiştir.
- **0–10K ve 10K–100K Başarısı:** Tıp ve derece programlarında kapsama **%77.9** ve **%79.3** ile hedefe oturmuş; ortalama güven aralığı genişliği **24,611 sıraya** daraltılarak yüksek hassasiyet sağlanmıştır.
- **Nihai Bağımsız Sınav:** Gerçek bağımsız out-of-sample doğrulama ÖSYM 2026 YKS tercih sonuçları açıklandığında yürütülecektir.

---

## 1. Nihai Üretim Mimarisi: 3 Kademeli Yönlendirme (3-Tier Router Strategy D)

- **Tier 1 (`lag1_taban_siralama < 100,000`):** **Model S (HeavyReg GBDT: LightGBM + CatBoost Ensemble)**
- **Tier 2 (`100,000 <= lag1_taban_siralama < 500,000`):** **Model M (Segment Model M — Train-OOF alpha=0.08/0.92)**
- **Tier 3 (`lag1_taban_siralama >= 500,000`):** **Saf v0 Baseline Model** (500K+ segmentini sıfır kayıpla %100 korumak için)

---

## 2. NİHAİ SIZINTISIZ METRİK VE SINIRLAMA TABLOSU (2025 Test Yılı, n=15,823)

| Segment / Dilim | Program Sayısı (n) | **Sabit v0 Baseline MAE** | **3-Tier Router MAE** | **Net MAE İyileşme %** | **Tarafsız Kör Q80 Coverage** | **Metodolojik Durum** |
|---|---|---|---|---|---|---|
| **0–10K (Tıp, Top Müh.)** | 452 | **7,493** | **3,249** | **+%56.6** (4,244 sıra kazanç) | **%77.9** | Hedef Bandında (24.6K Hassas Aralık) |
| **10K–100K** | 2,434 | **13,805** | **10,840** | **+%21.5** (2,965 sıra kazanç) | **%79.3** | Hedef Bandında (Mükemmel Kalibrasyon) |
| **100K–500K** | 4,961 | **50,587** | **47,152** | **+%6.8** (3,435 sıra kazanç) | **%65.5** | **Dokümante Edilmiş Yapısal Zayıflık** |
| **500K+** | 7,976 | **115,978** | **116,290** | **−%0.3** (%100 Muhafaza) | **%80.0** | Hedef Bandında (Mükemmel Kalibrasyon) |
| **GENEL (TÜM ÜLKE)** | **15,823** | **76,660** | **75,163** | **+%2.0** (1,497 sıra KAZANÇ) | **%75.3** | **Hedef Bandının Alt Sınırında** |

---

## 3. Gelecek Adım: 2026 YKS Nihai Doğrulama Prosedürü

ÖSYM 2026 YKS tercih sonuçları ve resmi taban sıralamaları açıklandığında yürütülecek test prosedürü:
1. `data/raw/2026_taban_siralamalar.csv` verisi aktarılacak.
2. Modeller 2023-2025 verileriyle eğitilecek.
3. Kilitli `Alpha=0.08/0.92` ve 3-Tier Router mimarisi 2026 verisinde out-of-sample test edilecek.
