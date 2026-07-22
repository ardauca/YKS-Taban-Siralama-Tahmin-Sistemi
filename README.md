# YKS Taban Sıralama Tahmin Sistemi

Türkiye üniversite bölümlerinin geçmiş yıl verilerinden öğrenip, **bu senenin açıklanan kontenjanlarına göre oluşacak taban sıralamalarını** tahmin eden üç aşamalı sistem.

## Mimari

```
1. Talep Tahmini      → regresyon modeli (CatBoost)
2. Yerleştirme Sim.   → kural tabanlı ÖSYM simülatörü
3. Belirsizlik Aralığı → quantile regresyon (LightGBM)
```

Her tahmin çıktısı: `point_estimate`, `lower_bound`, `upper_bound`, `confidence_level`

## Kurulum

```bash
# 1. Sanal ortam
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Bağımlılıklar
pip install -r requirements.txt

# 3. Playwright tarayıcısı
playwright install chromium

# 4. Ortam değişkenleri
cp .env.example .env
# .env dosyasını düzenle

# 5. Testleri çalıştır
pytest tests/ -v
```

## Klasör Yapısı

```
yks-tahmin/
├── data/{raw,interim,processed}/   # Ham ve işlenmiş veri
├── scraping/                        # Web scraper modülleri
├── notebooks/                       # Keşifsel analiz
├── src/{features,models,evaluation}/ # Core ML pipeline
├── api/                             # FastAPI servisi
├── frontend/                        # Streamlit / Next.js
├── mlruns/                          # MLflow deneyleri
└── docs/                            # Teknik dökümanlar
```

## Veri Kaynakları

| Kaynak | İçerik | Öncelik |
|---|---|---|
| YÖK Atlas | Taban sıralama, kontenjan, doluluk | Birincil |
| ÖSYM Kılavuzu | Resmi puan/sıralama | Birincil |
| TÜİK | Şehir ekonomi göstergeleri | İkincil |

## Katkı Kuralları

- Her commit Conventional Commits formatında
- Test yazmadan "tamamlandı" denmez
- Gizli anahtar asla koda girilmez
- Her model MLflow'a loglanır

## Lisans

MIT
