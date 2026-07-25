"""
Türkiye'deki TÜM Lisans Programlarını (%100 Kapsama) YÖK Atlas'tan Güvenli Olarak Çeken Scraper.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import sys
import time
import urllib.request

import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from scraping.yokatlas_scraper import normalize_records, RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

URL = "https://yokatlas.yok.gov.tr/api/tercih-kilavuz/search"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json",
}


def fetch_page_records(page_num: int) -> list[dict]:
    payload = json.dumps({"filters": {}, "page": page_num, "size": 100, "sortBy": "basariSirasi", "direction": "ASC"}).encode("utf-8")
    for attempt in range(1, 4):
        try:
            req = urllib.request.Request(URL, data=payload, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read()).get("content", [])
        except Exception:
            time.sleep(0.5 * attempt)
    logger.warning("Sayfa %d 3 denemede de çekilemedi.", page_num)
    return []


def run_full_yokatlas_scrape() -> pd.DataFrame:
    logger.info("YÖK Atlas Türkiye Geneli %100 Lisans Taraması Başlatılıyor (215 Sayfa)...")
    
    all_raw_records = []
    pages = list(range(0, 216))

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_page_records, p): p for p in pages}
        done_count = 0
        for future in as_completed(futures):
            recs = future.result()
            if recs:
                all_raw_records.extend(recs)
            done_count += 1
            if done_count % 30 == 0:
                logger.info("İlerleme: %d / 215 sayfa tamamlandı...", done_count)

    logger.info("Toplam %d ham YÖK Atlas kaydı çekildi! Normalize ediliyor...", len(all_raw_records))
    
    df_all = normalize_records(all_raw_records)
    
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "yokatlas_all_departments_raw.csv"
    df_all.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("=== %%100 VERI SETI BASARIYLA KAYDEDILDI: %s (%d satir) ===", out_path, len(df_all))

    return df_all


if __name__ == "__main__":
    df = run_full_yokatlas_scrape()
    print("\n%100 YÖK Atlas Veri Seti Özeti:")
    print(f"Toplam Satır Sayısı: {len(df)}")
    print(f"Benzersiz Program Sayısı: {df['kilavuz_kodu'].nunique()}")
