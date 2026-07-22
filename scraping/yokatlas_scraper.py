"""
YÖK Atlas Scraper — Bilgisayar Mühendisliği bölümü için taban sıralama/kontenjan verisi.

Kaynak: https://yokatlas.yok.gov.tr/api/tercih-kilavuz/search (POST JSON)
API'nin yeni (Nisan 2026 sonrası React SPA) versiyonu kullanılıyor.
Eski HTML scraping endpoint'leri kapalı.

Rate limiting: İstekler arası min. 1.5 sn (robots.txt uyumu + etik scraping).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Sabitler ──────────────────────────────────────────────────────────────────

BASE_URL = "https://yokatlas.yok.gov.tr"
USER_AGENT = (
    "YKS-Tahmin-Research/0.1 "
    "(github.com/ardauca/YKS-Taban-Siralama-Tahmin-Sistemi; "
    "veri bilimi arastirma projesi)"
)
# Bilgisayar Mühendisliği birimGrupId
BILGISAYAR_MUH_GRUP_ID = 2010
RATE_LIMIT_SECONDS = 0.2
MAX_RETRIES = 3
PAGE_SIZE = 100  # API'nin desteklediği max sayfa büyüklüğü

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "yokatlas"


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────


def _post_json(path: str, payload: dict[str, Any]) -> Any:
    """Rate-limited, retry-destekli POST isteği."""
    url = BASE_URL + path
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "User-Agent": USER_AGENT,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(RATE_LIMIT_SECONDS)
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            logger.warning("HTTP %s (deneme %d/%d): %s", exc.code, attempt, MAX_RETRIES, url)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RATE_LIMIT_SECONDS * attempt)
        except urllib.error.URLError as exc:
            logger.warning("Bağlantı hatası (deneme %d/%d): %s", attempt, MAX_RETRIES, exc.reason)
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RATE_LIMIT_SECONDS * attempt)


def _get_json(path: str) -> Any:
    """Rate-limited GET isteği."""
    url = BASE_URL + path
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    time.sleep(RATE_LIMIT_SECONDS)
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


# ── Veri çekme ────────────────────────────────────────────────────────────────


def fetch_all_programs(birim_grup_id: int = BILGISAYAR_MUH_GRUP_ID) -> list[dict]:
    """
    Verilen birimGrupId için tüm programları sayfalayarak çeker.
    Her kayıt: mevcut yıl + son 3 yılın taban puan / başarı sırası / kontenjan bilgisi.

    Returns:
        Ham API kayıtlarının listesi (normalize edilmemiş).
    """
    logger.info("Programlar çekiliyor (birimGrupId=%d)...", birim_grup_id)
    page = 0
    all_records: list[dict] = []

    while True:
        payload = {
            "filters": {"birimGrupId": [birim_grup_id]},
            "page": page,
            "size": PAGE_SIZE,
            "sortBy": "basariSirasi",
            "direction": "ASC",
        }
        data = _post_json("/api/tercih-kilavuz/search", payload)

        content = data.get("content", [])
        total = data.get("totalElements", 0)
        all_records.extend(content)
        logger.info(
            "Sayfa %d: %d kayıt alındı | toplam: %d/%d",
            page,
            len(content),
            len(all_records),
            total,
        )

        # Son sayfa kontrolü
        if len(all_records) >= total or not content:
            break
        page += 1

    logger.info("Toplam %d kayıt çekildi.", len(all_records))
    return all_records


# ── Veri dönüştürme ───────────────────────────────────────────────────────────


def normalize_records(raw_records: list[dict]) -> pd.DataFrame:
    """
    API'nin döndürdüğü ham kayıtları uzun formata (tidy data) dönüştürür.
    Her satır = bir program × bir yıl.

    API'nin yıl kodlaması:
        suffix yok  → mevcut yıl (2025)
        suffix "1"  → 1 yıl önce (2024)
        suffix "2"  → 2 yıl önce (2023)
        suffix "3"  → 3 yıl önce (2022)

    Returns:
        Normalize edilmiş DataFrame.
    """
    rows = []
    # Mevcut yıl API canlı veri döndürüyorsa 2025, değilse None
    current_year = 2025

    year_map = {
        "": current_year,
        "1": current_year - 1,
        "2": current_year - 2,
        "3": current_year - 3,
    }

    for rec in raw_records:
        base_info = {
            "kilavuz_kodu": rec.get("kilavuzKodu"),
            "universite_adi": rec.get("universiteAdi"),
            "birim_adi": rec.get("birimAdi"),
            "birim_grup_adi": rec.get("birimGrupAdi"),
            "il_kodu": rec.get("ilKodu"),
            "il_adi": rec.get("ilAdi"),
            "universite_turu": rec.get("universiteTuru"),  # Devlet / Vakıf
            "ogretim_turu": rec.get("ogretimTuru"),        # Örgün / Uzaktan
            "burs_orani": rec.get("bursOrani"),
            "puan_turu": rec.get("puanTuru"),
            "kaynak": "yokatlas_api",
            "cekme_tarihi": pd.Timestamp.now().date().isoformat(),
        }

        for suffix, year in year_map.items():
            gk_key = f"gk{suffix}" if suffix else "kontenjan"
            puan_key = f"minPuan{suffix}" if suffix else "minPuan"
            siralama_key = f"basariSirasi{suffix}" if suffix else "basariSirasi"

            # Yılın verisi API'de hiç yoksa satır ekleme
            puan_raw = rec.get(puan_key)
            siralama_raw = rec.get(siralama_key)
            genel_kontenjan = rec.get(gk_key)

            if puan_raw is None and siralama_raw is None and genel_kontenjan is None:
                continue  # Bu yılın verisi yok

            row = {
                **base_info,
                "yil": year,
                "genel_kontenjan": _safe_int(genel_kontenjan),
                "sehit_gazi_kontenjan": _safe_int(rec.get(f"sgy{suffix}")),
                "depremzede_kontenjan": _safe_int(rec.get(f"dprm{suffix}")),
                "okul_birincisi_kontenjan": _safe_int(rec.get(f"obk{suffix}")),
                "taban_puan": _safe_float(puan_raw),
                "taban_siralama": _safe_int(siralama_raw),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


# Ana Bölüm Grupları (SAY, EA, SÖZ, DİL)
MAJOR_DEPARTMENT_GROUPS = {
    # SAY
    "bilgisayar_muhendisligi": 2010,
    "tip": 5370,
    "elektrik_elektronik_muhendisligi": 2644,
    "makine_muhendisligi": 3987,
    "endustri_muhendisligi": 2704,
    "dis_hekimligi": 2445,
    "eczacilik": 2471,
    "yazilim_muhendisligi": 5821,
    "insaat_muhendisligi": 3468,
    "mimarlik": 4108,
    "ic_mimarlik": 3338,
    "hemsirelik": 3217,
    "ilkogretim_matematik_ogretmenligi": 3410,
    # EA
    "hukuk": 3309,
    "iktisat": 3353,
    "isletme": 3549,
    "psikoloji": 4679,
    "yonetim_bilisim_sistemleri": 5874,
    "siyaset_bilimi_ve_kamu_yonetimi": 4967,
    "sinif_ogretmenligi": 4915,
    "uluslararasi_iliskiler": 5581,
    # SÖZ
    "ozel_egitim_ogretmenligi": 8660,
    "tarih": 5163,
    "turk_dili_ve_edebiyati": 5468,
    # DİL
    "ingilizce_ogretmenligi": 3441,
    "ingiliz_dili_ve_edebiyati": 3431,
}


def _safe_int(val: Any) -> int | None:
    """Güvenli int dönüşümü — dönüşemezse None döner."""
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_float(val: Any) -> float | None:
    """Güvenli float dönüşümü — dönüşemezse None döner."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ── Ana akış ─────────────────────────────────────────────────────────────────


def run(grup_id: int = BILGISAYAR_MUH_GRUP_ID, filename: str = "bilgisayar_muhendisligi_raw.csv", save: bool = True) -> pd.DataFrame:
    """
    Belirtilen birimGrupId için scraping akışını çalıştırır.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    raw = fetch_all_programs(grup_id)
    df = normalize_records(raw)

    if save and not df.empty:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RAW_DIR / filename
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info("Veri kaydedildi: %s (%d satır)", out_path, len(df))

    return df


def run_all_departments(save: bool = True) -> pd.DataFrame:
    """
    Tüm ana bölüm aileleri için (Tıp, Hukuk, Mühendislikler, İktisat/İşletme vb.)
    verileri çeker ve konsolide ham veri setini kaydeder.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    all_dfs = []
    for dept_name, grup_id in MAJOR_DEPARTMENT_GROUPS.items():
        logger.info("\n=== Bölüm Çekiliyor: %s (ID: %d) ===", dept_name, grup_id)
        df_dept = run(grup_id=grup_id, filename=f"{dept_name}_raw.csv", save=save)
        all_dfs.append(df_dept)

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        if save:
            out_combined = RAW_DIR / "yokatlas_all_departments_raw.csv"
            combined.to_csv(out_combined, index=False, encoding="utf-8-sig")
            logger.info("\n=== TÜM BÖLÜMLER KONSOLİDE EDİLDİ: %s (%d satır) ===", out_combined, len(combined))
        return combined

    return pd.DataFrame()


if __name__ == "__main__":
    df_all = run_all_departments()
    print("\nKonsolide Veri Seti Özeti:")
    print(f"Toplam Satır: {len(df_all)}")
    if not df_all.empty:
        print(df_all.groupby(["birim_grup_adi", "yil"])["kilavuz_kodu"].count())
