"""
ÖSYM 2026-2027 YKS Ön Kontenjan Kılavuzu (Tablo-4) PDF Parser ve Kalite Kontrol Modülü.

Girdiler:
  PDF: C:/Users/ARDA/Downloads/kontkilavuz_yktd21072026.pdf (793 sayfa)
  Tablo-4 Aralığı: Sayfa 150 — Sayfa 554

Çıktı:
  CSV: data/raw/osym/kontenjan_kilavuzu_2026.csv
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scraping.quality import run_quality_checks

logger = logging.getLogger(__name__)

PDF_PATH = Path(r"C:\Users\ARDA\Downloads\kontkilavuz_yktd21072026.pdf")
OSYM_RAW_DIR = ROOT / "data" / "raw" / "osym"
YOKATLAS_ALL_CSV = ROOT / "data" / "raw" / "yokatlas" / "yokatlas_all_departments_raw.csv"
OUTPUT_CSV = OSYM_RAW_DIR / "kontenjan_kilavuzu_2026.csv"


def parse_osym_2026_pdf(pdf_path: Path = PDF_PATH) -> pd.DataFrame:
    """
    PyMuPDF metin-akış parser'ı ile ÖSYM 2026 Ön Kontenjan Kılavuzu (Tablo-4)
    tüm lisans programlarını 5 saniyede çıkarır.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not pdf_path.exists():
        raise FileNotFoundError(f"ÖSYM PDF kılavuzu bulunamadı: {pdf_path}")

    logger.info("ÖSYM 2026 PDF Okunuyor (Ultra-Fast Text Parser): %s", pdf_path)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    rows = []
    curr_uni = None
    curr_fak = None

    # Tablo-4: Sayfa 150 - 555 (index 149 - 554)
    for page_idx in range(149, min(555, total_pages)):
        raw_lines = [l.strip() for l in doc[page_idx].get_text("text").splitlines() if l.strip()]

        i = 0
        while i < len(raw_lines):
            line = raw_lines[i]

            # 1. Üniversite başlığı tespiti (Çoklu satır birleştirme)
            if any(kw in line.upper() for kw in ["ÜNİVERSİTESİ", "ÜNIVERSITESI", "DEVLET ÜNİVERSİTESİ", "VAKIF ÜNİVERSİTESİ"]):
                uni_candidate = line
                # Eğer önceki satır Üniversite adının başı ise birleştir
                if i > 0 and any(kw in raw_lines[i-1].upper() for kw in ["ÜNİVERSİTESİ", "ÜNIVERSITESI", "DEVLET", "VAKIF", "RÖKTÖRLÜĞÜ", "REKTORLUGU"]) or (i > 0 and len(raw_lines[i-1]) > 5 and not raw_lines[i-1].isdigit() and "TABLO" not in raw_lines[i-1].upper()):
                    if "TABLO" not in raw_lines[i-1].upper() and "LİSANS" not in raw_lines[i-1].upper() and "2025" not in raw_lines[i-1]:
                        uni_candidate = f"{raw_lines[i-1]} {line}"
                curr_uni = uni_candidate
                curr_fak = None
                i += 1
                continue

            # 2. Fakülte / Yüksekokul başlığı
            if any(kw in line.upper() for kw in ["FAKÜLTESİ", "FAKULTESI", "YÜKSEKOKULU", "YUKSEKOKULU", "ENSTİTÜSÜ"]):
                curr_fak = line
                i += 1
                continue

            # 3. 9 Haneli Program Kodu (Örn. 108410336)
            if re.match(r"^\d{9}$", line):
                pk = int(line)
                prog_name = raw_lines[i+1] if i+1 < len(raw_lines) else None
                sure = raw_lines[i+2] if i+2 < len(raw_lines) else None
                puan_turu = raw_lines[i+3] if i+3 < len(raw_lines) else None
                kontenjan = raw_lines[i+4] if i+4 < len(raw_lines) else None

                # Süre ve puan türü doğrulaması
                if sure and sure.isdigit() and puan_turu in ["SAY", "EA", "SÖZ", "DİL"]:
                    rows.append({
                        "program_kodu": pk,
                        "universite_adi": curr_uni,
                        "fakulte_adi": curr_fak,
                        "program_adi": prog_name,
                        "ogretim_suresi": int(sure),
                        "puan_turu": puan_turu,
                        "genel_kontenjan": int(kontenjan) if kontenjan and kontenjan.isdigit() else None,
                        "kaynak": "osym_2026_preliminary_pdf",
                    })
                    i += 5
                    continue

            i += 1

    doc.close()
    df = pd.DataFrame(rows)
    logger.info("Tablo-4 Lisans Programları Çıkarıldı: Toplam %d satır.", len(df))
    return df


def save_and_evaluate_osym_data(df: pd.DataFrame) -> dict:
    """
    ÖSYM verisini kaydeder, YÖK Atlas verileriyle eşleştirilebilirlik ve kalite raporu sunar.
    """
    OSYM_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    logger.info("ÖSYM 2026 Kontenjan Kılavuzu Kaydedildi: %s (%d satır)", OUTPUT_CSV, len(df))

    # Kalite kontrolü (Quality module - ÖSYM Kılavuz Şeması)
    quality_report = run_quality_checks(df, expected_min=3000, expected_max=20000, dataset_type="osym")

    # YÖK Atlas Eşleştirme Kontrolü
    match_summary = {}
    if YOKATLAS_ALL_CSV.exists():
        df_yok = pd.read_csv(YOKATLAS_ALL_CSV)
        yok_codes_2025 = set(df_yok[df_yok["yil"] == 2025]["kilavuz_kodu"].dropna().astype(int))
        osym_codes_2026 = set(df["program_kodu"].astype(int))

        matched_codes = yok_codes_2025.intersection(osym_codes_2026)
        match_rate = len(matched_codes) / len(yok_codes_2025) if yok_codes_2025 else 0.0

        # Kontenjan Değişim Analizi (Eşleşen programlar için 2025 YÖK Atlas vs 2026 ÖSYM)
        df_yok_2025 = df_yok[df_yok["yil"] == 2025][["kilavuz_kodu", "genel_kontenjan"]].rename(
            columns={"kilavuz_kodu": "program_kodu", "genel_kontenjan": "genel_kontenjan_2025"}
        )
        df_merged = df.merge(df_yok_2025, on="program_kodu", how="inner")
        df_merged["kontenjan_farki"] = df_merged["genel_kontenjan"] - df_merged["genel_kontenjan_2025"]

        artti = int((df_merged["kontenjan_farki"] > 0).sum())
        azaldi = int((df_merged["kontenjan_farki"] < 0).sum())
        ayni = int((df_merged["kontenjan_farki"] == 0).sum())

        match_summary = {
            "quality_report_passed": quality_report.passed,
            "yok_atlas_2025_program_count": len(yok_codes_2025),
            "osym_2026_total_programs": len(osym_codes_2026),
            "matched_programs_count": len(matched_codes),
            "matchability_rate": match_rate,
            "new_unmatched_in_osym": len(osym_codes_2026 - yok_codes_2025),
            "kontenjan_artti_count": artt_cnt if 'artt_cnt' in locals() else (artti if 'artti' in locals() else 0),
            "kontenjan_azaldi_count": azaldi,
            "kontenjan_ayni_count": ayni,
        }

    return match_summary


if __name__ == "__main__":
    df_osym = parse_osym_2026_pdf()
    summary = save_and_evaluate_osym_data(df_osym)

    print("\n" + "=" * 70)
    print(" ÖSYM 2026 TABLO-4 ÖN KONTENJAN KILAVUZU PARSE SONUÇLARI")
    print("=" * 70)
    print(f"Toplam Parse Edilen Lisans Programı : {len(df_osym):,}")
    print("\nPuan Türü Dağılımı:")
    print(df_osym["puan_turu"].value_counts().to_string())

    if summary:
        print("\n" + "=" * 70)
        print(" YÖK ATLAS İLE EŞLEŞTİRİLEBİLİRLİK VE KALİTE RAPORU")
        print("=" * 70)
        print(f"Kalite Kontrol Test Sonucu        : {'GEÇTİ' if summary['quality_report_passed'] else 'BAŞARISIZ'}")
        print(f"YÖK Atlas 2025 Program Sayısı      : {summary['yok_atlas_2025_program_count']:,}")
        print(f"Eşleşen Program Sayısı (Tam Kodu)  : {summary['matched_programs_count']:,}")
        print(f"Eşleşebilirlik Oranı (Match Rate)   : %{summary['matchability_rate']*100:.1f}")
        print(f"ÖSYM 2026'daki Yeni/Diğer Program  : {summary['new_unmatched_in_osym']:,}")
        print("\n2025 (YÖK Atlas) -> 2026 (ÖSYM Kılavuz) Kontenjan Değişim İstatistikleri:")
        print(f"  Kontenjanı Artan Programlar     : {summary['kontenjan_artti_count']:,}")
        print(f"  Kontenjanı Azalan Programlar    : {summary['kontenjan_azaldi_count']:,}")
        print(f"  Kontenjanı Değişmeyen Programlar: {summary['kontenjan_ayni_count']:,}")
