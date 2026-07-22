"""
Unit testler: scraping/quality.py modülü için.

Çalıştırma: pytest tests/test_scraper.py -v
"""

import pandas as pd
import pytest

# Proje root'unu Python path'ine ekle
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scraping.quality import (
    CRITICAL_COLUMNS,
    DUPLICATE_KEY,
    EXPECTED_COLUMNS,
    QualityReport,
    run_quality_checks,
)
from scraping.yokatlas_scraper import _safe_float, _safe_int, normalize_records


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_valid_df(n_programs: int = 50, years: list[int] | None = None) -> pd.DataFrame:
    """Test için geçerli bir DataFrame üretir."""
    if years is None:
        years = [2022, 2023, 2024, 2025]

    rows = []
    for i in range(n_programs):
        for year in years:
            rows.append({
                "kilavuz_kodu": 100000 + i,
                "universite_adi": f"Test Üniversitesi {i % 10}",
                "birim_adi": "Bilgisayar Mühendisliği",
                "birim_grup_adi": "Bilgisayar Mühendisliği",
                "il_kodu": "06",
                "il_adi": "Ankara",
                "universite_turu": "Devlet",
                "ogretim_turu": "Örgün",
                "burs_orani": None,
                "puan_turu": "SAY",
                "kaynak": "yokatlas_api",
                "cekme_tarihi": "2025-07-22",
                "yil": year,
                "genel_kontenjan": 50,
                "sehit_gazi_kontenjan": None,
                "depremzede_kontenjan": None,
                "okul_birincisi_kontenjan": 2,
                "taban_puan": 450.0 + i,
                "taban_siralama": 10000 + i * 100,
            })
    return pd.DataFrame(rows)


# ── Test: _safe_int ───────────────────────────────────────────────────────────

class TestSafeInt:
    def test_integer_input(self):
        assert _safe_int(42) == 42

    def test_string_integer(self):
        assert _safe_int("100") == 100

    def test_float_truncates(self):
        assert _safe_int(3.9) == 3

    def test_none_returns_none(self):
        assert _safe_int(None) is None

    def test_invalid_string_returns_none(self):
        assert _safe_int("abc") is None

    def test_empty_string_returns_none(self):
        assert _safe_int("") is None


# ── Test: _safe_float ─────────────────────────────────────────────────────────

class TestSafeFloat:
    def test_float_input(self):
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_string_float(self):
        assert _safe_float("547.69") == pytest.approx(547.69)

    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_invalid_returns_none(self):
        assert _safe_float("xyz") is None


# ── Test: normalize_records ───────────────────────────────────────────────────

class TestNormalizeRecords:
    def _make_api_record(self) -> dict:
        """Gerçek API yanıtını taklit eden ham kayıt."""
        return {
            "kilavuzKodu": 102210277,
            "universiteAdi": "Boğaziçi Üniversitesi",
            "birimAdi": "Bilgisayar Mühendisliği",
            "birimGrupAdi": "Bilgisayar Mühendisliği",
            "ilKodu": "34",
            "ilAdi": "İstanbul",
            "universiteTuru": "Devlet",
            "ogretimTuru": "Örgün",
            "bursOrani": None,
            "puanTuru": "SAY",
            # Mevcut yıl (2025)
            "minPuan": 547.69,
            "basariSirasi": 113,
            "gk": 18,
            "sgy": 16,
            "dprm": 89,
            # 2024
            "minPuan1": "550.88",
            "basariSirasi1": 122,
            "gk1": 18,
            "sgy1": 17,
            "dprm1": 94,
            # 2023
            "minPuan2": "552.72",
            "basariSirasi2": 98,
            "gk2": 15,
            # 2022
            "minPuan3": "550.07",
            "basariSirasi3": 156,
            "gk3": 16,
        }

    def test_produces_dataframe(self):
        df = normalize_records([self._make_api_record()])
        assert isinstance(df, pd.DataFrame)

    def test_creates_four_rows_for_full_data(self):
        """Bir program kaydı 4 yıl verisiyle 4 satır üretmeli."""
        df = normalize_records([self._make_api_record()])
        assert len(df) == 4

    def test_years_present(self):
        df = normalize_records([self._make_api_record()])
        years = set(df["yil"].tolist())
        assert 2025 in years
        assert 2024 in years
        assert 2023 in years
        assert 2022 in years

    def test_taban_siralama_numeric(self):
        df = normalize_records([self._make_api_record()])
        assert df["taban_siralama"].dtype in [int, "int64", "Int64", float]
        assert df.loc[df["yil"] == 2025, "taban_siralama"].iloc[0] == 113

    def test_taban_puan_numeric(self):
        df = normalize_records([self._make_api_record()])
        assert df.loc[df["yil"] == 2025, "taban_puan"].iloc[0] == pytest.approx(547.69, rel=1e-3)

    def test_empty_input_returns_empty_df(self):
        df = normalize_records([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_record_without_year3_skipped(self):
        """3. yıl verisi olmayan kayıtta 3 satır üretilmeli."""
        rec = self._make_api_record()
        del rec["minPuan3"]
        del rec["basariSirasi3"]
        del rec["gk3"]
        df = normalize_records([rec])
        assert len(df) == 3
        assert 2022 not in df["yil"].tolist()

    def test_kaynak_column_set(self):
        df = normalize_records([self._make_api_record()])
        assert (df["kaynak"] == "yokatlas_api").all()


# ── Test: run_quality_checks ─────────────────────────────────────────────────

class TestQualityChecks:
    def test_valid_df_passes(self):
        df = _make_valid_df(n_programs=100, years=[2022, 2023, 2024, 2025])
        report = run_quality_checks(df, expected_min=200, expected_max=1500)
        assert report.passed, f"Geçerli df başarısız: {report.errors}"

    def test_too_few_rows_fails(self):
        df = _make_valid_df(n_programs=5, years=[2025])  # 5 satır
        report = run_quality_checks(df, expected_min=200, expected_max=1500)
        assert not report.passed
        assert any("Satır sayısı" in e for e in report.errors)

    def test_too_many_rows_warns(self):
        # Çok fazla satır → uyarı (hata değil)
        df = _make_valid_df(n_programs=500, years=[2022, 2023, 2024, 2025])
        report = run_quality_checks(df, expected_min=200, expected_max=1500)
        # 500*4=2000 satır → max(1500) aşıldı → uyarı
        # Not: hata değil çünkü konteyner kontrolü uyarı veriyor
        assert report.row_count == 2000

    def test_missing_critical_column_fails(self):
        df = _make_valid_df(n_programs=100, years=[2022, 2023, 2024, 2025])
        # Kritik sütunu tamamen NaN yap (> %50 eksik)
        df["taban_siralama"] = None
        report = run_quality_checks(df, expected_min=200, expected_max=1500)
        assert not report.passed
        assert any("taban_siralama" in e for e in report.errors)

    def test_duplicate_detection(self):
        df = _make_valid_df(n_programs=5, years=[2022, 2023, 2024, 2025])
        # Aynı df'i iki kez birleştir (duplicate yarat)
        df_dup = pd.concat([df, df], ignore_index=True)
        report = run_quality_checks(df_dup, expected_min=200, expected_max=5000)
        assert report.duplicate_count > 0

    def test_schema_missing_column_fails(self):
        df = _make_valid_df(n_programs=50, years=[2022, 2023, 2024, 2025])
        df = df.drop(columns=["kilavuz_kodu"])
        report = run_quality_checks(df, expected_min=200, expected_max=1500)
        assert not report.passed
        assert "kilavuz_kodu" in report.missing_columns

    def test_year_distribution_reported(self):
        df = _make_valid_df(n_programs=50, years=[2022, 2023, 2024, 2025])
        report = run_quality_checks(df, expected_min=200, expected_max=1500)
        assert sorted(report.years_found) == [2022, 2023, 2024, 2025]
        assert report.year_row_counts[2022] == 50

    def test_report_summary_is_string(self):
        df = _make_valid_df(n_programs=100, years=[2022, 2023, 2024, 2025])
        report = run_quality_checks(df)
        summary = report.summary()
        assert isinstance(summary, str)
        assert "SONUÇ" in summary

    def test_no_rows_fails(self):
        df = pd.DataFrame()
        report = run_quality_checks(df, expected_min=200, expected_max=1500)
        assert not report.passed
        assert report.row_count == 0
