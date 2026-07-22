"""
Veri kalite kontrol modülü (Bölüm 7 gereksinimleri).

Her scraping/ETL görevinde zorunlu kontroller:
- Satır sayısı beklenen aralıkta mı?
- Eksik değer oranı sütun bazlı raporlanıyor mu?
- Duplicate kayıt var mı?
- Yeni yılın verisi önceki yıllarla şema uyumlu mu?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Türkiye'de Bilgisayar Mühendisliği program sayısı için beklenen aralık (program × yıl)
# ~200 program × 4 yıl ≈ 800 satır. Düşük sınır: 200, yüksek: 1500.
EXPECTED_ROW_MIN = 200
EXPECTED_ROW_MAX = 1500

# Kritik sütunlar — bunlarda eksik veri kabul edilemez (oran > 0.5 → uyarı)
CRITICAL_COLUMNS = ["kilavuz_kodu", "universite_adi", "birim_adi", "yil", "taban_siralama"]

# Duplicate için kullanılacak bileşik anahtar
DUPLICATE_KEY = ["kilavuz_kodu", "yil"]

# Beklenen sütunlar (şema uyumu kontrolü)
EXPECTED_COLUMNS = {
    "kilavuz_kodu",
    "universite_adi",
    "birim_adi",
    "birim_grup_adi",
    "il_kodu",
    "il_adi",
    "universite_turu",
    "ogretim_turu",
    "burs_orani",
    "puan_turu",
    "kaynak",
    "cekme_tarihi",
    "yil",
    "genel_kontenjan",
    "sehit_gazi_kontenjan",
    "depremzede_kontenjan",
    "okul_birincisi_kontenjan",
    "taban_puan",
    "taban_siralama",
}


@dataclass
class QualityReport:
    """Veri kalite raporu."""

    row_count: int = 0
    expected_row_min: int = EXPECTED_ROW_MIN
    expected_row_max: int = EXPECTED_ROW_MAX
    row_count_ok: bool = False

    missing_value_rates: dict[str, float] = field(default_factory=dict)
    critical_missing_warnings: list[str] = field(default_factory=list)

    duplicate_count: int = 0
    duplicate_rows: list[dict] | None = None  # Örnek duplikat (max 5)

    missing_columns: list[str] = field(default_factory=list)
    extra_columns: list[str] = field(default_factory=list)
    schema_ok: bool = False

    years_found: list[int] = field(default_factory=list)
    year_row_counts: dict[int, int] = field(default_factory=dict)

    passed: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """İnsan okunabilir özet."""
        lines = [
            "═══════════════════════════════════════",
            " VERİ KALİTE RAPORU",
            "═══════════════════════════════════════",
            f"  Satır sayısı   : {self.row_count} "
            f"({'✓' if self.row_count_ok else '✗'} beklenen: {self.expected_row_min}–{self.expected_row_max})",
            f"  Yıllar         : {sorted(self.years_found)}",
        ]
        for year, cnt in sorted(self.year_row_counts.items()):
            lines.append(f"    {year}: {cnt} satır")

        lines.append("  Eksik değer oranları (kritik sütunlar):")
        for col in CRITICAL_COLUMNS:
            rate = self.missing_value_rates.get(col, 0.0)
            flag = " ⚠" if rate > 0.05 else ""
            lines.append(f"    {col}: %{rate * 100:.1f}{flag}")

        lines.append(f"  Duplicate kayıt: {self.duplicate_count}")
        lines.append(f"  Şema uyumu     : {'✓' if self.schema_ok else '✗'}")

        if self.warnings:
            lines.append("  UYARILAR:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        if self.errors:
            lines.append("  HATALAR:")
            for e in self.errors:
                lines.append(f"    ✗ {e}")

        lines.append(f"  SONUÇ: {'GEÇTI ✓' if self.passed else 'BAŞARISIZ ✗'}")
        lines.append("═══════════════════════════════════════")
        return "\n".join(lines)


def run_quality_checks(
    df: pd.DataFrame,
    expected_min: int = EXPECTED_ROW_MIN,
    expected_max: int = EXPECTED_ROW_MAX,
) -> QualityReport:
    """
    DataFrame üzerinde tüm kalite kontrollerini çalıştırır.

    Args:
        df: Kontrol edilecek DataFrame.
        expected_min: Beklenen minimum satır sayısı.
        expected_max: Beklenen maksimum satır sayısı.

    Returns:
        Doldurulmuş QualityReport.
    """
    report = QualityReport(
        expected_row_min=expected_min,
        expected_row_max=expected_max,
    )

    # ── 1. Satır sayısı ──────────────────────────────────────────────────────
    report.row_count = len(df)
    report.row_count_ok = expected_min <= report.row_count <= expected_max

    if not report.row_count_ok:
        msg = (
            f"Satır sayısı ({report.row_count}) beklenen aralık dışında "
            f"({expected_min}–{expected_max})"
        )
        if report.row_count < expected_min:
            report.errors.append(msg)
        else:
            report.warnings.append(msg)
        logger.warning(msg)

    # ── 2. Eksik değer oranları ───────────────────────────────────────────────
    for col in df.columns:
        rate = df[col].isna().mean()
        report.missing_value_rates[col] = float(rate)

    for col in CRITICAL_COLUMNS:
        if col not in df.columns:
            continue
        rate = report.missing_value_rates.get(col, 0.0)
        if rate > 0.5:
            msg = f"KRİTİK sütun '{col}' eksik oranı yüksek: %{rate * 100:.1f}"
            report.critical_missing_warnings.append(msg)
            report.errors.append(msg)
            logger.error(msg)
        elif rate > 0.05:
            msg = f"'{col}' sütununda %{rate * 100:.1f} eksik değer"
            report.warnings.append(msg)
            logger.warning(msg)

    # ── 3. Duplicate kayıt ───────────────────────────────────────────────────
    dup_cols = [c for c in DUPLICATE_KEY if c in df.columns]
    if dup_cols:
        dups = df[df.duplicated(subset=dup_cols, keep=False)]
        report.duplicate_count = len(dups)
        if report.duplicate_count > 0:
            msg = f"{report.duplicate_count} duplicate kayıt bulundu (anahtar: {dup_cols})"
            report.warnings.append(msg)
            report.duplicate_rows = dups.head(5).to_dict("records")
            logger.warning(msg)

    # ── 4. Şema uyumu ────────────────────────────────────────────────────────
    actual_cols = set(df.columns)
    report.missing_columns = sorted(EXPECTED_COLUMNS - actual_cols)
    report.extra_columns = sorted(actual_cols - EXPECTED_COLUMNS)
    report.schema_ok = len(report.missing_columns) == 0

    if report.missing_columns:
        msg = f"Eksik sütunlar: {report.missing_columns}"
        report.errors.append(msg)
        logger.error(msg)
    if report.extra_columns:
        report.warnings.append(f"Beklenmedik sütunlar (önemli değil): {report.extra_columns}")

    # ── 5. Yıl dağılımı ──────────────────────────────────────────────────────
    if "yil" in df.columns:
        year_counts = df["yil"].value_counts().sort_index()
        report.years_found = sorted(year_counts.index.tolist())
        report.year_row_counts = {int(y): int(c) for y, c in year_counts.items()}

        if len(report.years_found) < 3:
            msg = f"Beklenen 3+ yıl verisi, bulunan: {report.years_found}"
            report.warnings.append(msg)
            logger.warning(msg)

    # ── Genel sonuç ──────────────────────────────────────────────────────────
    report.passed = len(report.errors) == 0
    return report
