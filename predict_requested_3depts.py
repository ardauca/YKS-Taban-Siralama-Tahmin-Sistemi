"""
İstenen 3 Yeni Bölüm (ÇEEİ, Yeni Medya ve İletişim, Havacılık Yönetimi)
2026 YKS Taban Sıralama Tahmin Raporu.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
SIMULATION_CSV = ROOT / "data" / "processed" / "simulasyon_2026_tahminleri.csv"


def generate_3depts_report() -> dict[str, pd.DataFrame]:
    if not SIMULATION_CSV.exists():
        print("Simülasyon CSV bulunamadı!")
        return {}

    df = pd.read_csv(SIMULATION_CSV)

    depts = {
        "Havacılık Yönetimi": df[df["birim_adi"].str.contains("Havacılık", case=False, na=False)].sort_values("2026_tahmini_siralama"),
        "Yeni Medya ve İletişim": df[df["birim_adi"].str.contains("Yeni Medya", case=False, na=False)].sort_values("2026_tahmini_siralama"),
        "Çalışma Ekonomisi ve Endüstri İlişkileri": df[df["birim_adi"].str.contains("Çalışma Ekonomisi", case=False, na=False)].sort_values("2026_tahmini_siralama"),
    }

    return depts


if __name__ == "__main__":
    depts = generate_3depts_report()

    for name, df_dept in depts.items():
        print("\n" + "=" * 95)
        print(f" {name.upper()} 2026 YKS TAHMINI TABAN SIRALAMALARI (TOP 10)")
        print("=" * 95)
        if not df_dept.empty:
            cols = ["universite_adi", "birim_adi", "2024_gerceklesen_siralama", "2025_gerceklesen_siralama", "2026_tahmini_siralama", "2026_guven_alt_sinir", "2026_guven_ust_sinir"]
            print(df_dept[[c for c in cols if c in df_dept.columns]].head(10).to_string(index=False))
        else:
            print(f"{name} programı bulunamadı.")
