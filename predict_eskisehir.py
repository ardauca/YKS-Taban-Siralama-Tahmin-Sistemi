"""
Eskişehir Üniversiteleri (Anadolu Üniversitesi & Eskişehir Osmangazi Üniversitesi)
2026 YKS Taban Sıralama ve Güven Aralığı Tahmin Raporu.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
SIMULATION_CSV = ROOT / "data" / "processed" / "simulasyon_2026_tahminleri.csv"


def generate_eskisehir_report() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SIMULATION_CSV.exists():
        print("Simülasyon CSV bulunamadı!")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.read_csv(SIMULATION_CSV)

    # Anadolu Üni & Osmangazi Üni filtresi
    mask_anadolu = df["universite_adi"].str.contains("ANADOLU", case=False, na=False) & ~df["universite_adi"].str.contains("KKTC", case=False, na=False)
    mask_osmangazi = df["universite_adi"].str.contains("OSMANGAZİ", case=False, na=False)

    df_anadolu = df[mask_anadolu].sort_values("2026_tahmini_siralama").copy()
    df_osmangazi = df[mask_osmangazi].sort_values("2026_tahmini_siralama").copy()

    return df_anadolu, df_osmangazi


if __name__ == "__main__":
    df_ana, df_osm = generate_eskisehir_report()

    print("\n" + "=" * 90)
    print(" ANADOLU UNIVERSITESI 2026 YKS TAHMINI TABAN SIRALAMALARI")
    print("=" * 90)
    if not df_ana.empty:
        cols = ["birim_adi", "puan_turu", "2025_gerceklesen_siralama", "2026_tahmini_siralama", "2026_guven_alt_sinir", "2026_guven_ust_sinir"]
        print(df_ana[[c for c in cols if c in df_ana.columns]].to_string(index=False))
    else:
        print("Anadolu Üniversitesi programı bulunamadı.")

    print("\n" + "=" * 90)
    print(" ESKISEHIR OSMANGAZI UNIVERSITESI 2026 YKS TAHMINI TABAN SIRALAMALARI")
    print("=" * 90)
    if not df_osm.empty:
        cols = ["birim_adi", "puan_turu", "2025_gerceklesen_siralama", "2026_tahmini_siralama", "2026_guven_alt_sinir", "2026_guven_ust_sinir"]
        print(df_osm[[c for c in cols if c in df_osm.columns]].to_string(index=False))
    else:
        print("Osmangazi Üniversitesi programı bulunamadı.")
