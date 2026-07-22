"""
YKS 2026 İnteraktif Tercih Danışmanı & Tahmin Konsolu.

Kullanımı:
  python tercih_danismani.py

İşlev:
  Kullanıcıdan YKS sıralamasını (örn: 180000) ve Puan Türünü (SAY, EA, SÖZ, DİL) alır.
  2026 ÖSYM Kılavuz simülasyonundan adaya özel:
  - 🟢 Garanti Tercihler
  - 🔵 Güvenli Tercihler
  - 🟡 İdeal / Hedef Tercihler
  - 🟠 Sürpriz Tercihler
  listesini üretir.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
SIMULATION_CSV = ROOT / "data" / "processed" / "simulasyon_2026_tahminleri.csv"


def classify_risk(user_rank: int, pred_med: int, pred_low: int, pred_upp: int) -> tuple[str, str]:
    """Adayın sıralamasına göre 2026 tercih risk kategorisi."""
    if user_rank <= pred_low:
        return "[GARANTI]", "Siralamaniz tahmin alt sinirindan daha iyi."
    elif user_rank <= pred_med:
        return "[GUVENLI]", "Siralamaniz tahmin edilen taban sirasindan iyi."
    elif user_rank <= pred_upp:
        return "[IDEAL/HEDEF]", "Siralamaniz tahmin edilen aralik icerisinde."
    elif user_rank <= pred_upp * 1.15:
        return "[SURPRIZ]", "Siralamaniz tahmin ust sinirina yakin, sansiniz var."
    else:
        return "[YUKSEK RISK]", "Siralamaniz tahmin araliginin gerisinde."


def recommend_preferences(user_rank: int, puan_turu: str, top_n: int = 15) -> pd.DataFrame:
    if not SIMULATION_CSV.exists():
        print("Hata: 2026 Simülasyon dosyası bulunamadı! Önce simülasyonu çalıştırın.")
        return pd.DataFrame()

    df = pd.read_csv(SIMULATION_CSV)
    
    if "puan_turu" in df.columns:
        df_filtered = df[df["puan_turu"].str.upper() == puan_turu.upper()].copy()
    else:
        df_filtered = df.copy()

    # Adayın sıralamasına yakın aralıktaki programları filtrele (0.5x - 1.5x)
    min_search = user_rank * 0.4
    max_search = user_rank * 1.6
    
    df_near = df_filtered[
        (df_filtered["2026_tahmini_siralama"] >= min_search) &
        (df_filtered["2026_tahmini_siralama"] <= max_search)
    ].copy()

    risk_labels = []
    risk_descs = []

    for _, row in df_near.iterrows():
        label, desc = classify_risk(
            user_rank,
            int(row["2026_tahmini_siralama"]),
            int(row["2026_guven_alt_sinir"]),
            int(row["2026_guven_ust_sinir"])
        )
        risk_labels.append(label)
        risk_descs.append(desc)

    df_near["risk_kategorisi"] = risk_labels
    df_near["risk_aciklamasi"] = risk_descs

    # Sıralama mesafesine göre en uygun tercihleri sırala
    df_near["mesafe"] = (df_near["2026_tahmini_siralama"] - user_rank).abs()
    df_sorted = df_near.sort_values("2026_tahmini_siralama", ascending=True)

    return df_sorted.head(top_n)


def main():
    print("=" * 75)
    print(" YKS 2026 INTERAKTIF TERCIH DANISMANI & TAHMIN KONSOLU")
    print("=" * 75)

    if len(sys.argv) >= 3:
        user_rank = int(sys.argv[1])
        puan_turu = sys.argv[2].upper()
    else:
        try:
            user_rank_str = input("\nYKS 2026 Tahmini / Gerceklesen Siralamaniz (Orn: 180000): ")
            user_rank = int(user_rank_str.strip())
            puan_turu = input("Puan Turunuz (EA, SAY, SOZ, DIL): ").strip().upper()
        except Exception:
            print("Gecersiz giris! Varsayilan olarak 180.000 EA siralamasi simule ediliyor...")
            user_rank = 180000
            puan_turu = "EA"

    print(f"\n[+] {user_rank:,} Siralama ({puan_turu}) icin 2026 Kilavuz Tercih Onerileri Hesaplaniyor...\n")

    recs = recommend_preferences(user_rank, puan_turu, top_n=20)

    if recs.empty:
        print("Uygun tercih bulunamadı.")
        return

    cols = ["universite_adi", "birim_adi", "2025_gerceklesen_siralama", "2026_tahmini_siralama", "2026_guven_alt_sinir", "2026_guven_ust_sinir", "risk_kategorisi"]
    
    print("-" * 110)
    for idx, (_, row) in enumerate(recs.iterrows(), 1):
        print(f"{idx:2d}. [{row['risk_kategorisi']:14s}] {row['universite_adi']} - {row['birim_adi']}")
        print(f"    2025 Sırası: {int(row['2025_gerceklesen_siralama']):,d}  |  2026 TAHMİNİ: {int(row['2026_tahmini_siralama']):,d}  (%80 Güven: [{int(row['2026_guven_alt_sinir']):,d} - {int(row['2026_guven_ust_sinir']):,d}])")
        print("-" * 110)


if __name__ == "__main__":
    main()
