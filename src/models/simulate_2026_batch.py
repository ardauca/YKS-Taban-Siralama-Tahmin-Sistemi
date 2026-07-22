"""
2026 YKS Kılavuz Tabanlı Toplu Simülasyon Motoru (Batch Prediction Engine).

Görev:
  11 Ana Bölüm Ailesindeki tüm 2,722 lisans programı için ÖSYM 2026 Kılavuz kontenjanları
  ve Hibrit Ensemble modelimiz ile 2026 YKS tahmini ve belirsizlik aralığı üretmek.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from api.main import _load_or_train_models, _lgb_med, _lgb_low, _lgb_upp, _cb_med, _cb_low, _cb_upp, BURS_ORANI_ORDERED, UNIVERSITE_TURU_MAP, OGRETIM_TURU_MAP, PUAN_TURU_MAP
from src.features.build_features import get_feature_columns, load_and_build
from src.models.train_quantile import enforce_quantile_constraints

logger = logging.getLogger(__name__)

SIMULATION_OUT_CSV = ROOT / "data" / "processed" / "simulasyon_2026_tahminleri.csv"


def run_2026_batch_simulation() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    logger.info("2026 YKS Toplu Simülasyon Motoru Başlatılıyor...")
    import api.main as api_main
    api_main._load_or_train_models()

    X, y, meta = load_and_build()
    feature_cols = [c for c in get_feature_columns() if c in X.columns]

    # Sadece 2025 verisine sahip aktif programlar (2026 tahmini için girdi olanlar)
    mask_2025 = (meta["yil"] == 2025) & X["lag1_taban_siralama"].notna()
    X_2026_input = X[mask_2025].copy()
    meta_2026 = meta[mask_2025].copy()

    # 2026 Yılı için güncelle
    X_2026_input["yil"] = 2026

    logger.info("Toplam %d lisans programı için 2026 simülasyonu çalıştırılıyor...", len(X_2026_input))

    X_eval = X_2026_input[feature_cols]

    raw_med = 0.5 * api_main._lgb_med.predict(X_eval) + 0.5 * api_main._cb_med.predict(X_eval)
    raw_low = 0.5 * api_main._lgb_low.predict(X_eval) + 0.5 * api_main._cb_low.predict(X_eval)
    raw_upp = 0.5 * api_main._lgb_upp.predict(X_eval) + 0.5 * api_main._cb_upp.predict(X_eval)

    clean_med, clean_low, clean_upp = enforce_quantile_constraints(raw_med, raw_low, raw_upp)

    sim_df = meta_2026.copy()
    sim_df["2025_gerceklesen_siralama"] = X_2026_input["lag1_taban_siralama"].values
    sim_df["2026_tahmini_siralama"] = np.round(clean_med).astype(int)
    sim_df["2026_guven_alt_sinir"] = np.round(clean_low).astype(int)
    sim_df["2026_guven_ust_sinir"] = np.round(clean_upp).astype(int)
    sim_df["tahmini_siralama_degisimi"] = sim_df["2025_gerceklesen_siralama"] - sim_df["2026_tahmini_siralama"]

    SIMULATION_OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    sim_df.to_csv(SIMULATION_OUT_CSV, index=False, encoding="utf-8-sig")
    logger.info("2026 Simülasyon Sonuçları Kaydedildi: %s (%d program)", SIMULATION_OUT_CSV, len(sim_df))

    return sim_df


if __name__ == "__main__":
    df_sim = run_2026_batch_simulation()

    print("\n" + "=" * 70)
    print(" 2026 YKS TAHMİNİ SIRALAMASI EN ÇOK YÜKSELECEK PROGRAMLAR (TOP 10)")
    print("=" * 70)
    top_up = df_sim.sort_values("tahmini_siralama_degisimi", ascending=False).head(10)
    print(top_up[["universite_adi", "birim_adi", "2025_gerceklesen_siralama", "2026_tahmini_siralama", "tahmini_siralama_degisimi"]].to_string(index=False))

    print("\n" + "=" * 70)
    print(" 2026 YKS TAHMİNİ SIRALAMASI EN ÇOK GERİLEYECEK PROGRAMLAR (TOP 10)")
    print("=" * 70)
    top_down = df_sim.sort_values("tahmini_siralama_degisimi", ascending=True).head(10)
    print(top_down[["universite_adi", "birim_adi", "2025_gerceklesen_siralama", "2026_tahmini_siralama", "tahmini_siralama_degisimi"]].to_string(index=False))
