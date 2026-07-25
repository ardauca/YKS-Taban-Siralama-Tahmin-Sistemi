"""
İki doğrulama analizi:
1) MAE'yi sıralama dilimine göre ayrı raporla (0-10K, 10K-100K, 100K-500K, 500K+)
2) 2025 test setinde null olan vs olmayan satırların özellik dağılımını karşılaştır
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
from catboost import CatBoostRegressor
from src.features.build_features import load_and_build, get_feature_columns
from src.models.train_quantile import enforce_quantile_constraints

# ── Feature matrix yükle ───────────────────────────────────────────────────────
X_all, y_all, meta_all = load_and_build()
feature_cols = [c for c in get_feature_columns() if c in X_all.columns]
X_all = X_all[feature_cols]

ALPHA_LOWER = 0.030
ALPHA_UPPER = 0.970

lgb_params = {
    "n_estimators": 179, "learning_rate": 0.030065, "num_leaves": 57,
    "min_child_samples": 23, "subsample": 0.7727, "colsample_bytree": 0.9634,
    "reg_alpha": 0.13255, "reg_lambda": 3.8551, "random_state": 42, "verbosity": -1,
}

# ── Fold 2: train=2023+2024, test=2025 (2024 yerine 2025'i analiz ediyoruz) ──
# Aynı zamanda Fold 1 (train=2023, test=2024) de çalıştır
FOLDS = [
    {"train_years": [2023], "test_year": 2024},
    {"train_years": [2023, 2024], "test_year": 2025},
]

for fold in FOLDS:
    train_years = fold["train_years"]
    test_year = fold["test_year"]

    # Train
    mask_train = meta_all["yil"].isin(train_years) & X_all["lag1_taban_siralama"].notna() & y_all.notna()
    X_train, y_train = X_all[mask_train], y_all[mask_train]

    # Test — TÜM 2025 satırları (null hedef dahil)
    mask_test_all = (meta_all["yil"] == test_year) & X_all["lag1_taban_siralama"].notna()
    X_test_all = X_all[mask_test_all]
    y_test_all = y_all[mask_test_all]
    meta_test_all = meta_all[mask_test_all]

    # Sadece null olmayan test satırları (değerlendirme için)
    eval_mask = y_test_all.notna()
    X_test_eval = X_test_all[eval_mask]
    y_test_eval = y_test_all[eval_mask]

    print(f"\n{'='*70}")
    print(f" TEST YILI: {test_year}  |  Train: {train_years}")
    print(f"{'='*70}")
    print(f"\nTest seti toplam (lag1 notna): {len(X_test_all)}")
    print(f"  - taban_siralama dolu (değerlendirmeye giren): {eval_mask.sum()}")
    print(f"  - taban_siralama null (değerlendirme dışı bırakılan): {(~eval_mask).sum()}")

    # ── NULL vs NON-NULL karşılaştırması ──────────────────────────────────────
    X_null = X_test_all[~eval_mask]
    X_nonnull = X_test_all[eval_mask]

    print(f"\n--- SORU 2: NULL vs NON-NULL {test_year} Satırlarının Özellik Dağılımı ---")
    check_cols = ["lag1_taban_siralama", "program_hist_medyan_siralama",
                  "univ_hist_medyan_siralama", "lag1_genel_kontenjan",
                  "universite_turu_enc", "puan_turu_enc"]

    rows_cmp = []
    for col in check_cols:
        null_vals = X_null[col].dropna()
        nonnull_vals = X_nonnull[col].dropna()
        rows_cmp.append({
            "feature": col,
            "null_target_n": len(null_vals),
            "null_target_mean": round(null_vals.mean(), 1) if len(null_vals) else float("nan"),
            "null_target_median": round(null_vals.median(), 1) if len(null_vals) else float("nan"),
            "nonnull_target_n": len(nonnull_vals),
            "nonnull_target_mean": round(nonnull_vals.mean(), 1) if len(nonnull_vals) else float("nan"),
            "nonnull_target_median": round(nonnull_vals.median(), 1) if len(nonnull_vals) else float("nan"),
        })

    cmp_df = pd.DataFrame(rows_cmp)
    print(cmp_df.to_string(index=False))

    # Özellikle: program_hist_medyan_siralama doluluk oranı
    phm_null_pct = X_null["program_hist_medyan_siralama"].notna().mean() * 100
    phm_nonnull_pct = X_nonnull["program_hist_medyan_siralama"].notna().mean() * 100
    print(f"\n  program_hist_medyan_siralama doluluk: null_target=%{phm_null_pct:.1f} | nonnull_target=%{phm_nonnull_pct:.1f}")

    lag1_null_median = X_null["lag1_taban_siralama"].median()
    lag1_nonnull_median = X_nonnull["lag1_taban_siralama"].median()
    print(f"  lag1_taban_siralama medyan: null_target={lag1_null_median:,.0f} | nonnull_target={lag1_nonnull_median:,.0f}")

    # ── Model eğit ────────────────────────────────────────────────────────────
    lgb_med = lgb.LGBMRegressor(objective="regression_l1", **lgb_params)
    lgb_med.fit(X_train, y_train)
    cb_med = CatBoostRegressor(loss_function="MAE", iterations=300, learning_rate=0.04,
                               depth=6, verbose=0, random_seed=42)
    cb_med.fit(X_train, y_train)

    pred_eval = 0.5 * lgb_med.predict(X_test_eval) + 0.5 * cb_med.predict(X_test_eval)

    y_true = y_test_eval.values
    err = np.abs(pred_eval - y_true)

    # ── SORU 1: Stratifiye MAE ─────────────────────────────────────────────────
    print(f"\n--- SORU 1: Sıralama Dilimine Göre Stratifiye MAE ({test_year} Test) ---")
    bins = [0, 10_000, 100_000, 500_000, float("inf")]
    labels = ["0–10K", "10K–100K", "100K–500K", "500K+"]

    rows_strat = []
    for i, label in enumerate(labels):
        lo, hi = bins[i], bins[i+1]
        mask_bin = (y_true >= lo) & (y_true < hi)
        n = mask_bin.sum()
        if n == 0:
            rows_strat.append({"dilim": label, "n_program": 0, "MAE": "—",
                                "RMSE": "—", "R2": "—", "MAE/medyan_pct": "—"})
            continue
        mae_bin = np.mean(err[mask_bin])
        rmse_bin = np.sqrt(np.mean((pred_eval[mask_bin] - y_true[mask_bin])**2))
        # R² hesabı
        ss_res = np.sum((y_true[mask_bin] - pred_eval[mask_bin])**2)
        ss_tot = np.sum((y_true[mask_bin] - y_true[mask_bin].mean())**2)
        r2_bin = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        medyan_bin = np.median(y_true[mask_bin])
        mae_pct = mae_bin / medyan_bin * 100
        rows_strat.append({
            "dilim": label,
            "n_program": n,
            "MAE": f"{mae_bin:,.0f}",
            "RMSE": f"{rmse_bin:,.0f}",
            "R2": f"{r2_bin:.3f}",
            "MAE/medyan_%": f"{mae_pct:.1f}%",
        })

    strat_df = pd.DataFrame(rows_strat)
    print(strat_df.to_string(index=False))

    overall_mae = np.mean(err)
    overall_r2 = 1 - np.sum((y_true - pred_eval)**2) / np.sum((y_true - y_true.mean())**2)
    print(f"\n  Genel MAE (tüm dilimler): {overall_mae:,.0f}")
    print(f"  Genel R²  (tüm dilimler): {overall_r2:.3f}")
    print(f"  Değerlendirilen program sayısı: {len(y_true)}")
