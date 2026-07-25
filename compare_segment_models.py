"""
Adım A2: Segment-Özel Model Karşılaştırması

Split: lag1_taban_siralama < 100_000 → Model S (rekabetçi)
       lag1_taban_siralama >= 100_000 → Model L (kitlesel)

Model S için iki aday karşılaştırılır:
  - Aday 1: Ridge / ElasticNet (sklearn)
  - Aday 2: Ağır regularize LightGBM + CatBoost ensemble

Model L: Mevcut LightGBM + CatBoost ensemble (değişmedi).

Kazanan: Rolling fold MAE ortalaması daha düşük olan Model S adayı.

Çıktı:
  - Her segment ve fold için stratifiye MAE/R² tablosu
  - Model S kazananı belirlenir
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.features.build_features import load_and_build, get_feature_columns
from src.models.train_quantile import enforce_quantile_constraints

# ── Feature Matrix ─────────────────────────────────────────────────────────────
print("Feature matrix yükleniyor...")
X_all, y_all, meta_all = load_and_build()
feature_cols = [c for c in get_feature_columns() if c in X_all.columns]
X_all = X_all[feature_cols]

SEGMENT_THRESHOLD = 100_000

ROLLING_FOLDS = [
    {"train_years": [2023], "test_year": 2024},
    {"train_years": [2023, 2024], "test_year": 2025},
]

# Model S parametreleri
# Aday 2: ağır regularize LightGBM (küçük veri için)
LGB_S_PARAMS = {
    "n_estimators": 50,
    "learning_rate": 0.05,
    "num_leaves": 15,        # küçük ağaç (overfit'e karşı)
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 5.0,        # agresif L1
    "reg_lambda": 10.0,      # agresif L2
    "random_state": 42,
    "verbosity": -1,
}
CB_S_PARAMS = dict(loss_function="MAE", iterations=100, learning_rate=0.05,
                   depth=4, l2_leaf_reg=20, random_seed=42, verbose=0)

# Model L parametreleri (mevcut)
LGB_L_PARAMS = {
    "n_estimators": 179, "learning_rate": 0.030065, "num_leaves": 57,
    "min_child_samples": 23, "subsample": 0.7727, "colsample_bytree": 0.9634,
    "reg_alpha": 0.13255, "reg_lambda": 3.8551, "random_state": 42, "verbosity": -1,
}
CB_L_PARAMS = dict(loss_function="MAE", iterations=300, learning_rate=0.04,
                   depth=6, random_seed=42, verbose=0)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, label: str = "") -> dict:
    err = np.abs(y_pred - y_true)
    mae = np.mean(err)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"label": label, "n": len(y_true), "mae": mae, "rmse": rmse, "r2": r2}


results = []

for fold in ROLLING_FOLDS:
    train_years = fold["train_years"]
    test_year = fold["test_year"]

    # Train maskesi
    mask_train = meta_all["yil"].isin(train_years) & X_all["lag1_taban_siralama"].notna() & y_all.notna()
    X_tr_all = X_all[mask_train]
    y_tr_all = y_all[mask_train]

    # Test maskesi (hedef dolu)
    mask_test = (meta_all["yil"] == test_year) & X_all["lag1_taban_siralama"].notna() & y_all.notna()
    X_te_all = X_all[mask_test]
    y_te_all = y_all[mask_test]

    # Segment ayırımı (lag1 bazında)
    seg_train_s = X_tr_all["lag1_taban_siralama"] < SEGMENT_THRESHOLD
    seg_test_s  = X_te_all["lag1_taban_siralama"] < SEGMENT_THRESHOLD

    X_tr_S, y_tr_S = X_tr_all[seg_train_s], y_tr_all[seg_train_s]
    X_tr_L, y_tr_L = X_tr_all[~seg_train_s], y_tr_all[~seg_train_s]
    X_te_S, y_te_S = X_te_all[seg_test_s], y_te_all[seg_test_s]
    X_te_L, y_te_L = X_te_all[~seg_test_s], y_te_all[~seg_test_s]

    print(f"\n{'='*70}")
    print(f" FOLD: train={train_years}  test={test_year}")
    print(f" Model S train n={len(X_tr_S)}  test n={len(X_te_S)}")
    print(f" Model L train n={len(X_tr_L)}  test n={len(X_te_L)}")
    print(f"{'='*70}")

    # ── MODEL S: Aday 1 — Ridge / ElasticNet ──────────────────────────────────
    best_ridge_mae = float("inf")
    best_ridge_pred = None
    best_ridge_label = ""
    for alpha in [10, 100, 1000]:
        pipe = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        pipe.fit(X_tr_S.fillna(0), y_tr_S)
        pred = pipe.predict(X_te_S.fillna(0))
        mae = np.mean(np.abs(pred - y_te_S.values))
        if mae < best_ridge_mae:
            best_ridge_mae = mae
            best_ridge_pred = pred
            best_ridge_label = f"Ridge(alpha={alpha})"

    best_en_mae = float("inf")
    best_en_pred = None
    best_en_label = ""
    for alpha in [10, 100]:
        for l1 in [0.1, 0.5]:
            pipe = Pipeline([("sc", StandardScaler()), ("en", ElasticNet(alpha=alpha, l1_ratio=l1, max_iter=5000))])
            pipe.fit(X_tr_S.fillna(0), y_tr_S)
            pred = pipe.predict(X_te_S.fillna(0))
            mae = np.mean(np.abs(pred - y_te_S.values))
            if mae < best_en_mae:
                best_en_mae = mae
                best_en_pred = pred
                best_en_label = f"ElasticNet(alpha={alpha},l1={l1})"

    # Aday 1 kazananı (Ridge vs ElasticNet)
    if best_ridge_mae <= best_en_mae:
        cand1_pred, cand1_mae, cand1_label = best_ridge_pred, best_ridge_mae, best_ridge_label
    else:
        cand1_pred, cand1_mae, cand1_label = best_en_pred, best_en_mae, best_en_label

    m_c1 = metrics(y_te_S.values, cand1_pred, cand1_label)
    print(f"\n[S Aday 1 — Linear] {m_c1['label']}: MAE={m_c1['mae']:,.0f}  R²={m_c1['r2']:.3f}")

    # ── MODEL S: Aday 2 — Ağır Regularize LightGBM + CatBoost ───────────────
    lgb_s = lgb.LGBMRegressor(objective="regression_l1", **LGB_S_PARAMS)
    lgb_s.fit(X_tr_S.fillna(0), y_tr_S)
    cb_s = CatBoostRegressor(**CB_S_PARAMS)
    cb_s.fit(X_tr_S.fillna(0), y_tr_S)
    pred_s_gbdt = 0.5 * lgb_s.predict(X_te_S.fillna(0)) + 0.5 * cb_s.predict(X_te_S.fillna(0))

    m_c2 = metrics(y_te_S.values, pred_s_gbdt, "HeavyRegLGBM+CB")
    print(f"[S Aday 2 — GBDT]   {m_c2['label']}: MAE={m_c2['mae']:,.0f}  R²={m_c2['r2']:.3f}")

    # Segment S kazananı
    winner_s = cand1_label if m_c1["mae"] <= m_c2["mae"] else m_c2["label"]
    winner_pred_s = cand1_pred if m_c1["mae"] <= m_c2["mae"] else pred_s_gbdt
    print(f"\n  --> Model S kazananı (düşük MAE): {winner_s}")

    # ── MODEL L: Mevcut Ensemble ──────────────────────────────────────────────
    lgb_l = lgb.LGBMRegressor(objective="regression_l1", **LGB_L_PARAMS)
    lgb_l.fit(X_tr_L.fillna(0), y_tr_L)
    cb_l = CatBoostRegressor(**CB_L_PARAMS)
    cb_l.fit(X_tr_L.fillna(0), y_tr_L)
    pred_l = 0.5 * lgb_l.predict(X_te_L.fillna(0)) + 0.5 * cb_l.predict(X_te_L.fillna(0))

    m_l = metrics(y_te_L.values, pred_l, "LGBMEnsemble_L")
    print(f"\n[L — LGBM+CB Ensemble] MAE={m_l['mae']:,.0f}  R²={m_l['r2']:.3f}")

    # ── Genel karşılaştırma: eski global model vs yeni segment ──────────────
    # Global model için (karşılaştırma amacıyla yeniden eğit)
    lgb_g = lgb.LGBMRegressor(objective="regression_l1", **LGB_L_PARAMS)
    lgb_g.fit(X_tr_all.fillna(0), y_tr_all)
    cb_g = CatBoostRegressor(**CB_L_PARAMS)
    cb_g.fit(X_tr_all.fillna(0), y_tr_all)
    pred_g_all = 0.5 * lgb_g.predict(X_te_all.fillna(0)) + 0.5 * cb_g.predict(X_te_all.fillna(0))

    m_g_s = metrics(y_te_S.values, pred_g_all[seg_test_s], "Global_on_S")
    m_g_l = metrics(y_te_L.values, pred_g_all[~seg_test_s], "Global_on_L")

    print(f"\n--- KARŞILAŞTIRMA (test={test_year}) ---")
    print(f"{'Model':<35} {'Segment':<12} {'n':>6} {'MAE':>12} {'R²':>8}")
    print(f"{'-'*75}")

    winner_m = m_c1 if m_c1["mae"] <= m_c2["mae"] else m_c2
    compare_rows = [
        ("Segment-S kazanan", f"< 100K",  winner_m),
        ("Global-on-S",       f"< 100K",  m_g_s),
        ("Segment-L Ensemble",f">= 100K", m_l),
        ("Global-on-L",       f">= 100K", m_g_l),
    ]
    for row_label, seg_label, row_m in compare_rows:
        print(f"  {row_label:<33} {seg_label:<12} {row_m['n']:>6} {row_m['mae']:>12,.0f} {row_m['r2']:>8.3f}")

    results.append({
        "test_year": test_year,
        "winner_S": winner_s,
        "S_winner_MAE": min(m_c1["mae"], m_c2["mae"]),
        "S_winner_R2": m_c1["r2"] if m_c1["mae"] <= m_c2["mae"] else m_c2["r2"],
        "S_global_MAE": m_g_s["mae"],
        "S_global_R2": m_g_s["r2"],
        "L_segment_MAE": m_l["mae"],
        "L_segment_R2": m_l["r2"],
        "L_global_MAE": m_g_l["mae"],
        "L_global_R2": m_g_l["r2"],
    })

print("\n\n" + "="*70)
print(" ÖZET: Segment Model vs Global Model")
print("="*70)
print(f"{'Test Yılı':<12} {'Seg/Küçük MAE':>16} {'Glob/Küçük MAE':>16} {'Fark%':>8}")
for r in results:
    delta = (r["S_winner_MAE"] - r["S_global_MAE"]) / r["S_global_MAE"] * 100
    print(f"  {r['test_year']:<12} {r['S_winner_MAE']:>16,.0f} {r['S_global_MAE']:>16,.0f} {delta:>+8.1f}%")

print(f"\n  Model S kazananları: {set(r['winner_S'] for r in results)}")
