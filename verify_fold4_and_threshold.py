"""
İki Son Doğrulama Noktası:
1. Fold 4 Neden Belirgin Şekilde Kötü? (Kök Neden Analizi)
   - Fold 4 kütlesindeki programların özellik dağılımlarını (üniversite türü, burs oranı, il, doluluk, baraj mesafesi) diğer fold'larla karşılaştır.
2. Router Eşiğinin (50K / 100K / 150K / 200K) SADECE Train Seti Üzerinde Doğrulanması (Data Snooping Önleme)
"""
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold

from src.features.build_features import load_and_build, get_feature_columns

X_raw, y_raw, meta_raw = load_and_build()
all_features = [c for c in get_feature_columns() if c in X_raw.columns]

# Train Seti: 2023-2024
mask_tr = meta_raw["yil"].isin([2023, 2024]) & X_raw["lag1_taban_siralama"].notna() & y_raw.notna()
X_tr, y_tr, meta_tr = X_raw[mask_tr], y_raw[mask_tr], meta_raw[mask_tr]

# Model S subset (< 100K)
seg_s = X_tr["lag1_taban_siralama"] < 100_000
X_tr_S, y_tr_S, meta_tr_S = X_tr[seg_s], y_tr[seg_s], meta_tr[seg_s]

# ── 1. FOLD 4 KÖK NEDEN ANALİZİ ─────────────────────────────────────────────
print("="*85)
print(" 1. FOLD 4 KÖK NEDEN VE SİSTEMATİK HATA ANALİZİ")
print("="*85)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

LGB_S_PARAMS = {
    "n_estimators": 50, "learning_rate": 0.05, "num_leaves": 15,
    "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 5.0, "reg_lambda": 10.0, "random_state": 42, "verbosity": -1,
}
CB_S_PARAMS = dict(loss_function="MAE", iterations=100, learning_rate=0.05,
                   depth=4, l2_leaf_reg=20, random_seed=42, verbose=0)

fold_analysis = []

for i_fold, (tr_idx, val_idx) in enumerate(kf.split(X_tr_S)):
    X_k_tr, y_k_tr = X_tr_S.iloc[tr_idx][all_features].fillna(0), y_tr_S.iloc[tr_idx]
    X_k_val, y_k_val = X_tr_S.iloc[val_idx][all_features].fillna(0), y_tr_S.iloc[val_idx]
    meta_k_val = meta_tr_S.iloc[val_idx].copy()
    
    lgb_s = lgb.LGBMRegressor(objective="regression_l1", **LGB_S_PARAMS).fit(X_k_tr, y_k_tr)
    cb_s = CatBoostRegressor(**CB_S_PARAMS).fit(X_k_tr, y_k_tr)
    pred_val = 0.5 * lgb_s.predict(X_k_val) + 0.5 * cb_s.predict(X_k_val)
    
    err = np.abs(pred_val - y_k_val.values)
    mae = np.mean(err)
    
    meta_k_val["y_true"] = y_k_val.values
    meta_k_val["y_pred"] = pred_val
    meta_k_val["err"] = err
    meta_k_val["univ_turu"] = X_tr_S.iloc[val_idx]["universite_turu_enc"].values
    meta_k_val["lag1"] = X_tr_S.iloc[val_idx]["lag1_taban_siralama"].values
    meta_k_val["burs"] = X_tr_S.iloc[val_idx]["burs_enc"].values
    
    vakif_pct = (meta_k_val["univ_turu"] == 1).mean() * 100
    med_lag1 = meta_k_val["lag1"].median()
    top_err_progs = meta_k_val.sort_values("err", ascending=False).head(3)
    
    fold_analysis.append({
        "fold": i_fold + 1,
        "n": len(val_idx),
        "mae": mae,
        "vakif_pct": vakif_pct,
        "med_lag1": med_lag1,
        "top_err_progs": top_err_progs[["universite_adi", "birim_adi", "lag1", "y_true", "y_pred", "err"]].to_dict("records")
    })

for fa in fold_analysis:
    print(f"\n--- FOLD {fa['fold']} --- (MAE: {fa['mae']:,.0f} | Vakıf Oranı: %{fa['vakif_pct']:.1f} | Medyan Lag1: {fa['med_lag1']:,.0f})")
    print("  En Çok Hata Yapılan Top 3 Program:")
    for p in fa["top_err_progs"]:
        print(f"    - {p['universite_adi'][:25]} | {p['birim_adi'][:30]} | Lag1: {p['lag1']:,.0f} | Gerçek: {p['y_true']:,.0f} | Tahmin: {p['y_pred']:,.0f} | Hata: {p['err']:,.0f}")

# ── 2. ROUTER EŞİĞİ SIKI TRAIN-CV DOĞRULAMASI (Data Snooping Önleme) ──────────
print("\n\n" + "="*85)
print(" 2. SADECE TRAIN SETİ ÜZERİNDE ROUTER EŞİĞİ (THRESH) SEÇİMİ (5-Fold CV)")
print("="*85)

threshold_candidates = [30_000, 50_000, 75_000, 100_000, 125_000, 150_000, 200_000]
thresh_cv_results = []

kf_all = KFold(n_splits=5, shuffle=True, random_state=42)

for thresh in threshold_candidates:
    cv_maes = []
    for tr_idx, val_idx in kf_all.split(X_tr):
        X_k_tr_raw, y_k_tr = X_tr.iloc[tr_idx], y_tr.iloc[tr_idx]
        X_k_val_raw, y_k_val = X_tr.iloc[val_idx], y_tr.iloc[val_idx]
        
        # Train split on thresh
        seg_k_tr_s = X_k_tr_raw["lag1_taban_siralama"] < thresh
        X_k_tr_S, y_k_tr_S = X_k_tr_raw[seg_k_tr_s][all_features].fillna(0), y_k_tr[seg_k_tr_s]
        
        # Models
        # Model S
        lgb_s = lgb.LGBMRegressor(objective="regression_l1", **LGB_S_PARAMS).fit(X_k_tr_S, y_k_tr_S)
        cb_s = CatBoostRegressor(**CB_S_PARAMS).fit(X_k_tr_S, y_k_tr_S)
        
        # Global Model (for >= thresh router)
        lgb_g = lgb.LGBMRegressor(objective="regression_l1", **LGB_S_PARAMS).fit(X_k_tr_raw[all_features].fillna(0), y_k_tr)
        cb_g = CatBoostRegressor(**CB_S_PARAMS).fit(X_k_tr_raw[all_features].fillna(0), y_k_tr)
        
        # Val prediction
        seg_k_val_s = X_k_val_raw["lag1_taban_siralama"] < thresh
        X_k_val_f = X_k_val_raw[all_features].fillna(0)
        
        pred_val = np.zeros(len(val_idx))
        if seg_k_val_s.sum() > 0:
            pred_val[seg_k_val_s] = 0.5 * lgb_s.predict(X_k_val_f[seg_k_val_s]) + 0.5 * cb_s.predict(X_k_val_f[seg_k_val_s])
        if (~seg_k_val_s).sum() > 0:
            pred_val[~seg_k_val_s] = 0.5 * lgb_g.predict(X_k_val_f[~seg_k_val_s]) + 0.5 * cb_g.predict(X_k_val_f[~seg_k_val_s])
            
        cv_maes.append(np.mean(np.abs(pred_val - y_k_val.values)))
        
    avg_mae = np.mean(cv_maes)
    thresh_cv_results.append({"threshold": thresh, "train_cv_mae": avg_mae})
    print(f"  Eşik: {thresh:>7,d} sıra  --> Train 5-Fold CV MAE: {avg_mae:,.0f}")

best_thresh = min(thresh_cv_results, key=lambda x: x["train_cv_mae"])
print(f"\n  --> SIZINTISIZ TRAIN CV EN OPTİMAL ROUTER EŞİĞİ: {best_thresh['threshold']:,d} sıra (MAE: {best_thresh['train_cv_mae']:,.0f})")
