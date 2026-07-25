"""
E-Adımı Pilot Denemesi: 3 Farklı Karakterde Bölüm Ailesinde Hibrit Router Modeli

Test edilen bölüm aileleri:
1. Yüksek Rekabetli / Sıralaması Yüksek: Tıp
2. Orta Rekabetli / Sosyal-EA: Hukuk
3. Düşük/Değişken Rekabetli: Gastronomi ve Mutfak Sanatları

Görev: Mimarinin (Model S < 100K/150K HeavyReg-GBDT + Global Model >= 100K/150K)
      Bilgisayar Mühendisliği dışındaki bölüm ailelerine aşırı uyum (overfit) sağlamadığını doğrulamak.
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
from src.features.build_features import load_and_build, get_feature_columns

print("Pilot değerlendirme için veriler yükleniyor...")
X_raw, y_raw, meta_raw = load_and_build()
all_features = [c for c in get_feature_columns() if c in X_raw.columns]

# birim_grup_adi'yı ham veriden al
df_all_depts = pd.read_csv(ROOT / "data" / "raw" / "yokatlas" / "yokatlas_all_departments_raw.csv")
dept_map = df_all_depts.set_index(["kilavuz_kodu", "yil"])["birim_grup_adi"].to_dict()
meta_raw["birim_grup_adi"] = meta_raw.set_index(["kilavuz_kodu", "yil"]).index.map(dept_map)

# Pilot bölüm aileleri
PILOT_DEPTS = ["Tıp", "Hukuk", "Gastronomi ve Mutfak Sanatları"]

# 2025 Test Yılı (Kontenjan Şoku Yılı)
mask_tr = meta_raw["yil"].isin([2023, 2024]) & X_raw["lag1_taban_siralama"].notna() & y_raw.notna()
mask_te = (meta_raw["yil"] == 2025) & X_raw["lag1_taban_siralama"].notna() & y_raw.notna()

X_tr, y_tr, meta_tr = X_raw[mask_tr], y_raw[mask_tr], meta_raw[mask_tr]
X_te, y_te, meta_te = X_raw[mask_te], y_raw[mask_te], meta_raw[mask_te]

# Model S ve Global Model Eğitimi (Sadece Train seti üzerinde)
SEG_THRESH = 100_000

LGB_S_PARAMS = {
    "n_estimators": 50, "learning_rate": 0.05, "num_leaves": 15,
    "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 5.0, "reg_lambda": 10.0, "random_state": 42, "verbosity": -1,
}
CB_S_PARAMS = dict(loss_function="MAE", iterations=100, learning_rate=0.05,
                   depth=4, l2_leaf_reg=20, random_seed=42, verbose=0)

LGB_L_PARAMS = {
    "n_estimators": 179, "learning_rate": 0.030065, "num_leaves": 57,
    "min_child_samples": 23, "subsample": 0.7727, "colsample_bytree": 0.9634,
    "reg_alpha": 0.13255, "reg_lambda": 3.8551, "random_state": 42, "verbosity": -1,
}
CB_L_PARAMS = dict(loss_function="MAE", iterations=300, learning_rate=0.04,
                   depth=6, random_seed=42, verbose=0)

# Global Model
lgb_g = lgb.LGBMRegressor(objective="regression_l1", **LGB_L_PARAMS).fit(X_tr[all_features].fillna(0), y_tr)
cb_g = CatBoostRegressor(**CB_L_PARAMS).fit(X_tr[all_features].fillna(0), y_tr)
pred_global_te = 0.5 * lgb_g.predict(X_te[all_features].fillna(0)) + 0.5 * cb_g.predict(X_te[all_features].fillna(0))

# Model S
seg_tr_s = X_tr["lag1_taban_siralama"] < SEG_THRESH
X_tr_S, y_tr_S = X_tr[seg_tr_s][all_features].fillna(0), y_tr[seg_tr_s]
lgb_s = lgb.LGBMRegressor(objective="regression_l1", **LGB_S_PARAMS).fit(X_tr_S, y_tr_S)
cb_s = CatBoostRegressor(**CB_S_PARAMS).fit(X_tr_S, y_tr_S)
pred_s_te = 0.5 * lgb_s.predict(X_te[all_features].fillna(0)) + 0.5 * cb_s.predict(X_te[all_features].fillna(0))

# Hibrit Router (Strategy C)
seg_te_s = X_te["lag1_taban_siralama"] < SEG_THRESH
pred_router_te = np.zeros(len(X_te))
pred_router_te[seg_te_s] = pred_s_te[seg_te_s]
pred_router_te[~seg_te_s] = pred_global_te[~seg_te_s]

# Metrikleri Bölüm Bazında İncele
print("\n" + "="*85)
print(" PILOT BÖLÜM AİLELERİ PERFORMANS DOĞRULAMASI (2025 Test Yılı)")
print("="*85)
print(f"{'Bölüm Ailesi':<32} {'n':>5} {'Eski Global MAE':>16} {'Hibrit Router MAE':>18} {'İyileşme %':>12}")
print("-" * 88)

for dept in PILOT_DEPTS:
    mask_dept = meta_te["birim_grup_adi"] == dept
    n_dept = mask_dept.sum()
    if n_dept == 0:
        print(f"  {dept:<30} {0:>5} {'—':>16} {'—':>18} {'—':>12}")
        continue
    
    y_true_d = y_te[mask_dept].values
    p_glob_d = pred_global_te[mask_dept]
    p_rout_d = pred_router_te[mask_dept]
    
    mae_glob = np.mean(np.abs(p_glob_d - y_true_d))
    mae_rout = np.mean(np.abs(p_rout_d - y_true_d))
    imp_pct = ((mae_glob - mae_rout) / mae_glob) * 100
    
    print(f"  {dept:<30} {n_dept:>5} {mae_glob:>16,.0f} {mae_rout:>18,.0f} {imp_pct:>+11.1f}%")

# Genel Tüm Veri (Referans)
mae_glob_all = np.mean(np.abs(pred_global_te - y_te.values))
mae_rout_all = np.mean(np.abs(pred_router_te - y_te.values))
imp_all = ((mae_glob_all - mae_rout_all) / mae_glob_all) * 100
print("-" * 88)
print(f"  {'TÜM 625 BÖLÜM AİLELERİ (GENEL)':<30} {len(y_te):>5} {mae_glob_all:>16,.0f} {mae_rout_all:>18,.0f} {imp_all:>+11.1f}%")
