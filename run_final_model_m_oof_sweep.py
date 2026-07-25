"""
Model M (100K-500K) İçin Sadece Train (2023-2024) 5-Fold OOF Genişletilmiş Alpha Taraması

Metodolojik Adımlar:
1. Alpha adaylarını [0.01 - 0.15] arası sadece 2023-2024 Train seti 5-Fold OOF üzerinde tara.
2. Train-OOF üzerinde %80 kapsama hedefine ulaşan bir alpha varsa kilitler, 2025 Test verisinde tek seferlik kör ölçeriz.
3. Yoksa (veya dar kalıyorsa) bunu Model M'nin yapısal bir sınırlaması olarak dokümante ederiz.
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
from sklearn.model_selection import KFold

from src.features.build_features import load_and_build
from src.models.train_quantile import enforce_quantile_constraints

print("="*85)
print(" MODEL M TRAIN-OOF GENİŞLETİLMİŞ ALPHA TARAMASI (SIZINTISIZ)")
print("="*85)

X_all, y_all, meta_all = load_and_build()

v0_features = [
    "lag1_taban_siralama", "lag1_taban_puan", "lag1_genel_kontenjan",
    "lag1_sehit_gazi_kontenjan", "lag1_depremzede_kontenjan", "lag1_okul_birincisi_kontenjan",
    "lag2_taban_siralama", "siralama_trend", "siralama_pct_change", "kontenjan_degisim_orani",
    "universite_turu_enc", "ogretim_turu_enc", "puan_turu_enc", "burs_enc", "il_kodu_num",
    "program_hist_medyan_siralama", "univ_hist_medyan_siralama", "kontenjan_kategori",
    "univ_trend_momentum", "sehir_tercih_indeksi", "kontenjan_farki_2026",
    "macro_puan_turu_degisim_orani", "macro_bolum_degisim_orani", "kontenjan_sok_faktoru",
    "baraj_mesafe_indeksi", "vakif_devlet_burs_gap", "puan_turu_rekabet_indeksi", "yil"
]
v0_features = [c for c in v0_features if c in X_all.columns]
final_features = [c for c in X_all.columns if c not in ["kilavuz_kodu", "taban_siralama", "universite_adi", "birim_adi", "birim_grup_adi"]]

mask_tr = meta_all["yil"].isin([2023, 2024]) & X_all["lag1_taban_siralama"].notna() & y_all.notna()
mask_te = (meta_all["yil"] == 2025) & X_all["lag1_taban_siralama"].notna() & y_all.notna()

X_tr, y_tr = X_all[mask_tr], y_all[mask_tr]
X_te, y_te = X_all[mask_te], y_all[mask_te]

LGB_L_PARAMS = {
    "n_estimators": 179, "learning_rate": 0.030065, "num_leaves": 57,
    "min_child_samples": 23, "subsample": 0.7727, "colsample_bytree": 0.9634,
    "reg_alpha": 0.13255, "reg_lambda": 3.8551, "random_state": 42, "verbosity": -1,
}

kf_oof = KFold(n_splits=5, shuffle=True, random_state=42)

seg_tr_M = (X_tr["lag1_taban_siralama"] >= 100_000) & (X_tr["lag1_taban_siralama"] < 500_000)
X_tr_M, y_tr_M = X_tr[seg_tr_M], y_tr[seg_tr_M]

alpha_candidates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
oof_results_m = {}

for a_low in alpha_candidates:
    a_upp = 1.0 - a_low
    oof_low = np.zeros(len(X_tr_M))
    oof_upp = np.zeros(len(X_tr_M))
    
    for tr_i, val_i in kf_oof.split(X_tr_M):
        X_k_tr, y_k_tr = X_tr_M.iloc[tr_i][final_features].fillna(0), y_tr_M.iloc[tr_i]
        X_k_val = X_tr_M.iloc[val_i][final_features].fillna(0)
        
        lgb_low_k = lgb.LGBMRegressor(objective="quantile", alpha=a_low, **LGB_L_PARAMS).fit(X_k_tr, y_k_tr)
        lgb_upp_k = lgb.LGBMRegressor(objective="quantile", alpha=a_upp, **LGB_L_PARAMS).fit(X_k_tr, y_k_tr)
        cb_low_k = CatBoostRegressor(loss_function=f"Quantile:alpha={a_low}", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_k_tr, y_k_tr)
        cb_upp_k = CatBoostRegressor(loss_function=f"Quantile:alpha={a_upp}", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_k_tr, y_k_tr)
        
        oof_low[val_i] = 0.5 * lgb_low_k.predict(X_k_val) + 0.5 * cb_low_k.predict(X_k_val)
        oof_upp[val_i] = 0.5 * lgb_upp_k.predict(X_k_val) + 0.5 * cb_upp_k.predict(X_k_val)
        
    oof_cov = float(np.mean((y_tr_M.values >= oof_low) & (y_tr_M.values <= oof_upp))) * 100
    w_oof = float(np.mean(oof_upp - oof_low))
    oof_results_m[a_low] = (oof_cov, w_oof)
    print(f"  Model M Train OOF Alpha={a_low:.2f}/{a_upp:.2f} --> OOF Kapsama Oranı: %{oof_cov:.1f} (Ort. Genişlik: {w_oof:,.0f} sıra)")

# Train OOF üzerinde %80'e en yakın alpha'yı seç
best_m_alpha_low = min(oof_results_m.keys(), key=lambda a: abs(oof_results_m[a][0] - 80.0))
best_m_alpha_upp = 1.0 - best_m_alpha_low

print(f"\n  --> TRAIN-OOF MODEL M İÇİN SEÇİLEN ALFA: Low={best_m_alpha_low:.2f} / Upp={best_m_alpha_upp:.2f} (OOF Kapsama: %{oof_results_m[best_m_alpha_low][0]:.1f})")

# ── 2. 2025 TEST VERİSİNDE TEK SEFERLİK KÖR ÖLÇÜM ────────────────────────────
print("\n--- 2025 TEST VERİSİNDE TEK SEFERLİK KÖR KONTROL ---")
seg_te_M = (X_te["lag1_taban_siralama"] >= 100_000) & (X_te["lag1_taban_siralama"] < 500_000)
y_te_M = y_te[seg_te_M].values

lgb_m_low_f = lgb.LGBMRegressor(objective="quantile", alpha=best_m_alpha_low, **LGB_L_PARAMS).fit(X_tr_M[final_features].fillna(0), y_tr_M)
lgb_m_upp_f = lgb.LGBMRegressor(objective="quantile", alpha=best_m_alpha_upp, **LGB_L_PARAMS).fit(X_tr_M[final_features].fillna(0), y_tr_M)
cb_m_low_f = CatBoostRegressor(loss_function=f"Quantile:alpha={best_m_alpha_low}", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr_M[final_features].fillna(0), y_tr_M)
cb_m_upp_f = CatBoostRegressor(loss_function=f"Quantile:alpha={best_m_alpha_upp}", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr_M[final_features].fillna(0), y_tr_M)

pm_low_test = 0.5 * lgb_m_low_f.predict(X_te[seg_te_M][final_features].fillna(0)) + 0.5 * cb_m_low_f.predict(X_te[seg_te_M][final_features].fillna(0))
pm_upp_test = 0.5 * lgb_m_upp_f.predict(X_te[seg_te_M][final_features].fillna(0)) + 0.5 * cb_m_upp_f.predict(X_te[seg_te_M][final_features].fillna(0))

test_cov_m = float(np.mean((y_te_M >= pm_low_test) & (y_te_M <= pm_upp_test))) * 100
test_w_m = float(np.mean(pm_upp_test - pm_low_test))

print(f"  Model M Kilitli Alpha ({best_m_alpha_low:.2f}/{best_m_alpha_upp:.2f}) 2025 Test Kapsaması: %{test_cov_m:.1f} (Ort. Genişlik: {test_w_m:,.0f} sıra)")
