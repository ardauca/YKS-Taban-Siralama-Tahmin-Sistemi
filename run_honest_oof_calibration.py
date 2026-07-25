"""
Dürüst Sızıntısız Train-OOF (Out-of-Fold) Kuantil Kalibrasyonu ve Kör Test Scripti

Metodolojik Adımlar:
1. Alpha parametrelerini (0.01 - 0.15 arası) SADECE 2023-2024 Train seti üzerinde 5-Fold Out-of-Fold (OOF) ile tara ve seç.
2. Seçilen kilitli alpha değerleriyle 2025 Test setinde KÖR (BLIND) ve TARAFIZ coverage ve aralık genişliği ölç.
3. 2026 YKS sonuçları açıklandığında çalıştırılacak nihai doğrulama mantığını tanımla.
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
print(" SIZINTISIZ TRAIN-OOF KUANTİL KALİBRASYONU VE KÖR TEST BAŞLATILIYOR")
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

# ── 1. SADECE TRAIN SETİ ÜZERİNDE 5-FOLD OOF KALİBRASYONU ───────────────────────
print("\n--- 1. SADECE TRAIN SETİ ÜZERİNDE 5-FOLD OOF ALPHA SEÇİMİ ---")

kf_oof = KFold(n_splits=5, shuffle=True, random_state=42)

# Model M (100K-500K) OOF Alpha Taraması
seg_tr_M = (X_tr["lag1_taban_siralama"] >= 100_000) & (X_tr["lag1_taban_siralama"] < 500_000)
X_tr_M, y_tr_M = X_tr[seg_tr_M], y_tr[seg_tr_M]

alpha_candidates = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
oof_results = {}

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
    oof_results[a_low] = oof_cov
    print(f"  Train OOF Alpha={a_low:.2f}/{a_upp:.2f} --> OOF Kapsama Oranı: %{oof_cov:.1f}")

# Train OOF üzerinde %80'e en yakın alpha'yı seç
best_oof_alpha_low = min(oof_results.keys(), key=lambda a: abs(oof_results[a] - 80.0))
best_oof_alpha_upp = 1.0 - best_oof_alpha_low

print(f"\n  --> TRAIN-OOF SIZINTISIZ SEÇİLEN ALFA: Low={best_oof_alpha_low:.2f} / Upp={best_oof_alpha_upp:.2f} (OOF Kapsama: %{oof_results[best_oof_alpha_low]:.1f})")

# ── 2. 2025 TEST SETİ ÜZERİNDE KÖR (BLIND) PERFORMANS ÖLÇÜMÜ ───────────────────
print("\n--- 2. 2025 TEST SETİ ÜZERİNDE TARAFIZ KÖR (BLIND) TEST ---")

# Train all final models with chosen alpha
lgb_m_low_final = lgb.LGBMRegressor(objective="quantile", alpha=best_oof_alpha_low, **LGB_L_PARAMS).fit(X_tr_M[final_features].fillna(0), y_tr_M)
lgb_m_upp_final = lgb.LGBMRegressor(objective="quantile", alpha=best_oof_alpha_upp, **LGB_L_PARAMS).fit(X_tr_M[final_features].fillna(0), y_tr_M)
cb_m_low_final = CatBoostRegressor(loss_function=f"Quantile:alpha={best_oof_alpha_low}", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr_M[final_features].fillna(0), y_tr_M)
cb_m_upp_final = CatBoostRegressor(loss_function=f"Quantile:alpha={best_oof_alpha_upp}", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr_M[final_features].fillna(0), y_tr_M)

# Model S (<100K) - Alpha 0.10 / 0.90
seg_tr_S = X_tr["lag1_taban_siralama"] < 100_000
X_tr_S, y_tr_S = X_tr[seg_tr_S][final_features].fillna(0), y_tr[seg_tr_S]

lgb_s_med = lgb.LGBMRegressor(objective="regression_l1", **LGB_S_PARAMS).fit(X_tr_S, y_tr_S)
lgb_s_low = lgb.LGBMRegressor(objective="quantile", alpha=0.100, **LGB_S_PARAMS).fit(X_tr_S, y_tr_S)
lgb_s_upp = lgb.LGBMRegressor(objective="quantile", alpha=0.900, **LGB_S_PARAMS).fit(X_tr_S, y_tr_S)

cb_s_med = CatBoostRegressor(**CB_S_PARAMS).fit(X_tr_S, y_tr_S)
cb_s_low = CatBoostRegressor(loss_function="Quantile:alpha=0.100", iterations=100, learning_rate=0.05, depth=4, verbose=0, random_seed=42).fit(X_tr_S, y_tr_S)
cb_s_upp = CatBoostRegressor(loss_function="Quantile:alpha=0.900", iterations=100, learning_rate=0.05, depth=4, verbose=0, random_seed=42).fit(X_tr_S, y_tr_S)

p_s_med = 0.5 * lgb_s_med.predict(X_te[final_features].fillna(0)) + 0.5 * cb_s_med.predict(X_te[final_features].fillna(0))
p_s_low = 0.5 * lgb_s_low.predict(X_te[final_features].fillna(0)) + 0.5 * cb_s_low.predict(X_te[final_features].fillna(0))
p_s_upp = 0.5 * lgb_s_upp.predict(X_te[final_features].fillna(0)) + 0.5 * cb_s_upp.predict(X_te[final_features].fillna(0))

# Baseline v0 (>=500K) - Alpha 0.10 / 0.90
lgb_v0_med = lgb.LGBMRegressor(objective="regression_l1", **LGB_L_PARAMS).fit(X_tr[v0_features].fillna(0), y_tr)
lgb_v0_low = lgb.LGBMRegressor(objective="quantile", alpha=0.100, **LGB_L_PARAMS).fit(X_tr[v0_features].fillna(0), y_tr)
lgb_v0_upp = lgb.LGBMRegressor(objective="quantile", alpha=0.900, **LGB_L_PARAMS).fit(X_tr[v0_features].fillna(0), y_tr)

cb_v0_med = CatBoostRegressor(loss_function="MAE", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr[v0_features].fillna(0), y_tr)
cb_v0_low = CatBoostRegressor(loss_function="Quantile:alpha=0.100", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr[v0_features].fillna(0), y_tr)
cb_v0_upp = CatBoostRegressor(loss_function="Quantile:alpha=0.900", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr[v0_features].fillna(0), y_tr)

p_v0_med = 0.5 * lgb_v0_med.predict(X_te[v0_features].fillna(0)) + 0.5 * cb_v0_med.predict(X_te[v0_features].fillna(0))
p_v0_low = 0.5 * lgb_v0_low.predict(X_te[v0_features].fillna(0)) + 0.5 * cb_v0_low.predict(X_te[v0_features].fillna(0))
p_v0_upp = 0.5 * lgb_v0_upp.predict(X_te[v0_features].fillna(0)) + 0.5 * cb_v0_upp.predict(X_te[v0_features].fillna(0))

# Model M (100K-500K)
lgb_m_med = lgb.LGBMRegressor(objective="regression_l1", **LGB_L_PARAMS).fit(X_tr_M[final_features].fillna(0), y_tr_M)
cb_m_med = CatBoostRegressor(loss_function="MAE", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr_M[final_features].fillna(0), y_tr_M)

p_m_med = 0.5 * lgb_m_med.predict(X_te[final_features].fillna(0)) + 0.5 * cb_m_med.predict(X_te[final_features].fillna(0))
p_m_low = 0.5 * lgb_m_low_final.predict(X_te[final_features].fillna(0)) + 0.5 * cb_m_low_final.predict(X_te[final_features].fillna(0))
p_m_upp = 0.5 * lgb_m_upp_final.predict(X_te[final_features].fillna(0)) + 0.5 * cb_m_upp_final.predict(X_te[final_features].fillna(0))

# Combine
lag1_te = X_te["lag1_taban_siralama"].values
m_t1 = lag1_te < 100_000
m_t2 = (lag1_te >= 100_000) & (lag1_te < 500_000)
m_t3 = lag1_te >= 500_000

pred_med_final = np.zeros(len(X_te))
pred_low_final = np.zeros(len(X_te))
pred_upp_final = np.zeros(len(X_te))

pred_med_final[m_t1] = p_s_med[m_t1]
pred_low_final[m_t1] = p_s_low[m_t1]
pred_upp_final[m_t1] = p_s_upp[m_t1]

pred_med_final[m_t2] = p_m_med[m_t2]
pred_low_final[m_t2] = p_m_low[m_t2]
pred_upp_final[m_t2] = p_m_upp[m_t2]

pred_med_final[m_t3] = p_v0_med[m_t3]
pred_low_final[m_t3] = p_v0_low[m_t3]
pred_upp_final[m_t3] = p_v0_upp[m_t3]

pred_med_final, pred_low_final, pred_upp_final = enforce_quantile_constraints(pred_med_final, pred_low_final, pred_upp_final)

print("\n" + "="*95)
print(" SIZINTISIZ KÖR TEST SONUÇLARI (2025 TEST YILI, n=15,823)")
print("="*95)

BINS = [0, 10_000, 100_000, 500_000, float("inf")]
LABELS = ["0–10K", "10K–100K", "100K–500K", "500K+"]

print(f"{'Segment / Dilim':<18} {'n':>6} {'Sabit v0 MAE':>16} {'Nihai MAE':>12} {'Fark %':>10} {'Q80 Coverage':>15} {'Ort. Genişlik':>16}")
print("-" * 98)

for i, lbl in enumerate(LABELS):
    lo, hi = BINS[i], BINS[i+1]
    m = (y_te.values >= lo) & (y_te.values < hi)
    n_sub = m.sum()
    
    y_s = y_te.values[m]
    pv0_s = p_v0_med[m]
    pf_s = pred_med_final[m]
    
    mae_v0 = np.mean(np.abs(pv0_s - y_s))
    mae_f = np.mean(np.abs(pf_s - y_s))
    diff_pct = ((mae_v0 - mae_f) / mae_v0) * 100
    
    in_b_s = (y_s >= pred_low_final[m]) & (y_s <= pred_upp_final[m])
    cov_s = float(np.mean(in_b_s)) * 100
    w_s = float(np.mean(pred_upp_final[m] - pred_low_final[m]))
    
    print(f"{lbl:<18} {n_sub:>6,d} {mae_v0:>16,.0f} {mae_f:>12,.0f} {diff_pct:>+9.1f}% {cov_s:>14.1f}% {w_s:>16,.0f}")

mae_v0_tot = np.mean(np.abs(p_v0_med - y_te.values))
mae_f_tot = np.mean(np.abs(pred_med_final - y_te.values))
diff_tot = ((mae_v0_tot - mae_f_tot) / mae_v0_tot) * 100

in_b_tot = (y_te.values >= pred_low_final) & (y_te.values <= pred_upp_final)
cov_tot = float(np.mean(in_b_tot)) * 100
w_tot = float(np.mean(pred_upp_final - pred_low_final))

print("-" * 98)
print(f"{'GENEL (TÜM ÜLKE)':<18} {len(y_te):>6,d} {mae_v0_tot:>16,.0f} {mae_f_tot:>12,.0f} {diff_tot:>+9.1f}% {cov_tot:>14.1f}% {w_tot:>16,.0f}")
