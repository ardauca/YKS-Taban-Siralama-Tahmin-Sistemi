"""
Dört İnce Kalibrasyon ve 3 Kademeli Router Doğrulama Scripti:

1. Stratifiye Q80 Kapsama Oranı (Coverage) ve Ortalama Aralık Genişliği (0-10K, 10K-100K, 100K-500K, 500K+, GENEL).
2. 3 Kademeli Router Mimarisi (Strategy D):
   - Tier 1 (< 100K): Model S (HeavyReg GBDT)
   - Tier 2 (100K - 500K): Model M (Segment Model M)
   - Tier 3 (>= 500K): Pure Baseline v0 (500K+ koruma)
3. Gerçek KFold Veri Bölünmesi Seed Varyansı Analizi (KFold shuffle random_state = s).
4. Point Estimate (alpha=0.50 L1) ile Kuantil Aralıklarının (alpha=0.10 / 0.90) Matematiksel Tutarlılık Doğrulaması.
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
print(" DÖRT İNCE KALİBRASYON VE 3 KADEMELİ ROUTER ANALİZİ BAŞLATILIYOR")
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

# ── 3. MADDE 3: KFOLD SEED SHUFFLE SEED SEED VARYANSI ANALİZİ ──────────────────
print("\n--- 3. KFOLD SHUFFLE RANDAMIZATION İLE SEED VARYANS ANALİZİ ---")

seeds = [42, 123, 456, 789, 2026]
seed_cv_maes = []

for s in seeds:
    kf_s = KFold(n_splits=5, shuffle=True, random_state=s)
    cv_m = []
    for tr_i, val_i in kf_s.split(X_tr):
        X_k_tr, y_k_tr = X_tr.iloc[tr_i], y_tr.iloc[tr_i]
        X_k_val, y_k_val = X_tr.iloc[val_i], y_tr.iloc[val_i]
        
        seg_s = X_k_tr["lag1_taban_siralama"] < 100_000
        X_tr_S, y_tr_S = X_k_tr[seg_s][final_features].fillna(0), y_k_tr[seg_s]
        
        lgb_s_params_s = dict(LGB_S_PARAMS, random_state=s)
        lgb_l_params_s = dict(LGB_L_PARAMS, random_state=s)
        
        lgb_s = lgb.LGBMRegressor(objective="regression_l1", **lgb_s_params_s).fit(X_tr_S, y_tr_S)
        lgb_v0_k = lgb.LGBMRegressor(objective="regression_l1", **lgb_l_params_s).fit(X_k_tr[v0_features].fillna(0), y_k_tr)
        
        seg_val_s = X_k_val["lag1_taban_siralama"] < 100_000
        pv = np.where(seg_val_s, lgb_s.predict(X_k_val[final_features].fillna(0)), lgb_v0_k.predict(X_k_val[v0_features].fillna(0)))
        cv_m.append(np.mean(np.abs(pv - y_k_val.values)))
        
    res_s = float(np.mean(cv_m))
    seed_cv_maes.append(res_s)
    print(f"  Seed {s:<5}: 5-Fold CV MAE = {res_s:,.0f}")

print(f"  --> KFold Shuffle Dahil Seed Varyansı: Ortalama = {np.mean(seed_cv_maes):,.0f} | std = {np.std(seed_cv_maes):.1f} (Bağıl varyans: %{np.std(seed_cv_maes)/np.mean(seed_cv_maes)*100:.2f})")

# ── 2. MADDE 2: 3 KADEMELİ ROUTER EĞİTİMİ (TIER 1: <100K, TIER 2: 100K-500K, TIER 3: >=500K) ──
print("\n--- 2. 3 KADEMELİ ROUTER MİMARİSİ EĞİTİMİ VE TESTİ ---")

# 1. Tier 3 Model: Pure Baseline v0 Model (Tüm veri ile v0 öznitelikleriyle eğitilmiş)
lgb_v0_med = lgb.LGBMRegressor(objective="regression_l1", **LGB_L_PARAMS).fit(X_tr[v0_features].fillna(0), y_tr)
lgb_v0_low = lgb.LGBMRegressor(objective="quantile", alpha=0.100, **LGB_L_PARAMS).fit(X_tr[v0_features].fillna(0), y_tr)
lgb_v0_upp = lgb.LGBMRegressor(objective="quantile", alpha=0.900, **LGB_L_PARAMS).fit(X_tr[v0_features].fillna(0), y_tr)

cb_v0_med = CatBoostRegressor(loss_function="MAE", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr[v0_features].fillna(0), y_tr)
cb_v0_low = CatBoostRegressor(loss_function="Quantile:alpha=0.100", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr[v0_features].fillna(0), y_tr)
cb_v0_upp = CatBoostRegressor(loss_function="Quantile:alpha=0.900", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr[v0_features].fillna(0), y_tr)

p_v0_med = 0.5 * lgb_v0_med.predict(X_te[v0_features].fillna(0)) + 0.5 * cb_v0_med.predict(X_te[v0_features].fillna(0))
p_v0_low = 0.5 * lgb_v0_low.predict(X_te[v0_features].fillna(0)) + 0.5 * cb_v0_low.predict(X_te[v0_features].fillna(0))
p_v0_upp = 0.5 * lgb_v0_upp.predict(X_te[v0_features].fillna(0)) + 0.5 * cb_v0_upp.predict(X_te[v0_features].fillna(0))

# 2. Tier 1 Model (<100K): Model S (HeavyReg GBDT)
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

# 3. Tier 2 Model (100K - 500K): Model M (Segment Model M)
seg_tr_M = (X_tr["lag1_taban_siralama"] >= 100_000) & (X_tr["lag1_taban_siralama"] < 500_000)
X_tr_M, y_tr_M = X_tr[seg_tr_M][final_features].fillna(0), y_tr[seg_tr_M]

lgb_m_med = lgb.LGBMRegressor(objective="regression_l1", **LGB_L_PARAMS).fit(X_tr_M, y_tr_M)
lgb_m_low = lgb.LGBMRegressor(objective="quantile", alpha=0.100, **LGB_L_PARAMS).fit(X_tr_M, y_tr_M)
lgb_m_upp = lgb.LGBMRegressor(objective="quantile", alpha=0.900, **LGB_L_PARAMS).fit(X_tr_M, y_tr_M)

cb_m_med = CatBoostRegressor(loss_function="MAE", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr_M, y_tr_M)
cb_m_low = CatBoostRegressor(loss_function="Quantile:alpha=0.100", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr_M, y_tr_M)
cb_m_upp = CatBoostRegressor(loss_function="Quantile:alpha=0.900", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr_M, y_tr_M)

p_m_med = 0.5 * lgb_m_med.predict(X_te[final_features].fillna(0)) + 0.5 * cb_m_med.predict(X_te[final_features].fillna(0))
p_m_low = 0.5 * lgb_m_low.predict(X_te[final_features].fillna(0)) + 0.5 * cb_m_low.predict(X_te[final_features].fillna(0))
p_m_upp = 0.5 * lgb_m_upp.predict(X_te[final_features].fillna(0)) + 0.5 * cb_m_upp.predict(X_te[final_features].fillna(0))

# 3 Kademeli Birleştirme (Strategy D - 3-Tier Router)
lag1_te = X_te["lag1_taban_siralama"].values
m_tier1 = lag1_te < 100_000
m_tier2 = (lag1_te >= 100_000) & (lag1_te < 500_000)
m_tier3 = lag1_te >= 500_000

pred_tier3_med = np.zeros(len(X_te))
pred_tier3_low = np.zeros(len(X_te))
pred_tier3_upp = np.zeros(len(X_te))

pred_tier3_med[m_tier1] = p_s_med[m_tier1]
pred_tier3_low[m_tier1] = p_s_low[m_tier1]
pred_tier3_upp[m_tier1] = p_s_upp[m_tier1]

pred_tier3_med[m_tier2] = p_m_med[m_tier2]
pred_tier3_low[m_tier2] = p_m_low[m_tier2]
pred_tier3_upp[m_tier2] = p_m_upp[m_tier2]

pred_tier3_med[m_tier3] = p_v0_med[m_tier3]
pred_tier3_low[m_tier3] = p_v0_low[m_tier3]
pred_tier3_upp[m_tier3] = p_v0_upp[m_tier3]

pred_tier3_med, pred_tier3_low, pred_tier3_upp = enforce_quantile_constraints(pred_tier3_med, pred_tier3_low, pred_tier3_upp)

# ── 1. MADDE 1 & MADDE 4: STRATİFİYE FİNAL TABLO (MAE, COVERAGE, INTERVAL WIDTH) ──
print("\n" + "="*95)
print(" SEGMENT BAZLI FİNAL TABLO (3 KADEMELİ ROUTER — STRATİFİYE COVERAGE, WIDTH VE MAE)")
print("="*95)

BINS = [0, 10_000, 100_000, 500_000, float("inf")]
LABELS = ["0–10K", "10K–100K", "100K–500K", "500K+"]

print(f"{'Segment / Dilim':<18} {'n':>6} {'Sabit v0 MAE':>16} {'3-Tier MAE':>14} {'Fark %':>10} {'Q80 Coverage':>15} {'Ort. Genişlik':>16}")
print("-" * 98)

for i, lbl in enumerate(LABELS):
    lo, hi = BINS[i], BINS[i+1]
    m = (y_te.values >= lo) & (y_te.values < hi)
    n_sub = m.sum()
    
    y_s = y_te.values[m]
    pv0_s = p_v0_med[m]
    pt3_s = pred_tier3_med[m]
    
    mae_v0 = np.mean(np.abs(pv0_s - y_s))
    mae_t3 = np.mean(np.abs(pt3_s - y_s))
    diff_pct = ((mae_v0 - mae_t3) / mae_v0) * 100
    
    in_b_s = (y_s >= pred_tier3_low[m]) & (y_s <= pred_tier3_upp[m])
    cov_s = float(np.mean(in_b_s)) * 100
    w_s = float(np.mean(pred_tier3_upp[m] - pred_tier3_low[m]))
    
    print(f"{lbl:<18} {n_sub:>6,d} {mae_v0:>16,.0f} {mae_t3:>14,.0f} {diff_pct:>+9.1f}% {cov_s:>14.1f}% {w_s:>16,.0f}")

# GENEL
mae_v0_tot = np.mean(np.abs(p_v0_med - y_te.values))
mae_t3_tot = np.mean(np.abs(pred_tier3_med - y_te.values))
diff_tot = ((mae_v0_tot - mae_t3_tot) / mae_v0_tot) * 100

in_b_tot = (y_te.values >= pred_tier3_low) & (y_te.values <= pred_tier3_upp)
cov_tot = float(np.mean(in_b_tot)) * 100
w_tot = float(np.mean(pred_tier3_upp - pred_tier3_low))

print("-" * 98)
print(f"{'GENEL (TÜM ÜLKE)':<18} {len(y_te):>6,d} {mae_v0_tot:>16,.0f} {mae_t3_tot:>14,.0f} {diff_tot:>+9.1f}% {cov_tot:>14.1f}% {w_tot:>16,.0f}")
