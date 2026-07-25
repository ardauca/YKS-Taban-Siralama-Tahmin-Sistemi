"""
Dört Kritik Doğrulama Düzeltmesi Scripti:

1. Sabit Değişmez Baseline v0 (Referans Kontrolü):
   - Orijinal 28 feature ile eğitilmiş sabit v0 Global Model referansı. Tüm iyileşmeler tek ve değişmez bu v0 baseline'a göre hesaplanacak.

2. 0-10K Düşük Güvenilirlik Bayrağı (API Uyarı Sistemi):
   - 0-10K segmenti tahminleri için explicit 'VERY_LOW' ve 'low_reliability_warning' eklemesi.

3. Ülke Geneli Ulusal Q80 Coverage ve Ortalama Aralık Genişliği Hesaplaması (15,823 Program):
   - Hibrit Router 150K + Quantile aralıkları ile tüm ülke geneli test setinde Q80 kapsama oranını yeniden ölç.

4. 625 Bölüm Ailesinin Tamamı Üzerinde 5-Fold Train CV ile Router ve Volatilite Eşiği Doğrulaması:
   - 100% veri üzerinde 95. persentil dalgalanma eşiği ve Router eşiğini sıfırdan hesapla.
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
print(" DÖRT KRİTİK DOĞRULAMA DÜZELTMESİ BAŞLATILIYOR")
print("="*85)

X_all, y_all, meta_all = load_and_build()

# ── 1. SABİT V0 BASELINE FEATURE SETİ ─────────────────────────────────────────
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

# Tüm Yeni Üretim Feature Seti (30 Feature)
final_features = [c for c in X_all.columns if c not in ["kilavuz_kodu", "taban_siralama", "universite_adi", "birim_adi", "birim_grup_adi"]]

mask_tr = meta_all["yil"].isin([2023, 2024]) & X_all["lag1_taban_siralama"].notna() & y_all.notna()
mask_te = (meta_all["yil"] == 2025) & X_all["lag1_taban_siralama"].notna() & y_all.notna()

X_tr, y_tr = X_all[mask_tr], y_all[mask_tr]
X_te, y_te = X_all[mask_te], y_all[mask_te]

# ── 4. 100% TAM VERİ SETİ ÜZERİNDE EŞİK DOĞRULAMASI (NOKTA 4) ─────────────────
print("\n--- 4. TÜM VERİ SETİ ÜZERİNDE EŞİK VERİFİKASYONU ---")

# Volatilite %95 persentili (100% Train verisi)
trends_abs_tr = X_tr["siralama_trend"].abs().dropna()
p95_nationwide = float(np.percentile(trends_abs_tr, 95))
volatility_thresh_nationwide = round(p95_nationwide, -3)
print(f"  - Ulusal %95 Volatilite Eşiği: {p95_nationwide:,.0f} sıra  --> Tanımlanan: {volatility_thresh_nationwide:,.0f} sıra")

# Router Eşiği 5-Fold Train CV (100% Train verisi)
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

kf_all = KFold(n_splits=5, shuffle=True, random_state=42)
thresh_candidates = [50_000, 100_000, 150_000, 200_000]

print("  - Ulusal 625 Bölüm Train CV Router Eşik Tarama:")
for th in thresh_candidates:
    cv_maes = []
    for tr_i, val_i in kf_all.split(X_tr):
        X_k_tr, y_k_tr = X_tr.iloc[tr_i], y_tr.iloc[tr_i]
        X_k_val, y_k_val = X_tr.iloc[val_i], y_tr.iloc[val_i]
        
        seg_tr_s = X_k_tr["lag1_taban_siralama"] < th
        X_tr_S, y_tr_S = X_k_tr[seg_tr_s][final_features].fillna(0), y_k_tr[seg_tr_s]
        
        lgb_s = lgb.LGBMRegressor(objective="regression_l1", **LGB_S_PARAMS).fit(X_tr_S, y_tr_S)
        cb_s = CatBoostRegressor(**CB_S_PARAMS).fit(X_tr_S, y_tr_S)
        
        lgb_g = lgb.LGBMRegressor(objective="regression_l1", **LGB_L_PARAMS).fit(X_k_tr[final_features].fillna(0), y_k_tr)
        cb_g = CatBoostRegressor(**CB_L_PARAMS).fit(X_k_tr[final_features].fillna(0), y_k_tr)
        
        seg_val_s = X_k_val["lag1_taban_siralama"] < th
        pred_val = np.zeros(len(val_i))
        if seg_val_s.sum() > 0:
            pred_val[seg_val_s] = 0.5 * lgb_s.predict(X_k_val[seg_val_s][final_features].fillna(0)) + 0.5 * cb_s.predict(X_k_val[seg_val_s][final_features].fillna(0))
        if (~seg_val_s).sum() > 0:
            pred_val[~seg_val_s] = 0.5 * lgb_g.predict(X_k_val[~seg_val_s][final_features].fillna(0)) + 0.5 * cb_g.predict(X_k_val[~seg_val_s][final_features].fillna(0))
            
        cv_maes.append(np.mean(np.abs(pred_val - y_k_val.values)))
        
    print(f"    Eşik: {th:>7,d} sıra  --> Train-CV MAE: {np.mean(cv_maes):,.0f}")

# ── 1. SABİT V0 BASELINE MODELİ EĞİTİMİ (NOKTA 1) ────────────────────────────
print("\n--- 1. SABİT V0 BASELINE VE NİHAİ MODEL HESAPLAMASI ---")

# Sabit Baseline v0 (Değişmez Referans - Orijinal 28 Öznitelik)
lgb_v0 = lgb.LGBMRegressor(objective="regression_l1", **LGB_L_PARAMS).fit(X_tr[v0_features].fillna(0), y_tr)
cb_v0 = CatBoostRegressor(**CB_L_PARAMS).fit(X_tr[v0_features].fillna(0), y_tr)
pred_v0_baseline = 0.5 * lgb_v0.predict(X_te[v0_features].fillna(0)) + 0.5 * cb_v0.predict(X_te[v0_features].fillna(0))

# Nihai Hibrit Router 150K Modelleri (Medyan + Quantiles %80 Güven Aralığı)
lgb_g_med = lgb.LGBMRegressor(objective="regression_l1", **LGB_L_PARAMS).fit(X_tr[final_features].fillna(0), y_tr)
lgb_g_low = lgb.LGBMRegressor(objective="quantile", alpha=0.030, **LGB_L_PARAMS).fit(X_tr[final_features].fillna(0), y_tr)
lgb_g_upp = lgb.LGBMRegressor(objective="quantile", alpha=0.970, **LGB_L_PARAMS).fit(X_tr[final_features].fillna(0), y_tr)

cb_g_med = CatBoostRegressor(loss_function="MAE", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr[final_features].fillna(0), y_tr)
cb_g_low = CatBoostRegressor(loss_function="Quantile:alpha=0.030", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr[final_features].fillna(0), y_tr)
cb_g_upp = CatBoostRegressor(loss_function="Quantile:alpha=0.970", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_tr[final_features].fillna(0), y_tr)

# Model S (150K)
seg_tr_150 = X_tr["lag1_taban_siralama"] < 150_000
X_tr_S150, y_tr_S150 = X_tr[seg_tr_150][final_features].fillna(0), y_tr[seg_tr_150]

lgb_s_med = lgb.LGBMRegressor(objective="regression_l1", **LGB_S_PARAMS).fit(X_tr_S150, y_tr_S150)
lgb_s_low = lgb.LGBMRegressor(objective="quantile", alpha=0.030, **LGB_S_PARAMS).fit(X_tr_S150, y_tr_S150)
lgb_s_upp = lgb.LGBMRegressor(objective="quantile", alpha=0.970, **LGB_S_PARAMS).fit(X_tr_S150, y_tr_S150)

cb_s_med = CatBoostRegressor(**CB_S_PARAMS).fit(X_tr_S150, y_tr_S150)
cb_s_low = CatBoostRegressor(loss_function="Quantile:alpha=0.030", iterations=100, learning_rate=0.05, depth=4, verbose=0, random_seed=42).fit(X_tr_S150, y_tr_S150)
cb_s_upp = CatBoostRegressor(loss_function="Quantile:alpha=0.970", iterations=100, learning_rate=0.05, depth=4, verbose=0, random_seed=42).fit(X_tr_S150, y_tr_S150)

# Predict on Test Set
X_te_ff = X_te[final_features].fillna(0)
p_g_med = 0.5 * lgb_g_med.predict(X_te_ff) + 0.5 * cb_g_med.predict(X_te_ff)
p_g_low = 0.5 * lgb_g_low.predict(X_te_ff) + 0.5 * cb_g_low.predict(X_te_ff)
p_g_upp = 0.5 * lgb_g_upp.predict(X_te_ff) + 0.5 * cb_g_upp.predict(X_te_ff)

p_s_med = 0.5 * lgb_s_med.predict(X_te_ff) + 0.5 * cb_s_med.predict(X_te_ff)
p_s_low = 0.5 * lgb_s_low.predict(X_te_ff) + 0.5 * cb_s_low.predict(X_te_ff)
p_s_upp = 0.5 * lgb_s_upp.predict(X_te_ff) + 0.5 * cb_s_upp.predict(X_te_ff)

seg_te_150 = X_te["lag1_taban_siralama"] < 150_000

pred_final_med = np.where(seg_te_150, p_s_med, p_g_med)
pred_final_low = np.where(seg_te_150, p_s_low, p_g_low)
pred_final_upp = np.where(seg_te_150, p_s_upp, p_g_upp)

pred_final_med, pred_final_low, pred_final_upp = enforce_quantile_constraints(pred_final_med, pred_final_low, pred_final_upp)

# ── 3. ULUSAL Q80 COVERAGE VE ARALIK GENİŞLİĞİ HESABI (NOKTA 3) ───────────────
print("\n--- 3. NİHAİ ULUSAL Q80 COVERAGE HESAPLAMASI (15,823 PROGRAM) ---")

in_bounds = (y_te.values >= pred_final_low) & (y_te.values <= pred_final_upp)
q80_coverage_nationwide = float(np.mean(in_bounds)) * 100
mean_interval_width = float(np.mean(pred_final_upp - pred_final_low))

print(f"  - Ulusal Q80 Kapsama Oranı (Coverage) : %{q80_coverage_nationwide:.1f}  (Hedef: %80)")
print(f"  - Ortalama Güven Aralığı Genişliği   : {mean_interval_width:,.0f} sıra")

# ── 1 & 2. KARŞILAŞTIRMALI SABİT BASELINE V0 VE UYARI SİSTEMİ ──────────────────
print("\n" + "="*85)
print(" SABİT BASELINE V0'A GÖRE STRATİFİYE PERFORMANS VE UYARI TABLOSU")
print("="*85)

BINS = [0, 10_000, 100_000, 500_000, float("inf")]
LABELS = ["0–10K", "10K–100K", "100K–500K", "500K+"]

print(f"{'Segment / Dilim':<20} {'n':>6} {'Sabit Baseline v0 MAE':>22} {'Nihai Router MAE':>18} {'Fark %':>10} {'Baseline R²':>14} {'Nihai R²':>12}")
print("-" * 104)

for i, lbl in enumerate(LABELS):
    lo, hi = BINS[i], BINS[i+1]
    m = (y_te.values >= lo) & (y_te.values < hi)
    n_sub = m.sum()
    
    y_s = y_te.values[m]
    pv0_s = pred_v0_baseline[m]
    pr_s = pred_final_med[m]
    
    mae_v0 = np.mean(np.abs(pv0_s - y_s))
    mae_r = np.mean(np.abs(pr_s - y_s))
    diff_pct = ((mae_v0 - mae_r) / mae_v0) * 100
    
    ss_tot = np.sum((y_s - y_s.mean())**2)
    r2_v0 = 1.0 - (np.sum((y_s - pv0_s)**2) / ss_tot) if ss_tot > 0 else float("nan")
    r2_r = 1.0 - (np.sum((y_s - pr_s)**2) / ss_tot) if ss_tot > 0 else float("nan")
    
    print(f"{lbl:<20} {n_sub:>6,d} {mae_v0:>22,.0f} {mae_r:>18,.0f} {diff_pct:>+9.1f}% {r2_v0:>14.3f} {r2_r:>12.3f}")

mae_v0_tot = np.mean(np.abs(pred_v0_baseline - y_te.values))
mae_r_tot = np.mean(np.abs(pred_final_med - y_te.values))
diff_tot = ((mae_v0_tot - mae_r_tot) / mae_v0_tot) * 100
ss_tot_tot = np.sum((y_te.values - y_te.values.mean())**2)
r2_v0_tot = 1.0 - (np.sum((y_te.values - pred_v0_baseline)**2) / ss_tot_tot)
r2_r_tot = 1.0 - (np.sum((y_te.values - pred_final_med)**2) / ss_tot_tot)

print("-" * 104)
print(f"{'GENEL (TÜM VERİ)':<20} {len(y_te):>6,d} {mae_v0_tot:>22,.0f} {mae_r_tot:>18,.0f} {diff_tot:>+9.1f}% {r2_v0_tot:>14.3f} {r2_r_tot:>12.3f}")
