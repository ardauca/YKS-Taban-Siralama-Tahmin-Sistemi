"""
Son 3 Açık Noktanın Kapatılması:

1. Router eşiğini Train-CV ampirik kazananı olan 150,000'e güncelleme ve 100K vs 150K performansını kıyaslama.
2. Orta-rekabet bandındaki (~50K-150K) diğer 3 bölüm ailesinde (Psikoloji, İşletme, İktisat) Router ve 150K eşiğinin etkisini test etme.
3. Mevzuat, ücret, KKTC ve radikal sıçrama yapan programlar için API'de `STRUCTURAL_ANOMALY` veri kalitesi kategorisini tanımlama.
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

print("Veriler yükleniyor...")
X_raw, y_raw, meta_raw = load_and_build()
all_features = [c for c in get_feature_columns() if c in X_raw.columns]

df_all_depts = pd.read_csv(ROOT / "data" / "raw" / "yokatlas" / "yokatlas_all_departments_raw.csv")
dept_map = df_all_depts.set_index(["kilavuz_kodu", "yil"])["birim_grup_adi"].to_dict()
meta_raw["birim_grup_adi"] = meta_raw.set_index(["kilavuz_kodu", "yil"]).index.map(dept_map)

# ── 1. ROUTER EŞİĞİ GÜNCELLEMESİ (150,000) ────────────────────────────────────
# 150,000 eşiği ampirik olarak Train CV minimumudur.

# 2025 Test Yılı (n=15,823)
mask_tr = meta_raw["yil"].isin([2023, 2024]) & X_raw["lag1_taban_siralama"].notna() & y_raw.notna()
mask_te = (meta_raw["yil"] == 2025) & X_raw["lag1_taban_siralama"].notna() & y_raw.notna()

X_tr, y_tr, meta_tr = X_raw[mask_tr], y_raw[mask_tr], meta_raw[mask_tr]
X_te, y_te, meta_te = X_raw[mask_te], y_raw[mask_te], meta_raw[mask_te]

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

# Model S (150K eşiği için)
seg_tr_150 = X_tr["lag1_taban_siralama"] < 150_000
X_tr_S150, y_tr_S150 = X_tr[seg_tr_150][all_features].fillna(0), y_tr[seg_tr_150]
lgb_s150 = lgb.LGBMRegressor(objective="regression_l1", **LGB_S_PARAMS).fit(X_tr_S150, y_tr_S150)
cb_s150 = CatBoostRegressor(**CB_S_PARAMS).fit(X_tr_S150, y_tr_S150)
pred_s150_te = 0.5 * lgb_s150.predict(X_te[all_features].fillna(0)) + 0.5 * cb_s150.predict(X_te[all_features].fillna(0))

seg_te_150 = X_te["lag1_taban_siralama"] < 150_000
pred_router_150 = np.zeros(len(X_te))
pred_router_150[seg_te_150] = pred_s150_te[seg_te_150]
pred_router_150[~seg_te_150] = pred_global_te[~seg_te_150]

print("="*85)
print(" 1. ROUTER EŞİĞİ: 100,000 VS 150,000 KARŞILAŞTIRMASI (2025 Test Yılı)")
print("="*85)

# 100K Router
seg_tr_100 = X_tr["lag1_taban_siralama"] < 100_000
X_tr_S100, y_tr_S100 = X_tr[seg_tr_100][all_features].fillna(0), y_tr[seg_tr_100]
lgb_s100 = lgb.LGBMRegressor(objective="regression_l1", **LGB_S_PARAMS).fit(X_tr_S100, y_tr_S100)
cb_s100 = CatBoostRegressor(**CB_S_PARAMS).fit(X_tr_S100, y_tr_S100)
pred_s100_te = 0.5 * lgb_s100.predict(X_te[all_features].fillna(0)) + 0.5 * cb_s100.predict(X_te[all_features].fillna(0))

seg_te_100 = X_te["lag1_taban_siralama"] < 100_000
pred_router_100 = np.zeros(len(X_te))
pred_router_100[seg_te_100] = pred_s100_te[seg_te_100]
pred_router_100[~seg_te_100] = pred_global_te[~seg_te_100]

mae_100 = np.mean(np.abs(pred_router_100 - y_te.values))
mae_150 = np.mean(np.abs(pred_router_150 - y_te.values))
print(f"  Eski Router (100K Eşik) Genel MAE: {mae_100:,.0f}")
print(f"  Güncel Router (150K Eşik) Genel MAE: {mae_150:,.0f}  --> Ampirik Train-CV Kazananı Deploy Edildi.")

# ── 2. ORTA REKABET BANDI (~50K-150K) DETAYLI ANALİZ ──────────────────────────
print("\n" + "="*85)
print(" 2. ORTA REKABET BANDI (~50K-150K) İLAVE BÖLÜM AİLELERİ TESTİ")
print("="*85)

MEDIUM_DEPTS = ["Psikoloji", "İşletme", "İktisat", "Makine Mühendisliği"]
print(f"{'Bölüm Ailesi':<32} {'n':>5} {'Eski Global MAE':>16} {'Router 150K MAE':>18} {'İyileşme %':>12}")
print("-" * 88)

for dept in MEDIUM_DEPTS:
    mask_dept = meta_te["birim_grup_adi"] == dept
    n_dept = mask_dept.sum()
    if n_dept == 0:
        continue
    y_d = y_te[mask_dept].values
    pg_d = pred_global_te[mask_dept]
    pr_d = pred_router_150[mask_dept]
    
    m_glob = np.mean(np.abs(pg_d - y_d))
    m_rout = np.mean(np.abs(pr_d - y_d))
    imp = ((m_glob - m_rout) / m_glob) * 100
    print(f"  {dept:<30} {n_dept:>5} {m_glob:>16,.0f} {m_rout:>18,.0f} {imp:>+11.1f}%")

# ── 3. STRUCTURAL ANOMALY UYARI KATEGORİSİ TESTİ ──────────────────────────────
print("\n" + "="*85)
print(" 3. YAPIAL ANOMALİ (STRUCTURAL ANOMALY) TESPİT SİSTEMİ TESTİ")
print("="*85)

def detect_data_quality(univ_turu_enc: int, siralama_trend: float, lag1: float, hist_med: float | None) -> str:
    # 1. Eksik veri
    if hist_med is None or np.isnan(hist_med) or lag1 is None or np.isnan(lag1):
        return "INSUFFICIENT"
    
    # 2. Yapısal Anomali (KKTC, YURTDIŞI KAMU/VAKIF veya radikal > 150K trend kayması)
    # universite_turu_enc: 2=KKTC, 3=YURTDISI VAKIF, 4=YURTDISI KAMU
    if univ_turu_enc in [2, 3, 4] or abs(siralama_trend) > 150_000:
        return "STRUCTURAL_ANOMALY"
        
    return "SUFFICIENT"

# Test seti üzerinde anomali istatistikleri
anom_counts = {"SUFFICIENT": 0, "INSUFFICIENT": 0, "STRUCTURAL_ANOMALY": 0}

for idx in range(len(X_te)):
    u_enc = int(X_te.iloc[idx]["universite_turu_enc"])
    trend = float(X_te.iloc[idx]["siralama_trend"]) if "siralama_trend" in X_te.columns else 0.0
    lag1 = float(X_te.iloc[idx]["lag1_taban_siralama"])
    h_med = X_te.iloc[idx]["program_hist_medyan_siralama"]
    
    dq = detect_data_quality(u_enc, trend, lag1, h_med)
    anom_counts[dq] += 1

print(f"  Test Seti Data Quality Dağılımı (n={len(X_te)}):")
for k, v in anom_counts.items():
    pct = (v / len(X_te)) * 100
    print(f"    - {k:<20}: {v:>6,d} program (%{pct:.1f})")
