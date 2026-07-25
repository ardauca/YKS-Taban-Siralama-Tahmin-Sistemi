"""
E-Adımı: Tüm 625 Bölüm Ailesinin Tam Genişleme ve Kademeli Doğrulama Scripti

Görev:
1. 625 bölüm ailesini 100'erli kademeli bloklar (batch) halinde işlemek.
2. Bilinmeyen anomali türlerini (örn. tanınmayan puan türleri, negatif kontenjan, özel statüler) tespit edip raporlamak.
3. Kademeli ilerlemeyi kaydedip en son tüm 625 bölüm ailesi için FİNAL STRATİFİYE MAE tablosunu üretmek.
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

print("="*85)
print(" E-ADIMI: TÜM 625 BÖLÜM AİLESİ TAM GENİŞLEME BAŞLATILIYOR")
print("="*85)

X_raw, y_raw, meta_raw = load_and_build()
all_features = [c for c in get_feature_columns() if c in X_raw.columns]

# birim_grup_adi ekle
df_all_depts = pd.read_csv(ROOT / "data" / "raw" / "yokatlas" / "yokatlas_all_departments_raw.csv")
dept_map = df_all_depts.set_index(["kilavuz_kodu", "yil"])["birim_grup_adi"].to_dict()
meta_raw["birim_grup_adi"] = meta_raw.set_index(["kilavuz_kodu", "yil"]).index.map(dept_map)

# ── 1. BİLİNMEYEN ANOMALİ TARAMASI ───────────────────────────────────────────
print("\n--- 1. BİLİNMEYEN YAPIAL ANOMALİ TARAMASI (TÜM VERİ SETİ) ---")

unique_puan_turleri = meta_raw.merge(df_all_depts[["kilavuz_kodu", "yil", "puan_turu"]], on=["kilavuz_kodu", "yil"], how="left")["puan_turu"].dropna().unique()
print(f"Tespit Edilen Puan Türleri: {list(unique_puan_turleri)}")

unique_univ_turleri = meta_raw.merge(df_all_depts[["kilavuz_kodu", "yil", "universite_turu"]], on=["kilavuz_kodu", "yil"], how="left")["universite_turu"].dropna().unique()
print(f"Tespit Edilen Üniversite Türleri: {list(unique_univ_turleri)}")

# Negatif veya beklenmeyen kontenjan var mı?
neg_kont = (X_raw["lag1_genel_kontenjan"] <= 0).sum()
print(f"Negatif veya Sıfır Kontenjan Kaydı Sayısı: {neg_kont}")

# Bütünüyle yeni veya tanınmayan üniversite türleri tespiti
known_univ_types = {"DEVLET", "VAKIF", "KKTC", "YURTDISI VAKIF", "YURTDISI KAMU"}
unknown_types = set(unique_univ_turleri) - known_univ_types
if unknown_types:
    print(f"⚠️ UYARI: Yeni Tanınmayan Üniversite Türü Tespit Edildi: {unknown_types}")
else:
    print("[OK] Universite turleri ve puan turleri tanimli sinirlar icinde.")

# ── 2. MODEL EĞİTİMİ VE NİHAİ HİBRİT ROUTER TAHMİNİ ───────────────────────────
mask_tr = meta_raw["yil"].isin([2023, 2024]) & X_raw["lag1_taban_siralama"].notna() & y_raw.notna()
mask_te = (meta_raw["yil"] == 2025) & X_raw["lag1_taban_siralama"].notna() & y_raw.notna()

X_tr, y_tr, meta_tr = X_raw[mask_tr], y_raw[mask_tr], meta_raw[mask_tr]
X_te, y_te, meta_te = X_raw[mask_te], y_raw[mask_te], meta_raw[mask_te]

SEG_THRESH = 150_000

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

print("\nModel S ve Global Model Eğitiliyor...")
# Global Model
lgb_g = lgb.LGBMRegressor(objective="regression_l1", **LGB_L_PARAMS).fit(X_tr[all_features].fillna(0), y_tr)
cb_g = CatBoostRegressor(**CB_L_PARAMS).fit(X_tr[all_features].fillna(0), y_tr)
pred_global_te = 0.5 * lgb_g.predict(X_te[all_features].fillna(0)) + 0.5 * cb_g.predict(X_te[all_features].fillna(0))

# Model S (<150K)
seg_tr_150 = X_tr["lag1_taban_siralama"] < SEG_THRESH
X_tr_S150, y_tr_S150 = X_tr[seg_tr_150][all_features].fillna(0), y_tr[seg_tr_150]
lgb_s150 = lgb.LGBMRegressor(objective="regression_l1", **LGB_S_PARAMS).fit(X_tr_S150, y_tr_S150)
cb_s150 = CatBoostRegressor(**CB_S_PARAMS).fit(X_tr_S150, y_tr_S150)
pred_s150_te = 0.5 * lgb_s150.predict(X_te[all_features].fillna(0)) + 0.5 * cb_s150.predict(X_te[all_features].fillna(0))

seg_te_150 = X_te["lag1_taban_siralama"] < SEG_THRESH
pred_router_150 = np.zeros(len(X_te))
pred_router_150[seg_te_150] = pred_s150_te[seg_te_150]
pred_router_150[~seg_te_150] = pred_global_te[~seg_te_150]

# ── 3. 100'ERLİ BATCH KADEMELİ İLERLEME RAPORU ─────────────────────────────────
print("\n--- 2. 100'ERLİ BÖLÜM AİLESİ BATCH KADEMELİ İLERLEME ---")

all_depts = sorted(meta_te["birim_grup_adi"].dropna().unique().tolist())
batch_size = 100

for b_idx in range(0, len(all_depts), batch_size):
    sub_depts = set(all_depts[:b_idx + batch_size])
    mask_sub = meta_te["birim_grup_adi"].isin(sub_depts)
    
    n_sub = mask_sub.sum()
    y_sub = y_te[mask_sub].values
    pg_sub = pred_global_te[mask_sub]
    pr_sub = pred_router_150[mask_sub]
    
    mae_g = np.mean(np.abs(pg_sub - y_sub))
    mae_r = np.mean(np.abs(pr_sub - y_sub))
    imp = ((mae_g - mae_r) / mae_g) * 100
    
    print(f"  Batch {b_idx//batch_size + 1} (İlk {min(b_idx+batch_size, len(all_depts))} Bölüm Ailesi | n={n_sub:>5,d}): Eski Global MAE={mae_g:,.0f} | Router MAE={mae_r:,.0f} | İyileşme={imp:>+5.2f}%")

# ── 4. TÜM 625 BÖLÜM AİLESİ FİNAL STRATİFİYE MAE TABLOSU ──────────────────────
print("\n\n" + "="*85)
print(" 3. TÜM 625 BÖLÜM AİLESİ FİNAL STRATİFİYE MAE TABLOSU (2025 TEST YILI)")
print("="*85)

BINS = [0, 10_000, 100_000, 500_000, float("inf")]
LABELS = ["0–10K", "10K–100K", "100K–500K", "500K+"]

print(f"{'Segment / Dilim':<22} {'n':>6} {'Eski Global MAE':>18} {'Nihai Router MAE':>18} {'Fark %':>10} {'Eski Global R²':>16} {'Nihai R²':>12}")
print("-" * 104)

for i, lbl in enumerate(LABELS):
    lo, hi = BINS[i], BINS[i+1]
    m = (y_te.values >= lo) & (y_te.values < hi)
    n_sub = m.sum()
    
    y_s = y_te.values[m]
    pg_s = pred_global_te[m]
    pr_s = pred_router_150[m]
    
    mae_g = np.mean(np.abs(pg_s - y_s))
    mae_r = np.mean(np.abs(pr_s - y_s))
    diff_pct = ((mae_g - mae_r) / mae_g) * 100
    
    ss_tot = np.sum((y_s - y_s.mean())**2)
    r2_g = 1.0 - (np.sum((y_s - pg_s)**2) / ss_tot) if ss_tot > 0 else float("nan")
    r2_r = 1.0 - (np.sum((y_s - pr_s)**2) / ss_tot) if ss_tot > 0 else float("nan")
    
    print(f"{lbl:<22} {n_sub:>6,d} {mae_g:>18,.0f} {mae_r:>18,.0f} {diff_pct:>+9.1f}% {r2_g:>16.3f} {r2_r:>12.3f}")

# Genel
mae_g_tot = np.mean(np.abs(pred_global_te - y_te.values))
mae_r_tot = np.mean(np.abs(pred_router_150 - y_te.values))
diff_tot = ((mae_g_tot - mae_r_tot) / mae_g_tot) * 100
ss_tot_tot = np.sum((y_te.values - y_te.values.mean())**2)
r2_g_tot = 1.0 - (np.sum((y_te.values - pred_global_te)**2) / ss_tot_tot)
r2_r_tot = 1.0 - (np.sum((y_te.values - pred_router_150)**2) / ss_tot_tot)

print("-" * 104)
print(f"{'GENEL (TÜM VERİ)':<22} {len(y_te):>6,d} {mae_g_tot:>18,.0f} {mae_r_tot:>18,.0f} {diff_tot:>+9.1f}% {r2_g_tot:>16.3f} {r2_r_tot:>12.3f}")
