"""
Üç Nihai Düzeltme Analizi:

1. Data Quality Ayrıştırması:
   - STRUCTURAL_ANOMALY: Gerçek KKTC, Yurt Dışı veya Burs/Statü Değişimleri.
   - HIGH_VOLATILITY: Sadece yüksek tarihsel dalgalanma/trend yaşayan standart programlar.
   - SUFFICIENT: Standart kararlı verili programlar.
   - INSUFFICIENT: Eksik verili programlar.

2. Dalgalanma (Volatility) Eşiğinin Tren Verisi İstatistiksel Dağılımından Bağımsız Türetilmesi:
   - abs(siralama_trend) özniteliğinin %95 persentilini (95th percentile) train seti üzerinde hesaplayıp ampirik eşiği belirleme.

3. Puan Türü Bazlı Router Davranış Analizi (SAY vs EA/SÖZ):
   - SAY (Tıp, Mühendislikler) vs EA/SÖZ (Psikoloji, Sosyoloji, Tarih, İletişim) router etkisini kıyaslama.
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

# ── 1. TREND DALGALANMA EŞİĞİNİN (VOLATILITY THRESHOLD) BAĞIMSIZ TÜRETİLMESİ ──
print("="*85)
print(" 1 & 2. İSTATİSTİKSEL VOLATİLİTE EŞİĞİ TÜRETİLMESİ VE ANOMALİ AYRIŞTIRMASI")
print("="*85)

mask_tr = meta_raw["yil"].isin([2023, 2024]) & X_raw["lag1_taban_siralama"].notna() & y_raw.notna()
X_tr = X_raw[mask_tr]

# abs(siralama_trend) persentillerini train verisi üzerinde hesapla
trends_abs_tr = X_tr["siralama_trend"].abs().dropna()
p90 = float(np.percentile(trends_abs_tr, 90))
p95 = float(np.percentile(trends_abs_tr, 95))
p97_5 = float(np.percentile(trends_abs_tr, 97.5))

print(f"Train Seti (2023-2024) Trend Dalgalanma Persentilleri:")
print(f"  - %90 Persentil  : {p90:,.0f} sıra")
print(f"  - %95 Persentil  : {p95:,.0f} sıra  --> Sezgisellikten bağımsız ampirik HIGH_VOLATILITY eşiği!")
print(f"  - %97.5 Persentil: {p97_5:,.0f} sıra")

VOLATILITY_THRESH = round(p95, -3)  # %95 persentili en yakın binliğe yuvarla

# 2025 Test Seti Üzerinde Yeni 4-Lü Etiketleme
mask_te = (meta_raw["yil"] == 2025) & X_raw["lag1_taban_siralama"].notna() & y_raw.notna()
X_te = X_raw[mask_te]

def classify_data_quality_v2(univ_turu_enc: int, siralama_trend: float, lag1: float, hist_med: float | None) -> str:
    # 1. Eksik Veri
    if hist_med is None or np.isnan(hist_med) or lag1 is None or np.isnan(lag1):
        return "INSUFFICIENT"
    
    # 2. Gerçek Yapısal Anomali (KKTC, Yurt Dışı Kamu/Vakıf)
    if univ_turu_enc in [2, 3, 4]:
        return "STRUCTURAL_ANOMALY"
        
    # 3. Yüksek Dalgalanma (Persentil %95 aşımı)
    if abs(siralama_trend) > VOLATILITY_THRESH:
        return "HIGH_VOLATILITY"
        
    return "SUFFICIENT"

counts_v2 = {"SUFFICIENT": 0, "INSUFFICIENT": 0, "STRUCTURAL_ANOMALY": 0, "HIGH_VOLATILITY": 0}
for idx in range(len(X_te)):
    u_enc = int(X_te.iloc[idx]["universite_turu_enc"])
    trend = float(X_te.iloc[idx]["siralama_trend"]) if "siralama_trend" in X_te.columns else 0.0
    lag1 = float(X_te.iloc[idx]["lag1_taban_siralama"])
    h_med = X_te.iloc[idx]["program_hist_medyan_siralama"]
    
    dq = classify_data_quality_v2(u_enc, trend, lag1, h_med)
    counts_v2[dq] += 1

print(f"\n2025 Test Seti Ayrıştırılmış Veri Kalitesi Dağılımı (n={len(X_te)}):")
for k, v in counts_v2.items():
    pct = (v / len(X_te)) * 100
    print(f"  - {k:<20}: {v:>6,d} program (%{pct:.1f})")

# ── 3. PUAN TÜRÜ BAZLI ROUTER ANALİZİ (SAY vs EA/SÖZ) ─────────────────────────
print("\n" + "="*85)
print(" 3. PUAN TÜRÜ BAZLI ROUTER ANALİZİ (SAY vs EA / SÖZ)")
print("="*85)

y_tr, meta_tr = y_raw[mask_tr], meta_raw[mask_tr]
y_te, meta_te = y_raw[mask_te], meta_raw[mask_te]

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

# Model S (150K)
seg_tr_150 = X_tr["lag1_taban_siralama"] < 150_000
X_tr_S150, y_tr_S150 = X_tr[seg_tr_150][all_features].fillna(0), y_tr[seg_tr_150]
lgb_s150 = lgb.LGBMRegressor(objective="regression_l1", **LGB_S_PARAMS).fit(X_tr_S150, y_tr_S150)
cb_s150 = CatBoostRegressor(**CB_S_PARAMS).fit(X_tr_S150, y_tr_S150)
pred_s150_te = 0.5 * lgb_s150.predict(X_te[all_features].fillna(0)) + 0.5 * cb_s150.predict(X_te[all_features].fillna(0))

seg_te_150 = X_te["lag1_taban_siralama"] < 150_000
pred_router_150 = np.zeros(len(X_te))
pred_router_150[seg_te_150] = pred_s150_te[seg_te_150]
pred_router_150[~seg_te_150] = pred_global_te[~seg_te_150]

# Puan Türü Haritası (0: SAY, 1: SÖZ, 2: EA, 3: DİL, 4: TYT)
puan_map = {0: "SAY", 1: "SÖZ", 2: "EA", 3: "DİL", 4: "TYT"}
X_te_puan_str = X_te["puan_turu_enc"].map(puan_map)

print(f"{'Puan Türü':<12} {'n':>6} {'Eski Global MAE':>18} {'Router 150K MAE':>18} {'İyileşme %':>12}")
print("-" * 72)

for p_code, p_name in puan_map.items():
    mask_p = X_te["puan_turu_enc"] == p_code
    n_p = mask_p.sum()
    if n_p == 0:
        continue
    y_p = y_te[mask_p].values
    pg_p = pred_global_te[mask_p]
    pr_p = pred_router_150[mask_p]
    
    m_glob = np.mean(np.abs(pg_p - y_p))
    m_rout = np.mean(np.abs(pr_p - y_p))
    imp = ((m_glob - m_rout) / m_glob) * 100
    print(f"  {p_name:<10} {n_p:>6} {m_glob:>18,.0f} {m_rout:>18,.0f} {imp:>+11.1f}%")

# İlave Sosyal Bölüm Aileleri Testi
ADDITIONAL_SOC_DEPTS = ["Sosyoloji", "Tarih", "Yeni Medya ve İletişim"]
print("\nİlave Sosyal Bölüm Aileleri Performansı:")
print(f"{'Bölüm Ailesi':<32} {'n':>5} {'Eski Global MAE':>16} {'Router 150K MAE':>18} {'İyileşme %':>12}")
print("-" * 88)

for dept in ADDITIONAL_SOC_DEPTS:
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
