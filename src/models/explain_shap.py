"""
SHAP (SHapley Additive exPlanations) & Öznitelik Önem Analizi — YKS Taban Sıralama Tahmin Sistemi.

Görev:
  28 özniteliğin model tahminleri üzerindeki etkilerini SHAP değerleri ile analiz etmek.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.features.build_features import get_feature_columns, load_and_build

logger = logging.getLogger(__name__)


def run_shap_analysis() -> pd.DataFrame:
    import lightgbm as lgb
    import shap
    from catboost import CatBoostRegressor

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    X, y, meta = load_and_build()
    feature_cols = [c for c in get_feature_columns() if c in X.columns]
    X = X[feature_cols]

    mask = y.notna() & X["lag1_taban_siralama"].notna()
    X_train, y_train = X[mask], y[mask]

    logger.info("SHAP Analizi için Model Eğitiliyor (n=%d, %d feature)...", len(X_train), len(feature_cols))

    lgb_params = {
        "n_estimators": 179,
        "learning_rate": 0.030065,
        "num_leaves": 57,
        "min_child_samples": 23,
        "subsample": 0.7727,
        "colsample_bytree": 0.9634,
        "reg_alpha": 0.13255,
        "reg_lambda": 3.8551,
        "random_state": 42,
        "verbosity": -1,
    }

    model = lgb.LGBMRegressor(objective="regression_l1", **lgb_params)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df["etki_yuzdesi"] = (importance_df["mean_abs_shap"] / importance_df["mean_abs_shap"].sum()) * 100

    return importance_df


if __name__ == "__main__":
    df_imp = run_shap_analysis()
    print("\n" + "=" * 70)
    print(" SHAP ÖZNİTELİK ETKİ VE ÖNEM SIRALAMASI (TOP 15)")
    print("=" * 70)
    print(df_imp.head(15).to_string(index=False))
