"""
Zamana Duyarlı Kör (Blind) Walk-Forward Backtest & Gelecek Simülasyon Motoru.

Görev:
  Modelin hiçbir t anındaki taban puanı veya sıralamayı görmeden, sadece:
  - t-1 ve t-2 yılı geçmiş sıralamalarını
  - t yılının YENİ KONTENJANLARINI (genel, okul birincisi, depremzede vb.)
  - t yılının makro kontenjan kısıntı şoklarını
  bilerek t yılının taban sıralamasını tam "kör" (blind) şekilde tahmin etmesi.

Bu test her geçmiş yıl (2023, 2024, 2025) için ayrı ayrı simüle edilip değerlendirilir.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from api.main import _lgb_med, _cb_med
from src.features.build_features import get_feature_columns, load_and_build
from src.models.train_quantile import enforce_quantile_constraints

logger = logging.getLogger(__name__)


def run_walkforward_backtest() -> dict[int, dict[str, float]]:
    import lightgbm as lgb
    from catboost import CatBoostRegressor

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    X, y, meta = load_and_build()
    feature_cols = [c for c in get_feature_columns() if c in X.columns]

    years_to_test = [2024, 2025]
    results = {}

    for test_year in years_to_test:
        train_mask = (meta["yil"] < test_year) & y.notna() & X["lag1_taban_siralama"].notna()
        test_mask = (meta["yil"] == test_year) & y.notna() & X["lag1_taban_siralama"].notna()

        X_train, y_train = X[train_mask][feature_cols], y[train_mask]
        X_test, y_test = X[test_mask][feature_cols], y[test_mask]

        logger.info("\n=======================================================")
        logger.info(" WALK-FORWARD KÖR BAKIŞ TESTİ: %d YILI SİMÜLASYONU", test_year)
        logger.info(" Eğitilen Yıllar: < %d (n=%d)", test_year, len(X_train))
        logger.info(" Kör Tahmin Yapılan Yıl: %d (n=%d)", test_year, len(X_test))
        logger.info("=======================================================")

        # LightGBM Quantile Blending Model
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

        m_lgb_med = lgb.LGBMRegressor(objective="regression_l1", **lgb_params)
        m_lgb_low = lgb.LGBMRegressor(objective="quantile", alpha=0.030, **lgb_params)
        m_lgb_upp = lgb.LGBMRegressor(objective="quantile", alpha=0.970, **lgb_params)

        m_lgb_med.fit(X_train, y_train)
        m_lgb_low.fit(X_train, y_train)
        m_lgb_upp.fit(X_train, y_train)

        cb_params = {
            "iterations": 250,
            "learning_rate": 0.04,
            "depth": 6,
            "l2_leaf_reg": 4.0,
            "random_seed": 42,
            "verbose": 0,
        }

        m_cb_med = CatBoostRegressor(loss_function="MAE", **cb_params)
        m_cb_low = CatBoostRegressor(loss_function="Quantile:alpha=0.030", **cb_params)
        m_cb_upp = CatBoostRegressor(loss_function="Quantile:alpha=0.970", **cb_params)

        m_cb_med.fit(X_train, y_train)
        m_cb_low.fit(X_train, y_train)
        m_cb_upp.fit(X_train, y_train)

        # Tahmin üret
        p_med = 0.5 * m_lgb_med.predict(X_test) + 0.5 * m_cb_med.predict(X_test)
        p_low = 0.5 * m_lgb_low.predict(X_test) + 0.5 * m_cb_low.predict(X_test)
        p_upp = 0.5 * m_lgb_upp.predict(X_test) + 0.5 * m_cb_upp.predict(X_test)

        p_med, p_low, p_upp = enforce_quantile_constraints(p_med, p_low, p_upp)

        mae = mean_absolute_error(y_test, p_med)
        rmse = np.sqrt(mean_squared_error(y_test, p_med))
        r2 = r2_score(y_test, p_med)

        in_interval = (y_test >= p_low) & (y_test <= p_upp)
        coverage = np.mean(in_interval) * 100
        mean_width = np.mean(p_upp - p_low)

        results[test_year] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "Coverage": coverage,
            "Width": mean_width,
        }

        print(f"  --> {test_year} KÖR TAHMİN MAE     : {mae:,.0f} sıra")
        print(f"  --> {test_year} KÖR TAHMİN RMSE    : {rmse:,.0f} sıra")
        print(f"  --> {test_year} KÖR TAHMİN R² SKORU: %{r2*100:.1f}")
        print(f"  --> {test_year} Q80 KAPSAMA ORANI  : %{coverage:.1f}")
        print(f"  --> {test_year} ARALIK GENİŞLİĞİ   : {mean_width:,.0f} sıra")

    return results


if __name__ == "__main__":
    res = run_walkforward_backtest()
    print("\n" + "=" * 70)
    print(" ZAMANA DUYARLI KÖR (WALK-FORWARD) BACKTEST ÖZETİ")
    print("=" * 70)
    for yr, metrics in res.items():
        print(f"{yr} Yılı Kör Tahmini -> MAE: {metrics['MAE']:,.0f} | R²: %{metrics['R2']*100:.1f} | Güven Kapsama: %{metrics['Coverage']:.1f}")
