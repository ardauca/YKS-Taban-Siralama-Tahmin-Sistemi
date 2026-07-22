"""
Optuna ile Hiperparametre Optimizasyonu — YKS Taban Sıralama Tahmin Sistemi.

Görev:
  CatBoost ve LightGBM modelleri için hiperparametre arama uzayı (Optuna study)

Amaç:
  Rolling backtest ortalama MAE değerini minimize ederken %80 kalibre kapsama oranını (Q80 Coverage >= %75) korumak.

MLflow Entegrasyonu:
  Her trial ve en iyi parametreler MLflow'a loglanır.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.features.build_features import get_feature_columns, load_and_build
from src.evaluation.metrics import compute_quantile_coverage, compute_regression_metrics
from src.models.train_quantile import ALPHA_LOWER, ALPHA_UPPER, enforce_quantile_constraints

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

_mlruns_path = ROOT / "mlruns"
_mlruns_path.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + str(_mlruns_path / "mlflow.db").replace("\\", "/"),
)
EXPERIMENT_NAME = "yks-taban-siralama"

ROLLING_FOLDS = [
    {"train": [2023], "test": 2024},
    {"train": [2023, 2024], "test": 2025},
]


def _get_usable_data(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, years: list[int]):
    mask = meta["yil"].isin(years)
    X_f, y_f = X[mask].copy(), y[mask].copy()
    lag_mask = X_f["lag1_taban_siralama"].notna()
    X_f, y_f = X_f[lag_mask], y_f[lag_mask]
    target_mask = y_f.notna()
    return X_f[target_mask], y_f[target_mask]


def objective_lightgbm(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> float:
    """LightGBM hiperparametre arama hedefi (MAE + Coverage cezası)."""
    import lightgbm as lgb

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 80, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    maes = []
    coverages = []

    for fold in ROLLING_FOLDS:
        X_tr, y_tr = _get_usable_data(X, y, meta, fold["train"])
        X_te, y_te = _get_usable_data(X, y, meta, [fold["test"]])

        m_med = lgb.LGBMRegressor(objective="regression_l1", **params, random_state=42, verbosity=-1).fit(X_tr, y_tr)
        m_low = lgb.LGBMRegressor(objective="quantile", alpha=ALPHA_LOWER, **params, random_state=42, verbosity=-1).fit(X_tr, y_tr)
        m_upp = lgb.LGBMRegressor(objective="quantile", alpha=ALPHA_UPPER, **params, random_state=42, verbosity=-1).fit(X_tr, y_tr)

        p_med, p_low, p_upp = enforce_quantile_constraints(m_med.predict(X_te), m_low.predict(X_te), m_upp.predict(X_te))

        mae = compute_regression_metrics(y_te.values, p_med)["mae"]
        cov = compute_quantile_coverage(y_te.values, p_low, p_upp)

        maes.append(mae)
        coverages.append(cov)

    mean_mae = float(np.mean(maes))
    mean_cov = float(np.mean(coverages))

    # Kalibrasyon cezası: Coverage %75'in altındaysa büyük ceza ver
    if mean_cov < 0.75:
        penalty = (0.75 - mean_cov) * 100000.0
        return mean_mae + penalty

    return mean_mae


def run_optuna_study(n_trials: int = 30, log_to_mlflow: bool = True) -> tuple[dict, float]:
    """Optuna çalışmasını yürütür ve en iyi hiperparametreleri döner."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    X, y, meta = load_and_build()
    feature_cols = [c for c in get_feature_columns() if c in X.columns]
    X = X[feature_cols]

    logger.info("Optuna Calismasi Baslatiliyor (%d trial)...", n_trials)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: objective_lightgbm(t, X, y, meta), n_trials=n_trials)

    best_params = study.best_params
    best_value = study.best_value

    logger.info("Optuna Tamamlandi!")
    logger.info("En iyi Skor (MAE): %.0f", best_value)
    logger.info("En iyi Parametreler:\n%s", best_params)

    if log_to_mlflow:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name="optuna_lightgbm_hyperparameter_tuning") as run:
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_mae", best_value)
            mlflow.log_metric("n_trials", n_trials)
            logger.info("MLflow Optuna run_id: %s", run.info.run_id)

    return best_params, best_value


if __name__ == "__main__":
    best_params, best_score = run_optuna_study(n_trials=25, log_to_mlflow=True)
    print("\n" + "=" * 60)
    print(" OPTUNA OPTİMİZASYON SONUÇLARI")
    print("=" * 60)
    print(f"En İyi MAE (CV) : {best_score:,.0f}")
    print("\nEn İyi Hiperparametreler:")
    for k, v in best_params.items():
        print(f"  {k:20s}: {v}")
