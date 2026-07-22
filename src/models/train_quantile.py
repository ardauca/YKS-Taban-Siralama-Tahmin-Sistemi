"""
Quantile Regression (Belirsizlik Aralık Tahmini) — YKS Taban Sıralama Tahmin Sistemi.

Görev:
  Nokta tahmin (MAE minimize) + %80 Güven Aralığı (Lower alpha=0.10, Upper alpha=0.90)

Modeller:
  LightGBM / CatBoost Quantile Regressors

Gereksinimler (Bölüm 8):
  - Quantile Coverage Rate (Q80 Hedef: %80 kapsama)
  - Quantile Crossing önleme: lower <= median <= upper garantisi
  - MLflow loglama (coverage, interval_width, run_id)
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Proje root'u path'e ekle
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.features.build_features import get_feature_columns, load_and_build
from src.evaluation.metrics import (
    ModelMetrics,
    compute_quantile_coverage,
    compute_regression_metrics,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Sabitler ─────────────────────────────────────────────────────────────────

MODEL_NAME = "quantile_lightgbm_v1"
_mlruns_path = ROOT / "mlruns"
_mlruns_path.mkdir(parents=True, exist_ok=True)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + str(_mlruns_path / "mlflow.db").replace("\\", "/"),
)
EXPERIMENT_NAME = "yks-taban-siralama"

ALPHA_LOWER = 0.030
ALPHA_UPPER = 0.970

ROLLING_FOLDS = [
    {"train": [2023], "test": 2024},
    {"train": [2023, 2024], "test": 2025},
]


def _filter_usable_rows(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    years: list[int],
    require_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Verilen yılları filtreler."""
    mask = meta["yil"].isin(years)
    X_f, y_f = X[mask].copy(), y[mask].copy()

    lag_mask = X_f["lag1_taban_siralama"].notna()
    X_f, y_f = X_f[lag_mask], y_f[lag_mask]

    if require_target:
        target_mask = y_f.notna()
        X_f, y_f = X_f[target_mask], y_f[target_mask]

    return X_f, y_f


def enforce_quantile_constraints(
    pred_median: np.ndarray,
    pred_lower: np.ndarray,
    pred_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quantile crossing önleme:
    lower <= median <= upper eşitsizliğini garanti eder.
    """
    pred_lower_clean = np.minimum(pred_lower, pred_median)
    pred_upper_clean = np.maximum(pred_upper, pred_median)
    # Negatif sıralama olamaz (min 1)
    pred_lower_clean = np.maximum(1.0, pred_lower_clean)
    pred_median_clean = np.maximum(1.0, pred_median)
    pred_upper_clean = np.maximum(1.0, pred_upper_clean)
    return pred_median_clean, pred_lower_clean, pred_upper_clean


def train_and_evaluate_quantile(
    csv_path: Path | None = None,
    log_to_mlflow: bool = True,
) -> tuple[list[ModelMetrics], dict]:
    """
    Quantile regression ile nokta tahmini + %80 belirsizlik aralığı üretir.

    Returns:
        fold_metrics: Her fold için metrikler
        results_summary: Q80 coverage ve ortalama aralık genişliği özeti
    """
    import lightgbm as lgb

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    X, y, meta = load_and_build() if csv_path is None else load_and_build(csv_path)
    feature_cols = [c for c in get_feature_columns() if c in X.columns]
    X = X[feature_cols]

    logger.info("Quantile Feature Matrix: %d satir x %d feature", len(X), len(feature_cols))

    fold_metrics: list[ModelMetrics] = []
    coverage_scores: list[float] = []
    interval_widths: list[float] = []

    for fold_idx, fold in enumerate(ROLLING_FOLDS):
        train_years = fold["train"]
        test_year = fold["test"]

        X_train, y_train = _filter_usable_rows(X, y, meta, train_years, require_target=True)
        X_test, y_test = _filter_usable_rows(X, y, meta, [test_year], require_target=False)

        if len(X_train) < 10:
            continue

        # LightGBM Quantile Modelleri
        # 1. Median (alpha = 0.50)
        model_med = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=150,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
        )
        model_med.fit(X_train, y_train)
        pred_med = model_med.predict(X_test)

        # 2. Lower bound (alpha = 0.10)
        model_low = lgb.LGBMRegressor(
            objective="quantile",
            alpha=ALPHA_LOWER,
            n_estimators=150,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
        )
        model_low.fit(X_train, y_train)
        pred_low = model_low.predict(X_test)

        # 3. Upper bound (alpha = 0.90)
        model_upp = lgb.LGBMRegressor(
            objective="quantile",
            alpha=ALPHA_UPPER,
            n_estimators=150,
            learning_rate=0.05,
            random_state=42,
            verbosity=-1,
        )
        model_upp.fit(X_train, y_train)
        pred_upp = model_upp.predict(X_test)

        # Post-process: quantile crossing fix
        pred_med, pred_low, pred_upp = enforce_quantile_constraints(pred_med, pred_low, pred_upp)

        # Test hedefleri ile metrik hesapla (null hariç)
        eval_mask = y_test.notna()
        y_true_valid = y_test[eval_mask].values
        p_med_valid = pred_med[eval_mask]
        p_low_valid = pred_low[eval_mask]
        p_upp_valid = pred_upp[eval_mask]

        reg_metrics = compute_regression_metrics(y_true_valid, p_med_valid)
        coverage = compute_quantile_coverage(y_true_valid, p_low_valid, p_upp_valid)
        mean_width = float(np.mean(p_upp_valid - p_low_valid))

        coverage_scores.append(coverage)
        interval_widths.append(mean_width)

        fm = ModelMetrics(
            model_name=MODEL_NAME,
            train_years=train_years,
            test_year=test_year,
            n_train=len(X_train),
            n_test=int(reg_metrics["n"]),
            mae=reg_metrics["mae"],
            rmse=reg_metrics["rmse"],
            r2=reg_metrics["r2"],
            quantile_coverage_80=coverage,
        )
        fold_metrics.append(fm)

        logger.info(
            "Fold %d (%d Test): MAE=%.0f | RMSE=%.0f | R²=%.3f | Q80 Coverage=%.1f%% | Mean Interval Width=%.0f",
            fold_idx + 1, test_year, reg_metrics["mae"], reg_metrics["rmse"],
            reg_metrics["r2"], coverage * 100, mean_width,
        )

    # Özet sonuçlar
    mean_mae = float(np.mean([fm.mae for fm in fold_metrics]))
    mean_rmse = float(np.mean([fm.rmse for fm in fold_metrics]))
    mean_cov = float(np.mean(coverage_scores))
    mean_width = float(np.mean(interval_widths))

    summary = {
        "mean_mae": mean_mae,
        "mean_rmse": mean_rmse,
        "mean_q80_coverage": mean_cov,
        "mean_interval_width": mean_width,
        "n_folds": len(fold_metrics),
    }

    logger.info(
        "\nQuantile Regression Ozet:\n  Mean MAE: %.0f\n  Mean RMSE: %.0f\n  Mean Q80 Coverage: %.1f%%\n  Mean Interval Width: %.0f",
        mean_mae, mean_rmse, mean_cov * 100, mean_width,
    )

    # MLflow Loglama
    if log_to_mlflow and fold_metrics:
        import mlflow

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"{MODEL_NAME}_rolling_backtest") as run:
            mlflow.log_params({
                "model_type": "LightGBM_Quantile",
                "alpha_lower": ALPHA_LOWER,
                "alpha_upper": ALPHA_UPPER,
                "n_estimators": 150,
                "learning_rate": 0.05,
            })
            mlflow.log_metrics({
                "mean_mae": mean_mae,
                "mean_rmse": mean_rmse,
                "mean_q80_coverage": mean_cov,
                "mean_interval_width": mean_width,
            })
            for i, fm in enumerate(fold_metrics):
                mlflow.log_metric(f"fold_{i+1}_mae", fm.mae)
                mlflow.log_metric(f"fold_{i+1}_q80_coverage", fm.quantile_coverage_80 or 0.0)

            logger.info("MLflow run_id: %s", run.info.run_id)

    return fold_metrics, summary


if __name__ == "__main__":
    fold_metrics, summary = train_and_evaluate_quantile(log_to_mlflow=True)
    print("\n" + "=" * 60)
    print(" QUANTILE REGRESSION (LIGHTGBM %80 GUVEN ARALIGI) SONUCLARI")
    print("=" * 60)
    for fm in fold_metrics:
        print(f"  Test {fm.test_year}: MAE={fm.mae:,.0f} | RMSE={fm.rmse:,.0f} | Q80 Coverage={fm.quantile_coverage_80:.1%}")
    print(f"\nOrtalama MAE             : {summary['mean_mae']:,.0f}")
    print(f"Ortalama RMSE            : {summary['mean_rmse']:,.0f}")
    print(f"Ortalama Q80 Coverage    : {summary['mean_q80_coverage']:.1%} (Hedef %80)")
    print(f"Ortalama Aralik Genisligi: {summary['mean_interval_width']:,.0f} sira")
