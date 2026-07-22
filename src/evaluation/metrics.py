"""
Model değerlendirme metrikleri.

Her model varyantı için zorunlu rapor (Bölüm 8):
- MAE / RMSE (nokta tahmin)
- Quantile coverage (%80 aralık gerçekten %80'i kapsıyor mu)
- Rolling backtest sonucu (en az 3 yıl)
- SHAP ile en etkili 10 feature
- Önceki en iyi modelle kıyas
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Tek bir model değerlendirmesinin sonuçları."""
    model_name: str
    train_years: list[int]
    test_year: int
    n_train: int
    n_test: int

    mae: float = float("nan")
    rmse: float = float("nan")
    mae_pct: float = float("nan")   # MAE / medyan sıralama
    r2: float = float("nan")

    # Quantile coverage (quantile model için)
    quantile_coverage_80: float | None = None
    quantile_lower_q: float = 0.10
    quantile_upper_q: float = 0.90

    # Top feature'lar (SHAP)
    top_features: list[tuple[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"{'='*50}",
            f" Model: {self.model_name}",
            f" Eğitim: {self.train_years} → Test: {self.test_year}",
            f" n_train={self.n_train}, n_test={self.n_test}",
            f"{'='*50}",
            f" MAE          : {self.mae:,.0f}",
            f" RMSE         : {self.rmse:,.0f}",
            f" MAE %        : {self.mae_pct:.1%}",
            f" R²           : {self.r2:.3f}",
        ]
        if self.quantile_coverage_80 is not None:
            lines.append(f" Q80 coverage : {self.quantile_coverage_80:.1%} (hedef: %80)")
        if self.top_features:
            lines.append(" Top-10 feature (SHAP):")
            for rank, (feat, imp) in enumerate(self.top_features[:10], 1):
                lines.append(f"   {rank:2d}. {feat:40s} {imp:.1f}")
        lines.append(f"{'='*50}")
        return "\n".join(lines)


def compute_regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float]:
    """MAE, RMSE, R² hesaplar. NaN'ları otomatik filtreler."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "r2": float("nan"), "n": 0}

    residuals = y_true - y_pred
    mae = float(np.abs(residuals).mean())
    rmse = float(np.sqrt((residuals ** 2).mean()))
    ss_res = float((residuals ** 2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae_pct = mae / np.median(y_true) if np.median(y_true) > 0 else float("nan")

    return {"mae": mae, "rmse": rmse, "r2": r2, "mae_pct": mae_pct, "n": len(y_true)}


def rolling_backtest_metrics(
    all_metrics: list[ModelMetrics],
) -> dict[str, float]:
    """Birden fazla backtest katmanının ağırlıklı ortalamasını hesaplar."""
    if not all_metrics:
        return {}
    maes = [m.mae for m in all_metrics if not np.isnan(m.mae)]
    rmses = [m.rmse for m in all_metrics if not np.isnan(m.rmse)]
    return {
        "mean_mae": float(np.mean(maes)) if maes else float("nan"),
        "mean_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "n_folds": len(all_metrics),
    }


def compute_quantile_coverage(
    y_true: np.ndarray | pd.Series,
    y_lower: np.ndarray | pd.Series,
    y_upper: np.ndarray | pd.Series,
) -> float:
    """Gerçek değerlerin [lower, upper] aralığında kalma oranı."""
    mask = ~np.isnan(np.asarray(y_true, dtype=float))
    y_true = np.asarray(y_true, dtype=float)[mask]
    y_lower = np.asarray(y_lower, dtype=float)[mask]
    y_upper = np.asarray(y_upper, dtype=float)[mask]
    if len(y_true) == 0:
        return float("nan")
    return float(((y_true >= y_lower) & (y_true <= y_upper)).mean())
