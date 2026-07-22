"""
CatBoost regresyon prototipi — YKS Taban Sıralama Tahmini.

Mimari (3 aşamalı sistemin Aşama 1'i):
  Bu modül: Talep/Sıralama Tahmini → regresyon

Rolling backtest stratejisi:
  Fold 1: train=[2022]       → test=2023
  Fold 2: train=[2022,2023]  → test=2024
  Fold 3: train=[2022-2024]  → test=2025 (prediction — 2025 null ise sadece tahmin)

Her deney MLflow'a loglanır. Log olmayan deney "yapılmamış" sayılır (Bölüm 4, madde 4).
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
    compute_regression_metrics,
    rolling_backtest_metrics,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Sabitler ─────────────────────────────────────────────────────────────────

MODEL_NAME = "catboost_v1_baseline"
_mlruns_path = ROOT / "mlruns"
_mlruns_path.mkdir(parents=True, exist_ok=True)
# MLflow filesystem backend deprecated — SQLite kullanıyoruz (hafif, local)
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "sqlite:///" + str(_mlruns_path / "mlflow.db").replace("\\", "/"),
)
EXPERIMENT_NAME = "yks-taban-siralama"

# CatBoost hiperparametreleri (başlangıç — Optuna ile ileride optimize edilecek)
CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "loss_function": "RMSE",
    "eval_metric": "MAE",
    "random_seed": 42,
    "verbose": 0,
    "allow_writing_files": False,
}

# Fold tasarımı:
# Lag feature'lar bir önceki yıl gerektirir.
# 2022 verisi lag1 üretemez (2021 yok) → eğitilebilir ilk satırlar 2023+
# Fold 1: train=2023 (lag1=2022), test=2024
# Fold 2: train=2023+2024 (lag1=2022/2023), test=2025
# Not: Az fold ama veri kısıtı gereği — gelecekte daha eski yıllar eklendikçe artacak
ROLLING_FOLDS = [
    {"train": [2023], "test": 2024},
    {"train": [2023, 2024], "test": 2025},
]


# ── Yardımcı fonksiyonlar ─────────────────────────────────────────────────────


def _filter_usable_rows(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    years: list[int],
    require_target: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Verilen yılları filtreler.
    require_target=True → hedef null olan satırları da atar (eğitim için).
    require_target=False → null hedefli satırları tutar (prediction için).
    """
    mask = meta["yil"].isin(years)
    X_f, y_f = X[mask].copy(), y[mask].copy()

    # En az lag1 gerekli — lag1_taban_siralama null ise satır kullanılamaz
    lag_mask = X_f["lag1_taban_siralama"].notna()
    X_f, y_f = X_f[lag_mask], y_f[lag_mask]

    if require_target:
        target_mask = y_f.notna()
        X_f, y_f = X_f[target_mask], y_f[target_mask]

    return X_f, y_f


def _compute_shap(model, X_sample: pd.DataFrame) -> list[tuple[str, float]]:
    """CatBoost SHAP feature importance (mean |SHAP|)."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        mean_abs = np.abs(shap_values).mean(axis=0)
        pairs = sorted(
            zip(X_sample.columns, mean_abs),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(str(f), float(v)) for f, v in pairs]
    except Exception as exc:
        logger.warning("SHAP hesaplanamadı: %s", exc)
        return []


def _log_to_mlflow(
    run_name: str,
    params: dict,
    metrics: dict,
    model,
    feature_cols: list[str],
    fold_metrics: list[ModelMetrics],
    shap_top: list[tuple[str, float]],
) -> str:
    """MLflow'a deney loglar, run_id döner."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_params({"feature_count": len(feature_cols)})
        mlflow.log_metrics(metrics)

        # Her fold'un metriklerini de logla
        for i, fm in enumerate(fold_metrics):
            mlflow.log_metric(f"fold_{i+1}_mae", fm.mae)
            mlflow.log_metric(f"fold_{i+1}_rmse", fm.rmse)
            mlflow.log_metric(f"fold_{i+1}_test_year", fm.test_year)
            mlflow.log_metric(f"fold_{i+1}_n_test", fm.n_test)

        # SHAP top-10
        for rank, (feat, val) in enumerate(shap_top[:10], 1):
            mlflow.log_metric(f"shap_rank_{rank:02d}_{feat[:30]}", val)

        # Feature listesini artifact olarak kaydet
        feat_txt = "\n".join(f"{i+1}. {f}" for i, f in enumerate(feature_cols))
        mlflow.log_text(feat_txt, "features.txt")

        # Modeli kaydet
        try:
            mlflow.catboost.log_model(model, "model")
        except Exception:
            logger.warning("MLflow CatBoost model log başarısız (catboost mlflow integration eksik)")

        run_id = run.info.run_id
        logger.info("MLflow run loglandı: %s (run_id=%s)", run_name, run_id)
        return run_id


# ── Ana eğitim fonksiyonu ─────────────────────────────────────────────────────


def train_and_evaluate(
    csv_path: Path | None = None,
    log_to_mlflow: bool = True,
) -> tuple[list[ModelMetrics], object, list[tuple[str, float]]]:
    """
    Rolling backtest ile CatBoost modelini eğitir ve değerlendirir.

    Returns:
        fold_metrics: Her fold için ModelMetrics listesi
        final_model: Son fold'un (tüm veri üzerinde) eğitilmiş modeli
        shap_top: SHAP feature önem sıralaması
    """
    from catboost import CatBoostRegressor

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Veriyi yükle
    X, y, meta = load_and_build() if csv_path is None else load_and_build(csv_path)
    feature_cols = [c for c in get_feature_columns() if c in X.columns]
    X = X[feature_cols]

    logger.info("Feature matrix: %d satir x %d feature", len(X), len(feature_cols))

    fold_metrics: list[ModelMetrics] = []
    final_model = None
    shap_top: list[tuple[str, float]] = []

    # ── Rolling backtest ────────────────────────────────────────────────────
    for fold_idx, fold in enumerate(ROLLING_FOLDS):
        train_years = fold["train"]
        test_year = fold["test"]

        X_train, y_train = _filter_usable_rows(X, y, meta, train_years, require_target=True)
        X_test, y_test = _filter_usable_rows(X, y, meta, [test_year], require_target=False)

        logger.info(
            "Fold %d: train=%s (n=%d) → test=%d (n=%d, target_null=%d)",
            fold_idx + 1, train_years, len(X_train),
            test_year, len(X_test), y_test.isna().sum(),
        )

        if len(X_train) < 10:
            logger.warning("Fold %d: yetersiz eğitim verisi, atlanıyor.", fold_idx + 1)
            continue

        # Modeli eğit
        model = CatBoostRegressor(**CATBOOST_PARAMS)
        # eval_set: sadece hedefi null olmayan test satırları
        eval_mask = y_test.notna()
        eval_set = (X_test[eval_mask], y_test[eval_mask]) if eval_mask.any() else None
        model.fit(X_train, y_train, eval_set=eval_set)

        # Tahmin
        y_pred = model.predict(X_test)

        # Metrikler (sadece null olmayan test hedefleri)
        raw = compute_regression_metrics(y_test, y_pred)
        fm = ModelMetrics(
            model_name=MODEL_NAME,
            train_years=train_years,
            test_year=test_year,
            n_train=len(X_train),
            n_test=int(raw["n"]),
            mae=raw["mae"],
            rmse=raw["rmse"],
            mae_pct=raw.get("mae_pct", float("nan")),
            r2=raw["r2"],
        )
        fold_metrics.append(fm)
        final_model = model

        # Son fold'dan SHAP hesapla (en fazla 200 satır)
        if fold_idx == len(ROLLING_FOLDS) - 1 or fold_idx == len(ROLLING_FOLDS) - 2:
            sample = X_train.sample(min(200, len(X_train)), random_state=42)
            shap_top = _compute_shap(model, sample)
            fm.top_features = shap_top[:10]

        logger.info("\n%s", fm.summary())

    # ── Rolling backtest özet ───────────────────────────────────────────────
    agg = rolling_backtest_metrics(fold_metrics)
    logger.info(
        "Rolling backtest ozet: mean_MAE=%.0f | mean_RMSE=%.0f | folds=%d",
        agg.get("mean_mae", float("nan")),
        agg.get("mean_rmse", float("nan")),
        agg.get("n_folds", 0),
    )

    # ── MLflow loglama ──────────────────────────────────────────────────────
    if log_to_mlflow and fold_metrics and final_model is not None:
        valid_folds = [fm for fm in fold_metrics if not np.isnan(fm.mae)]
        agg_mae = float(np.mean([fm.mae for fm in valid_folds])) if valid_folds else float("nan")
        agg_rmse = float(np.mean([fm.rmse for fm in valid_folds])) if valid_folds else float("nan")

        run_id = _log_to_mlflow(
            run_name=f"{MODEL_NAME}_rolling_backtest",
            params={**CATBOOST_PARAMS, "model_version": "v1", "bolum": "bilgisayar_muh"},
            metrics={
                "mean_mae": agg_mae,
                "mean_rmse": agg_rmse,
                "n_folds": len(valid_folds),
            },
            model=final_model,
            feature_cols=feature_cols,
            fold_metrics=fold_metrics,
            shap_top=shap_top,
        )
        logger.info("MLflow run_id: %s", run_id)
        logger.info("MLflow UI icin: mlflow ui --backend-store-uri %s", MLFLOW_TRACKING_URI)

    return fold_metrics, final_model, shap_top


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fold_metrics, model, shap_top = train_and_evaluate(log_to_mlflow=True)

    print("\n" + "=" * 60)
    print(" ROLLING BACKTEST SONUCLARI")
    print("=" * 60)
    for fm in fold_metrics:
        print(f"  Test {fm.test_year}: MAE={fm.mae:,.0f} | RMSE={fm.rmse:,.0f} | R²={fm.r2:.3f} | n={fm.n_test}")

    print("\nSHAP Top-10 Feature:")
    for rank, (feat, val) in enumerate(shap_top[:10], 1):
        bar = "#" * int(val / max(v for _, v in shap_top[:1]) * 30 + 1) if shap_top else ""
        print(f"  {rank:2d}. {feat:45s} {val:8.1f}  {bar}")

    agg = rolling_backtest_metrics(fold_metrics)
    print(f"\nOrtalama MAE  : {agg.get('mean_mae', float('nan')):,.0f}")
    print(f"Ortalama RMSE : {agg.get('mean_rmse', float('nan')):,.0f}")
