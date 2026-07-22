"""
Unit testler: src/features/build_features.py ve src/evaluation/metrics.py için.

Çalıştırma: pytest tests/test_features.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_features import (
    _encode_burs,
    build_lag_features,
    build_static_features,
    build_derived_features,
    build_feature_matrix,
    get_feature_columns,
    UNIVERSITE_TURU_MAP,
)
from src.evaluation.metrics import (
    compute_regression_metrics,
    compute_quantile_coverage,
    rolling_backtest_metrics,
    ModelMetrics,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _make_raw_df(n_programs: int = 10, years: list[int] | None = None) -> pd.DataFrame:
    """Test için minimal ham veri üretir."""
    if years is None:
        years = [2022, 2023, 2024, 2025]
    rows = []
    for i in range(n_programs):
        base_rank = 50000 + i * 10000
        for j, year in enumerate(years):
            rows.append({
                "kilavuz_kodu": 100000 + i,
                "universite_adi": f"Uni {i % 3}",
                "birim_adi": "Bilgisayar Mühendisliği",
                "birim_grup_adi": "Bilgisayar Mühendisliği",
                "il_kodu": str(float(6 + i % 5)),
                "il_adi": ["Ankara", "İstanbul", "İzmir", "Bursa", "Antalya"][i % 5],
                "universite_turu": ["DEVLET", "VAKIF"][i % 2],
                "ogretim_turu": "Örgün",
                "burs_orani": [None, "Burslu", "%50 İndirimli"][i % 3],
                "puan_turu": "SAY",
                "kaynak": "yokatlas_api",
                "cekme_tarihi": "2025-07-22",
                "yil": year,
                "genel_kontenjan": 50 + i * 5,
                "sehit_gazi_kontenjan": None,
                "depremzede_kontenjan": None,
                "okul_birincisi_kontenjan": 2,
                "taban_puan": 400.0 + i * 10 + j * 2,
                "taban_siralama": float(base_rank - j * 1000),
            })
    return pd.DataFrame(rows)


# ── Test: _encode_burs ────────────────────────────────────────────────────────

class TestEncodeBurs:
    def test_none_returns_zero(self):
        assert _encode_burs(None) == 0

    def test_burslu_returns_five(self):
        assert _encode_burs("Burslu") == 5

    def test_yuzde50_returns_three(self):
        assert _encode_burs("%50 İndirimli") == 3

    def test_ucretli_returns_one(self):
        assert _encode_burs("Ücretli") == 1

    def test_unknown_returns_zero(self):
        assert _encode_burs("Bilinmeyen") == 0


# ── Test: build_lag_features ──────────────────────────────────────────────────

class TestBuildLagFeatures:
    def test_lag1_correct_value(self):
        df = _make_raw_df(n_programs=3, years=[2022, 2023, 2024])
        result = build_lag_features(df)
        # Program 0, 2023 yılı: lag1_taban_siralama = 2022 değeri
        prog0_2023 = result[(result["kilavuz_kodu"] == 100000) & (result["yil"] == 2023)]
        prog0_2022 = df[(df["kilavuz_kodu"] == 100000) & (df["yil"] == 2022)]
        assert not prog0_2023.empty
        assert prog0_2023["lag1_taban_siralama"].iloc[0] == prog0_2022["taban_siralama"].iloc[0]

    def test_lag1_first_year_is_nan(self):
        df = _make_raw_df(n_programs=3, years=[2022, 2023, 2024])
        result = build_lag_features(df)
        first_year = result[result["yil"] == 2022]
        assert first_year["lag1_taban_siralama"].isna().all()

    def test_lag2_second_year_is_nan(self):
        df = _make_raw_df(n_programs=3, years=[2022, 2023, 2024])
        result = build_lag_features(df)
        second_year = result[result["yil"] == 2023]
        assert second_year["lag2_taban_siralama"].isna().all()

    def test_siralama_trend_computed(self):
        df = _make_raw_df(n_programs=2, years=[2022, 2023, 2024])
        result = build_lag_features(df)
        year2024 = result[result["yil"] == 2024]
        # Trend = lag1 - lag2, her ikisi de mevcut olmalı
        assert year2024["siralama_trend"].notna().any()

    def test_kontenjan_degisim_orani_computed(self):
        df = _make_raw_df(n_programs=2, years=[2022, 2023, 2024])
        result = build_lag_features(df)
        year2024 = result[result["yil"] == 2024]
        assert "kontenjan_degisim_orani" in result.columns
        assert year2024["kontenjan_degisim_orani"].notna().any()

    def test_no_cross_program_leakage(self):
        """Farklı programların lag değerleri birbirine karışmamalı."""
        df = _make_raw_df(n_programs=5, years=[2022, 2023])
        result = build_lag_features(df)
        for kod in result["kilavuz_kodu"].unique():
            prog = result[result["kilavuz_kodu"] == kod].sort_values("yil")
            if len(prog) >= 2:
                yr2022_rank = prog[prog["yil"] == 2022]["taban_siralama"].iloc[0]
                yr2023_lag1 = prog[prog["yil"] == 2023]["lag1_taban_siralama"].iloc[0]
                assert yr2022_rank == yr2023_lag1


# ── Test: build_static_features ──────────────────────────────────────────────

class TestBuildStaticFeatures:
    def test_universite_turu_encoded(self):
        df = _make_raw_df(n_programs=4, years=[2023])
        result = build_static_features(df)
        assert "universite_turu_enc" in result.columns
        devlet_rows = result[result["universite_turu"] == "DEVLET"]
        assert (devlet_rows["universite_turu_enc"] == 0).all()
        vakif_rows = result[result["universite_turu"] == "VAKIF"]
        assert (vakif_rows["universite_turu_enc"] == 1).all()

    def test_burs_enc_column_exists(self):
        df = _make_raw_df(n_programs=3, years=[2023])
        result = build_static_features(df)
        assert "burs_enc" in result.columns

    def test_il_kodu_num_numeric(self):
        df = _make_raw_df(n_programs=3, years=[2023])
        result = build_static_features(df)
        assert pd.api.types.is_integer_dtype(result["il_kodu_num"])

    def test_puan_turu_enc_say_is_zero(self):
        df = _make_raw_df(n_programs=3, years=[2023])
        result = build_static_features(df)
        say_rows = result[result["puan_turu"] == "SAY"]
        assert (say_rows["puan_turu_enc"] == 0).all()


# ── Test: build_derived_features ─────────────────────────────────────────────

class TestBuildDerivedFeatures:
    def _prep(self):
        df = _make_raw_df(n_programs=5, years=[2022, 2023, 2024])
        df = build_lag_features(df)
        df = build_static_features(df)
        return df

    def test_hist_medyan_siralama_exists(self):
        df = self._prep()
        result = build_derived_features(df)
        assert "program_hist_medyan_siralama" in result.columns

    def test_univ_hist_medyan_exists(self):
        df = self._prep()
        result = build_derived_features(df)
        assert "univ_hist_medyan_siralama" in result.columns

    def test_kontenjan_kategori_valid_values(self):
        df = self._prep()
        result = build_derived_features(df)
        valid = {-1, 0, 1, 2}
        assert set(result["kontenjan_kategori"].unique()).issubset(valid)


# ── Test: build_feature_matrix ────────────────────────────────────────────────

class TestBuildFeatureMatrix:
    def test_returns_three_outputs(self):
        df = _make_raw_df(n_programs=5, years=[2022, 2023, 2024])
        X, y, meta = build_feature_matrix(df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert isinstance(meta, pd.DataFrame)

    def test_feature_columns_subset_of_expected(self):
        df = _make_raw_df(n_programs=5, years=[2022, 2023, 2024])
        X, _, _ = build_feature_matrix(df)
        expected = set(get_feature_columns())
        assert set(X.columns).issubset(expected)

    def test_meta_has_kilavuz_kodu_yil(self):
        df = _make_raw_df(n_programs=5, years=[2022, 2023, 2024])
        _, _, meta = build_feature_matrix(df)
        assert "kilavuz_kodu" in meta.columns
        assert "yil" in meta.columns

    def test_row_count_preserved(self):
        df = _make_raw_df(n_programs=5, years=[2022, 2023, 2024])
        X, y, meta = build_feature_matrix(df)
        assert len(X) == len(y) == len(meta) == len(df)


# ── Test: compute_regression_metrics ─────────────────────────────────────────

class TestComputeRegressionMetrics:
    def test_perfect_predictions(self):
        y = np.array([100, 200, 300, 400], dtype=float)
        result = compute_regression_metrics(y, y)
        assert result["mae"] == pytest.approx(0.0)
        assert result["rmse"] == pytest.approx(0.0)
        assert result["r2"] == pytest.approx(1.0)

    def test_nan_filtered(self):
        y_true = np.array([100, np.nan, 300])
        y_pred = np.array([110, 200, 290])
        result = compute_regression_metrics(y_true, y_pred)
        assert result["n"] == 2  # NaN satırı atıldı

    def test_mae_positive(self):
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        result = compute_regression_metrics(y_true, y_pred)
        assert result["mae"] > 0
        assert result["rmse"] >= result["mae"]  # RMSE ≥ MAE

    def test_empty_returns_nan(self):
        result = compute_regression_metrics(np.array([]), np.array([]))
        assert np.isnan(result["mae"])


# ── Test: compute_quantile_coverage ──────────────────────────────────────────

class TestComputeQuantileCoverage:
    def test_perfect_coverage(self):
        y = np.array([100, 200, 300])
        lower = np.array([50, 150, 250])
        upper = np.array([150, 250, 350])
        assert compute_quantile_coverage(y, lower, upper) == pytest.approx(1.0)

    def test_zero_coverage(self):
        y = np.array([100, 200, 300])
        lower = np.array([200, 300, 400])
        upper = np.array([300, 400, 500])
        assert compute_quantile_coverage(y, lower, upper) == pytest.approx(0.0)

    def test_partial_coverage(self):
        y = np.array([100, 200, 300, 400])
        lower = np.array([50, 50, 50, 50])
        upper = np.array([150, 150, 150, 150])
        # Sadece 100 aralıkta
        assert compute_quantile_coverage(y, lower, upper) == pytest.approx(0.25)


# ── Test: rolling_backtest_metrics ────────────────────────────────────────────

class TestRollingBacktestMetrics:
    def _make_fm(self, mae, rmse, test_year=2023):
        return ModelMetrics(
            model_name="test", train_years=[2022], test_year=test_year,
            n_train=100, n_test=50, mae=mae, rmse=rmse,
        )

    def test_mean_mae(self):
        metrics = [self._make_fm(1000, 1200), self._make_fm(2000, 2400)]
        result = rolling_backtest_metrics(metrics)
        assert result["mean_mae"] == pytest.approx(1500.0)
        assert result["mean_rmse"] == pytest.approx(1800.0)

    def test_empty_returns_empty(self):
        result = rolling_backtest_metrics([])
        assert result == {}

    def test_n_folds_correct(self):
        metrics = [self._make_fm(1000, 1200), self._make_fm(2000, 2400), self._make_fm(1500, 1800)]
        result = rolling_backtest_metrics(metrics)
        assert result["n_folds"] == 3
