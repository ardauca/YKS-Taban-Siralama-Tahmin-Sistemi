"""
Unit testler: src/models/train_quantile.py ve train_catboost.py için.

Çalıştırma: pytest tests/test_models.py -v
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.train_quantile import enforce_quantile_constraints


class TestQuantileConstraints:
    def test_enforces_ordering(self):
        pred_med = np.array([100.0, 200.0, 300.0])
        pred_low = np.array([120.0, 150.0, 350.0])  # 120 > 100 and 350 > 300 (crossing)
        pred_upp = np.array([80.0, 250.0, 290.0])   # 80 < 100 and 290 < 300 (crossing)

        clean_med, clean_low, clean_upp = enforce_quantile_constraints(pred_med, pred_low, pred_upp)

        assert (clean_low <= clean_med).all()
        assert (clean_med <= clean_upp).all()
        assert (clean_low >= 1.0).all()

    def test_valid_inputs_unchanged(self):
        pred_med = np.array([100.0, 200.0])
        pred_low = np.array([80.0, 150.0])
        pred_upp = np.array([120.0, 250.0])

        clean_med, clean_low, clean_upp = enforce_quantile_constraints(pred_med, pred_low, pred_upp)

        np.testing.assert_array_almost_equal(clean_med, pred_med)
        np.testing.assert_array_almost_equal(clean_low, pred_low)
        np.testing.assert_array_almost_equal(clean_upp, pred_upp)
