"""
Unit testler: api/main.py (FastAPI servisi için).

Çalıştırma: pytest tests/test_api.py -v
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)


def test_health_check():
    """/health endpoint kontrolü."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "YKS" in data["service"]


def test_predict_rank_success():
    """/api/v1/predict endpoint tahmini kontrolü."""
    payload = {
        "kilavuz_kodu": 203910363,
        "universite_turu": "VAKIF",
        "ogretim_turu": "Örgün",
        "burs_orani": "Burslu",
        "puan_turu": "SAY",
        "il_kodu": "34",
        "yil": 2025,
        "lag1_taban_siralama": 122.0,
        "lag1_taban_puan": 550.88,
        "lag1_genel_kontenjan": 18.0,
        "lag2_taban_siralama": 98.0,
        "program_hist_medyan_siralama": 110.0,
        "univ_hist_medyan_siralama": 150.0,
    }

    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "success"
    assert data["kilavuz_kodu"] == 203910363

    pred = data["prediction"]
    assert "point_estimate" in pred
    assert "lower_bound" in pred
    assert "upper_bound" in pred
    assert pred["confidence_level"] == 0.80
    assert pred["unit"] == "siralama"

    # Quantile constraints: lower <= point <= upper
    assert pred["lower_bound"] <= pred["point_estimate"]
    assert pred["point_estimate"] <= pred["upper_bound"]
    assert pred["lower_bound"] >= 1
