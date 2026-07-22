"""
FastAPI Tahmin Servis Katmanı — YKS Taban Sıralama Tahmin Sistemi.

Endpoint'ler:
  GET  /health          → Servis sağlık ve versiyon durumu
  POST /api/v1/predict  → Taban sıralama nokta tahmini + %80 belirsizlik aralığı

Yanıt Yapısı (Zorunlu Sözleşme):
  - point_estimate    (nokta tahmin / medyan)
  - lower_bound       (alt sınır / %80 güven aralığı)
  - upper_bound       (üst sınır / %80 güven aralığı)
  - confidence_level  (0.80)
  - unit              ("siralama")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Proje root'u
ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(ROOT))

from src.features.build_features import (
    BURS_ORANI_ORDERED,
    OGRETIM_TURU_MAP,
    PUAN_TURU_MAP,
    UNIVERSITE_TURU_MAP,
    get_feature_columns,
)
from src.models.train_quantile import enforce_quantile_constraints

logger = logging.getLogger(__name__)

app = FastAPI(
    title="YKS Taban Sıralama Tahmin API",
    description="Bilgisayar Mühendisliği ve diğer YKS programları için taban sıralama nokta tahmini ve belirsizlik aralık servisi.",
    version="1.0.0",
)

# ── Pydantic Şemaları ─────────────────────────────────────────────────────────


class PredictionRequest(BaseModel):
    kilavuz_kodu: int = Field(..., example=203910363, description="ÖSYM / YÖK Program Kılavuz Kodu")
    universite_turu: str = Field("DEVLET", example="VAKIF", description="DEVLET, VAKIF, KKTC vb.")
    ogretim_turu: str = Field("Örgün", example="Örgün", description="Örgün, İkinci Öğretim vb.")
    burs_orani: str | None = Field(None, example="Burslu", description="Burslu, %50 İndirimli, Ücretli vb.")
    puan_turu: str = Field("SAY", example="SAY", description="SAY, EA, SÖZ, DİL")
    il_kodu: str = Field("34", example="34", description="İl Kodu")
    yil: int = Field(2025, example=2025, description="Tahmin Yapılan Yıl")

    # Lag-1 Metrikleri (Y-1)
    lag1_taban_siralama: float = Field(..., example=122.0, description="Geçen Yıl Taban Sıralaması")
    lag1_taban_puan: float = Field(..., example=550.88, description="Geçen Yıl Taban Puanı")
    lag1_genel_kontenjan: float = Field(..., example=18.0, description="Geçen Yıl Genel Kontenjanı")
    lag1_sehit_gazi_kontenjan: float | None = Field(0.0, example=2.0)
    lag1_depremzede_kontenjan: float | None = Field(0.0, example=1.0)
    lag1_okul_birincisi_kontenjan: float | None = Field(0.0, example=0.0)

    # Lag-2 & Trend
    lag2_taban_siralama: float | None = Field(None, example=98.0, description="2 Yıl Önceki Taban Sıralaması")
    lag2_kontenjan: float | None = Field(None, example=15.0)

    # Historical Derived
    program_hist_medyan_siralama: float | None = Field(None, example=110.0)
    univ_hist_medyan_siralama: float | None = Field(None, example=150.0)


class PredictionValues(BaseModel):
    point_estimate: int = Field(..., description="Medyan / En olası sıralama tahmini")
    lower_bound: int = Field(..., description="Alt Sınır (İyimser / En yüksek başarı sıralaması)")
    upper_bound: int = Field(..., description="Üst Sınır (Kötümser / En düşük başarı sıralaması)")
    confidence_level: float = Field(0.80, description="Güven Aralığı Seviyesi (%80)")
    unit: str = Field("siralama", description="Birim")


class MetadataInfo(BaseModel):
    model_version: str = Field("quantile_lightgbm_v1_optuna")
    timestamp: str = Field(..., description="ISO 8601 UTC Tarih/Saat")


class PredictionResponse(BaseModel):
    status: str = Field("success")
    kilavuz_kodu: int
    prediction: PredictionValues
    metadata: MetadataInfo


# ── Model Ön Yükleme / Tahmin Motoru ───────────────────────────────────────

_models_loaded = False
_model_med = None
_model_low = None
_model_upp = None


def _load_or_train_models():
    global _models_loaded, _lgb_med, _lgb_low, _lgb_upp, _cb_med, _cb_low, _cb_upp
    if _models_loaded:
        return

    import lightgbm as lgb
    from catboost import CatBoostRegressor
    from src.features.build_features import load_and_build

    X, y, meta = load_and_build()
    feature_cols = [c for c in get_feature_columns() if c in X.columns]
    X = X[feature_cols]

    mask = y.notna() & X["lag1_taban_siralama"].notna()
    X_train, y_train = X[mask], y[mask]

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

    logger.info("FastAPI: Hybrid LightGBM+CatBoost modelleri egitiliyor (train n=%d)...", len(X_train))

    _lgb_med = lgb.LGBMRegressor(objective="regression_l1", **lgb_params).fit(X_train, y_train)
    _lgb_low = lgb.LGBMRegressor(objective="quantile", alpha=0.030, **lgb_params).fit(X_train, y_train)
    _lgb_upp = lgb.LGBMRegressor(objective="quantile", alpha=0.970, **lgb_params).fit(X_train, y_train)

    _cb_med = CatBoostRegressor(loss_function="MAE", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_train, y_train)
    _cb_low = CatBoostRegressor(loss_function="Quantile:alpha=0.030", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_train, y_train)
    _cb_upp = CatBoostRegressor(loss_function="Quantile:alpha=0.970", iterations=300, learning_rate=0.04, depth=6, verbose=0, random_seed=42).fit(X_train, y_train)

    _models_loaded = True
    logger.info("FastAPI: Hybrid Ensemble Modelleri yuklendi!")


_lgb_med, _lgb_low, _lgb_upp = None, None, None
_cb_med, _cb_low, _cb_upp = None, None, None


# ── Endpoint'ler ─────────────────────────────────────────────────────────────


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Servis sağlık ve durum kontrolü."""
    return {
        "status": "healthy",
        "service": "YKS Taban Sıralama Tahmin API",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/v1/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict_rank(req: PredictionRequest):
    """
    Verilen program özellikleri için taban sıralama tahmini ve %80 belirsizlik aralığı üretir.
    """
    try:
        _load_or_train_models()

        # Enkodlamalar
        burs_enc = 0
        if req.burs_orani:
            for i, label in enumerate(BURS_ORANI_ORDERED):
                if label and label in str(req.burs_orani):
                    burs_enc = i
                    break

        il_num = float(req.il_kodu) if req.il_kodu.replace(".", "", 1).isdigit() else -1.0

        lag2_rank = req.lag2_taban_siralama if req.lag2_taban_siralama is not None else req.lag1_taban_siralama
        siralama_trend = req.lag1_taban_siralama - lag2_rank
        pct_change = (siralama_trend / lag2_rank) if lag2_rank > 0 else 0.0

        lag2_kont = req.lag2_kontenjan if req.lag2_kontenjan is not None else req.lag1_genel_kontenjan
        kont_degisim = (req.lag1_genel_kontenjan - lag2_kont) / lag2_kont if lag2_kont > 0 else 0.0

        prog_med = req.program_hist_medyan_siralama if req.program_hist_medyan_siralama is not None else req.lag1_taban_siralama
        univ_med = req.univ_hist_medyan_siralama if req.univ_hist_medyan_siralama is not None else req.lag1_taban_siralama

        kont_kat = 0 if req.lag1_genel_kontenjan <= 30 else (1 if req.lag1_genel_kontenjan <= 80 else 2)

        big_city_codes = {34, 6, 35}
        mid_city_codes = {16, 41, 7, 1}
        sehir_idx = 1.0 if il_num in big_city_codes else (0.5 if il_num in mid_city_codes else 0.0)

        feat_dict = {
            "lag1_taban_siralama": [req.lag1_taban_siralama],
            "lag1_taban_puan": [req.lag1_taban_puan],
            "lag1_genel_kontenjan": [req.lag1_genel_kontenjan],
            "lag1_sehit_gazi_kontenjan": [req.lag1_sehit_gazi_kontenjan or 0.0],
            "lag1_depremzede_kontenjan": [req.lag1_depremzede_kontenjan or 0.0],
            "lag1_okul_birincisi_kontenjan": [req.lag1_okul_birincisi_kontenjan or 0.0],
            "lag2_taban_siralama": [lag2_rank],
            "siralama_trend": [siralama_trend],
            "siralama_pct_change": [pct_change],
            "kontenjan_degisim_orani": [kont_degisim],
            "universite_turu_enc": [UNIVERSITE_TURU_MAP.get(req.universite_turu, 0)],
            "ogretim_turu_enc": [OGRETIM_TURU_MAP.get(req.ogretim_turu, 0)],
            "puan_turu_enc": [PUAN_TURU_MAP.get(req.puan_turu, 0)],
            "burs_enc": [burs_enc],
            "il_kodu_num": [il_num],
            "program_hist_medyan_siralama": [prog_med],
            "univ_hist_medyan_siralama": [univ_med],
            "kontenjan_kategori": [kont_kat],
            "univ_trend_momentum": [0.0],
            "sehir_tercih_indeksi": [sehir_idx],
            "kontenjan_farki_2026": [0.0],
            "macro_puan_turu_degisim_orani": [0.0],
            "macro_bolum_degisim_orani": [0.0],
            "kontenjan_sok_faktoru": [0.0],
            "yil": [req.yil],
        }

        df_feat = pd.DataFrame(feat_dict)
        feature_cols = get_feature_columns()
        df_feat = df_feat[[c for c in feature_cols if c in df_feat.columns]]

        # Hybrid Ensemble Tahminleri (50% LightGBM + 50% CatBoost)
        raw_med = 0.5 * _lgb_med.predict(df_feat)[0] + 0.5 * _cb_med.predict(df_feat)[0]
        raw_low = 0.5 * _lgb_low.predict(df_feat)[0] + 0.5 * _cb_low.predict(df_feat)[0]
        raw_upp = 0.5 * _lgb_upp.predict(df_feat)[0] + 0.5 * _cb_upp.predict(df_feat)[0]

        clean_med, clean_low, clean_upp = enforce_quantile_constraints(
            np.array([raw_med]), np.array([raw_low]), np.array([raw_upp])
        )

        return PredictionResponse(
            status="success",
            kilavuz_kodu=req.kilavuz_kodu,
            prediction=PredictionValues(
                point_estimate=int(round(clean_med[0])),
                lower_bound=int(round(clean_low[0])),
                upper_bound=int(round(clean_upp[0])),
                confidence_level=0.80,
                unit="siralama",
            ),
            metadata=MetadataInfo(
                model_version="quantile_lightgbm_v1_optuna",
                timestamp=datetime.now(timezone.utc).isoformat(),
            ),
        )

    except Exception as exc:
        logger.error("Tahmin hatasi: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tahmin uretilirken hata olustu: {str(exc)}",
        )
