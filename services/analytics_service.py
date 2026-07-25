"""
Trend Analizi ve Türkiye Geneli İstatistik Servisi.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl

from services.search_service import get_polars_master_df

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Trend ve istatistik analizi servisi."""

    @staticmethod
    def get_nationwide_stats() -> Dict[str, Any]:
        """Türkiye geneli makro istatistik özetini döndürür."""
        df_pl = get_polars_master_df()
        
        total_programs = len(df_pl)
        total_universities = df_pl["universite_adi"].n_unique()
        total_departments = df_pl["birim_grup_adi"].n_unique()

        mean_rank = float(df_pl["lag1_taban_siralama"].mean()) if "lag1_taban_siralama" in df_pl.columns else 0.0
        min_rank = float(df_pl["lag1_taban_siralama"].min()) if "lag1_taban_siralama" in df_pl.columns else 0.0
        max_rank = float(df_pl["lag1_taban_siralama"].max()) if "lag1_taban_siralama" in df_pl.columns else 0.0

        total_quota = float(df_pl["lag1_genel_kontenjan"].sum()) if "lag1_genel_kontenjan" in df_pl.columns else 0.0

        # Puan türü dağılımı
        pt_counts = df_pl.group_by("puan_turu").count().to_dicts() if "puan_turu" in df_pl.columns else []

        return {
            "total_programs": total_programs,
            "total_universities": total_universities,
            "total_departments": total_departments,
            "mean_rank": mean_rank,
            "min_rank": min_rank,
            "max_rank": max_rank,
            "total_quota": total_quota,
            "point_type_counts": {r["puan_turu"]: r["count"] for r in pt_counts if r.get("puan_turu")},
        }

    @staticmethod
    def get_top_risers(limit: int = 15) -> List[Dict[str, Any]]:
        """Taban sıralaması en çok yükselen (başarı sırası iyileşen) programlar."""
        df_pl = get_polars_master_df()
        if "siralama_trend" not in df_pl.columns:
            return []
        
        # siralama_trend negatif ise sıralama yükselmiş (iyileşmiş) demektir (ör. 50k -> 30k = -20k)
        risers = df_pl.filter(pl.col("siralama_trend").not_null()).sort("siralama_trend", descending=False).head(limit)
        return risers.to_dicts()

    @staticmethod
    def get_top_decliners(limit: int = 15) -> List[Dict[str, Any]]:
        """Taban sıralaması en çok düşen (gerileyen) programlar."""
        df_pl = get_polars_master_df()
        if "siralama_trend" not in df_pl.columns:
            return []
        
        decliners = df_pl.filter(pl.col("siralama_trend").not_null()).sort("siralama_trend", descending=True).head(limit)
        return decliners.to_dicts()

    @staticmethod
    def get_most_stable(limit: int = 15) -> List[Dict[str, Any]]:
        """En kararlı / stabil sıralamaya sahip programlar (abs(siralama_trend) en küçük)."""
        df_pl = get_polars_master_df()
        if "siralama_trend" not in df_pl.columns:
            return []
        
        df_abs = df_pl.filter(pl.col("siralama_trend").not_null()).with_columns(
            pl.col("siralama_trend").abs().alias("abs_trend")
        ).sort("abs_trend", descending=False).head(limit)
        return df_abs.to_dicts()

    @staticmethod
    def get_most_volatile(limit: int = 15) -> List[Dict[str, Any]]:
        """En yüksek dalgalanma / oynaklığa sahip programlar."""
        df_pl = get_polars_master_df()
        if "siralama_trend" not in df_pl.columns:
            return []
        
        df_abs = df_pl.filter(pl.col("siralama_trend").not_null()).with_columns(
            pl.col("siralama_trend").abs().alias("abs_trend")
        ).sort("abs_trend", descending=True).head(limit)
        return df_abs.to_dicts()
