"""
Trend Analizi ve Türkiye Geneli İstatistik Servisi.
Artık 2026 ML simülasyon verisini de kullanır.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import polars as pl

from services.search_service import get_master_df

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Trend ve istatistik analizi servisi."""

    @staticmethod
    def get_nationwide_stats() -> Dict[str, Any]:
        """Türkiye geneli makro istatistik özetini döndürür."""
        df = get_master_df()

        total_programs = len(df)
        total_universities = df["universite_adi"].n_unique()
        total_departments = df["birim_grup_adi"].n_unique() if "birim_grup_adi" in df.columns else 0

        mean_rank = float(df["lag1_taban_siralama"].mean()) if "lag1_taban_siralama" in df.columns else 0.0
        min_rank = float(df["lag1_taban_siralama"].min()) if "lag1_taban_siralama" in df.columns else 0.0
        max_rank = float(df["lag1_taban_siralama"].max()) if "lag1_taban_siralama" in df.columns else 0.0
        total_quota = float(df["lag1_genel_kontenjan"].sum()) if "lag1_genel_kontenjan" in df.columns else 0.0

        # ML tahmin istatistikleri
        sim_count = 0
        mean_pred = 0.0
        if "pred_2026" in df.columns:
            pred_valid = df.filter(pl.col("pred_2026") > 0)
            sim_count = len(pred_valid)
            mean_pred = float(pred_valid["pred_2026"].mean()) if sim_count > 0 else 0.0

        # Üniversite türü dağılımı
        uni_turu_counts: dict = {}
        if "universite_turu" in df.columns:
            try:
                counts = df.group_by("universite_turu").len().to_dicts()
                uni_turu_counts = {r["universite_turu"]: r.get("len", r.get("count", 0)) for r in counts if r.get("universite_turu")}
            except Exception:
                pass

        # Puan türü dağılımı
        pt_counts: dict = {}
        if "puan_turu" in df.columns:
            try:
                counts = df.group_by("puan_turu").len().to_dicts()
                pt_counts = {r["puan_turu"]: r.get("len", r.get("count", 0)) for r in counts if r.get("puan_turu")}
            except Exception:
                pass

        return {
            "total_programs": total_programs,
            "total_universities": total_universities,
            "total_departments": total_departments,
            "mean_rank": mean_rank,
            "min_rank": min_rank,
            "max_rank": max_rank,
            "total_quota": total_quota,
            "sim_program_count": sim_count,
            "mean_pred_2026": mean_pred,
            "uni_turu_counts": uni_turu_counts,
            "point_type_counts": pt_counts,
        }

    @staticmethod
    def get_top_risers(limit: int = 15) -> List[Dict[str, Any]]:
        """
        2026 ML tahminine göre en çok sıralaması iyileşen programlar.
        pred_degisim negatif = sıralama düştü = daha iyiye gitti.
        """
        df = get_master_df()
        sort_col = "pred_degisim" if "pred_degisim" in df.columns else "siralama_trend"
        if sort_col not in df.columns:
            return []
        return (
            df.filter(pl.col(sort_col).is_not_null() & (pl.col(sort_col) < 0))
            .sort(sort_col, descending=False)
            .head(limit)
            .to_dicts()
        )

    @staticmethod
    def get_top_decliners(limit: int = 15) -> List[Dict[str, Any]]:
        """2026 ML tahminine göre en çok gerileyen programlar."""
        df = get_master_df()
        sort_col = "pred_degisim" if "pred_degisim" in df.columns else "siralama_trend"
        if sort_col not in df.columns:
            return []
        return (
            df.filter(pl.col(sort_col).is_not_null() & (pl.col(sort_col) > 0))
            .sort(sort_col, descending=True)
            .head(limit)
            .to_dicts()
        )

    @staticmethod
    def get_most_stable(limit: int = 15) -> List[Dict[str, Any]]:
        """En kararlı / stabil sıralamaya sahip programlar."""
        df = get_master_df()
        sort_col = "pred_degisim" if "pred_degisim" in df.columns else "siralama_trend"
        if sort_col not in df.columns:
            return []
        return (
            df.filter(pl.col(sort_col).is_not_null())
            .with_columns(pl.col(sort_col).abs().alias("_abs_degisim"))
            .sort("_abs_degisim")
            .head(limit)
            .to_dicts()
        )

    @staticmethod
    def get_most_volatile(limit: int = 15) -> List[Dict[str, Any]]:
        """En yüksek dalgalanmaya sahip programlar."""
        df = get_master_df()
        sort_col = "pred_degisim" if "pred_degisim" in df.columns else "siralama_trend"
        if sort_col not in df.columns:
            return []
        return (
            df.filter(pl.col(sort_col).is_not_null())
            .with_columns(pl.col(sort_col).abs().alias("_abs_degisim"))
            .sort("_abs_degisim", descending=True)
            .head(limit)
            .to_dicts()
        )

    @staticmethod
    def get_university_summary(universite_adi: str) -> Dict[str, Any]:
        """Bir üniversitenin tüm programlarının özet istatistiklerini döndürür."""
        from services.search_service import SearchService
        programs = SearchService.get_programs_by_university(universite_adi)

        if not programs:
            return {"error": f"'{universite_adi}' ile eşleşen üniversite bulunamadı."}

        # programs listesinden Polars DataFrame oluştur
        import polars as pl
        uni_df = pl.from_dicts(programs)

        mean_2025 = float(uni_df["lag1_taban_siralama"].mean()) if "lag1_taban_siralama" in uni_df.columns else 0.0
        mean_pred = float(uni_df["pred_2026"].mean()) if "pred_2026" in uni_df.columns else 0.0
        total_quota = float(uni_df["lag1_genel_kontenjan"].sum()) if "lag1_genel_kontenjan" in uni_df.columns else 0.0

        # Risk dağılımı
        risk_dist: dict = {}
        if "risk_renk" in uni_df.columns:
            try:
                counts = uni_df.group_by("risk_renk").len().to_dicts()
                risk_dist = {r["risk_renk"]: r.get("len", 0) for r in counts}
            except Exception:
                pass

        return {
            "universite_adi": str(programs[0].get("universite_adi", universite_adi)),
            "total_programs": len(programs),
            "mean_rank_2025": mean_2025,
            "mean_pred_2026": mean_pred,
            "total_quota": total_quota,
            "risk_distribution": risk_dist,
            "programs": programs,
        }
