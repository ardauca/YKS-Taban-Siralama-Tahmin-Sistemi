"""
Polars Destekli Yüksek Performanslı Çoklu-Filtreleme Arama Motoru.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import polars as pl

from src.features.build_features import load_and_build

logger = logging.getLogger(__name__)

# Polars LazyFrame Singleton Önbelleği
_search_df_cache: Optional[pl.DataFrame] = None


def get_polars_master_df() -> pl.DataFrame:
    """Master veri kümesini Polars DataFrame olarak döndürür ve bellek içi önbelleğe alır."""
    global _search_df_cache
    if _search_df_cache is not None:
        return _search_df_cache

    logger.info("SearchService: Master veri kümesi Polars hafızasına yükleniyor...")
    X, y, meta = load_and_build()

    # Master birleşik DataFrame oluştur
    df_combined = pd.concat([meta, X, y.rename("taban_siralama")], axis=1)
    
    # Tekrarlayan sütunları temizle
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]

    # 2025 yılı verisine odaklan (Test yılı)
    df_2025 = df_combined[df_combined["yil"] == 2025].copy()

    # Eksik değerleri temizle / varsayılan atamalar
    if "universite_adi" not in df_2025.columns:
        df_2025["universite_adi"] = "Bilinmiyor"
    
    if "birim_grup_adi" not in df_2025.columns or (df_2025["birim_grup_adi"] == "Bilinmiyor").all():
        if "birim_adi" in df_2025.columns:
            df_2025["birim_grup_adi"] = df_2025["birim_adi"]
        else:
            df_2025["birim_grup_adi"] = "Bilinmiyor"

    # Polars DataFrame'e dönüştür
    _search_df_cache = pl.from_pandas(df_2025)
    logger.info("SearchService: Polars master DataFrame hazır! Total n=%d", len(_search_df_cache))
    return _search_df_cache


class SearchService:
    """Polars reaktif çoklu-filtreleme arama servisi."""

    @staticmethod
    def search_programs(
        universite_adi: Optional[str] = None,
        birim_grup_adi: Optional[str] = None,
        il_adi: Optional[str] = None,
        puan_turu: Optional[str] = None,
        universite_turu: Optional[str] = None,  # DEVLET, VAKIF, KKTC
        min_rank: Optional[float] = None,
        max_rank: Optional[float] = None,
        min_quota: Optional[float] = None,
        max_quota: Optional[float] = None,
        search_query: Optional[str] = None,  # Hızlı arama kelimesi
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Çoklu filtreleri Polars üstünde anlık çalıştırıp sonuçları liste olarak döndürür."""
        df_pl = get_polars_master_df()
        
        # Filtreleme zinciri (expressions)
        exprs = []

        if search_query and search_query.strip():
            q = search_query.strip()
            exprs.append(
                pl.col("universite_adi").str.contains(f"(?i){q}")
                | pl.col("birim_grup_adi").str.contains(f"(?i){q}")
            )

        if universite_adi and universite_adi.strip():
            u = universite_adi.strip()
            exprs.append(pl.col("universite_adi").str.contains(f"(?i){u}"))

        if birim_grup_adi and birim_grup_adi.strip():
            b = birim_grup_adi.strip()
            exprs.append(pl.col("birim_grup_adi").str.contains(f"(?i){b}"))

        if il_adi and il_adi.strip():
            i = il_adi.strip()
            if "il_adi" in df_pl.columns:
                exprs.append(pl.col("il_adi").str.contains(f"(?i){i}"))

        if puan_turu and puan_turu.strip() and puan_turu.upper() != "TÜMÜ":
            pt = puan_turu.strip().upper()
            if "puan_turu" in df_pl.columns:
                exprs.append(pl.col("puan_turu") == pt)

        if min_rank is not None and min_rank > 0:
            exprs.append(pl.col("lag1_taban_siralama") >= min_rank)

        if max_rank is not None and max_rank > 0:
            exprs.append(pl.col("lag1_taban_siralama") <= max_rank)

        if min_quota is not None and min_quota > 0:
            exprs.append(pl.col("lag1_genel_kontenjan") >= min_quota)

        if max_quota is not None and max_quota > 0:
            exprs.append(pl.col("lag1_genel_kontenjan") <= max_quota)

        # Filtreleri uygula
        filtered_df = df_pl
        for e in exprs:
            filtered_df = filtered_df.filter(e)

        # Sıralama sütununa göre küçükten büyüğe sırala
        filtered_df = filtered_df.sort("lag1_taban_siralama", descending=False).head(limit)

        results = filtered_df.to_dicts()
        return results

    @staticmethod
    def get_program_by_code(kilavuz_kodu: int) -> Optional[Dict[str, Any]]:
        """Kılavuz koduna göre tek bir program detayını döndürür."""
        df_pl = get_polars_master_df()
        res = df_pl.filter(pl.col("kilavuz_kodu") == kilavuz_kodu)
        if len(res) > 0:
            return res.to_dicts()[0]
        return None
