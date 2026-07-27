"""
Polars Destekli Yüksek Performanslı Çoklu-Filtreleme Arama Motoru.

Veri Mimarisi — 3 kaynak JOIN:
  1. Feature Matrix (build_features.py)    → ML özellikleri + 2025 sıralama
  2. Ham CSV (yokatlas_all_departments_raw.csv) → il_adi, puan_turu, universite_turu,
                                                    ogretim_turu (string)
  3. 2026 Simülasyon CSV                   → CatBoost ML tahminleri + güven aralığı

Türkçe Karakter Sorununa Çözüm:
  Polars'ın (?i) flag'i Türkçe özgü karakterleri (İ/i, I/ı, Ğ/ğ vb.) unicode
  düzeyinde desteklemiyor. Çözüm: build aşamasında `_search_col` adında
  ASCII-normalize edilmiş arama kolonu oluşturuluyor; tüm metin filtreleri
  bu kolonda çalışıyor.
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

# ── Dosya Yolları ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
RAW_CSV = _ROOT / "data" / "raw" / "yokatlas" / "yokatlas_all_departments_raw.csv"
SIMULASYON_CSV = _ROOT / "data" / "processed" / "simulasyon_2026_tahminleri.csv"

# Bellek içi önbellek
_master_df_cache: Optional[pl.DataFrame] = None

# ── Türkçe Karakter Normalizer ────────────────────────────────────────────────
# Türkçe → ASCII benzeri karakter tablosu (dict ile güvenli)
_TR_MAP = str.maketrans({
    "İ": "i", "I": "i",   # Büyük İ ve I → küçük i
    "Ş": "s", "ş": "s",
    "Ğ": "g", "ğ": "g",
    "Ü": "u", "ü": "u",
    "Ö": "o", "ö": "o",
    "Ç": "c", "ç": "c",
    "ı": "i",              # Türkçe küçük ı → i
})


def _tr_norm(s: str) -> str:
    """
    Türkçe metni ASCII-benzeri forma normalize eder.

    Önemli: önce translate() sonra lower() çağrılır.
    Neden? Python'da 'İ'.lower() = 'i̇' (2 karakter: i + U+0307 combining dot above).
    İönce translate() ile 'İ' → 'i' dönüştürülünce lower() güvenle çalışır.

    Örnek: 'Tıp' → 'tip' | 'ORTA DOĞU' → 'orta dogu' | 'İSTANBUL' → 'istanbul'
    """
    return s.translate(_TR_MAP).lower()



def _build_master_df() -> pl.DataFrame:
    """
    3 veri kaynağını JOIN'leyerek zengin master Polars DataFrame oluşturur.

    Sütun grupları:
      - Kimlik      : kilavuz_kodu, universite_adi, birim_adi, birim_grup_adi
      - Ham string  : il_adi, puan_turu, universite_turu, ogretim_turu
      - Burs        : burs_orani (burs_enc encode değerinden türetilir)
      - ML features : lag1_taban_siralama, siralama_trend, ...
      - 2026 Tahmin : pred_2026, pred_lower, pred_upper, pred_degisim
      - Arama yrd.  : _search_col (Türkçe normalize edilmiş birleşik metin)
    """
    logger.info("Master DF oluşturuluyor — 3 kaynak JOIN başlatılıyor...")

    # ── 1. Feature Matrix (2025 yılı) ─────────────────────────────────────────
    X, y, meta = load_and_build()
    df_feat = pd.concat(
        [
            meta.reset_index(drop=True),
            X.reset_index(drop=True),
            y.rename("taban_siralama_2025").reset_index(drop=True),
        ],
        axis=1,
    )
    # Duplicate kolonları temizle
    df_feat = df_feat.loc[:, ~df_feat.columns.duplicated()]
    df_feat = df_feat[df_feat["yil"] == 2025].copy().reset_index(drop=True)

    # ── 2. Ham CSV — string kolonlar ──────────────────────────────────────────
    # Not: burs_orani ham CSV'de YOK — burs_enc encode'dan türetilir
    ham_cols = [
        "kilavuz_kodu", "il_adi", "puan_turu", "universite_turu",
        "ogretim_turu", "birim_grup_adi", "genel_kontenjan",
    ]
    if RAW_CSV.exists():
        raw = pd.read_csv(RAW_CSV)
        available_ham = [c for c in ham_cols if c in raw.columns]
        raw_2025 = (
            raw[raw["yil"] == 2025][available_ham]
            .drop_duplicates("kilavuz_kodu")
            .reset_index(drop=True)
        )
        df_feat = df_feat.merge(raw_2025, on="kilavuz_kodu", how="left", suffixes=("", "_raw"))

        # birim_grup_adi: ham CSV'deki değer varsa onu tercih et
        if "birim_grup_adi_raw" in df_feat.columns:
            df_feat["birim_grup_adi"] = (
                df_feat["birim_grup_adi_raw"]
                .fillna(df_feat.get("birim_grup_adi", ""))
            )
            df_feat.drop(columns=["birim_grup_adi_raw"], inplace=True)
        if "genel_kontenjan_raw" in df_feat.columns:
            df_feat.drop(columns=["genel_kontenjan_raw"], inplace=True)

        # ── Tarihsel lag verileri: lag2_puan, lag3 (2023), lag4 (2022) ──────────
        # Feature matrix zaten lag1_taban_siralama/puan ve lag2_taban_siralama
        # içeriyor. Eksik olanları ham CSV'den tamamlıyoruz.
        HIST_LAGS = [
            (2, 2024, ["taban_siralama", "taban_puan"]),   # lag2_puan eksik
            (3, 2023, ["taban_siralama", "taban_puan"]),
            (4, 2022, ["taban_siralama", "taban_puan"]),
        ]
        for lag_n, year, cols in HIST_LAGS:
            avail = [c for c in ["kilavuz_kodu"] + cols if c in raw.columns]
            year_df = (
                raw[raw["yil"] == year][avail]
                .drop_duplicates("kilavuz_kodu")
                .reset_index(drop=True)
                .rename(columns={c: f"lag{lag_n}_{c}" for c in cols if c in avail})
            )
            # Mevcut kolon varsa üstüne yazma (lag2_taban_siralama feature matrix'te var)
            new_cols = [c for c in year_df.columns if c != "kilavuz_kodu" and c not in df_feat.columns]
            if new_cols:
                df_feat = df_feat.merge(
                    year_df[["kilavuz_kodu"] + new_cols],
                    on="kilavuz_kodu", how="left",
                )
    else:
        logger.warning("Ham CSV bulunamadı: %s", RAW_CSV)
        for col in ["il_adi", "puan_turu", "universite_turu", "ogretim_turu"]:
            if col not in df_feat.columns:
                df_feat[col] = ""


    # ── Burs oranı: burs_enc integer encode'dan metin etiketi türet ───────────
    # Neden bu yol? Ham CSV'de burs_orani verisi hiç yok (tüm satırlar NaN).
    # Feature pipeline bu bilgiyi ÖSYM/YÖK kılavuzundan encode ediyor.
    _BURS_LABELS: Dict[int, str] = {
        0: "-",
        1: "Ücretli",
        2: "%25 İndirimli",
        3: "%50 İndirimli",
        4: "%75 İndirimli",
        5: "Burslu",
    }
    if "burs_enc" in df_feat.columns:
        # burs_enc numpy int64 olabilir — Python int'e çevir
        df_feat["burs_orani"] = (
            df_feat["burs_enc"].astype(int).map(_BURS_LABELS).fillna("-")
        )
    else:
        df_feat["burs_orani"] = "-"

    # ── 3. 2026 ML Simülasyon CSV ─────────────────────────────────────────────
    sim_cols_map = {
        "kilavuz_kodu": "kilavuz_kodu",
        "2026_tahmini_siralama": "pred_2026",
        "2026_guven_alt_sinir": "pred_lower",
        "2026_guven_ust_sinir": "pred_upper",
        "tahmini_siralama_degisimi": "pred_degisim",
        "2025_gerceklesen_siralama": "sim_2025_siralama",
    }
    if SIMULASYON_CSV.exists():
        sim = pd.read_csv(SIMULASYON_CSV)
        sim_available = {k: v for k, v in sim_cols_map.items() if k in sim.columns}
        sim_sub = (
            sim[list(sim_available.keys())]
            .rename(columns=sim_available)
            .drop_duplicates("kilavuz_kodu")
        )
        df_feat = df_feat.merge(sim_sub, on="kilavuz_kodu", how="left")
    else:
        logger.warning("Simülasyon CSV bulunamadı: %s", SIMULASYON_CSV)
        for col in ["pred_2026", "pred_lower", "pred_upper", "pred_degisim"]:
            df_feat[col] = np.nan

    # ── Fallback: ML tahmini eksik satırlar için heuristic ───────────────────
    if "pred_2026" in df_feat.columns:
        mask_null = df_feat["pred_2026"].isna()
        if mask_null.any() and "lag1_taban_siralama" in df_feat.columns:
            fallback = (
                df_feat["lag1_taban_siralama"].fillna(0)
                + df_feat.get("siralama_trend", pd.Series(0, index=df_feat.index)).fillna(0) * 0.3
            )
            df_feat.loc[mask_null, "pred_2026"] = fallback[mask_null]
            df_feat.loc[mask_null, "pred_lower"] = fallback[mask_null] * 0.8
            df_feat.loc[mask_null, "pred_upper"] = fallback[mask_null] * 1.25

    # ── Risk Kategorisi ───────────────────────────────────────────────────────
    if "lag1_taban_siralama" in df_feat.columns and "pred_2026" in df_feat.columns:
        df_feat["pred_degisim"] = df_feat["pred_degisim"].fillna(
            df_feat["pred_2026"].fillna(0) - df_feat["lag1_taban_siralama"].fillna(0)
        )
        df_feat["risk_renk"] = df_feat["pred_degisim"].apply(_risk_label)

    # ── Sayısal kolon temizliği ───────────────────────────────────────────────
    for num_col in ["pred_2026", "pred_lower", "pred_upper", "pred_degisim"]:
        if num_col in df_feat.columns:
            df_feat[num_col] = (
                pd.to_numeric(df_feat[num_col], errors="coerce").fillna(0).astype(int)
            )

    # ── String boşluk doldur ─────────────────────────────────────────────────
    str_defaults = {
        "universite_turu": "DEVLET",
        "ogretim_turu": "Örgün",
        "burs_orani": "-",
        "il_adi": "",
        "puan_turu": "",
        "birim_grup_adi": "",
        "birim_adi": "",
        "universite_adi": "",
        "risk_renk": "🟡 STABIL",
    }
    for col, default in str_defaults.items():
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].fillna(default).astype(str)

    # ── Türkçe Normalize Arama Kolonu (_search_col) ───────────────────────────
    # Polars'ın (?i) flag'i Türkçe karakterleri desteklemez.
    # Tüm metin filtresi bu normalize edilmiş kolonda çalışır.
    def _build_search_col(row: pd.Series) -> str:
        parts = [
            str(row.get("universite_adi", "")),
            str(row.get("birim_adi", "")),
            str(row.get("birim_grup_adi", "")),
            str(row.get("il_adi", "")),
        ]
        return _tr_norm(" ".join(p for p in parts if p))

    df_feat["_search_col"] = df_feat.apply(_build_search_col, axis=1)

    # Ayrı ayrı normalize kolonlar (filtre sütunları)
    df_feat["_univ_norm"] = df_feat["universite_adi"].apply(_tr_norm)
    df_feat["_bolum_norm"] = df_feat["birim_grup_adi"].apply(_tr_norm)
    df_feat["_il_norm"] = df_feat["il_adi"].apply(_tr_norm)

    # ── Polars'a çevir ───────────────────────────────────────────────────────
    master = pl.from_pandas(df_feat)
    burs_count = int((master["burs_orani"] == "Burslu").sum()) if "burs_orani" in master.columns else 0
    ml_count = int((master["pred_2026"] > 0).sum()) if "pred_2026" in master.columns else 0
    logger.info(
        "Master DF hazır: %d program | ML tahmin: %d | Burslu: %d",
        len(master), ml_count, burs_count,
    )
    return master


def _risk_label(degisim: float) -> str:
    """Tahmini sıralama değişimine göre risk etiketi döndürür."""
    if degisim <= -20000:
        return "🟢 ÇOK İYİ"
    elif degisim <= -5000:
        return "🟢 İYİ"
    elif degisim <= 5000:
        return "🟡 STABIL"
    elif degisim <= 20000:
        return "🔴 GERİLİYOR"
    else:
        return "🔴 HIZLI GERİLEME"


def get_master_df() -> pl.DataFrame:
    """Önbellekli master Polars DataFrame döndürür."""
    global _master_df_cache
    if _master_df_cache is None:
        _master_df_cache = _build_master_df()
    return _master_df_cache


def get_polars_master_df() -> pl.DataFrame:
    """Alias: get_master_df() — geriye dönük uyumluluk."""
    return get_master_df()


class SearchService:
    """Polars reaktif çoklu-filtreleme arama servisi."""

    @staticmethod
    def search_programs(
        universite_adi: Optional[str] = None,
        birim_grup_adi: Optional[str] = None,
        il_adi: Optional[str] = None,
        puan_turu: Optional[str] = None,
        universite_turu: Optional[str] = None,
        ogretim_turu: Optional[str] = None,
        burs_orani: Optional[str] = None,
        min_rank: Optional[float] = None,
        max_rank: Optional[float] = None,
        min_pred: Optional[float] = None,
        max_pred: Optional[float] = None,
        min_quota: Optional[float] = None,
        max_quota: Optional[float] = None,
        search_query: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Çoklu filtreleri Polars üstünde anlık çalıştırıp sonuçları liste döndürür.

        Metin aramaları `_search_col` (Türkçe-normalize) kolonu üzerinde yapılır.
        Bu sayede 'tip' → 'Tıp', 'orta dogu' → 'ORTA DOĞU' gibi eşleşmeler çalışır.
        """
        df = get_master_df()
        exprs = []

        # ── Hızlı birleşik metin arama (üniversite + bölüm + şehir) ──────────
        if search_query and search_query.strip():
            q_norm = _tr_norm(search_query.strip())
            exprs.append(pl.col("_search_col").str.contains(q_norm))

        # ── Üniversite adı ─────────────────────────────────────────────────────
        if universite_adi and universite_adi.strip():
            q_norm = _tr_norm(universite_adi.strip())
            exprs.append(pl.col("_univ_norm").str.contains(q_norm))

        # ── Bölüm adı ─────────────────────────────────────────────────────────
        if birim_grup_adi and birim_grup_adi.strip():
            q_norm = _tr_norm(birim_grup_adi.strip())
            exprs.append(pl.col("_bolum_norm").str.contains(q_norm))

        # ── Şehir ─────────────────────────────────────────────────────────────
        if il_adi and il_adi.strip() and "_il_norm" in df.columns:
            q_norm = _tr_norm(il_adi.strip())
            exprs.append(pl.col("_il_norm").str.contains(q_norm))

        # ── Puan türü (tam eşleşme, ASCII) ────────────────────────────────────
        if puan_turu and puan_turu.strip() and puan_turu.upper() not in ("TÜMÜ", "TUMU", ""):
            exprs.append(pl.col("puan_turu") == puan_turu.strip().upper())

        # ── Üniversite türü ───────────────────────────────────────────────────
        if universite_turu and universite_turu.strip() and universite_turu.upper() not in ("TÜMÜ", "TUMU", ""):
            ut = universite_turu.strip().upper()
            if "universite_turu" in df.columns:
                exprs.append(pl.col("universite_turu").str.to_uppercase().str.contains(ut))

        # ── Öğretim türü ──────────────────────────────────────────────────────
        if ogretim_turu and ogretim_turu.strip() and ogretim_turu not in ("TÜMÜ", "TUMU", ""):
            ot_norm = _tr_norm(ogretim_turu.strip())
            if "ogretim_turu" in df.columns:
                exprs.append(
                    pl.col("ogretim_turu").map_elements(
                        lambda x: ot_norm in _tr_norm(str(x)), return_dtype=pl.Boolean
                    )
                )

        # ── Burs oranı ────────────────────────────────────────────────────────
        if burs_orani and burs_orani.strip() and burs_orani not in ("TÜMÜ", "TUMU", ""):
            b_norm = _tr_norm(burs_orani.strip())
            if "burs_orani" in df.columns:
                exprs.append(
                    pl.col("burs_orani").map_elements(
                        lambda x: b_norm in _tr_norm(str(x)), return_dtype=pl.Boolean
                    )
                )

        # ── Sayısal filtreler ─────────────────────────────────────────────────
        if min_rank is not None and min_rank > 0 and "lag1_taban_siralama" in df.columns:
            exprs.append(pl.col("lag1_taban_siralama") >= min_rank)
        if max_rank is not None and max_rank > 0 and "lag1_taban_siralama" in df.columns:
            exprs.append(pl.col("lag1_taban_siralama") <= max_rank)
        if min_pred is not None and min_pred > 0 and "pred_2026" in df.columns:
            exprs.append(pl.col("pred_2026") >= min_pred)
        if max_pred is not None and max_pred > 0 and "pred_2026" in df.columns:
            # pred_2026=0 olan satırları dışla (ML tahmini olmayan programlar)
            exprs.append((pl.col("pred_2026") <= max_pred) & (pl.col("pred_2026") > 0))
        if min_quota is not None and min_quota > 0 and "lag1_genel_kontenjan" in df.columns:
            exprs.append(pl.col("lag1_genel_kontenjan") >= min_quota)
        if max_quota is not None and max_quota > 0 and "lag1_genel_kontenjan" in df.columns:
            exprs.append(pl.col("lag1_genel_kontenjan") <= max_quota)

        # ── Filtreleri uygula ─────────────────────────────────────────────────
        filtered = df
        for expr in exprs:
            filtered = filtered.filter(expr)

        sort_col = "pred_2026" if "pred_2026" in filtered.columns else "lag1_taban_siralama"
        filtered = filtered.sort(sort_col, descending=False, nulls_last=True).head(limit)

        return filtered.to_dicts()

    @staticmethod
    def get_program_by_code(kilavuz_kodu: int) -> Optional[Dict[str, Any]]:
        """Kılavuz koduna göre tek bir programın tüm zengin verisini döndürür."""
        df = get_master_df()
        res = df.filter(pl.col("kilavuz_kodu") == kilavuz_kodu)
        return res.to_dicts()[0] if len(res) > 0 else None

    @staticmethod
    def get_programs_by_university(universite_adi: str) -> List[Dict[str, Any]]:
        """Bir üniversitenin tüm programlarını döndürür."""
        df = get_master_df()
        q_norm = _tr_norm(universite_adi.strip())
        res = df.filter(
            pl.col("_univ_norm").str.contains(q_norm)
        ).sort("pred_2026", nulls_last=True)
        return res.to_dicts()
