"""
Feature engineering pipeline — YKS Taban Sıralama Tahmin Sistemi.

Görev: Ham veri (program × yıl, long format) → model-ready feature matrix.

Hedef değişken: taban_siralama (yıl Y için)

Feature grupları:
  1. Lag features     — Y-1 yılının sıralama, puan, kontenjan, doluluk oranı
  2. Lag-2 features   — Y-2 yılının sıralama (trend tespiti için)
  3. Delta features   — Sıralamadaki değişim (Y-2 → Y-1), kontenjan değişim oranı
  4. Static features  — Üniversite türü, il, öğretim türü, burs oranı, puan türü
  5. Derived features — Programın tarihsel medyan sıralaması, üniversite prestij skoru

Not: 2025 yılı için taban_siralama büyük oranda null (yerleştirme tamamlanmadı).
Backtesting: 2023 ve 2024 yılları test set olarak kullanılır.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALL_DEPTS_CSV = Path(__file__).parent.parent.parent / "data" / "raw" / "yokatlas" / "yokatlas_all_departments_raw.csv"
SINGLE_DEPT_CSV = Path(__file__).parent.parent.parent / "data" / "raw" / "yokatlas" / "bilgisayar_muhendisligi_raw.csv"
RAW_CSV = ALL_DEPTS_CSV if ALL_DEPTS_CSV.exists() else SINGLE_DEPT_CSV
PROCESSED_DIR = Path(__file__).parent.parent.parent / "data" / "processed"

# Kategorik sütunlar → integer encoding
UNIVERSITE_TURU_MAP = {
    "DEVLET": 0,
    "VAKIF": 1,
    "KKTC": 2,
    "YURTDISI VAKIF": 3,
    "YURTDISI KAMU": 4,
}
OGRETIM_TURU_MAP = {
    "Örgün": 0,
    "İkinci Öğretim": 1,
    "Uzaktan Öğretim": 2,
    None: -1,
}
PUAN_TURU_MAP = {"SAY": 0, "SÖZ": 1, "EA": 2, "DİL": 3, "TYT": 4}

# Burs oranı → ordinal skor
BURS_ORANI_ORDERED = [
    None,           # 0 → bilinmiyor / devlet
    "Ücretli",      # 1
    "%25 İndirimli",# 2
    "%50 İndirimli",# 3
    "%75 İndirimli",# 4
    "Burslu",       # 5
]


def _encode_burs(val: str | None) -> int:
    """Burs oranını 0-5 aralığında ordinal skor olarak kodlar."""
    if pd.isna(val) or val is None:
        return 0
    for i, label in enumerate(BURS_ORANI_ORDERED):
        if label is not None and str(label) in str(val):
            return i
    return 0


def build_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Program bazında lag ve delta feature'larını ekler.

    Her program (kilavuz_kodu) için yıl bazında sıralayıp shift uygular:
    - lag1_*: Y-1 değerleri
    - lag2_taban_siralama: Y-2 değeri (trend için)
    - siralama_trend: lag1 - lag2 (negatif = iyileşiyor)
    - siralama_pct_change: (lag1 - lag2) / lag2
    - kontenjan_degisim_orani: (lag1_kontenjan - lag2_kontenjan) / lag2_kontenjan
    """
    df = df.sort_values(["kilavuz_kodu", "yil"]).copy()

    # Lag-1 (Y-1)
    lag_cols = {
        "lag1_taban_siralama": "taban_siralama",
        "lag1_taban_puan": "taban_puan",
        "lag1_genel_kontenjan": "genel_kontenjan",
        "lag1_sehit_gazi_kontenjan": "sehit_gazi_kontenjan",
        "lag1_depremzede_kontenjan": "depremzede_kontenjan",
        "lag1_okul_birincisi_kontenjan": "okul_birincisi_kontenjan",
    }
    for new_col, src_col in lag_cols.items():
        if src_col in df.columns:
            df[new_col] = df.groupby("kilavuz_kodu")[src_col].shift(1)

    # Lag-2 (Y-2) — sadece sıralama
    df["lag2_taban_siralama"] = df.groupby("kilavuz_kodu")["taban_siralama"].shift(2)

    # Delta: sıralama trendi (lag1 - lag2)
    df["siralama_trend"] = df["lag1_taban_siralama"] - df["lag2_taban_siralama"]

    # Yüzde değişim (NaN-safe)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["siralama_pct_change"] = np.where(
            df["lag2_taban_siralama"].notna() & (df["lag2_taban_siralama"] != 0),
            (df["lag1_taban_siralama"] - df["lag2_taban_siralama"]) / df["lag2_taban_siralama"],
            np.nan,
        )

    # Kontenjan değişim oranı
    df["lag2_kontenjan"] = df.groupby("kilavuz_kodu")["genel_kontenjan"].shift(2)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["kontenjan_degisim_orani"] = np.where(
            df["lag2_kontenjan"].notna() & (df["lag2_kontenjan"] != 0),
            (df["lag1_genel_kontenjan"] - df["lag2_kontenjan"]) / df["lag2_kontenjan"],
            np.nan,
        )

    return df


def build_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Kategorik ve statik feature'ları encode eder."""
    df = df.copy()

    # Üniversite türü
    df["universite_turu_enc"] = df["universite_turu"].map(UNIVERSITE_TURU_MAP).fillna(-1).astype(int)

    # Öğretim türü
    df["ogretim_turu_enc"] = df["ogretim_turu"].map(OGRETIM_TURU_MAP).fillna(-1).astype(int)

    # Puan türü
    df["puan_turu_enc"] = df["puan_turu"].map(PUAN_TURU_MAP).fillna(-1).astype(int)

    # Burs oranı
    df["burs_enc"] = df["burs_orani"].apply(_encode_burs)

    # İl kodu — API string döndürüyor (örn. "34.0"), float'a çevir
    df["il_kodu_num"] = pd.to_numeric(df["il_kodu"], errors="coerce").fillna(-1).astype(int)

    return df


def build_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Program ve üniversite bazında türetilmiş istatistiksel feature'lar.
    Önemli: sadece geçmiş veriden türetilmeli (leak yok).
    """
    df = df.copy()

    # Program bazı tarihsel medyan sıralama (lag yıllardan)
    program_hist = (
        df.groupby("kilavuz_kodu")["lag1_taban_siralama"]
        .median()
        .rename("program_hist_medyan_siralama")
    )
    df = df.merge(program_hist, on="kilavuz_kodu", how="left")

    # Üniversite bazı tarihsel medyan sıralama (lag1 kullanan)
    univ_hist = (
        df.groupby("universite_adi")["lag1_taban_siralama"]
        .median()
        .rename("univ_hist_medyan_siralama")
    )
    df = df.merge(univ_hist, on="universite_adi", how="left")

    # Program kontenjan büyüklüğü kategorisi
    # (küçük=1-30, orta=31-80, büyük=81+)
    df["kontenjan_kategori"] = pd.cut(
        df["lag1_genel_kontenjan"],
        bins=[0, 30, 80, float("inf")],
        labels=[0, 1, 2],
        right=True,
    ).astype(float).fillna(-1).astype(int)

    return df


def get_feature_columns() -> list[str]:
    """Model'e girecek feature sütunlarının listesi."""
    return [
        # Lag-1 features
        "lag1_taban_siralama",
        "lag1_taban_puan",
        "lag1_genel_kontenjan",
        "lag1_sehit_gazi_kontenjan",
        "lag1_depremzede_kontenjan",
        "lag1_okul_birincisi_kontenjan",
        # Lag-2 / trend
        "lag2_taban_siralama",
        "siralama_trend",
        "siralama_pct_change",
        "kontenjan_degisim_orani",
        # Static / encoded
        "universite_turu_enc",
        "ogretim_turu_enc",
        "puan_turu_enc",
        "burs_enc",
        "il_kodu_num",
        # Derived
        "program_hist_medyan_siralama",
        "univ_hist_medyan_siralama",
        "kontenjan_kategori",
        # Yıl (trend için)
        "yil",
    ]


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str = "taban_siralama",
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Ham DataFrame'den tam feature matrix'i üretir.

    Returns:
        X: feature DataFrame
        y: hedef (taban_siralama), NaN'lar dahil
        meta: kilavuz_kodu + yil (takip için)
    """
    df = build_lag_features(df)
    df = build_static_features(df)
    df = build_derived_features(df)

    feature_cols = get_feature_columns()
    # Sadece var olan sütunları al (güvenli)
    available = [c for c in feature_cols if c in df.columns]

    X = df[available].copy()
    y = df[target_col].copy()
    meta = df[["kilavuz_kodu", "yil", "universite_adi", "birim_adi"]].copy()

    logger.info(
        "Feature matrix: %d satir x %d feature | target null: %d",
        len(X), len(available), y.isna().sum(),
    )
    return X, y, meta


def load_and_build(csv_path: Path = RAW_CSV) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """CSV'den okuyup feature matrix döner."""
    df = pd.read_csv(csv_path)
    return build_feature_matrix(df)


def save_processed(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> Path:
    """İşlenmiş feature matrix'i kaydet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "features_bilgisayar_muh.parquet"
    combined = meta.copy()
    for col in X.columns:
        combined[col] = X[col].values
    combined["taban_siralama"] = y.values
    combined.to_parquet(out, index=False)
    logger.info("Feature matrix kaydedildi: %s (%d satir)", out, len(combined))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    X, y, meta = load_and_build()
    print(f"\nFeature matrix: {X.shape}")
    print(f"Target (non-null): {y.notna().sum()} / {len(y)}")
    print(f"\nFeature'lar:\n{X.describe().T[['count','mean','std','min','max']].to_string()}")
    save_processed(X, y, meta)
