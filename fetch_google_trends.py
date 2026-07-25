"""
Google Trends 2 Katmanlı Çekme ve Önbellekleme Scripti — YKS Talep Sinyali

- 429 (Too Many Requests) hatalarını önlemek için istekler arasında bekleme (5 sn) ve üssel geri çekilme (exponential backoff) uygulanır.
- Başarılı çekilen sorgular 'data/raw/demand/trends_cache.json' dosyasına önbelleklenir, yeniden çalıştırmalarda tekrar istek atılmaz.
- Katman 1: 15 kaba kategori.
- Katman 2: Popüler bölümler.
- 429 aşılması durumunda Katman 1 varsayılan değerleri veya güvenli nötr sinyal (0.0 YoY) kullanılır.
"""
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "data" / "raw" / "demand"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = OUTPUT_DIR / "trends_cache.json"
OUTPUT_CSV = OUTPUT_DIR / "google_trends.csv"

# ── Katman 1: 15 Kaba Kategori Tanımları ────────────────────────────────────
CATEGORY_MAP = {
    "tip": ["Tıp", "Diş Hekimliği"],
    "hukuk": ["Hukuk"],
    "muhendislik_sayisal": [
        "Bilgisayar Mühendisliği", "Yazılım Mühendisliği",
        "Elektrik-Elektronik Mühendisliği", "Makine Mühendisliği",
        "Mekatronik Mühendisliği", "Uçak Mühendisliği",
    ],
    "muhendislik_ea": [
        "Endüstri Mühendisliği", "İnşaat Mühendisliği",
        "Çevre Mühendisliği", "Jeoloji Mühendisliği",
    ],
    "iktisadi_idari": [
        "İşletme", "İktisat", "Uluslararası İlişkiler",
        "Kamu Yönetimi", "Yönetim Bilişim Sistemleri",
        "Çalışma Ekonomisi ve Endüstri İlişkileri",
    ],
    "psikoloji_sosyal": [
        "Psikoloji", "Sosyoloji", "Sosyal Hizmet",
        "Sosyal Bilgiler Öğretmenliği",
    ],
    "egitim": [
        "İlköğretim Matematik Öğretmenliği", "Türkçe Öğretmenliği",
        "Rehberlik ve Psikolojik Danışmanlık", "Okul Öncesi Öğretmenliği",
        "Sınıf Öğretmenliği",
    ],
    "saglik": [
        "Hemşirelik", "Fizyoterapi ve Rehabilitasyon",
        "Beslenme ve Diyetetik", "Eczacılık",
        "Tıbbi Laboratuvar Teknikleri",
    ],
    "mimarlik_tasarim": [
        "Mimarlık", "İç Mimarlık", "Peyzaj Mimarlığı",
        "Grafik Tasarım",
    ],
    "iletisim": [
        "Gazetecilik", "Yeni Medya ve İletişim",
        "Radyo, Televizyon ve Sinema",
    ],
    "guzel_sanatlar": [
        "Sinema ve Televizyon", "Müzik", "Sahne Sanatları",
        "Güzel Sanatlar",
    ],
    "tarim_orman": [
        "Ziraat Mühendisliği", "Orman Mühendisliği",
        "Gıda Mühendisliği",
    ],
    "turizm": [
        "Turizm İşletmeciliği", "Gastronomi ve Mutfak Sanatları",
        "Otel Yönetimi",
    ],
    "havacilik": [
        "Pilotaj", "Havacılık Yönetimi",
        "Uçak Teknolojisi",
    ],
    "diger": [],
}

BOLUM_TO_KATEGORI = {}
for kat, bolumler in CATEGORY_MAP.items():
    for b in bolumler:
        BOLUM_TO_KATEGORI[b] = kat

YEARS = [2022, 2023, 2024, 2025]

def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache: dict):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def fetch_trends_safe(query: str, year: int, cache: dict, max_retries: int = 3) -> float | None:
    cache_key = f"{query}_{year}"
    if cache_key in cache:
        return cache[cache_key]

    from pytrends.request import TrendReq
    backoff = 5.0
    for attempt in range(max_retries):
        try:
            pt = TrendReq(hl="tr-TR", tz=180, timeout=(10, 25))
            timeframe = f"{year}-01-01 {year}-12-31"
            pt.build_payload([query], timeframe=timeframe, geo="TR")
            df = pt.interest_over_time()
            if df is not None and not df.empty and query in df.columns:
                val = float(df[query].mean())
                res = val if val > 0 else None
                cache[cache_key] = res
                save_cache(cache)
                time.sleep(3.0)  # Politeness sleep
                return res
            else:
                cache[cache_key] = None
                save_cache(cache)
                time.sleep(2.0)
                return None
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "ResponseError" in err_str:
                print(f"    [429 Rate Limit] {query} ({year}) — {backoff}s bekleniyor (Deneme {attempt+1}/{max_retries})...")
                time.sleep(backoff)
                backoff *= 2.0
            else:
                print(f"    [Hata] {query} ({year}): {e}")
                cache[cache_key] = None
                save_cache(cache)
                return None

    cache[cache_key] = None
    save_cache(cache)
    return None


def run_collection():
    raw_csv = ROOT / "data" / "raw" / "yokatlas" / "yokatlas_all_departments_raw.csv"
    df_raw = pd.read_csv(raw_csv)
    dept_families = sorted(df_raw["birim_grup_adi"].dropna().unique().tolist())

    cache = load_cache()
    print(f"Google Trends çekme başlatılıyor. Toplam bölüm ailesi: {len(dept_families)}")
    print(f"Mevcut önbellek boyutu: {len(cache)} kayıt.\n")

    # Katman 1: 15 Kategori
    print("Katman 1: 15 Ana Kategori Verisi Çekiliyor...")
    cat_scores = {}
    for kat in CATEGORY_MAP:
        if kat == "diger":
            continue
        scores = {}
        for yr in YEARS:
            val = fetch_trends_safe(kat, yr, cache)
            if val is not None:
                scores[yr] = val
        cat_scores[kat] = scores
        print(f"  Katman 1 [{kat}]: {scores}")

    # Katman 2: Sadece BILGISAYAR, TIP, HUKUK, PSIKOLOJI, ELEKTRIK vb. en popüler 20 bölüm ailesi için granüler
    priority_depts = [
        "Tıp", "Hukuk", "Bilgisayar Mühendisliği", "Psikoloji", "Diş Hekimliği",
        "Elektrik-Elektronik Mühendisliği", "Endüstri Mühendisliği", "Hemşirelik",
        "Mimarlık", "İşletme", "Eczacılık", "Yazılım Mühendisliği", "İktisat"
    ]
    
    print("\nKatman 2: Öncelikli Popüler Bölümler Çekiliyor...")
    granular_scores = {}
    for dept in priority_depts:
        if dept in dept_families:
            scores = {}
            for yr in YEARS:
                val = fetch_trends_safe(dept, yr, cache)
                if val is not None:
                    scores[yr] = val
            if len(scores) >= 2:
                granular_scores[dept] = scores
                print(f"  Katman 2 [{dept}]: {scores}")

    # Sonuçları DataFrame'e dönüştür
    rows = []
    for dept in dept_families:
        kat = BOLUM_TO_KATEGORI.get(dept, "diger")
        if dept in granular_scores:
            src_scores = granular_scores[dept]
            layer = 2
            query_used = dept
        else:
            src_scores = cat_scores.get(kat, {})
            layer = 1
            query_used = kat

        for yr in YEARS:
            val = src_scores.get(yr, None)
            rows.append({
                "birim_grup_adi": dept,
                "yil": yr,
                "trends_skoru": val,
                "katman": layer,
                "query_used": query_used,
            })

    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(["birim_grup_adi", "yil"])

    # YoY Değişim hesapla
    df_out["trends_prev"] = df_out.groupby("birim_grup_adi")["trends_skoru"].shift(1)
    df_out["trends_yoy_degisim"] = np.where(
        df_out["trends_prev"].notna() & (df_out["trends_prev"] > 0) & df_out["trends_skoru"].notna(),
        (df_out["trends_skoru"] - df_out["trends_prev"]) / df_out["trends_prev"],
        0.0
    )
    df_out["trends_yoy_degisim"] = df_out["trends_yoy_degisim"].fillna(0.0)
    df_out.drop(columns=["trends_prev"], inplace=True)

    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\nİşlem tamamlandı. Trends verisi kaydedildi: {OUTPUT_CSV}")
    print(f"Katman 2 kullanılan bölüm sayısı: {len(granular_scores)}")

if __name__ == "__main__":
    run_collection()
