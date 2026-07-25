"""
Data quality report — per birim_grup_adi (department family)
"""
import pandas as pd

df = pd.read_csv('data/raw/yokatlas/yokatlas_all_departments_raw.csv')

rows = []
for grp_name, grp in df.groupby('birim_grup_adi'):
    n_rows = len(grp)
    n_uniq_prog = grp['kilavuz_kodu'].nunique()
    n_yillar = sorted(grp['yil'].unique())
    missing_siralama = grp['taban_siralama'].isna().mean() * 100
    missing_puan = grp['taban_puan'].isna().mean() * 100
    missing_kontenjan = grp['genel_kontenjan'].isna().mean() * 100
    n_dups = grp.duplicated(subset=['kilavuz_kodu', 'yil'], keep=False).sum()
    rows.append({
        'birim_grup_adi': grp_name[:40],
        'n_satir': n_rows,
        'n_program': n_uniq_prog,
        'yil_sayisi': len(n_yillar),
        'eksik_siralama_pct': round(missing_siralama, 1),
        'eksik_puan_pct': round(missing_puan, 1),
        'eksik_kontenjan_pct': round(missing_kontenjan, 1),
        'duplicate_sayisi': n_dups,
    })

report = pd.DataFrame(rows).sort_values('n_satir', ascending=False)

print('=== TOPLAM OZET ===')
print(f'Toplam bolum ailesi (birim_grup_adi): {len(report)}')
print(f'Toplam kayit (satir): {len(df)}')

dup_count = df.duplicated(subset=['kilavuz_kodu', 'yil'], keep=False).sum()
print(f'Duplicate kayit (kilavuz_kodu+yil): {dup_count}')
print(f'Ort. siralama eksik orani: {report["eksik_siralama_pct"].mean():.1f}%')
print(f'siralama eksik >= 80%% olan bolum ailesi: {(report["eksik_siralama_pct"] >= 80).sum()}')
print(f'siralama eksik >= 50%% olan bolum ailesi: {(report["eksik_siralama_pct"] >= 50).sum()}')
print()

print('=== TOP 30 EN COK SATIRI OLAN BOLUM AILESI ===')
print(report.head(30).to_string(index=False))
print()

print('=== KOTU KALITE: eksik_siralama >= 80% VE n_satir >= 20 ===')
bad = report[(report['eksik_siralama_pct'] >= 80) & (report['n_satir'] >= 20)]
print(f'{len(bad)} bolum ailesi bu kriteri karsilamakta:')
print(bad.to_string(index=False))
