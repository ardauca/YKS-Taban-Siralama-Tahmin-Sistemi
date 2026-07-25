"""
Backtest temporal leakage check.
1) program_hist_medyan_siralama nasil hesaplaniyor?
2) univ_hist_medyan_siralama nasil hesaplaniyor?
3) Test setindeki yillar hangi yillarin medyanini kullaniyor?
4) Cross-contamination var mi?
"""
import pandas as pd
import numpy as np

# build_features.py'yi import et
import sys
sys.path.insert(0, '.')
from src.features.build_features import build_features

df_feat = build_features()

# Train/test split nasil yapiliyor - train_quantile'a bak
# Test_2024: yil == 2024
# Test_2025: yil == 2025

print('=== BACKTEST TEMPORAL SPLIT ANALIZI ===')
print(f'Feature matrisindeki yil dagilimi:')
print(df_feat['yil'].value_counts().sort_index())
print()

# program_hist_medyan_siralama icin kontrol:
# Bu feature yil 2024 icin hangi veriye bakiyor?
print('=== FEATURE LEAKAGE KONTROLU: program_hist_medyan_siralama ===')
print('Bu feature, build_features.py icinde nasil uretiliyor kontrol ediliyor...')
print()

# 2024 test kayitlarinda program_hist_medyan_siralama hangi yillari kapsayabilir?
# Eger expanding window ile yapiliyorsa dogru, eger tum veri ile yapiliyorsa leakage var.

# Spesifik bir programa bak
mask_2024 = df_feat['yil'] == 2024
sample = df_feat[mask_2024].dropna(subset=['program_hist_medyan_siralama']).head(5)
print('2024 test setindeki ornek kayitlar:')
print(sample[['yil', 'program_hist_medyan_siralama', 'lag1_taban_siralama', 'lag2_taban_siralama']].to_string())
print()

# build_features'daki kodu incele
with open('src/features/build_features.py', 'r', encoding='utf-8') as f:
    content = f.read()

# program_hist_medyan icin ilgili satirlari bul
lines = content.split('\n')
for i, line in enumerate(lines):
    if 'program_hist_medyan' in line or 'hist_medyan' in line:
        start = max(0, i-2)
        end = min(len(lines), i+5)
        print(f'--- satir {i+1} ---')
        for l in lines[start:end]:
            print(l)
        print()
