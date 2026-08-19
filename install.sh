#!/usr/bin/env bash
set -e

echo "================================================================="
echo " 🎓 YKS 2026 TAHMİN SİSTEMİ — OTOMATİK KURULUM (LINUX / MACOS)"
echo "================================================================="
echo ""

if [ ! -d ".venv" ]; then
    echo "📦 Python Sanal Ortamı (.venv) Oluşturuluyor..."
    python3 -m venv .venv
fi

echo "📥 Kütüphaneler Yükleniyor (pip install)..."
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo ""
echo "================================================================="
echo "✅ KURULUM BAŞARIYLA TAMAMLANDI!"
echo ""
echo "Uygulamayı başlatmak için:"
echo "  python3 baslat.py"
echo "  veya terminalden: yks-tahmin"
echo "================================================================="
