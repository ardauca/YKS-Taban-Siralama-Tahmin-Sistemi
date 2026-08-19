@echo off
chcp 65001 > nul
echo =================================================================
echo  🎓 YKS 2026 TAHMİN SİSTEMİ — OTOMATİK KURULUM (WINDOWS)
echo =================================================================
echo.

if not exist ".venv" (
    echo 📦 Python Sanal Ortamı (.venv) Oluşturuluyor...
    python -m venv .venv
    if errorlevel 1 (
        echo ❌ Python bulunamadı! Lütfen Python 3.10 veya üzerini yükleyin.
        pause
        exit /b 1
    )
)

echo 📥 Kütüphaneler Yükleniyor (pip install)...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo.
echo =================================================================
echo ✅ KURULUM BAŞARIYLA TAMAMLANDI!
echo.
echo Uygulamayı başlatmak için:
echo   python baslat.py
echo   veya terminalden: yks-tahmin
echo =================================================================
echo.
pause
