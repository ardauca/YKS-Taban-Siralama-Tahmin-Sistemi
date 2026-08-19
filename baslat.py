import sys
import shutil
import subprocess
from pathlib import Path


def get_python_exe() -> str:
    """Gerekli kütüphanelerin (typer, textual, polars) yüklü olduğu Python ortamını otomatik tespit eder."""
    root = Path(__file__).parent
    venv_win = root / ".venv" / "Scripts" / "python.exe"
    venv_posix = root / ".venv" / "bin" / "python"

    candidates = [
        sys.executable,
        str(venv_win),
        str(venv_posix),
        r"C:\Users\ARDA\AppData\Local\spyder-6\python.exe",
        shutil.which("python3") or "",
        shutil.which("python") or "python",
    ]
    for py in candidates:
        if py and Path(py).exists():
            try:
                res = subprocess.run(
                    [py, "-c", "import typer, textual, polars; print('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0 and "OK" in res.stdout:
                    return py
            except Exception:
                pass
    return sys.executable



def main():
    py_exe = get_python_exe()
    while True:
        print("\n" + "=" * 65)
        print("🎓 YKS 2026 TABAN SIRALAMA TAHMİN SİSTEMİ — KOLAY BAŞLATICI")
        print("=" * 65)
        print("Lütfen yapmak istediğiniz işlemi seçiniz:\n")
        print(" [1] 🖥️  Grafik Ekranı Aç (TUI — Butonlar, Tablolar ve Renkli Ekranlar)")
        print(" [2] 🎯  Hızlı Tercih Danışmanı (Sıralamanızı girip anında öneri alın)")
        print(" [3] 🔎  Hızlı Bölüm / Üniversite Arama")
        print(" [4] 📊  Türkiye Geneli İstatistik Özetini Göster")
        print(" [5] 📄  Tercih Listesini PDF / Markdown Olarak Aktar")
        print(" [0] 🚪  Çıkış")
        print("-" * 65)

        secim = input("Seçiminiz (0-5): ").strip()

        if secim == "1":
            print("\n🚀 Grafik Ekran (TUI) Başlatılıyor...\n")
            subprocess.run([py_exe, "cli/app.py", "tui"])
        elif secim == "2":
            rank = input("👉 YKS Başarı Sıralamanızı girin (ör. 180000): ").strip()
            pt = input("👉 Puan Türü girin (SAY / EA / SÖZ / DİL) [Varsayılan: EA]: ").strip().upper() or "EA"
            if rank.isdigit():
                subprocess.run([py_exe, "tercih_danismani.py", rank, pt])
            else:
                print("❌ Geçersiz sıralama! Lütfen sayısal bir değer girin.")
        elif secim == "3":
            query = input("👉 Aramak istediğiniz Üniversite veya Bölüm adı (ör. Bilgisayar / Boğaziçi): ").strip()
            if query:
                subprocess.run([py_exe, "cli/app.py", "search", "--query", query, "--limit", "10"])
        elif secim == "4":
            subprocess.run([py_exe, "cli/app.py", "stats"])
        elif secim == "5":
            out_name = input("👉 Rapor dosya adı girin [Varsayılan: tercih_listem_2026]: ").strip() or "tercih_listem_2026"
            subprocess.run([py_exe, "cli/app.py", "export", "--list-id", "1", "--format", "pdf", "--output", out_name])
        elif secim == "0":
            print("\n👋 İyi günler dileriz!\n")
            break
        else:
            print("\n⚠️ Geçersiz seçim, lütfen 0-5 arasında bir rakam girin.")


if __name__ == "__main__":
    main()

