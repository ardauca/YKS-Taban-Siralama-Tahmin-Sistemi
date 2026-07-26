"""
YKS 2026 Taban Sıralama Tahmin Sistemi — Kolay Başlatıcı Menüsü (Basit & Kullanıcı Dostu)
"""
import sys
import subprocess
from pathlib import Path

def main():
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
            subprocess.run([sys.executable, "cli/app.py", "tui"])
        elif secim == "2":
            rank = input("👉 YKS Başarı Sıralamanızı girin (ör. 180000): ").strip()
            pt = input("👉 Puan Türü girin (SAY / EA / SÖZ / DİL) [Varsayılan: EA]: ").strip().upper() or "EA"
            if rank.isdigit():
                subprocess.run([sys.executable, "tercih_danismani.py", rank, pt])
            else:
                print("❌ Geçersiz sıralama! Lütfen sayısal bir değer girin.")
        elif secim == "3":
            query = input("👉 Aramak istediğiniz Üniversite veya Bölüm adı (ör. Bilgisayar / Boğaziçi): ").strip()
            if query:
                subprocess.run([sys.executable, "cli/app.py", "search", "--query", query, "--limit", "10"])
        elif secim == "4":
            subprocess.run([sys.executable, "cli/app.py", "stats"])
        elif secim == "5":
            out_name = input("👉 Rapor dosya adı girin [Varsayılan: tercih_listem_2026]: ").strip() or "tercih_listem_2026"
            subprocess.run([sys.executable, "cli/app.py", "export", "--list-id", "1", "--format", "pdf", "--output", out_name])
        elif secim == "0":
            print("\n👋 İyi günler dileriz!\n")
            break
        else:
            print("\n⚠️ Geçersiz seçim, lütfen 0-5 arasında bir rakam girin.")

if __name__ == "__main__":
    main()
