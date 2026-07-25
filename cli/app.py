"""
Typer CLI Giriş Noktası ve Komut Satırı Araçları.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Root import yolunu ayarla
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from services.analytics_service import AnalyticsService
from services.export_service import ExportService
from services.preference_service import PreferenceService
from services.search_service import SearchService

app = typer.Typer(
    name="agy-yks",
    help="🎓 YKS 2026 Taban Sıralama Tahmin ve Tercih Yönetim Sistemi",
    add_completion=False,
)
console = Console()


@app.command("tui")
def launch_tui():
    """Textual TUI İnteraktif Terminal Arayüzünü Başlatır."""
    try:
        from tui.app import YKSTahminApp
        tui_app = YKSTahminApp()
        tui_app.run()
    except Exception as e:
        console.print(f"[bold red]TUI Başlatılamadı:[/bold red] {e}")


@app.command("search")
def search_cli(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Arama kelimesi (Üniversite veya Bölüm)"),
    city: Optional[str] = typer.Option(None, "--city", "-c", help="Şehir adı"),
    point_type: Optional[str] = typer.Option(None, "--point-type", "-p", help="Puan Türü (SAY, EA, SÖZ, DİL)"),
    limit: int = typer.Option(15, "--limit", "-l", help="Sonuç limiti"),
):
    """Programlarda hızlı filtreli arama yapar ve Rich tablosu olarak basar."""
    results = SearchService.search_programs(
        search_query=query,
        il_adi=city,
        puan_turu=point_type,
        limit=limit,
    )

    if not results:
        console.print("[yellow]Arama kriterlerine uygun program bulunamadı.[/yellow]")
        return

    table = Table(title=f"🔎 YKS Program Arama Sonuçları ({len(results)} Program)", header_style="bold magenta")
    table.add_column("Kılavuz Kodu", style="cyan", justify="center")
    table.add_column("Üniversite", style="bold white")
    table.add_column("Bölüm", style="green")
    table.add_column("Şehir", style="yellow")
    table.add_column("Puan Türü", style="magenta", justify="center")
    table.add_column("2025 Taban Sıra", style="bold cyan", justify="right")
    table.add_column("Kontenjan", justify="right")

    for r in results:
        lag1_rank = float(r.get('lag1_taban_siralama') or 0)
        lag1_k = float(r.get('lag1_genel_kontenjan') or 0)
        table.add_row(
            str(r.get("kilavuz_kodu", "")),
            str(r.get("universite_adi", "")),
            str(r.get("birim_grup_adi", "")),
            str(r.get("il_adi", "")),
            str(r.get("puan_turu", "")),
            f"{lag1_rank:,.0f}",
            f"{lag1_k:.0f}",
        )

    console.print(table)


@app.command("stats")
def stats_cli():
    """Türkiye geneli makro istatistik özetini gösterir."""
    stats = AnalyticsService.get_nationwide_stats()
    
    panel_text = f"""
    [bold cyan]Toplam Program Sayısı:[/bold cyan] {stats['total_programs']:,d}
    [bold cyan]Toplam Üniversite Sayısı:[/bold cyan] {stats['total_universities']:,d}
    [bold cyan]Toplam Bölüm Ailesi Sayısı:[/bold cyan] {stats['total_departments']:,d}
    [bold cyan]Ortalama Taban Sıralama:[/bold cyan] {stats['mean_rank']:,.0f}
    [bold cyan]Toplam Kontenjan Hacmi:[/bold cyan] {stats['total_quota']:,.0f}
    """
    console.print(Panel(panel_text, title="📊 Türkiye Geneli Makro İstatistik Özeti", border_style="green"))


@app.command("export")
def export_cli(
    list_id: int = typer.Option(..., "--list-id", "-i", help="Dışa aktarılacak Tercih Listesi ID"),
    format_type: str = typer.Option("markdown", "--format", "-f", help="Format: csv, json, markdown, excel, pdf"),
    output: str = typer.Option("tercih_listesi_raporu", "--output", "-o", help="Çıktı dosya adı"),
):
    """Tercih listesini istenen formatta dışa aktarır."""
    analysis = PreferenceService.analyze_preference_list(list_id)
    if "error" in analysis:
        console.print(f"[bold red]Hata:[/bold red] {analysis['error']}")
        return

    ext_map = {"csv": ".csv", "json": ".json", "markdown": ".md", "excel": ".xlsx", "pdf": ".pdf"}
    ext = ext_map.get(format_type.lower(), ".md")
    target_path = f"{output}{ext}"

    if format_type.lower() == "csv":
        res_path = ExportService.export_to_csv(analysis["items"], target_path)
    elif format_type.lower() == "json":
        res_path = ExportService.export_to_json(analysis, target_path)
    elif format_type.lower() == "excel":
        res_path = ExportService.export_to_excel(analysis["items"], target_path)
    elif format_type.lower() == "pdf":
        res_path = ExportService.export_to_pdf(analysis, target_path)
    else:
        res_path = ExportService.export_to_markdown(analysis, target_path)

    console.print(f"[bold green]✅ Başarıyla Dışa Aktarıldı:[/bold green] {res_path}")


if __name__ == "__main__":
    app()
