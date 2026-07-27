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


@app.command("detail")
def detail_cli(
    code: int = typer.Argument(..., help="Kılavuz kodu (ör. 203111263)"),
):
    """Bir programa ait 4 yıllık tarihsel tablo + 2026 ML tahminini gösterir."""
    prog = SearchService.get_program_by_code(code)
    if not prog:
        console.print(f"[bold red]Hata:[/bold red] {code} kodu bulunamadı.")
        raise typer.Exit(1)

    u_name = str(prog.get("universite_adi", "-"))
    b_name = str(prog.get("birim_grup_adi") or prog.get("birim_adi", "-"))
    il     = str(prog.get("il_adi", "-"))
    pt     = str(prog.get("puan_turu", "-"))
    u_turu = str(prog.get("universite_turu", "-"))
    ogr_t  = str(prog.get("ogretim_turu", "-"))
    burs   = str(prog.get("burs_orani", "-"))
    k_say  = float(prog.get("lag1_genel_kontenjan") or 0)
    risk   = str(prog.get("risk_renk") or "🟡 STABIL")

    info = (
        f"\n[bold]Üniversite:[/bold] {u_name}\n"
        f"[bold]Bölüm:[/bold]      {b_name}\n"
        f"[bold]Şehir:[/bold] {il}  |  [bold]Puan:[/bold] {pt}  |  "
        f"[bold]Tür:[/bold] {u_turu}  |  [bold]Öğretim:[/bold] {ogr_t}\n"
        f"[bold]Burs:[/bold] {burs}  |  [bold]Kontenjan:[/bold] {k_say:.0f}  |  "
        f"[bold]Risk:[/bold] {risk}\n"
    )
    console.print(Panel(info, title=f"📌 Program Detayı — {code}", border_style="cyan"))

    # ── 4 Yıllık Tarihsel Tablo ───────────────────────────────────────────────
    hist_table = Table(header_style="bold magenta", title="📊 4 Yıllık Taban Sıralama ve Puan Geçmişi")
    hist_table.add_column("Yıl",              justify="center", style="bold")
    hist_table.add_column("Taban Sıralama",   justify="right",  style="cyan")
    hist_table.add_column("Değişim",          justify="right")
    hist_table.add_column("Taban Puanı",      justify="right",  style="yellow")
    hist_table.add_column("Kaynak")

    HIST = [
        (2022, "lag4_taban_siralama", "lag4_taban_puan", "Ham CSV"),
        (2023, "lag3_taban_siralama", "lag3_taban_puan", "Ham CSV"),
        (2024, "lag2_taban_siralama", "lag2_taban_puan", "Ham CSV"),
        (2025, "lag1_taban_siralama", "lag1_taban_puan", "ÖSYM 2025"),
    ]

    prev_rank = 0.0
    for year, sira_col, puan_col, kaynak in HIST:
        sira = float(prog.get(sira_col) or 0.0)
        puan = float(prog.get(puan_col) or 0.0)
        if prev_rank > 0 and sira > 0:
            deg = sira - prev_rank
            if deg < -2000:
                deg_str = f"[green]{deg:+,.0f}[/green]"
            elif deg > 2000:
                deg_str = f"[red]{deg:+,.0f}[/red]"
            else:
                deg_str = f"[yellow]{deg:+,.0f}[/yellow]"
        else:
            deg_str = "-"
        hist_table.add_row(
            str(year),
            f"{sira:,.0f}" if sira > 0 else "[dim]Veri yok[/dim]",
            deg_str,
            f"{puan:.3f}" if puan > 0 else "[dim]-[/dim]",
            kaynak,
        )
        if sira > 0:
            prev_rank = sira

    # 2026 ML satırı
    pred  = float(prog.get("pred_2026") or 0)
    lower = float(prog.get("pred_lower") or 0)
    upper = float(prog.get("pred_upper") or 0)
    deg26 = float(prog.get("pred_degisim") or 0)
    ml_src = "CatBoost ML" if pred > 0 else "Fallback"
    if not (pred > 0):
        lag1  = float(prog.get("lag1_taban_siralama") or 0)
        trend = float(prog.get("siralama_trend") or 0)
        pred  = max(1.0, lag1 + trend * 0.3) if lag1 > 0 else lag1
        lower, upper = pred * 0.8, pred * 1.25
        deg26 = pred - lag1

    deg26_str = (
        f"[green]{deg26:+,.0f}[/green]" if deg26 < -5000
        else f"[red]{deg26:+,.0f}[/red]" if deg26 > 5000
        else f"[yellow]{deg26:+,.0f}[/yellow]"
    )
    hist_table.add_row(
        "[bold yellow]2026 ✨[/bold yellow]",
        f"[bold cyan]{pred:,.0f}[/bold cyan]  ([dim]{lower:,.0f}–{upper:,.0f}[/dim])",
        deg26_str,
        "[dim]~[/dim]",
        f"✅ {ml_src}",
    )
    console.print(hist_table)


@app.command("search")
def search_cli(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Arama kelimesi (Üniversite veya Bölüm)"),
    city: Optional[str] = typer.Option(None, "--city", "-c", help="Şehir adı"),
    point_type: Optional[str] = typer.Option(None, "--point-type", "-p", help="Puan Türü (SAY, EA, SÖZ, DİL)"),
    uni_turu: Optional[str] = typer.Option(None, "--uni-turu", "-u", help="Üni Türü (DEVLET, VAKIF)"),
    burs: Optional[str] = typer.Option(None, "--burs", "-b", help="Burs (Burslu, Ücretli, %50 İndirimli...)"),
    max_rank: Optional[float] = typer.Option(None, "--max-rank", "-r", help="Max 2025 taban sıralaması"),
    max_pred: Optional[float] = typer.Option(None, "--max-pred", "-m", help="Max 2026 ML tahmini"),
    limit: int = typer.Option(15, "--limit", "-l", help="Sonuç limiti"),
):
    """Programlarda hızlı filtreli arama yapar — 2026 ML tahmini dahil."""
    results = SearchService.search_programs(
        search_query=query,
        il_adi=city,
        puan_turu=point_type,
        universite_turu=uni_turu,
        burs_orani=burs,
        max_rank=max_rank,
        max_pred=max_pred,
        limit=limit,
    )

    if not results:
        console.print("[yellow]Arama kriterlerine uygun program bulunamadı.[/yellow]")
        return

    table = Table(
        title=f"🔎 YKS Arama — {len(results)} Program",
        header_style="bold magenta",
    )
    table.add_column("Kılavuz Kodu", style="cyan", justify="center")
    table.add_column("Üniversite", style="bold white")
    table.add_column("Bölüm", style="green")
    table.add_column("Şehir", style="yellow")
    table.add_column("Puan", justify="center")
    table.add_column("Tür", justify="center")
    table.add_column("Burs", style="cyan")
    table.add_column("2025 Sıra", justify="right")
    table.add_column("2026 ML Tahmin", style="bold cyan", justify="right")
    table.add_column("Değişim", justify="right")

    for r in results:
        lag1_rank = float(r.get("lag1_taban_siralama") or 0)
        r_pred = int(r.get("pred_2026") or 0)
        r_deg = int(r.get("pred_degisim") or 0)
        deg_str = f"{r_deg:+,}" if r_deg != 0 else "-"
        table.add_row(
            str(r.get("kilavuz_kodu", "")),
            str(r.get("universite_adi", ""))[:35],
            str(r.get("birim_grup_adi") or r.get("birim_adi", ""))[:32],
            str(r.get("il_adi", "")),
            str(r.get("puan_turu", "")),
            str(r.get("universite_turu", ""))[:6],
            str(r.get("burs_orani", "")),
            f"{lag1_rank:,.0f}" if lag1_rank > 0 else "-",
            f"{r_pred:,}" if r_pred > 0 else "-",
            deg_str,
        )
    console.print(table)


@app.command("stats")
def stats_cli():
    """Türkiye geneli makro istatistik özetini gösterir."""
    stats = AnalyticsService.get_nationwide_stats()

    uni_turu = stats.get("uni_turu_counts", {})
    devlet = uni_turu.get("DEVLET", 0)
    vakif = uni_turu.get("VAKIF", 0)

    panel_text = (
        f"\n[bold cyan]Toplam Program:[/bold cyan] {stats['total_programs']:,d}\n"
        f"[bold cyan]Toplam Üniversite:[/bold cyan] {stats['total_universities']:,d}\n"
        f"[bold cyan]Devlet / Vakıf:[/bold cyan] {devlet:,d} / {vakif:,d}\n"
        f"[bold cyan]Ortalama Taban Sıralama:[/bold cyan] {stats['mean_rank']:,.0f}\n"
        f"[bold cyan]Toplam Kontenjan:[/bold cyan] {stats['total_quota']:,.0f}\n"
        f"\n[bold green]ML 2026 Kapsam:[/bold green] {stats.get('sim_program_count', 0):,d} program\n"
        f"[bold green]Ort. 2026 ML Tahmini:[/bold green] {stats.get('mean_pred_2026', 0):,.0f}"
    )
    console.print(Panel(panel_text, title="📊 Türkiye Geneli İstatistik Özeti", border_style="green"))


@app.command("simulate")
def simulate_cli(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Bölüm veya üniversite filtresi"),
    point_type: Optional[str] = typer.Option(None, "--point-type", "-p", help="Puan Türü"),
    trend: Optional[str] = typer.Option(
        None, "--trend", "-t", help="Trend: yukselenler | gerileyenler | stabil"
    ),
    limit: int = typer.Option(20, "--limit", "-l", help="Sonuç limiti"),
):
    """2026 CatBoost ML Simülasyon Sonuçlarını Gösterir."""
    results = SearchService.search_programs(
        search_query=query, puan_turu=point_type, limit=500
    )

    # Trend filtresi
    if trend == "yukselenler":
        results = [r for r in results if int(r.get("pred_degisim") or 0) < -2000]
    elif trend == "gerileyenler":
        results = [r for r in results if int(r.get("pred_degisim") or 0) > 2000]
    elif trend == "stabil":
        results = [r for r in results if abs(int(r.get("pred_degisim") or 0)) <= 2000]

    results = results[:limit]
    ml_count = len([r for r in results if int(r.get("pred_2026") or 0) > 0])

    table = Table(
        title=f"🤖 2026 ML Simülasyon — {len(results)} Program | {ml_count} ML Tahmini",
        header_style="bold magenta",
    )
    table.add_column("Üniversite", style="bold white")
    table.add_column("Bölüm", style="green")
    table.add_column("Puan", justify="center")
    table.add_column("Şehir")
    table.add_column("2025 Sıra", justify="right")
    table.add_column("2026 ML", style="bold cyan", justify="right")
    table.add_column("Alt", style="green", justify="right")
    table.add_column("Üst", style="red", justify="right")
    table.add_column("Değişim", justify="right")
    table.add_column("Risk", justify="center")

    for r in results:
        lag1 = float(r.get("lag1_taban_siralama") or 0)
        pred = int(r.get("pred_2026") or 0)
        lower = int(r.get("pred_lower") or 0)
        upper = int(r.get("pred_upper") or 0)
        deg = int(r.get("pred_degisim") or 0)
        risk = str(r.get("risk_renk") or "-")
        table.add_row(
            str(r.get("universite_adi", ""))[:32],
            str(r.get("birim_grup_adi") or r.get("birim_adi", ""))[:30],
            str(r.get("puan_turu", "")),
            str(r.get("il_adi", "")),
            f"{lag1:,.0f}" if lag1 > 0 else "-",
            f"{pred:,}" if pred > 0 else "[dim]Yok[/dim]",
            f"{lower:,}" if lower > 0 else "-",
            f"{upper:,}" if upper > 0 else "-",
            f"{deg:+,}" if deg != 0 else "-",
            risk[:12],
        )
    console.print(table)


@app.command("university")
def university_cli(
    name: str = typer.Argument(..., help="Üniversite adı (kısmi eşleşme desteklenir)"),
):
    """Bir üniversitenin tüm programlarını ve 2026 ML tahminlerini gösterir."""
    from services.analytics_service import AnalyticsService
    summary = AnalyticsService.get_university_summary(name)

    if "error" in summary:
        console.print(f"[bold red]Hata:[/bold red] {summary['error']}")
        return

    mean_r = float(summary.get("mean_rank_2025") or 0)
    mean_p = float(summary.get("mean_pred_2026") or 0)
    total_k = float(summary.get("total_quota") or 0)
    direction = "↑ İyileşiyor" if mean_p < mean_r else ("↓ Geriliyor" if mean_p > mean_r else "→ Stabil")

    info = (
        f"\n[bold]Toplam Program:[/bold] {summary.get('total_programs', 0)}\n"
        f"[bold]2025 Ort. Taban Sıra:[/bold] {mean_r:,.0f}\n"
        f"[bold]2026 ML Ort. Tahmin:[/bold] {mean_p:,.0f}\n"
        f"[bold]Genel Yönelim:[/bold] {direction}\n"
        f"[bold]Toplam Kontenjan:[/bold] {total_k:.0f}\n"
    )
    risk_dist = summary.get("risk_distribution", {})
    for label, cnt in sorted(risk_dist.items(), key=lambda x: x[1], reverse=True):
        info += f"\n  {label}: {cnt} program"

    console.print(Panel(
        info,
        title=f"🏛️ {summary.get('universite_adi', name)}",
        border_style="cyan",
    ))

    table = Table(header_style="bold magenta")
    table.add_column("Bölüm", style="green")
    table.add_column("Puan")
    table.add_column("Burs", style="cyan")
    table.add_column("2025 Sıra", justify="right")
    table.add_column("2026 ML", style="bold cyan", justify="right")
    table.add_column("Değişim", justify="right")
    table.add_column("Risk")

    for p in summary.get("programs", []):
        lag1 = float(p.get("lag1_taban_siralama") or 0)
        pred = int(p.get("pred_2026") or 0)
        deg = int(p.get("pred_degisim") or 0)
        table.add_row(
            str(p.get("birim_grup_adi") or p.get("birim_adi", ""))[:38],
            str(p.get("puan_turu", "")),
            str(p.get("burs_orani", "")),
            f"{lag1:,.0f}" if lag1 > 0 else "-",
            f"{pred:,}" if pred > 0 else "-",
            f"{deg:+,}" if deg != 0 else "-",
            str(p.get("risk_renk") or "")[:12],
        )
    console.print(table)


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

