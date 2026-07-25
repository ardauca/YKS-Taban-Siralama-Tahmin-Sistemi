"""
Textual TUI — Makro İstatistikler Ekranı.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Header, Footer, Label, Static

from services.analytics_service import AnalyticsService


class StatsScreen(Screen):
    """İstatistikler Ekranı."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("📊 TÜRKİYE GENELİ YKS MAKRO İSTATİSTİKLERİ", id="title_stats")

        with Vertical():
            yield Static(id="macro_stats_panel")

        yield Footer()

    def on_mount(self) -> None:
        stats = AnalyticsService.get_nationwide_stats()
        panel_text = (
            f"• [bold cyan]Toplam Program Sayısı:[/bold cyan] {stats['total_programs']:,d}\n"
            f"• [bold cyan]Toplam Üniversite Sayısı:[/bold cyan] {stats['total_universities']:,d}\n"
            f"• [bold cyan]Toplam Bölüm Ailesi Sayısı:[/bold cyan] {stats['total_departments']:,d}\n"
            f"• [bold cyan]Ortalama Taban Sıralama:[/bold cyan] {stats['mean_rank']:,.0f}\n"
            f"• [bold cyan]En Yüksek (En Başarılı) Sıralama:[/bold cyan] {stats['min_rank']:,.0f}\n"
            f"• [bold cyan]En Düşük Taban Sıralama:[/bold cyan] {stats['max_rank']:,.0f}\n"
            f"• [bold cyan]Toplam Kontenjan Hacmi:[/bold cyan] {stats['total_quota']:,.0f} Öğrenci\n\n"
            f"[bold magenta]📌 PUAN TÜRÜ DAĞILIMI:[/bold magenta]\n"
        )
        for pt, count in stats.get("point_type_counts", {}).items():
            panel_text += f"• [bold]{pt}:[/bold] {count:,d} Program\n"

        self.query_one("#macro_stats_panel", Static).update(panel_text)
