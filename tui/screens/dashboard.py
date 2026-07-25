"""
Textual TUI — Dashboard Ekranı.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Input, Label, Static, DataTable
from rich.panel import Panel

from services.analytics_service import AnalyticsService
from db.repository import FavoriteRepository, SearchHistoryRepository


class DashboardScreen(Screen):
    """Ana Dashboard Ekranı."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("🎓 YKS 2026 TABAN SIRALAMA TAHMİN SİSTEMİ — DASHBOARD", id="title_dashboard")

        with Grid(id="dash_grid"):
            # Makro Özet Kartları
            with Vertical(classes="dash_card"):
                yield Label("📊 TÜRKİYE GENELİ ÖZET", classes="card_header")
                yield Static(id="stats_summary_static")

            with Vertical(classes="dash_card"):
                yield Label("⭐ FAVORİ PROGRAMLARIM", classes="card_header")
                yield Static(id="favorites_static")

            with Vertical(classes="dash_card"):
                yield Label("🕒 SON İŞLEMLER", classes="card_header")
                yield Static(id="history_static")

        yield Label("🔥 TREND ÖZETİ (EN ÇOK YÜKSELEN 5 PROGRAM)", classes="section_title")
        yield DataTable(id="dash_trends_table")

        yield Footer()

    def on_mount(self) -> None:
        self.load_dashboard_data()

    def load_dashboard_data(self) -> None:
        """Dashboard verilerini servislerden yükler."""
        # 1. Makro İstatistik
        stats = AnalyticsService.get_nationwide_stats()
        stats_text = (
            f"• Toplam Program: [bold cyan]{stats['total_programs']:,d}[/bold cyan]\n"
            f"• Toplam Üniversite: [bold cyan]{stats['total_universities']:,d}[/bold cyan]\n"
            f"• Toplam Bölüm: [bold cyan]{stats['total_departments']:,d}[/bold cyan]\n"
            f"• Ortalama Taban Sıralama: [bold cyan]{stats['mean_rank']:,.0f}[/bold cyan]\n"
            f"• Toplam Kontenjan Hacmi: [bold cyan]{stats['total_quota']:,.0f}[/bold cyan]"
        )
        self.query_one("#stats_summary_static", Static).update(stats_text)

        # 2. Favoriler
        favs = FavoriteRepository.get_all_favorites()
        if not favs:
            fav_text = "[italic yellow]Henüz favori program eklenmedi.[/italic yellow]"
        else:
            fav_text = "\n".join([f"★ [bold]{f['universite_adi']}[/bold] - {f['birim_grup_adi']}" for f in favs[:5]])
        self.query_one("#favorites_static", Static).update(fav_text)

        # 3. Son İşlemler
        hist = SearchHistoryRepository.get_recent_history(5)
        if not hist:
            hist_text = "[italic yellow]Arama geçmişi bulunmuyor.[/italic yellow]"
        else:
            hist_text = "\n".join([f"• [{h['created_at']}] {h['summary']}" for h in hist])
        self.query_one("#history_static", Static).update(hist_text)

        # 4. Trends Tablosu
        table = self.query_one("#dash_trends_table", DataTable)
        table.clear()
        table.add_columns("Üniversite", "Bölüm", "Puan", "2025 Sıralama", "Sıralama Trend")
        
        top_risers = AnalyticsService.get_top_risers(5)
        for r in top_risers:
            table.add_row(
                str(r.get("universite_adi", "")),
                str(r.get("birim_grup_adi", "")),
                str(r.get("puan_turu", "")),
                f"{float(r.get('lag1_taban_siralama', 0)):,.0f}",
                f"[green]{float(r.get('siralama_trend', 0)):,.0f} sıra iyileşme[/green]"
            )
