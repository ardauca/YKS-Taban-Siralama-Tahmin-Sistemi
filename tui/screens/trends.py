"""
Textual TUI — Trend Analizi Ekranı.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Label, Static, DataTable

from services.analytics_service import AnalyticsService


class TrendsScreen(Screen):
    """Trend Analizi Ekranı."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("🔥 YKS TREND VE DALGALANMA ANALİZ MODÜLÜ", id="title_trends")

        with Horizontal():
            with Vertical(classes="trend_col"):
                yield Label("🚀 EN ÇOK YÜKSELENLER (BAŞARI ARTIŞI)", classes="section_title")
                yield DataTable(id="table_risers")

            with Vertical(classes="trend_col"):
                yield Label("📉 EN ÇOK DÜŞENLER (GERİLEYENLER)", classes="section_title")
                yield DataTable(id="table_decliners")

        yield Footer()

    def on_mount(self) -> None:
        tr_table = self.query_one("#table_risers", DataTable)
        tr_table.add_columns("Üniversite", "Bölüm", "2025 Sıra", "İyileşme Trendi")

        dec_table = self.query_one("#table_decliners", DataTable)
        dec_table.add_columns("Üniversite", "Bölüm", "2025 Sıra", "Gerileme Trendi")

        self.load_trends()

    def load_trends(self) -> None:
        risers = AnalyticsService.get_top_risers(12)
        tr_table = self.query_one("#table_risers", DataTable)
        tr_table.clear()
        for r in risers:
            tr_table.add_row(
                str(r.get("universite_adi", "")),
                str(r.get("birim_grup_adi", "")),
                f"{float(r.get('lag1_taban_siralama', 0)):,.0f}",
                f"[green]{float(r.get('siralama_trend', 0)):,.0f} sıra[/green]"
            )

        decliners = AnalyticsService.get_top_decliners(12)
        dec_table = self.query_one("#table_decliners", DataTable)
        dec_table.clear()
        for d in decliners:
            dec_table.add_row(
                str(d.get("universite_adi", "")),
                str(d.get("birim_grup_adi", "")),
                f"{float(d.get('lag1_taban_siralama', 0)):,.0f}",
                f"[red]+{float(d.get('siralama_trend', 0)):,.0f} sıra[/red]"
            )
