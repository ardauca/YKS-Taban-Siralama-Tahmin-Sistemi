"""
Textual TUI — 2026 ML Simülasyon Sonuçları Ekranı.
Tüm 16,957 program için CatBoost tahminlerini interaktif tabloda gösterir.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Input, Label, Select, Static, DataTable
from textual.widgets.select import NoSelection

from services.search_service import SearchService
from tui.screens.detail import DetailScreen


class SimulationScreen(Screen):
    """2026 ML Simülasyon Ekranı — Tüm CatBoost Tahminleri."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("🤖 2026 YKS ML SİMÜLASYON SONUÇLARI (CatBoost Tahminleri)", id="title_search")

        with Horizontal(id="filter_bar"):
            yield Input(placeholder="Üniversite veya Bölüm filtrele...", id="sim_query")
            yield Select(
                options=[
                    ("Tüm Puan Türleri", "TÜMÜ"),
                    ("SAY", "SAY"), ("EA", "EA"), ("SÖZ", "SÖZ"), ("DİL", "DİL"),
                ],
                value="TÜMÜ",
                id="sim_puan_turu",
                allow_blank=False,
            )
            yield Select(
                options=[
                    ("Tüm Üni Türleri", "TÜMÜ"),
                    ("DEVLET", "DEVLET"),
                    ("VAKIF", "VAKIF"),
                ],
                value="TÜMÜ",
                id="sim_uni_turu",
                allow_blank=False,
            )
            yield Select(
                options=[
                    ("Tüm Trendler", "TÜMÜ"),
                    ("🟢 Yükselen", "YUKSELENLER"),
                    ("🔴 Gerileyen", "GERILEYENLER"),
                    ("🟡 Stabil", "STABIL"),
                ],
                value="TÜMÜ",
                id="sim_trend_filter",
                allow_blank=False,
            )
            yield Button("🔍 Filtrele", variant="primary", id="btn_sim_search")

        yield Static(id="sim_info", classes="card_header")
        yield DataTable(id="sim_datatable", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#sim_datatable", DataTable)
        table.add_columns(
            "Kod",
            "Üniversite",
            "Bölüm",
            "Şehir",
            "Puan",
            "Tür",
            "2025 Sıra",
            "2026 Tahmin",
            "Alt Sınır",
            "Üst Sınır",
            "Değişim",
            "Risk",
        )
        self._load_data()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_sim_search":
            self._load_data()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._load_data()

    def _get_select(self, widget_id: str) -> str | None:
        val = self.query_one(widget_id, Select).value
        if isinstance(val, NoSelection) or str(val) in ("TÜMÜ", ""):
            return None
        return str(val)

    def _load_data(self) -> None:
        """Filtrelere göre simülasyon verilerini yükler."""
        query = self.query_one("#sim_query", Input).value.strip() or None
        puan_turu = self._get_select("#sim_puan_turu")
        uni_turu = self._get_select("#sim_uni_turu")
        trend_filter = self._get_select("#sim_trend_filter")

        # Trend filtresini sıralama filtrelerine çevir
        min_pred: float | None = None
        max_pred: float | None = None
        min_rank: float | None = None
        max_rank: float | None = None

        results = SearchService.search_programs(
            search_query=query,
            puan_turu=puan_turu,
            universite_turu=uni_turu,
            limit=500,
        )

        # Trend filtresi uygula (client tarafı)
        if trend_filter == "YUKSELENLER":
            results = [r for r in results if int(r.get("pred_degisim") or 0) < -2000]
        elif trend_filter == "GERILEYENLER":
            results = [r for r in results if int(r.get("pred_degisim") or 0) > 2000]
        elif trend_filter == "STABIL":
            results = [r for r in results if abs(int(r.get("pred_degisim") or 0)) <= 2000]

        table = self.query_one("#sim_datatable", DataTable)
        table.clear()

        for r in results[:400]:  # Tablo performansı için limit
            r_rank = float(r.get("lag1_taban_siralama") or 0.0)
            r_pred = int(r.get("pred_2026") or 0)
            r_lower = int(r.get("pred_lower") or 0)
            r_upper = int(r.get("pred_upper") or 0)
            r_deg = int(r.get("pred_degisim") or 0)
            risk_renk = str(r.get("risk_renk") or "🟡")

            if r_deg < -5000:
                deg_str = f"[green]{r_deg:+,}[/green]"
            elif r_deg > 5000:
                deg_str = f"[red]{r_deg:+,}[/red]"
            else:
                deg_str = f"[yellow]{r_deg:+,}[/yellow]"

            table.add_row(
                str(r.get("kilavuz_kodu", "")),
                str(r.get("universite_adi", ""))[:35],
                str(r.get("birim_grup_adi") or r.get("birim_adi", ""))[:32],
                str(r.get("il_adi", "")),
                str(r.get("puan_turu", "")),
                str(r.get("universite_turu", ""))[:6],
                f"{r_rank:,.0f}" if r_rank > 0 else "-",
                f"{r_pred:,}" if r_pred > 0 else "[dim]Yok[/dim]",
                f"{r_lower:,}" if r_lower > 0 else "-",
                f"{r_upper:,}" if r_upper > 0 else "-",
                deg_str,
                risk_renk[:10],
            )

        info = self.query_one("#sim_info", Static)
        ml_count = len([r for r in results if int(r.get("pred_2026") or 0) > 0])
        info.update(
            f"[bold cyan]{len(results)} program[/bold cyan] | "
            f"[bold green]{ml_count} ML tahmini[/bold green] mevcut"
        )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        try:
            row_data = self.query_one("#sim_datatable", DataTable).get_row(event.row_key)
            if row_data:
                self.app.push_screen(DetailScreen(kilavuz_kodu=int(row_data[0])))
        except (ValueError, IndexError, Exception):
            pass
