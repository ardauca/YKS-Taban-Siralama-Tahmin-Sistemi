"""
Textual TUI — Gerçek Zamanlı Çoklu-Filtreleme Arama Ekranı.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Input, Label, Select, Static, DataTable

from services.search_service import SearchService
from db.repository import SearchHistoryRepository


class SearchScreen(Screen):
    """Çoklu-filtreleme Arama Ekranı."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("🔎 YKS PROGRAM ARAMA VE ÇOKLU FİLTRELEME MOTORU", id="title_search")

        with Horizontal(id="filter_bar"):
            yield Input(placeholder="Kelime arama (Üniversite veya Bölüm)...", id="input_query")
            yield Input(placeholder="Şehir (ör. İstanbul)", id="input_city")
            yield Select(
                options=[("Tümü", "TÜMÜ"), ("SAY", "SAY"), ("EA", "EA"), ("SÖZ", "SÖZ"), ("DİL", "DİL")],
                value="TÜMÜ",
                id="select_point_type"
            )
            yield Input(placeholder="Max Sıralama (ör. 50000)", id="input_max_rank")
            yield Button("Ara / Filtrele", variant="primary", id="btn_search")

        yield DataTable(id="search_datatable", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#search_datatable", DataTable)
        table.add_columns(
            "Kılavuz Kodu",
            "Üniversite Adı",
            "Bölüm / Birim Grubu",
            "Şehir",
            "Puan Türü",
            "Üniverite Türü",
            "2025 Taban Sıralama",
            "Genel Kontenjan"
        )
        self.perform_search()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_search":
            self.perform_search()

    def on_input_submit(self, event: Input.Submitted) -> None:
        self.perform_search()

    def perform_search(self) -> None:
        """Filtre girdilerini okur ve Polars search servisiyle sonuçları günceller."""
        query = self.query_one("#input_query", Input).value.strip()
        city = self.query_one("#input_city", Input).value.strip()
        pt_val = self.query_one("#select_point_type", Select).value
        
        max_rank_str = self.query_one("#input_max_rank", Input).value.strip()
        max_rank = float(max_rank_str) if max_rank_str.isdigit() else None

        results = SearchService.search_programs(
            search_query=query if query else None,
            il_adi=city if city else None,
            puan_turu=str(pt_val) if pt_val != "TÜMÜ" else None,
            max_rank=max_rank,
            limit=150,
        )

        table = self.query_one("#search_datatable", DataTable)
        table.clear()

        for r in results:
            table.add_row(
                str(r.get("kilavuz_kodu", "")),
                str(r.get("universite_adi", "")),
                str(r.get("birim_grup_adi", "")),
                str(r.get("il_adi", "")),
                str(r.get("puan_turu", "")),
                str(r.get("universite_turu", "")),
                f"{float(r.get('lag1_taban_siralama', 0)):,.0f}",
                f"{float(r.get('lag1_genel_kontenjan', 0)):.0f}",
            )

        if query or city:
            SearchHistoryRepository.add_history("SEARCH", f"Arama: {query} | Şehir: {city} | Sonuç: {len(results)}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Tablodan bir satır seçildiğinde Program Detay ekranına geçer."""
        row_key = event.row_key
        table = self.query_one("#search_datatable", DataTable)
        row_data = table.get_row(row_key)
        
        if row_data:
            kilavuz_kodu = int(row_data[0])
            self.app.push_screen("detail", kilavuz_kodu=kilavuz_kodu)
