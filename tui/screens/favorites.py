"""
Textual TUI — Favori Programlar Ekranı.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Label, DataTable

from db.repository import FavoriteRepository


class FavoritesScreen(Screen):
    """Favoriler Ekranı."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("⭐ FAVORİ PROGRAMLARIM", id="title_favs")

        with Horizontal():
            yield Button("❌ Favorilerden Çıkar", id="btn_remove_fav", variant="error")
            yield Button("📋 Tercih Listeme Ekle", id="btn_fav_to_list", variant="success")

        yield DataTable(id="favs_datatable", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#favs_datatable", DataTable)
        table.add_columns("Kılavuz Kodu", "Üniversite", "Bölüm", "Puan Türü", "Şehir", "Ekleme Tarihi")
        self.load_favorites()

    def load_favorites(self) -> None:
        favs = FavoriteRepository.get_all_favorites()
        table = self.query_one("#favs_datatable", DataTable)
        table.clear()

        for f in favs:
            table.add_row(
                str(f["kilavuz_kodu"]),
                str(f["universite_adi"]),
                str(f["birim_grup_adi"]),
                str(f["puan_turu"]),
                str(f["il_adi"]),
                str(f["created_at"])[:10],
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        table = self.query_one("#favs_datatable", DataTable)
        if table.cursor_row is not None:
            row_data = table.get_row_at(table.cursor_row)
            if row_data:
                k_kodu = int(row_data[0])
                if event.button.id == "btn_remove_fav":
                    FavoriteRepository.remove_favorite(k_kodu)
                    self.notify("⭐ Favorilerden çıkarıldı.", title="Başarılı")
                    self.load_favorites()
