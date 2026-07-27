"""
Textual TUI — Favori Programlar Ekranı.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Label, DataTable, Static

from db.repository import FavoriteRepository, PreferenceListRepository
from tui.screens.detail import DetailScreen


class FavoritesScreen(Screen):
    """Favoriler Ekranı."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("⭐ FAVORİ PROGRAMLARIM", id="title_favs")

        with Horizontal():
            yield Button("❌ Favorilerden Çıkar", id="btn_remove_fav", variant="error")
            yield Button("📋 Tercih Listeme Ekle", id="btn_fav_to_list", variant="success")
            yield Button("🔍 Detayları Gör", id="btn_fav_detail", variant="primary")

        yield Static(id="favs_info", classes="section_title")
        yield DataTable(id="favs_datatable", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#favs_datatable", DataTable)
        table.add_columns(
            "Kılavuz Kodu",
            "Üniversite",
            "Bölüm",
            "Puan Türü",
            "Şehir",
            "Ekleme Tarihi",
        )
        self.load_favorites()

    def load_favorites(self) -> None:
        """Favori programları yeniden yükler."""
        favs = FavoriteRepository.get_all_favorites()
        table = self.query_one("#favs_datatable", DataTable)
        table.clear()

        if not favs:
            self.query_one("#favs_info", Static).update(
                "[italic yellow]Henüz favori program eklenmedi. "
                "Arama ekranında bir programa tıklayıp ⭐ Favorilere Ekle butonunu kullanın.[/italic yellow]"
            )
        else:
            self.query_one("#favs_info", Static).update(
                f"[bold cyan]{len(favs)} favori program[/bold cyan] kayıtlı."
            )
            for f in favs:
                table.add_row(
                    str(f["kilavuz_kodu"]),
                    str(f["universite_adi"]),
                    str(f["birim_grup_adi"]),
                    str(f["puan_turu"]),
                    str(f["il_adi"]),
                    str(f["created_at"])[:10],
                )

    def _get_selected_kilavuz_kodu(self) -> int | None:
        """Seçili satırdan kılavuz kodunu döndürür."""
        table = self.query_one("#favs_datatable", DataTable)
        if table.cursor_row is None:
            self.notify("⚠️ Lütfen önce bir program satırı seçin.", title="Uyarı")
            return None
        try:
            row_data = table.get_row_at(table.cursor_row)
            return int(row_data[0])
        except (ValueError, IndexError):
            return None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_remove_fav":
            k_kodu = self._get_selected_kilavuz_kodu()
            if k_kodu:
                FavoriteRepository.remove_favorite(k_kodu)
                self.notify("🗑️ Favorilerden çıkarıldı.", title="Başarılı")
                self.load_favorites()

        elif event.button.id == "btn_fav_to_list":
            k_kodu = self._get_selected_kilavuz_kodu()
            if k_kodu:
                lists = PreferenceListRepository.get_all_lists()
                if not lists:
                    plist = PreferenceListRepository.create_list(
                        "Tercih Listem 2026", target_rank=180000, point_type="EA"
                    )
                    list_id = plist.id
                else:
                    list_id = lists[0]["id"]
                PreferenceListRepository.add_item_to_list(list_id, k_kodu)
                self.notify(f"📋 Tercih listesine eklendi! (Liste ID: {list_id})", title="Başarılı")

        elif event.button.id == "btn_fav_detail":
            k_kodu = self._get_selected_kilavuz_kodu()
            if k_kodu:
                self.app.push_screen(DetailScreen(kilavuz_kodu=k_kodu))
