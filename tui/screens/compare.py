"""
Textual TUI — Çoklu Program Karşılaştırma Ekranı.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Input, Label, Static, DataTable

from services.search_service import SearchService


class CompareScreen(Screen):
    """Karşılaştırma Ekranı."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("⚔️ PROGRAM YAN YANA KARŞILAŞTIRMA MODÜLÜ", id="title_compare")

        with Horizontal(id="compare_bar"):
            yield Input(placeholder="Kılavuz Kodları (virgülle ayırın, ör. 101011005, 101010001)", id="input_compare_codes")
            yield Button("Karşılaştır", id="btn_compare", variant="primary")

        yield DataTable(id="compare_datatable")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#compare_datatable", DataTable)
        table.add_columns(
            "Metrik / Özellik",
            "Program 1",
            "Program 2",
            "Program 3"
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_compare":
            codes_str = self.query_one("#input_compare_codes", Input).value.strip()
            codes = [int(c.strip()) for c in codes_str.split(",") if c.strip().isdigit()]
            
            if len(codes) == 0:
                self.notify("⚠️ Lütfen en az bir kılavuz kodu giriniz.", title="Uyarı")
                return

            progs = [SearchService.get_program_by_code(c) for c in codes[:3]]
            progs = [p for p in progs if p is not None]

            table = self.query_one("#compare_datatable", DataTable)
            table.clear()

            metrics = [
                ("Kılavuz Kodu", lambda p: str(p.get("kilavuz_kodu", ""))),
                ("Üniversite Adı", lambda p: str(p.get("universite_adi", ""))),
                ("Bölüm / Birim", lambda p: str(p.get("birim_grup_adi", ""))),
                ("Şehir", lambda p: str(p.get("il_adi", ""))),
                ("Puan Türü", lambda p: str(p.get("puan_turu", ""))),
                ("Üniversite Türü", lambda p: str(p.get("universite_turu", ""))),
                ("2025 Taban Sıra", lambda p: f"{float(p.get('lag1_taban_siralama', 0)):,.0f}"),
                ("Genel Kontenjan", lambda p: f"{float(p.get('lag1_genel_kontenjan', 0)):.0f}"),
                ("Sıralama Trendi", lambda p: f"{float(p.get('siralama_trend', 0)):,.0f} sıra"),
            ]

            for m_title, m_func in metrics:
                row_vals = [m_title]
                for p in progs:
                    row_vals.append(m_func(p))
                # Fill remaining columns if less than 3
                while len(row_vals) < 4:
                    row_vals.append("-")
                table.add_row(*row_vals)
