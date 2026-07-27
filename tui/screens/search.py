"""
Textual TUI — Çoklu-Filtreleme Arama Ekranı.
Her filtre açıkça etiketlenmiş, dikey blok düzeninde.
Tüm filtreler aynı anda birlikte çalışır.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Header, Footer, Input, Label, Select, Static
from textual.widgets.select import NoSelection

from services.search_service import SearchService
from db.repository import SearchHistoryRepository
from tui.screens.detail import DetailScreen


class SearchScreen(Screen):
    """Çoklu-filtreleme Arama Ekranı — Etiketli Filtre Paneli."""

    CSS = """
    #search_filter_panel {
        height: auto;
        border: solid $accent;
        padding: 1 2;
        margin-bottom: 1;
    }

    #filter_row1, #filter_row2, #filter_row3 {
        height: auto;
        margin-bottom: 1;
    }

    .filter_block {
        width: 1fr;
        margin-right: 2;
        height: auto;
    }

    .filter_label {
        height: 1;
        color: $warning;
        text-style: bold;
        margin-bottom: 0;
    }

    Input {
        height: 3;
    }

    Select {
        height: 3;
    }

    #btn_search_row {
        height: 4;
        align: right middle;
        margin-top: 1;
    }

    #search_info {
        height: 2;
        padding: 0 1;
        margin-bottom: 1;
        color: $text;
    }

    #search_hint {
        height: 3;
        padding: 0 1;
        margin-bottom: 1;
        color: $text-muted;
        background: $panel;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("🔎 YKS PROGRAM ARAMA — ÇOKLU FİLTRELEME", id="title_search")

        with Vertical(id="search_filter_panel"):
            # Satır 1: Metin aramaları
            with Horizontal(id="filter_row1"):
                with Vertical(classes="filter_block"):
                    yield Label("📝 Bölüm / Üniversite Adı", classes="filter_label")
                    yield Input(
                        placeholder="ör: Tıp  |  Hukuk  |  Boğaziçi  |  Mühendislik",
                        id="input_query",
                    )
                with Vertical(classes="filter_block"):
                    yield Label("🏙️ Şehir", classes="filter_label")
                    yield Input(
                        placeholder="ör: Ankara  |  Eskisehir  |  Istanbul  |  Izmir",
                        id="input_city",
                    )

            # Satır 2: Dropdown filtreler
            with Horizontal(id="filter_row2"):
                with Vertical(classes="filter_block"):
                    yield Label("📐 Puan Türü", classes="filter_label")
                    yield Select(
                        options=[
                            ("Tüm Puan Türleri", "TÜMÜ"),
                            ("SAY", "SAY"),
                            ("EA", "EA"),
                            ("SÖZ", "SÖZ"),
                            ("DİL", "DİL"),
                        ],
                        value="TÜMÜ",
                        id="select_point_type",
                        allow_blank=False,
                    )
                with Vertical(classes="filter_block"):
                    yield Label("🏛️ Üniversite Türü", classes="filter_label")
                    yield Select(
                        options=[
                            ("Devlet + Vakıf", "TÜMÜ"),
                            ("DEVLET", "DEVLET"),
                            ("VAKIF", "VAKIF"),
                            ("KKTC", "KKTC"),
                        ],
                        value="TÜMÜ",
                        id="select_uni_turu",
                        allow_blank=False,
                    )
                with Vertical(classes="filter_block"):
                    yield Label("📚 Öğretim Türü", classes="filter_label")
                    yield Select(
                        options=[
                            ("Tüm Öğretim", "TÜMÜ"),
                            ("Örgün", "Örgün"),
                            ("İkinci Öğretim", "İkinci Öğretim"),
                        ],
                        value="TÜMÜ",
                        id="select_ogretim",
                        allow_blank=False,
                    )
                with Vertical(classes="filter_block"):
                    yield Label("💰 Burs", classes="filter_label")
                    yield Select(
                        options=[
                            ("Tüm Burslar", "TÜMÜ"),
                            ("Burslu", "Burslu"),
                            ("%75 İndirimli", "%75"),
                            ("%50 İndirimli", "%50"),
                            ("%25 İndirimli", "%25"),
                            ("Ücretli", "Ücretli"),
                        ],
                        value="TÜMÜ",
                        id="select_burs",
                        allow_blank=False,
                    )

            # Satır 3: Sayısal aralık + Arama butonu
            with Horizontal(id="filter_row3"):
                with Vertical(classes="filter_block"):
                    yield Label("📊 Max 2025 Taban Sıralaması", classes="filter_label")
                    yield Input(
                        placeholder="ör: 100000  (bu sayının altındakiler gelir)",
                        id="input_max_rank",
                    )
                with Vertical(classes="filter_block"):
                    yield Label("🤖 Max 2026 ML Tahmini", classes="filter_label")
                    yield Input(
                        placeholder="ör: 120000  (bu sayının altındakiler gelir)",
                        id="input_max_pred",
                    )
                with Vertical(id="btn_search_row"):
                    yield Button("🔍  ARA / FİLTRELE  (veya Enter)", variant="primary", id="btn_search")

        # Sonuç bilgisi
        yield Static(id="search_hint")
        yield Static(id="search_info", classes="card_header")
        yield DataTable(id="search_datatable", cursor_type="row")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#search_datatable", DataTable)
        table.add_columns(
            "Kod", "Üniversite", "Bölüm", "Şehir",
            "Puan", "Tür", "Burs",
            "2025 Sıra", "2026 Tahmin", "Trend",
        )
        self.query_one("#search_hint", Static).update(
            "[dim]💡 Her filtre birbirinden bağımsızdır. "
            "Doldurmadığın filtreler uygulanmaz (= hepsi gelir). "
            "Birden fazlasını aynı anda kullanabilirsin.[/dim]"
        )
        self.perform_search()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_search":
            self.perform_search()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.perform_search()

    def _get_select_value(self, widget_id: str) -> str | None:
        """Select widget'tan değer okur; NoSelection veya TÜMÜ için None döner."""
        val = self.query_one(widget_id, Select).value
        if isinstance(val, NoSelection) or str(val) in ("TÜMÜ", ""):
            return None
        return str(val)

    def perform_search(self) -> None:
        """Filtre girdilerini okur, SearchService'e gönderir, tabloyu günceller."""
        query     = self.query_one("#input_query", Input).value.strip() or None
        city      = self.query_one("#input_city", Input).value.strip() or None
        puan_turu = self._get_select_value("#select_point_type")
        uni_turu  = self._get_select_value("#select_uni_turu")
        ogretim   = self._get_select_value("#select_ogretim")
        burs      = self._get_select_value("#select_burs")

        max_rank_str = self.query_one("#input_max_rank", Input).value.strip()
        max_rank: float | None = float(max_rank_str) if max_rank_str.isdigit() else None

        max_pred_str = self.query_one("#input_max_pred", Input).value.strip()
        max_pred: float | None = float(max_pred_str) if max_pred_str.isdigit() else None

        results = SearchService.search_programs(
            search_query=query,
            il_adi=city,
            puan_turu=puan_turu,
            universite_turu=uni_turu,
            ogretim_turu=ogretim,
            burs_orani=burs,
            max_rank=max_rank,
            max_pred=max_pred,
            limit=300,
        )

        # Tablo güncelle
        table = self.query_one("#search_datatable", DataTable)
        table.clear()

        for r in results:
            r_rank = float(r.get("lag1_taban_siralama") or 0.0)
            r_pred = int(r.get("pred_2026") or 0)
            r_deg  = int(r.get("pred_degisim") or 0)

            if r_deg < -5000:
                trend_str = f"[green]{r_deg:+,}[/green]"
            elif r_deg > 5000:
                trend_str = f"[red]{r_deg:+,}[/red]"
            else:
                trend_str = f"[yellow]{r_deg:+,}[/yellow]" if r_deg else "-"

            table.add_row(
                str(r.get("kilavuz_kodu", "")),
                str(r.get("universite_adi", ""))[:38],
                str(r.get("birim_grup_adi") or r.get("birim_adi", ""))[:35],
                str(r.get("il_adi", "")),
                str(r.get("puan_turu", "")),
                str(r.get("universite_turu", ""))[:6],
                str(r.get("burs_orani", ""))[:12],
                f"{r_rank:,.0f}" if r_rank > 0 else "-",
                f"{r_pred:,}"    if r_pred > 0 else "-",
                trend_str,
            )

        # Aktif filtre özeti
        aktif = []
        if query:     aktif.append(f"Bölüm/Üni='{query}'")
        if city:      aktif.append(f"Şehir='{city}'")
        if puan_turu: aktif.append(f"Puan={puan_turu}")
        if uni_turu:  aktif.append(f"Tür={uni_turu}")
        if ogretim:   aktif.append(f"Öğretim={ogretim}")
        if burs:      aktif.append(f"Burs={burs}")
        if max_rank:  aktif.append(f"MaxSıra≤{int(max_rank):,}")
        if max_pred:  aktif.append(f"MaxPred≤{int(max_pred):,}")

        filtre_str = "  │  ".join(aktif) if aktif else "[dim]Filtre yok — tüm programlar[/dim]"
        self.query_one("#search_info", Static).update(
            f"[bold cyan]{len(results):,} program[/bold cyan] bulundu  │  "
            f"[bold]Aktif:[/bold] {filtre_str}"
        )

        if query or city:
            SearchHistoryRepository.add_history(
                "SEARCH",
                f"'{query or city}' → {len(results)} sonuç",
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Satır seçildiğinde Program Detay ekranı açılır."""
        try:
            row_data = self.query_one("#search_datatable", DataTable).get_row(event.row_key)
            if row_data:
                self.app.push_screen(DetailScreen(kilavuz_kodu=int(row_data[0])))
        except (ValueError, IndexError, Exception):
            pass
