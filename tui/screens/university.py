"""
Textual TUI — Üniversite Bazlı Analiz Ekranı.
Kombine filtreler: üniversite adı + şehir + puan türü + üni türü + öğretim türü.
Programları 4 yıllık tarihsel verilerle gösterir.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Header, Footer, Input, Label, Select, Static
from textual.widgets.select import NoSelection

from services.analytics_service import AnalyticsService
from services.search_service import SearchService
from tui.screens.detail import DetailScreen


class UniversityScreen(Screen):
    """Üniversite Bazlı Analiz — Kombine Filtreli Program Listesi."""

    CSS = """
    #uni_filter_panel {
        height: auto;
        border: solid $accent;
        padding: 1 2;
        margin-bottom: 1;
    }

    #uni_filter_row1, #uni_filter_row2 {
        height: auto;
        margin-bottom: 1;
    }

    .uni_filter_block {
        width: 1fr;
        margin-right: 2;
        height: auto;
    }

    .uni_filter_label {
        height: 1;
        color: $warning;
        text-style: bold;
        margin-bottom: 0;
    }

    #uni_btn_row {
        height: 4;
        align: right middle;
        margin-top: 1;
    }

    #prog_count_static {
        height: 2;
        padding: 0 1;
        color: $text;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("🏛️ ÜNİVERSİTE & PROGRAM ARAMA — KOMBİNE FİLTRELER", id="title_stats")

        with Vertical(id="uni_filter_panel"):
            with Horizontal(id="uni_filter_row1"):
                with Vertical(classes="uni_filter_block"):
                    yield Label("🏛️ Üniversite Adı", classes="uni_filter_label")
                    yield Input(
                        placeholder="ör. Anadolu, ODTÜ, Boğaziçi...",
                        id="input_uni_name",
                    )
                with Vertical(classes="uni_filter_block"):
                    yield Label("🏙️ Şehir", classes="uni_filter_label")
                    yield Input(
                        placeholder="ör. Eskişehir, İstanbul, Ankara...",
                        id="input_il_adi",
                    )
                with Vertical(id="uni_btn_row"):
                    yield Button("🔍 Ara", variant="primary", id="btn_search")

            with Horizontal(id="uni_filter_row2"):
                with Vertical(classes="uni_filter_block"):
                    yield Label("📐 Puan Türü", classes="uni_filter_label")
                    yield Select(
                        options=[
                            ("Tüm Puan Türleri", "TÜMÜ"),
                            ("SAY", "SAY"), ("EA", "EA"), ("SÖZ", "SÖZ"), ("DİL", "DİL"),
                        ],
                        value="TÜMÜ", id="sel_puan", allow_blank=False,
                    )
                with Vertical(classes="uni_filter_block"):
                    yield Label("🏢 Tür", classes="uni_filter_label")
                    yield Select(
                        options=[
                            ("Devlet + Vakıf", "TÜMÜ"),
                            ("DEVLET", "DEVLET"),
                            ("VAKIF", "VAKIF"),
                        ],
                        value="TÜMÜ", id="sel_uni_turu", allow_blank=False,
                    )
                with Vertical(classes="uni_filter_block"):
                    yield Label("📚 Öğretim Türü", classes="uni_filter_label")
                    yield Select(
                        options=[
                            ("Tüm Öğretim", "TÜMÜ"),
                            ("Örgün", "Örgün"),
                            ("İkinci Öğretim", "İkinci Öğretim"),
                        ],
                        value="TÜMÜ", id="sel_ogretim", allow_blank=False,
                    )
                with Vertical(classes="uni_filter_block"):
                    yield Label("🎯 Sıralama Aralığı", classes="uni_filter_label")
                    yield Select(
                        options=[
                            ("Tüm Sıralama", "TÜMÜ"),
                            ("İlk 10,000", "TOP10K"),
                            ("10K–50K", "MID"),
                            ("50K+", "LOWER"),
                        ],
                        value="TÜMÜ", id="sel_rank_range", allow_blank=False,
                    )


        # ── Sonuç alanı ────────────────────────────────────────────────────────
        with Horizontal():
            with Vertical(classes="detail_left"):
                yield Label("📊 SEÇILI ÜNİVERSİTE ÖZETİ", classes="section_title")
                yield Static(id="uni_summary_static")

                yield Label("🎯 RİSK DAĞILIMI", classes="section_title")
                yield Static(id="uni_risk_static")

                yield Label("💡 FİLTRE İPUCU", classes="section_title")
                yield Static(
                    "[dim]Birden fazla filtre aynı anda çalışır.\n\n"
                    "[bold]Örnek:[/bold]\n"
                    "  Şehir: Eskişehir\n"
                    "  Puan: EA\n"
                    "  Tür: DEVLET\n\n"
                    "→ Eskişehir'deki tüm DEVLET\n"
                    "  üniversitelerinin EA programları[/dim]",
                    id="tip_static",
                )

            with Vertical(classes="detail_right"):
                yield Label("📋 PROGRAM LİSTESİ — 2022→2025→2026 ML TAHMİN", classes="section_title")
                yield Static(id="prog_count_static")
                yield DataTable(id="uni_programs_table", cursor_type="row")

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#uni_programs_table", DataTable)
        table.add_columns(
            "Bölüm",
            "Öğr.",
            "Puan",
            "2022",
            "2023",
            "2024",
            "2025",
            "2026 ML",
            "Değişim",
            "Risk",
        )
        self.query_one("#uni_summary_static", Static).update(
            "[dim]Filtre uygulayıp 'Ara' butonuna basın.[/dim]"
        )

    def _get_select(self, widget_id: str) -> str | None:
        val = self.query_one(widget_id, Select).value
        if isinstance(val, NoSelection) or str(val) in ("TÜMÜ", ""):
            return None
        return str(val)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_search":
            self._search()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._search()

    def _search(self) -> None:
        uni_name   = self.query_one("#input_uni_name", Input).value.strip() or None
        il_adi     = self.query_one("#input_il_adi", Input).value.strip() or None
        puan_turu  = self._get_select("#sel_puan")
        uni_turu   = self._get_select("#sel_uni_turu")
        ogretim    = self._get_select("#sel_ogretim")
        rank_range = self._get_select("#sel_rank_range")

        # Sıralama aralığı filtresi
        min_rank: float | None = None
        max_rank: float | None = None
        if rank_range == "TOP10K":
            max_rank = 10000.0
        elif rank_range == "MID":
            min_rank, max_rank = 10000.0, 50000.0
        elif rank_range == "LOWER":
            min_rank = 50000.0

        results = SearchService.search_programs(
            universite_adi=uni_name,
            il_adi=il_adi,
            puan_turu=puan_turu,
            universite_turu=uni_turu,
            ogretim_turu=ogretim,
            min_rank=min_rank,
            max_rank=max_rank,
            limit=300,
        )

        # ── Özet Panel ────────────────────────────────────────────────────────
        if results:
            unis = set(r.get("universite_adi", "") for r in results)
            iller = set(r.get("il_adi", "") for r in results)
            total_k = sum(float(r.get("lag1_genel_kontenjan") or 0) for r in results)
            ml_count = sum(1 for r in results if int(r.get("pred_2026") or 0) > 0)

            summary = (
                f"• [bold cyan]{len(results)}[/bold cyan] program bulundu\n"
                f"• [bold cyan]{len(unis)}[/bold cyan] üniversite\n"
                f"• Şehirler: [yellow]{', '.join(sorted(iller)[:5])}[/yellow]\n"
                f"• Toplam Kontenjan: [bold]{total_k:.0f}[/bold]\n"
                f"• ML Tahminli: [bold green]{ml_count}[/bold green] program"
            )

            # En sık geçen üniversite
            from collections import Counter
            uni_counts = Counter(r.get("universite_adi", "") for r in results)
            top_uni = uni_counts.most_common(3)
            if top_uni:
                summary += "\n\n[bold]En fazla program:[/bold]"
                for uname, cnt in top_uni:
                    summary += f"\n  • {uname[:32]}: {cnt}"
        else:
            summary = "[dim]Sonuç bulunamadı.\nFiltreleri gevşetin.[/dim]"

        self.query_one("#uni_summary_static", Static).update(summary)

        # ── Risk Dağılımı ─────────────────────────────────────────────────────
        if results:
            from collections import Counter
            risk_counts = Counter(
                str(r.get("risk_renk") or "🟡 STABIL") for r in results
            )
            risk_text = "\n".join(
                f"• {label}: [bold]{cnt}[/bold] program"
                for label, cnt in sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)
            )
        else:
            risk_text = "[dim]Veri yok[/dim]"
        self.query_one("#uni_risk_static", Static).update(risk_text)

        # ── Program Sayısı ────────────────────────────────────────────────────
        self.query_one("#prog_count_static", Static).update(
            f"[bold cyan]{len(results)}[/bold cyan] program | "
            f"İlk 300 gösteriliyor"
        )

        # ── Program Tablosu ───────────────────────────────────────────────────
        table = self.query_one("#uni_programs_table", DataTable)
        table.clear()

        for r in results[:300]:
            lag4 = float(r.get("lag4_taban_siralama") or 0)
            lag3 = float(r.get("lag3_taban_siralama") or 0)
            lag2 = float(r.get("lag2_taban_siralama") or 0)
            lag1 = float(r.get("lag1_taban_siralama") or 0)
            pred = int(r.get("pred_2026") or 0)
            deg  = int(r.get("pred_degisim") or 0)
            risk = str(r.get("risk_renk") or "🟡")

            if deg < -5000:
                deg_str = f"[green]{deg:+,}[/green]"
            elif deg > 5000:
                deg_str = f"[red]{deg:+,}[/red]"
            else:
                deg_str = f"[yellow]{deg:+,}[/yellow]" if deg != 0 else "-"

            table.add_row(
                str(r.get("birim_grup_adi") or r.get("birim_adi", ""))[:32],
                str(r.get("ogretim_turu", ""))[:5],
                str(r.get("puan_turu", "")),
                f"{lag4:,.0f}" if lag4 > 0 else "[dim]-[/dim]",
                f"{lag3:,.0f}" if lag3 > 0 else "[dim]-[/dim]",
                f"{lag2:,.0f}" if lag2 > 0 else "[dim]-[/dim]",
                f"{lag1:,.0f}" if lag1 > 0 else "[dim]-[/dim]",
                f"[bold cyan]{pred:,}[/bold cyan]" if pred > 0 else "[dim]Yok[/dim]",
                deg_str,
                risk[:12],
            )

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Seçili satırın programını detay ekranında aç."""
        try:
            uni_name = self.query_one("#input_uni_name", Input).value.strip()
            il_adi   = self.query_one("#input_il_adi", Input).value.strip() or None
            puan     = self._get_select("#sel_puan")
            u_turu   = self._get_select("#sel_uni_turu")
            ogretim  = self._get_select("#sel_ogretim")

            results = SearchService.search_programs(
                universite_adi=uni_name or None,
                il_adi=il_adi,
                puan_turu=puan,
                universite_turu=u_turu,
                ogretim_turu=ogretim,
                limit=300,
            )
            if results and event.cursor_row < len(results):
                k_kodu = int(results[event.cursor_row].get("kilavuz_kodu", 0))
                if k_kodu > 0:
                    self.app.push_screen(DetailScreen(kilavuz_kodu=k_kodu))
        except Exception:
            pass
