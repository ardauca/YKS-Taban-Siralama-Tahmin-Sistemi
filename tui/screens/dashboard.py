"""
Textual TUI — Ana Dashboard Ekranı.
Makro istatistikler, favoriler, son işlemler ve 2026 ML tahmin özeti.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, Grid
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Label, Static, DataTable

from services.analytics_service import AnalyticsService
from db.repository import FavoriteRepository, SearchHistoryRepository


class DashboardScreen(Screen):
    """Ana Dashboard Ekranı."""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label(
            "🎓 YKS 2026 TABAN SIRALAMA TAHMİN SİSTEMİ — DASHBOARD",
            id="title_dashboard",
        )

        with Grid(id="dash_grid"):
            # Kart 1: Makro İstatistikler
            with Vertical(classes="dash_card"):
                yield Label("📊 TÜRKİYE GENELİ ÖZET", classes="card_header")
                yield Static(id="stats_summary_static")

            # Kart 2: 2026 ML Özeti
            with Vertical(classes="dash_card"):
                yield Label("🤖 2026 ML TAHMİN ÖZETİ", classes="card_header")
                yield Static(id="ml_summary_static")

            # Kart 3: Son İşlemler
            with Vertical(classes="dash_card"):
                yield Label("🕒 SON İŞLEMLER", classes="card_header")
                yield Static(id="history_static")

        yield Label(
            "🔥 2026 ML TAHMİNİNE GÖRE EN ÇOK İYİLEŞEN 5 PROGRAM",
            classes="section_title",
        )
        yield DataTable(id="dash_trends_table")

        with Horizontal(id="dash_actions"):
            yield Button("🔎 Arama  (F)", id="btn_goto_search", variant="primary")
            yield Button("🤖 ML Tahminler  (M)", id="btn_goto_sim", variant="warning")
            yield Button("🏛️ Üniversite  (U)", id="btn_goto_uni", variant="default")
            yield Button("📋 Tercih Listem  (L)", id="btn_goto_pref", variant="success")
            yield Button("📊 İstatistikler  (S)", id="btn_goto_stats", variant="default")

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#dash_trends_table", DataTable)
        table.add_columns(
            "Üniversite",
            "Bölüm",
            "Puan",
            "Şehir",
            "2025 Sıra",
            "2026 ML Tahmin",
            "Değişim",
        )
        self.load_dashboard_data()

    def load_dashboard_data(self) -> None:
        """Dashboard verilerini servislerden yükler."""
        # ── 1. Makro İstatistik ─────────────────────────────────────────────
        try:
            stats = AnalyticsService.get_nationwide_stats()
            # Üniversite türü dağılımı
            uni_turu = stats.get("uni_turu_counts", {})
            devlet = uni_turu.get("DEVLET", 0)
            vakif = uni_turu.get("VAKIF", 0)

            stats_text = (
                f"• Toplam Program: [bold cyan]{stats['total_programs']:,d}[/bold cyan]\n"
                f"• Toplam Üniversite: [bold cyan]{stats['total_universities']:,d}[/bold cyan]\n"
                f"• Devlet / Vakıf: [bold]{devlet:,d}[/bold] / [bold]{vakif:,d}[/bold]\n"
                f"• Ort. Taban Sıralama: [bold cyan]{stats['mean_rank']:,.0f}[/bold cyan]\n"
                f"• Toplam Kontenjan: [bold cyan]{stats['total_quota']:,.0f}[/bold cyan]"
            )
        except Exception as e:
            stats_text = f"[red]Yüklenemedi: {e}[/red]"
        self.query_one("#stats_summary_static", Static).update(stats_text)

        # ── 2. 2026 ML Tahmin Özeti ─────────────────────────────────────────
        try:
            stats = AnalyticsService.get_nationwide_stats()
            sim_count = stats.get("sim_program_count", 0)
            mean_pred = stats.get("mean_pred_2026", 0.0)

            # En çok iyileşen ve gerileyen bölüm ailelerini al
            risers = AnalyticsService.get_top_risers(3)
            decliners = AnalyticsService.get_top_decliners(3)

            ml_lines = [
                f"• ML Tahmin Kapsamı: [bold green]{sim_count:,d}[/bold green] program\n"
                f"• Ort. 2026 Tahmini: [bold cyan]{mean_pred:,.0f}[/bold cyan]\n"
                f"\n[bold green]▲ EN ÇOK İYİLEŞECEK:[/bold green]",
            ]
            for r in risers:
                b = str(r.get("birim_grup_adi") or r.get("birim_adi", ""))[:28]
                d = int(r.get("pred_degisim") or 0)
                ml_lines.append(f"  [green]{d:+,}[/green] {b}")

            ml_lines.append(f"[bold red]▼ EN ÇOK GERİLEYECEK:[/bold red]")
            for r in decliners:
                b = str(r.get("birim_grup_adi") or r.get("birim_adi", ""))[:28]
                d = int(r.get("pred_degisim") or 0)
                ml_lines.append(f"  [red]{d:+,}[/red] {b}")

            ml_text = "\n".join(ml_lines)
        except Exception as e:
            ml_text = f"[red]ML özeti yüklenemedi: {e}[/red]"
        self.query_one("#ml_summary_static", Static).update(ml_text)

        # ── 3. Son İşlemler ─────────────────────────────────────────────────
        try:
            hist = SearchHistoryRepository.get_recent_history(5)
            if not hist:
                hist_text = (
                    "[italic yellow]Arama geçmişi boş.\n\n"
                    "Arama ekranını (F) kullandıkça\nburadaki liste dolacak.[/italic yellow]"
                )
            else:
                hist_text = "\n".join([
                    f"[dim]{h['created_at']}[/dim] {h['summary'][:42]}"
                    for h in hist
                ])
        except Exception as e:
            hist_text = f"[red]Geçmiş yüklenemedi: {e}[/red]"
        self.query_one("#history_static", Static).update(hist_text)

        # ── 4. 2026 ML'e Göre En Çok İyileşenler ───────────────────────────
        try:
            table = self.query_one("#dash_trends_table", DataTable)
            table.clear()
            top_risers = AnalyticsService.get_top_risers(5)
            for r in top_risers:
                r_rank = float(r.get("lag1_taban_siralama") or 0.0)
                r_pred = int(r.get("pred_2026") or 0)
                r_deg = int(r.get("pred_degisim") or 0)
                table.add_row(
                    str(r.get("universite_adi", ""))[:38],
                    str(r.get("birim_grup_adi") or r.get("birim_adi", ""))[:32],
                    str(r.get("puan_turu", "")),
                    str(r.get("il_adi", "")),
                    f"{r_rank:,.0f}" if r_rank > 0 else "-",
                    f"{r_pred:,}" if r_pred > 0 else "[dim]Yok[/dim]",
                    f"[green]{r_deg:+,}[/green]",
                )
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        nav_map = {
            "btn_goto_search": "search",
            "btn_goto_sim": "simulation",
            "btn_goto_uni": "university",
            "btn_goto_pref": "preference_list",
            "btn_goto_stats": "stats",
        }
        target = nav_map.get(event.button.id)
        if target:
            self.app.switch_screen(target)
