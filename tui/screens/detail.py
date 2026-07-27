"""
Textual TUI — Program Detay Ekranı.
Gerçek CatBoost ML tahminleri, güven aralığı ve 4 yıllık tarihsel tablo.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, DataTable, Header, Footer, Label, Static

from services.search_service import SearchService
from services.chart_service import ChartService
from db.repository import FavoriteRepository, PreferenceListRepository, SearchHistoryRepository


class DetailScreen(Screen):
    """Program Detay Ekranı — 4 Yıl Tarih + Gerçek ML Tahmini."""

    def __init__(self, kilavuz_kodu: int = 101011005, **kwargs):
        super().__init__(**kwargs)
        self.kilavuz_kodu = kilavuz_kodu

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label(f"📌 PROGRAM DETAYI — Kılavuz: {self.kilavuz_kodu}", id="title_detail")

        with Horizontal(id="detail_header_bar"):
            yield Button("⭐ Favori Ekle/Çıkar", id="btn_fav", variant="warning")
            yield Button("📋 Tercih Listeme Ekle", id="btn_add_list", variant="success")
            yield Button("🔙 Geri Dön", id="btn_back", variant="default")

        with Horizontal():
            with Vertical(classes="detail_left"):
                yield Label("🏛️ PROGRAM BİLGİLERİ", classes="section_title")
                yield Static(id="prog_info_static")

                yield Label("🤖 2026 ML TAHMİNİ (CatBoost)", classes="section_title")
                yield Static(id="prediction_static")

            with Vertical(classes="detail_right"):
                yield Label("📊 4 YILLIK TARİHSEL TABLO (2022 → 2026 Tahmin)", classes="section_title")
                yield DataTable(id="hist_table", cursor_type="none")

                yield Label("📈 TABAN SIRALAMA GRAFİĞİ", classes="section_title")
                yield Static(id="chart_static")

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#hist_table", DataTable)
        table.add_columns("Yıl", "Taban Sıralama", "Değişim", "Taban Puanı", "Kaynak")
        self.load_program_details()

    def load_program_details(self) -> None:
        """Zengin master DF'den program bilgilerini yükler."""
        prog = SearchService.get_program_by_code(self.kilavuz_kodu)
        if not prog:
            self.query_one("#prog_info_static", Static).update(
                "[bold red]Program bulunamadı.[/bold red]\n"
                "Geçerli bir kılavuz kodu ile arama yapın."
            )
            return

        # ── Program Bilgileri ──────────────────────────────────────────────────
        u_name  = str(prog.get("universite_adi", "Bilinmiyor"))
        b_name  = str(prog.get("birim_grup_adi") or prog.get("birim_adi", "Bilinmiyor"))
        il      = str(prog.get("il_adi", "-"))
        pt      = str(prog.get("puan_turu", "-"))
        u_turu  = str(prog.get("universite_turu", "DEVLET"))
        ogr_t   = str(prog.get("ogretim_turu", "Örgün"))
        burs    = str(prog.get("burs_orani", "-"))
        genel_k = float(prog.get("lag1_genel_kontenjan") or 0)

        is_fav  = FavoriteRepository.is_favorite(self.kilavuz_kodu)
        fav_b   = " [bold yellow]★ FAVORİ[/bold yellow]" if is_fav else ""

        info_text = (
            f"• [bold]Üniversite:[/bold] {u_name}{fav_b}\n"
            f"• [bold]Bölüm:[/bold] {b_name}\n"
            f"• [bold]Şehir:[/bold] {il}\n"
            f"• [bold]Puan Türü:[/bold] [magenta]{pt}[/magenta]\n"
            f"• [bold]Üni. Türü:[/bold] {u_turu}\n"
            f"• [bold]Öğretim Türü:[/bold] {ogr_t}\n"
            f"• [bold]Burs Oranı:[/bold] [cyan]{burs}[/cyan]\n"
            f"• [bold]2025 Kontenjan:[/bold] {genel_k:.0f} öğrenci"
        )
        self.query_one("#prog_info_static", Static).update(info_text)

        # ── 4 Yıllık Tarihsel Veri ────────────────────────────────────────────
        # lag4=2022, lag3=2023, lag2=2024, lag1=2025, pred=2026
        hist_rows = [
            (4, 2022, "lag4_taban_siralama", "lag4_taban_puan", "Ham CSV"),
            (3, 2023, "lag3_taban_siralama", "lag3_taban_puan", "Ham CSV"),
            (2, 2024, "lag2_taban_siralama", "lag2_taban_puan", "Ham CSV"),
            (1, 2025, "lag1_taban_siralama", "lag1_taban_puan", "ÖSYM 2025"),
        ]

        table = self.query_one("#hist_table", DataTable)
        table.clear()

        ranks_for_chart: list[float] = []
        years_for_chart: list[int] = []
        prev_rank: float = 0.0

        for _, year, sira_col, puan_col, kaynak in hist_rows:
            sira  = float(prog.get(sira_col) or 0.0)
            puan  = float(prog.get(puan_col) or 0.0)

            if sira > 0:
                ranks_for_chart.append(sira)
                years_for_chart.append(year)

            # Değişim hesapla
            if prev_rank > 0 and sira > 0:
                degisim = sira - prev_rank
                if degisim < -2000:
                    deg_str = f"[green]{degisim:+,.0f}[/green]"
                elif degisim > 2000:
                    deg_str = f"[red]{degisim:+,.0f}[/red]"
                else:
                    deg_str = f"[yellow]{degisim:+,.0f}[/yellow]"
            else:
                deg_str = "-"

            sira_str = f"{sira:,.0f}" if sira > 0 else "[dim]Veri yok[/dim]"
            puan_str = f"{puan:.3f}" if puan > 0 else "[dim]-[/dim]"

            table.add_row(str(year), sira_str, deg_str, puan_str, kaynak)
            if sira > 0:
                prev_rank = sira

        # 2026 ML Tahmini satırı
        pred_2026   = float(prog.get("pred_2026") or 0)
        pred_lower  = float(prog.get("pred_lower") or 0)
        pred_upper  = float(prog.get("pred_upper") or 0)
        pred_degisim = float(prog.get("pred_degisim") or 0)
        ml_available = pred_2026 > 0

        if not ml_available:
            trend      = float(prog.get("siralama_trend") or 0.0)
            lag1_rank  = float(prog.get("lag1_taban_siralama") or 0.0)
            pred_2026  = max(1.0, lag1_rank + trend * 0.3) if lag1_rank > 0 else lag1_rank
            pred_lower = pred_2026 * 0.80
            pred_upper = pred_2026 * 1.25
            pred_degisim = pred_2026 - lag1_rank
            ml_label   = "[dim](Fallback — ML Verisi Yok)[/dim]"
        else:
            ml_label = "[bold green](✅ CatBoost)[/bold green]"

        if pred_2026 > 0:
            ranks_for_chart.append(pred_2026)
            years_for_chart.append(2026)

        # 2026 satırını renkli ekle
        if pred_degisim < -5000:
            pred_deg_str = f"[green]{pred_degisim:+,.0f}[/green]"
        elif pred_degisim > 5000:
            pred_deg_str = f"[red]{pred_degisim:+,.0f}[/red]"
        else:
            pred_deg_str = f"[yellow]{pred_degisim:+,.0f}[/yellow]"

        pred_sira_str = (
            f"[bold cyan]{pred_2026:,.0f}[/bold cyan]  "
            f"([dim]{pred_lower:,.0f}–{pred_upper:,.0f}[/dim])"
            if pred_2026 > 0 else "[dim]Yok[/dim]"
        )
        table.add_row("2026 ✨", pred_sira_str, pred_deg_str, "[dim]~[/dim]", ml_label)

        # ── 2026 ML Tahmin Paneli ──────────────────────────────────────────────
        risk_renk = str(prog.get("risk_renk") or "🟡 STABIL")
        lag1_rank  = float(prog.get("lag1_taban_siralama") or 0.0)

        if pred_degisim < -20000:
            change_str = f"[bold green]{pred_degisim:+,.0f}[/bold green] sıra (Çok İyi ↗)"
        elif pred_degisim < -5000:
            change_str = f"[green]{pred_degisim:+,.0f}[/green] sıra (İyi ↑)"
        elif pred_degisim < 5000:
            change_str = f"[yellow]{pred_degisim:+,.0f}[/yellow] sıra (Stabil →)"
        elif pred_degisim < 20000:
            change_str = f"[red]{pred_degisim:+,.0f}[/red] sıra (Geriliyor ↓)"
        else:
            change_str = f"[bold red]{pred_degisim:+,.0f}[/bold red] sıra (Hızlı Gerileme ↘)"

        pred_text = (
            f"• [bold]Tahmin Kaynağı:[/bold] {ml_label}\n"
            f"• [bold cyan]2025 Gerçekleşen:[/bold cyan] {lag1_rank:,.0f}\n"
            f"• [bold yellow]2026 Nokta Tahmini:[/bold yellow] [bold]{pred_2026:,.0f}[/bold]\n"
            f"• [bold green]%80 Alt Sınır:[/bold green] {pred_lower:,.0f}\n"
            f"• [bold red]%80 Üst Sınır:[/bold red] {pred_upper:,.0f}\n"
            f"• [bold]Değişim:[/bold] {change_str}\n"
            f"• [bold]Risk:[/bold] {risk_renk}"
        )
        self.query_one("#prediction_static", Static).update(pred_text)

        # ── Grafik ────────────────────────────────────────────────────────────
        if len(years_for_chart) >= 2:
            try:
                chart_str = ChartService.render_rank_history_chart(
                    years_for_chart, ranks_for_chart,
                    title=f"{b_name[:28]} — {years_for_chart[0]}→2026",
                    width=58, height=12,
                )
            except Exception as e:
                chart_str = f"[dim]Grafik oluşturulamadı: {e}[/dim]"
        else:
            chart_str = "[dim]Yeterli tarihsel veri yok.[/dim]"
        self.query_one("#chart_static", Static).update(chart_str)

        SearchHistoryRepository.add_history("PREDICT", f"{u_name[:30]} - {b_name[:25]}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_fav":
            prog = SearchService.get_program_by_code(self.kilavuz_kodu)
            if prog:
                if FavoriteRepository.is_favorite(self.kilavuz_kodu):
                    FavoriteRepository.remove_favorite(self.kilavuz_kodu)
                    self.notify("🗑️ Favorilerden çıkarıldı.", title="Güncellendi")
                else:
                    FavoriteRepository.add_favorite(
                        kilavuz_kodu=self.kilavuz_kodu,
                        universite_adi=str(prog.get("universite_adi", "")),
                        birim_grup_adi=str(prog.get("birim_grup_adi") or prog.get("birim_adi", "")),
                        puan_turu=str(prog.get("puan_turu", "")),
                        il_adi=str(prog.get("il_adi", "")),
                    )
                    self.notify("⭐ Favorilere eklendi!", title="Başarılı")
                self.load_program_details()

        elif event.button.id == "btn_add_list":
            lists = PreferenceListRepository.get_all_lists()
            list_id = lists[0]["id"] if lists else PreferenceListRepository.create_list(
                "Tercih Listem 2026", target_rank=180000, point_type="EA"
            ).id
            PreferenceListRepository.add_item_to_list(list_id, self.kilavuz_kodu)
            self.notify(f"📋 Tercih listesine eklendi! (Liste ID: {list_id})", title="Başarılı")

        elif event.button.id == "btn_back":
            self.app.pop_screen()
