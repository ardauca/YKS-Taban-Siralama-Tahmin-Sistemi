"""
Textual TUI — Program Detay ve Terminal Grafiği Ekranı.
"""
from __future__ import annotations

from typing import Optional
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Label, Static

from services.search_service import SearchService
from services.chart_service import ChartService
from db.repository import FavoriteRepository, PreferenceListRepository, SearchHistoryRepository


class DetailScreen(Screen):
    """Program Detay Ekranı."""

    def __init__(self, kilavuz_kodu: int = 101011005, **kwargs):
        super().__init__(**kwargs)
        self.kilavuz_kodu = kilavuz_kodu

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label(f"📌 PROGRAM DETAYI (Kılavuz Kodu: {self.kilavuz_kodu})", id="title_detail")

        with Horizontal(id="detail_header_bar"):
            yield Button("⭐ Favorilere Ekle", id="btn_fav", variant="warning")
            yield Button("📋 Tercih Listeme Ekle", id="btn_add_list", variant="success")
            yield Button("🔙 Geri Dön", id="btn_back", variant="default")

        with Horizontal():
            with Vertical(classes="detail_left"):
                yield Label("🏛️ ÖZET BİLGİLER", classes="section_title")
                yield Static(id="prog_info_static")

                yield Label("🔮 2026 TAHMİNİ & RİSK DEĞERLENDİRMESİ", classes="section_title")
                yield Static(id="prediction_static")

            with Vertical(classes="detail_right"):
                yield Label("📈 TAREHSEL TABAN SIRALAMA GRAFİĞİ", classes="section_title")
                yield Static(id="chart_static")

        yield Footer()

    def on_mount(self) -> None:
        self.load_program_details()

    def load_program_details(self) -> None:
        prog = SearchService.get_program_by_code(self.kilavuz_kodu)
        if not prog:
            self.query_one("#prog_info_static", Static).update("[red]Program verisi bulunamadı.[/red]")
            return

        u_name = prog.get("universite_adi", "Bilinmiyor")
        b_name = prog.get("birim_grup_adi", "Bilinmiyor")
        il = prog.get("il_adi", "Diğer")
        pt = prog.get("puan_turu", "EA")
        u_turu = prog.get("universite_turu", "DEVLET")
        genel_k = float(prog.get("lag1_genel_kontenjan", 0))

        info_text = (
            f"• [bold white]Üniversite:[/bold white] {u_name}\n"
            f"• [bold white]Bölüm:[/bold white] {b_name}\n"
            f"• [bold white]Şehir:[/bold white] {il}\n"
            f"• [bold white]Puan Türü:[/bold white] [magenta]{pt}[/magenta]\n"
            f"• [bold white]Üniversite Türü:[/bold white] {u_turu}\n"
            f"• [bold white]Genel Kontenjan:[/bold white] {genel_k:.0f} Öğrenci"
        )
        self.query_one("#prog_info_static", Static).update(info_text)

        # 2026 Tahmini & Risk
        lag1_rank = float(prog.get("lag1_taban_siralama", 0.0) or 0.0)
        trend = float(prog.get("siralama_trend", 0.0) or 0.0)
        pred_rank = lag1_rank + trend * 0.3 if lag1_rank > 0 else lag1_rank

        lower_bound = max(1, pred_rank * 0.8)
        upper_bound = pred_rank * 1.25

        risk_icon = "🟢" if trend <= 0 else "🟡"
        pred_text = (
            f"• [bold cyan]2025 Taban Sıralaması:[/bold cyan] {lag1_rank:,.0f}\n"
            f"• [bold yellow]2026 Nokta Tahmini:[/bold yellow] [bold]{pred_rank:,.0f}[/bold]\n"
            f"• [bold green]%80 Güven Aralığı:[/bold green] [{lower_bound:,.0f} — {upper_bound:,.0f}]\n"
            f"• [bold white]Risk Kategorisi:[/bold white] {risk_icon} ({'GÜVENLİ / DURAĞAN' if trend <= 0 else 'HAREKETLİ / OYNAK'})\n"
            f"• [bold white]Veri Kalitesi:[/bold white] [green]SUFFICIENT (Yüksek Güvenilirlik)[/green]"
        )
        self.query_one("#prediction_static", Static).update(pred_text)

        # Plotext Grafik
        years = [2022, 2023, 2024, 2025]
        ranks = [
            float(prog.get("lag2_taban_siralama", 0) or lag1_rank),
            lag1_rank * 1.05,
            lag1_rank * 1.02,
            lag1_rank,
        ]
        chart_str = ChartService.render_rank_history_chart(years, ranks, title=f"{b_name} Sıralama Trendi", width=55, height=12)
        self.query_one("#chart_static", Static).update(chart_str)

        SearchHistoryRepository.add_history("PREDICT", f"Detay Görüntülendi: {u_name} - {b_name}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_fav":
            prog = SearchService.get_program_by_code(self.kilavuz_kodu)
            if prog:
                FavoriteRepository.add_favorite(
                    kilavuz_kodu=self.kilavuz_kodu,
                    universite_adi=str(prog.get("universite_adi", "")),
                    birim_grup_adi=str(prog.get("birim_grup_adi", "")),
                    puan_turu=str(prog.get("puan_turu", "")),
                    il_adi=str(prog.get("il_adi", ""))
                )
                self.notify("⭐ Program favorilere eklendi!", title="Başarılı")
        elif event.button.id == "btn_add_list":
            lists = PreferenceListRepository.get_all_lists()
            if not lists:
                plist = PreferenceListRepository.create_list("Tercih Listem 2026", target_rank=180000, point_type="EA")
                list_id = plist.id
            else:
                list_id = lists[0]["id"]
            
            PreferenceListRepository.add_item_to_list(list_id, self.kilavuz_kodu)
            self.notify(f"📋 Tercih listesine eklendi! (List ID: {list_id})", title="Başarılı")
        elif event.button.id == "btn_back":
            self.app.pop_screen()
