"""
Textual TUI — Program Detay Ekranı.
Gerçek CatBoost ML tahminleri, 4 yıllık tarihsel tablo (Sıralama + Puan + Kontenjan)
ve Model Tahmin Sebepleri (Rasyoneli).
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
    """Program Detay Ekranı — 4 Yıl Tarih + Kontenjanlar + Tahmin Sebepleri."""

    CSS = """
    DetailScreen .detail_left {
        width: 44%;
        border: solid $primary;
        padding: 1;
        margin-right: 1;
        overflow-y: auto;
    }

    DetailScreen .detail_right {
        width: 56%;
        border: solid $secondary;
        padding: 1;
        overflow-y: auto;
    }

    DetailScreen #hist_table {
        height: 11;
        min-height: 11;
        margin-bottom: 1;
    }

    DetailScreen #chart_static {
        height: auto;
        border: solid $accent;
        padding: 1;
        margin-top: 1;
        background: $panel;
    }
    """

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

                yield Label("💡 2026 TAHMİN SEBEPLERİ VE ETKEN FAKTÖRLER", classes="section_title")
                yield Static(id="reasons_static")

            with Vertical(classes="detail_right"):
                yield Label("📊 4 YILLIK TARİHSEL TABLO (Sıralama, Puan & Kontenjan)", classes="section_title")
                yield DataTable(id="hist_table", cursor_type="none")

                yield Label("📈 TABAN SIRALAMA GRAFİĞİ", classes="section_title")
                yield Static(id="chart_static")

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#hist_table", DataTable)
        table.add_columns("Yıl", "Taban Sıralama", "Değişim", "Taban Puanı", "Kontenjan", "Kaynak")
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

        # ── 4 Yıllık Tarihsel Tablo (2022, 2023, 2024, 2025, 2026 Tahmin) ──────────
        hist_rows = [
            (2022, "sira_2022", "puan_2022", "kont_2022", "YÖK Atlas 2022"),
            (2023, "sira_2023", "puan_2023", "kont_2023", "YÖK Atlas 2023"),
            (2024, "sira_2024", "puan_2024", "kont_2024", "YÖK Atlas 2024"),
            (2025, "sira_2025", "puan_2025", "kont_2025", "ÖSYM 2025"),
        ]

        table = self.query_one("#hist_table", DataTable)
        table.clear()

        ranks_for_chart: list[float] = []
        years_for_chart: list[int] = []
        prev_rank: float = 0.0

        for year, sira_col, puan_col, k_col, kaynak in hist_rows:
            sira = float(prog.get(sira_col) or 0.0)
            puan = float(prog.get(puan_col) or 0.0)
            kont = float(prog.get(k_col) or 0.0)

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
            puan_str = f"{puan:.3f}"  if puan > 0 else "[dim]-[/dim]"
            kont_str = f"{kont:.0f}"   if kont > 0 else "[dim]-[/dim]"

            table.add_row(str(year), sira_str, deg_str, puan_str, kont_str, kaynak)
            if sira > 0:
                prev_rank = sira

        # 2026 Tahmini Kontenjan ve Tahmin Satırı
        k_fark_2026 = float(prog.get("kontenjan_farki_2026") or 0.0)
        pred_kont   = max(1.0, genel_k + k_fark_2026) if genel_k > 0 else 0.0
        
        pred_2026    = float(prog.get("pred_2026") or 0)
        pred_lower   = float(prog.get("pred_lower") or 0)
        pred_upper   = float(prog.get("pred_upper") or 0)
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
        pred_kont_str = (
            f"[bold yellow]{pred_kont:.0f}[/bold yellow] "
            f"({k_fark_2026:+.0f})" if genel_k > 0 else "[dim]~[/dim]"
        )
        table.add_row("2026 ✨", pred_sira_str, pred_deg_str, "[dim]~[/dim]", pred_kont_str, ml_label)

        # ── 2026 ML Tahmin Paneli ──────────────────────────────────────────────
        risk_renk = str(prog.get("risk_renk") or "🟡 STABIL")
        lag1_rank = float(prog.get("lag1_taban_siralama") or 0.0)

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
            f"• [bold]Risk Durumu:[/bold] {risk_renk}"
        )
        self.query_one("#prediction_static", Static).update(pred_text)

        # ── 💡 Tahmin Sebepleri & Etken Faktörler (Model Rasyoneli) ─────────────
        trend_val   = float(prog.get("siralama_trend") or 0.0)
        trends_yoy  = float(prog.get("trends_yoy_degisim") or 0.0)
        rekabet_idx = float(prog.get("puan_turu_rekabet_indeksi") or 0.0)
        sehir_idx   = float(prog.get("sehir_tercih_indeksi") or 0.0)
        k_sok       = float(prog.get("kontenjan_sok_faktoru") or 0.0)

        # Kontenjan etkisi yorumu
        if k_fark_2026 > 0:
            k_effect = f"[green]+{k_fark_2026:.0f} Kontenjan Artışı[/green] (Sıralamayı esneten/gerileten etki)"
        elif k_fark_2026 < 0:
            k_effect = f"[red]{k_fark_2026:.0f} Kontenjan Daralması[/red] (Sıralamayı öne çeken etki)"
        else:
            k_effect = "[yellow]Kontenjan Değişimsiz[/yellow] (Kontenjan şoku beklenmiyor)"

        # Trend ivmesi yorumu
        if trend_val < -2000:
            t_effect = f"[green]{trend_val:+,.0f} sıra/yıl[/green] (Son 3 yıl yükseliş trendinde)"
        elif trend_val > 2000:
            t_effect = f"[red]{trend_val:+,.0f} sıra/yıl[/red] (Son 3 yıl gerileme trendinde)"
        else:
            t_effect = f"[yellow]{trend_val:+,.0f} sıra/yıl[/yellow] (Yatay/Stabil iz seyri)"

        reasons_text = (
            f"• [bold]1. Kontenjan Etkisi:[/bold] {k_effect}\n"
            f"• [bold]2. Geçmiş Trend İvmesi:[/bold] {t_effect}\n"
            f"• [bold]3. Puan Türü Rekabeti:[/bold] {pt} grubu rekabet indeksi ({rekabet_idx:,.0f})\n"
            f"• [bold]4. Şehir Popülerliği:[/bold] {il} şehri tercih yoğunluğu ({sehir_idx:.2f})\n"
            f"• [bold]5. Dijital Arama İlgisi:[/bold] Google Trends YoY değişimi ({trends_yoy:+.1f}%)"
        )
        self.query_one("#reasons_static", Static).update(reasons_text)

        # ── Grafik ────────────────────────────────────────────────────────────
        if len(years_for_chart) >= 2:
            try:
                chart_str = ChartService.render_rank_history_chart(
                    years_for_chart, ranks_for_chart,
                    title=f"{b_name[:28]} — {years_for_chart[0]}→2026",
                    width=58, height=10,
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
