"""
Textual TUI — Tercih Listeleri Yönetim ve Strateji Analiz Ekranı.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Header, Footer, Input, Label, Static, DataTable

from db.repository import PreferenceListRepository
from services.preference_service import PreferenceService
from services.export_service import ExportService


class PreferenceListScreen(Screen):
    """Tercih Listesi Ekranı."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.active_list_id = 1

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Label("📋 YKS TERCIH LİSTESİ VE STRATEJİ ANALİZ MODÜLÜ", id="title_pref")

        with Horizontal(id="pref_action_bar"):
            yield Input(placeholder="Öğrenci Sıralaması (ör. 180000)", id="input_candidate_rank", value="180000")
            yield Button("🔍 Stratejiyi Yeniden Hesapla", id="btn_calc_strategy", variant="primary")
            yield Button("🔼 Yukarı Taşı", id="btn_move_up", variant="default")
            yield Button("🔽 Aşağı Taşı", id="btn_move_down", variant="default")
            yield Button("❌ Satırı Sil", id="btn_delete_item", variant="error")
            yield Button("📄 Raporu Dışa Aktar (MD/PDF)", id="btn_export", variant="success")

        with Horizontal():
            with Vertical(classes="pref_left"):
                yield Label("📊 STRATEJİ & RISK ANALİZİ", classes="section_title")
                yield Static(id="strategy_summary_static")

            with Vertical(classes="pref_right"):
                yield Label("📋 TERCIH SATIRLARI", classes="section_title")
                yield DataTable(id="pref_datatable", cursor_type="row")

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#pref_datatable", DataTable)
        table.add_columns(
            "Sıra",
            "Kılavuz Kodu",
            "Üniversite",
            "Bölüm",
            "Şehir",
            "Puan",
            "2025 Sıra",
            "2026 Tahmin",
            "İhtimal",
            "Risk"
        )
        self.ensure_default_list()
        self.load_list_data()

    def ensure_default_list(self) -> None:
        lists = PreferenceListRepository.get_all_lists()
        if not lists:
            plist = PreferenceListRepository.create_list("2026 Ana Tercih Listem", target_rank=180000, point_type="EA")
            self.active_list_id = plist.id
        else:
            self.active_list_id = lists[0]["id"]

    def load_list_data(self) -> None:
        rank_str = self.query_one("#input_candidate_rank", Input).value.strip()
        c_rank = int(rank_str) if rank_str.isdigit() else 180000

        analysis = PreferenceService.analyze_preference_list(self.active_list_id, candidate_rank=c_rank)
        if "error" in analysis:
            self.query_one("#strategy_summary_static", Static).update(f"[red]{analysis['error']}[/red]")
            return

        # Strateji Kartı
        strat_text = (
            f"• [bold white]Hedef Öğrenci Sıralaması:[/bold white] [bold cyan]{c_rank:,d}[/bold cyan]\n"
            f"• [bold white]Toplam Tercih Sayısı:[/bold white] {analysis['total_items']}\n"
            f"• [bold green]Garanti / Güvenli Tercihler:[/bold green] {analysis['safe_count']}\n"
            f"• [bold yellow]İdeal / Hedef Tercihler:[/bold yellow] {analysis['balanced_count']}\n"
            f"• [bold red]Sürpriz / Riskli Tercihler:[/bold red] {analysis['risky_count'] + analysis['dream_count']}\n"
            f"• [bold white]Devlet / Vakıf Oranı:[/bold white] {analysis['state_count']} Devlet / {analysis['foundation_count']} Vakıf\n"
            f"• [bold white]Ortalama Tahmini Sıralama:[/bold white] {analysis['avg_predicted_rank']:,d}\n\n"
            f"[bold magenta]💡 STRATEJİ ÖNERİLERİ:[/bold magenta]\n"
        )
        for rec in analysis.get("recommendations", []):
            strat_text += f"• {rec}\n"

        self.query_one("#strategy_summary_static", Static).update(strat_text)

        # Datatable Satırları
        table = self.query_one("#pref_datatable", DataTable)
        table.clear()

        for item in analysis.get("items", []):
            risk_color = "green" if item["risk_level"] in ["GARANTİ", "GÜVENLİ"] else ("yellow" if item["risk_level"] == "İDEAL/HEDEF" else "red")
            table.add_row(
                str(item["position"]),
                str(item["kilavuz_kodu"]),
                str(item["universite_adi"]),
                str(item["birim_grup_adi"]),
                str(item["il_adi"]),
                str(item["puan_turu"]),
                f"{item['lag1_taban_siralama']:,.0f}",
                f"{item['pred_2026_siralama']:,.0f}",
                str(item["admission_probability"]),
                f"[{risk_color}]{item['risk_level']}[/{risk_color}]",
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_calc_strategy":
            self.load_list_data()
        elif event.button.id in ["btn_move_up", "btn_move_down"]:
            table = self.query_one("#pref_datatable", DataTable)
            if table.cursor_row is not None:
                row_data = table.get_row_at(table.cursor_row)
                if row_data:
                    k_kodu = int(row_data[1])
                    direction = "up" if event.button.id == "btn_move_up" else "down"
                    PreferenceListRepository.move_item(self.active_list_id, k_kodu, direction)
                    self.load_list_data()
        elif event.button.id == "btn_delete_item":
            table = self.query_one("#pref_datatable", DataTable)
            if table.cursor_row is not None:
                row_data = table.get_row_at(table.cursor_row)
                if row_data:
                    k_kodu = int(row_data[1])
                    PreferenceListRepository.remove_item_from_list(self.active_list_id, k_kodu)
                    self.notify("❌ Tercih satırı silindi.", title="Başarılı")
                    self.load_list_data()
        elif event.button.id == "btn_export":
            analysis = PreferenceService.analyze_preference_list(self.active_list_id)
            res_path = ExportService.export_to_markdown(analysis, "tercih_listem_2026.md")
            self.notify(f"📄 Rapor dışa aktarıldı: {res_path}", title="Dışa Aktarma Başarılı")
