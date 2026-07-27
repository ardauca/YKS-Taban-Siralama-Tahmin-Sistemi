"""
Main Textual Application Class for YKS Tahmin TUI.
"""
from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding

from tui.screens.dashboard import DashboardScreen
from tui.screens.search import SearchScreen
from tui.screens.compare import CompareScreen
from tui.screens.trends import TrendsScreen
from tui.screens.stats import StatsScreen
from tui.screens.preference_list import PreferenceListScreen
from tui.screens.favorites import FavoritesScreen
from tui.screens.simulation import SimulationScreen
from tui.screens.university import UniversityScreen


class YKSTahminApp(App):
    """YKS Taban Sıralama Tahmin ve Tercih Yönetim Sistemi TUI Uygulaması."""

    TITLE = "YKS 2026 — Taban Sıralama Tahmin & Tercih Yönetim Sistemi"
    SUB_TITLE = "CatBoost ML • Polars • Textual TUI"

    CSS = """
    Screen {
        background: $surface;
    }

    #title_dashboard, #title_search, #title_detail, #title_compare,
    #title_trends, #title_stats, #title_pref, #title_favs {
        background: $primary;
        color: $text;
        text-align: center;
        text-style: bold;
        padding: 1;
        margin-bottom: 1;
    }

    #dash_grid {
        layout: grid;
        grid-size: 3 1;
        grid-columns: 1fr 1fr 1fr;
        height: 13;
        margin-bottom: 1;
    }

    .dash_card {
        border: solid $accent;
        padding: 1;
        height: 100%;
        background: $panel;
    }

    .card_header, .section_title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    #filter_bar, #filter_bar2, #detail_header_bar, #pref_action_bar, #compare_bar {
        height: 4;
        margin-bottom: 1;
        align: center middle;
    }

    .detail_left, .pref_left {
        width: 44%;
        border: solid $primary;
        padding: 1;
        margin-right: 1;
        overflow-y: auto;
    }

    .detail_right, .pref_right {
        width: 56%;
        border: solid $secondary;
        padding: 1;
        overflow-y: auto;
    }

    .trend_col {
        width: 50%;
        border: solid $accent;
        padding: 1;
    }

    DataTable {
        height: 1fr;
    }

    #dash_actions {
        height: 4;
        margin-top: 1;
        align: center middle;
    }

    #search_info, #sim_info {
        height: 2;
        margin-bottom: 1;
        padding: 0 1;
    }

    #uni_summary_static, #uni_risk_static {
        overflow-y: auto;
        height: auto;
        max-height: 20;
    }
    """

    BINDINGS = [
        Binding("d", "goto_dashboard", "Dashboard", show=True),
        Binding("f", "goto_search", "Arama", show=True),
        Binding("m", "goto_simulation", "ML 2026", show=True),
        Binding("u", "goto_university", "Üniversite", show=True),
        Binding("l", "goto_preference_list", "Tercih Listesi", show=True),
        Binding("t", "goto_trends", "Trendler", show=True),
        Binding("s", "goto_stats", "İstatistikler", show=True),
        Binding("v", "goto_favorites", "Favoriler", show=True),
        Binding("c", "goto_compare", "Karşılaştırma", show=True),
        Binding("q", "quit", "Çıkış", show=True),
    ]

    SCREENS = {
        "dashboard": DashboardScreen,
        "search": SearchScreen,
        "simulation": SimulationScreen,
        "university": UniversityScreen,
        "compare": CompareScreen,
        "trends": TrendsScreen,
        "stats": StatsScreen,
        "preference_list": PreferenceListScreen,
        "favorites": FavoritesScreen,
    }

    # DetailScreen push_screen ile dinamik olarak açılır (constructor arg alır)

    def action_goto_dashboard(self) -> None:
        self.switch_screen("dashboard")

    def action_goto_search(self) -> None:
        self.switch_screen("search")

    def action_goto_simulation(self) -> None:
        self.switch_screen("simulation")

    def action_goto_university(self) -> None:
        self.switch_screen("university")

    def action_goto_preference_list(self) -> None:
        self.switch_screen("preference_list")

    def action_goto_trends(self) -> None:
        self.switch_screen("trends")

    def action_goto_stats(self) -> None:
        self.switch_screen("stats")

    def action_goto_favorites(self) -> None:
        self.switch_screen("favorites")

    def action_goto_compare(self) -> None:
        self.switch_screen("compare")

    def on_mount(self) -> None:
        self.push_screen("dashboard")
