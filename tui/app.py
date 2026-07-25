"""
Main Textual Application Class for YKS Tahmin TUI.
"""
from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding

from tui.screens.dashboard import DashboardScreen
from tui.screens.search import SearchScreen
from tui.screens.detail import DetailScreen
from tui.screens.compare import CompareScreen
from tui.screens.trends import TrendsScreen
from tui.screens.stats import StatsScreen
from tui.screens.preference_list import PreferenceListScreen
from tui.screens.favorites import FavoritesScreen


class YKSTahminApp(App):
    """YKS Taban Sıralama Tahmin ve Tercih Yönetim Sistemi TUI Uygulaması."""

    TITLE = "YKS 2026 Taban Sıralama Tahmin & Tercih Yönetim Sistemi"
    SUB_TITLE = "Textual TUI Pro-Engine"

    CSS = """
    Screen {
        background: $surface;
    }

    #title_dashboard, #title_search, #title_detail, #title_compare, #title_trends, #title_stats, #title_pref, #title_favs {
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
        height: 14;
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

    #filter_bar, #detail_header_bar, #pref_action_bar, #compare_bar {
        height: 4;
        margin-bottom: 1;
        align: center middle;
    }

    .detail_left, .pref_left {
        width: 45%;
        border: solid $primary;
        padding: 1;
        margin-right: 1;
    }

    .detail_right, .pref_right {
        width: 55%;
        border: solid $secondary;
        padding: 1;
    }

    .trend_col {
        width: 50%;
        border: solid $accent;
        padding: 1;
    }

    DataTable {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("d", "switch_screen('dashboard')", "Dashboard (Ctrl+D)", show=True),
        Binding("f", "switch_screen('search')", "Arama (Ctrl+F)", show=True),
        Binding("l", "switch_screen('preference_list')", "Tercih Listesi (Ctrl+L)", show=True),
        Binding("t", "switch_screen('trends')", "Trendler (Ctrl+T)", show=True),
        Binding("s", "switch_screen('stats')", "İstatistikler (Ctrl+S)", show=True),
        Binding("v", "switch_screen('favorites')", "Favoriler (Ctrl+V)", show=True),
        Binding("c", "switch_screen('compare')", "Karşılaştırma (Ctrl+C)", show=True),
        Binding("q", "quit", "Çıkış (Q)", show=True),
    ]

    SCREENS = {
        "dashboard": DashboardScreen,
        "search": SearchScreen,
        "detail": DetailScreen,
        "compare": CompareScreen,
        "trends": TrendsScreen,
        "stats": StatsScreen,
        "preference_list": PreferenceListScreen,
        "favorites": FavoritesScreen,
    }

    def on_mount(self) -> None:
        self.push_screen("dashboard")
