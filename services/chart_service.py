"""
Plotext Destekli Terminal İçi Vektörel Grafik Çizim Servisi.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import plotext as plt

logger = logging.getLogger(__name__)


class ChartService:
    """Plotext terminal grafik servisi."""

    @staticmethod
    def render_rank_history_chart(
        years: List[int],
        ranks: List[float],
        title: str = "Taban Sıralama Tarihsel Trendi (2019-2025)",
        width: int = 70,
        height: int = 15,
    ) -> str:
        """Tarihsel sıralama çizgi grafiğini terminal dizgisi (string) olarak döndürür."""
        if not years or not ranks:
            return "Grafik çizmek için veri yetersiz."

        plt.clf()
        plt.plotsize(width, height)
        plt.title(title)

        # Temiz yıl ve sıralama serisi
        valid_x = []
        valid_y = []
        for y_val, r_val in zip(years, ranks):
            if r_val is not None and r_val > 0:
                valid_x.append(y_val)
                valid_y.append(r_val)

        if not valid_x:
            return "Grafik için geçerli sıralama verisi bulunamadı."

        plt.plot(valid_x, valid_y, marker="dot", color="cyan")
        plt.xlabel("Yıl")
        plt.ylabel("Taban Sıralama")
        
        # Plotext safe yaxis reversal: try ylimits or safe call
        try:
            if hasattr(plt, "invert_yaxis"):
                plt.invert_yaxis()
            else:
                # Manual inverted y limits if valid_y exists
                min_y, max_y = min(valid_y), max(valid_y)
                if min_y != max_y:
                    margin = (max_y - min_y) * 0.1
                    plt.ylim(max_y + margin, max(0, min_y - margin))
        except Exception:
            pass

        return plt.build()

    @staticmethod
    def render_bar_chart(
        categories: List[str],
        values: List[float],
        title: str = "Kategori Dağılımı",
        width: int = 70,
        height: int = 15,
    ) -> str:
        """Çubuk grafiği (bar chart) çizer."""
        if not categories or not values:
            return "Grafik için veri yok."

        plt.clf()
        plt.plotsize(width, height)
        plt.title(title)

        plt.bar(categories, values, color="green")
        return plt.build()
