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
        
        # Y eksenini ters çevirme mantığı (sıralama düştükçe başarı artar, yani 10k üstte 100k altta)
        plt.invert_yaxis()

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
