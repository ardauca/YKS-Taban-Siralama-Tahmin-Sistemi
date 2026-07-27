"""
Rich & Plotext Destekli Terminal İçi Vektörel Grafik Çizim Servisi.
Textual TUI karanlık temalarıyla %100 uyumlu natif Rich visualizer ve Plotext desteği.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import plotext as plt

logger = logging.getLogger(__name__)


class ChartService:
    """Terminal grafik servisi — Rich Natif & Plotext Vektörel."""

    @staticmethod
    def render_rank_history_chart(
        years: List[int],
        ranks: List[float],
        title: str = "Taban Sıralama Tarihsel Trendi (2022-2026)",
        width: int = 50,
        height: int = 10,
    ) -> str:
        """
        Textual TUI karanlık temasıyla %100 uyumlu Rich metin tabanlı trend görselleştiricisi.
        Göreceli başarı barları, renkli değişim rozetleri ve temiz hizalama sunar.
        """
        if not years or not ranks:
            return "[dim]Grafik çizmek için veri yetersiz.[/dim]"

        valid_data = [(y, r) for y, r in zip(years, ranks) if r is not None and r > 0]
        if not valid_data:
            return "[dim]Geçerli sıralama verisi bulunamadı.[/dim]"

        min_r = min(r for _, r in valid_data)
        max_r = max(r for _, r in valid_data)
        range_r = max_r - min_r if max_r != min_r else 1.0

        lines = [f"[bold cyan]📈 {title}[/bold cyan]", ""]

        prev_r: float | None = None
        for y, r in valid_data:
            # 0 = max_r (en düşük başarı/en yüksek sayı), 1 = min_r (en yüksek başarı/en küçük sayı)
            norm = 1.0 - ((r - min_r) / range_r) if range_r > 0 else 0.5
            bar_len = int(norm * 16) + 4
            bar_str = "█" * bar_len + "░" * (20 - bar_len)

            if prev_r is not None:
                diff = r - prev_r
                if diff < -2000:
                    badge = f"[bold green]{diff:+,.0f} ↗[/bold green]"
                elif diff > 2000:
                    badge = f"[bold red]{diff:+,.0f} ↘[/bold red]"
                else:
                    badge = f"[yellow]{diff:+,.0f} →[/yellow]"
            else:
                badge = "[dim]Başlangıç[/dim]"

            is_pred = (y == 2026)
            y_str = f"[bold yellow]{y} ✨[/bold yellow]" if is_pred else f"[bold]{y}[/bold]"
            r_str = f"[bold cyan]{r:,.0f}[/bold cyan]" if is_pred else f"{r:,.0f}"

            lines.append(f"  {y_str} │ {r_str:>10} │ [cyan]{bar_str}[/cyan] │ {badge}")
            prev_r = r

        lines.append("")
        lines.append("[dim]İpucu: Doluluk oranı göreceli başarı ivmesini temsil eder.[/dim]")
        return "\n".join(lines)

    @staticmethod
    def render_plotext_chart(
        years: List[int],
        ranks: List[float],
        title: str = "Tarihsel Sıralama Grafiği",
        width: int = 50,
        height: int = 10,
    ) -> str:
        """Plotext ile vektörel grafik çizer (temiz temayla)."""
        valid_x = []
        valid_y = []
        for y_val, r_val in zip(years, ranks):
            if r_val is not None and r_val > 0:
                valid_x.append(y_val)
                valid_y.append(r_val)

        if not valid_x:
            return "Grafik için veri yok."

        plt.clf()
        plt.theme("clear")
        plt.plotsize(width, height)
        plt.title(title)
        plt.plot(valid_x, valid_y, color="cyan", marker="dot")
        plt.xlabel("Yıl")
        plt.ylabel("Taban Sıralama")
        return plt.build()
