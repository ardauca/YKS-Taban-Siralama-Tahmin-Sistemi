"""
Tercih Listesi ve Arama Sonuçları Dışa Aktarma Servisi (CSV, JSON, Markdown, Excel, PDF).
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ExportService:
    """Çoklu format dışa aktarma servisi."""

    @staticmethod
    def export_to_csv(data: List[Dict[str, Any]], filepath: str) -> str:
        """Veriyi CSV dosyası olarak kaydeder."""
        if not data:
            return "Veri yok"

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        keys = data[0].keys()
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)

        return str(path.absolute())

    @staticmethod
    def export_to_json(data: Any, filepath: str) -> str:
        """Veriyi JSON dosyası olarak kaydeder."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return str(path.absolute())

    @staticmethod
    def export_to_markdown(analysis_data: Dict[str, Any], filepath: str) -> str:
        """Tercih listesi analizini Markdown raporu olarak kaydeder."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        md = []
        md.append(f"# 🎓 YKS 2026 Tercih Listesi Raporu — {analysis_data.get('title', 'Tercih Listem')}\n")
        md.append(f"- **Öğrenci Sıralaması:** {analysis_data.get('candidate_rank', 'Bilinmiyor'):,d}")
        md.append(f"- **Toplam Tercih Sayısı:** {analysis_data.get('total_items', 0)}")
        md.append(f"- **Ortalama Tahmini Sıralama:** {analysis_data.get('avg_predicted_rank', 0):,d}\n")

        md.append("## 📊 Strateji Dağılım Özeti")
        md.append(f"- 🟢 **Garanti / Güvenli Tercih:** {analysis_data.get('safe_count', 0)}")
        md.append(f"- 🟡 **İdeal / Hedef Tercih:** {analysis_data.get('balanced_count', 0)}")
        md.append(f"- 🔴 **Sürpriz / Riskli Tercih:** {analysis_data.get('risky_count', 0)}")
        md.append(f"- 🏛️ **Devlet / Vakıf Oranı:** {analysis_data.get('state_count', 0)} Devlet / {analysis_data.get('foundation_count', 0)} Vakıf\n")

        md.append("## 💡 Tercih Stratejisi Önerileri")
        for rec in analysis_data.get("recommendations", []):
            md.append(f"- {rec}")
        md.append("")

        md.append("## 📋 Tercih Listesi Detay Tablosu\n")
        md.append("| Sıra | Üniversite | Bölüm | Şehir | Puan | Kontenjan | 2025 Taban Sıra | 2026 Tahmini Sıra | İhtimal | Risk |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")

        for item in analysis_data.get("items", []):
            md.append(
                f"| {item['position']} | {item['universite_adi']} | {item['birim_grup_adi']} | {item['il_adi']} | {item['puan_turu']} | {item['genel_kontenjan']} | {item['lag1_taban_siralama']:,d} | {item['pred_2026_siralama']:,d} | {item['admission_probability']} | {item['risk_level']} |"
            )

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        return str(path.absolute())

    @staticmethod
    def export_to_excel(data: List[Dict[str, Any]], filepath: str) -> str:
        """Veriyi Excel (.xlsx) dosyası olarak kaydeder."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import openpyxl
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "Tercih Listesi"

            if data:
                headers = list(data[0].keys())
                ws.append(headers)
                for row in data:
                    ws.append([row.get(h, "") for h in headers])

            wb.save(path)
            return str(path.absolute())
        except Exception as e:
            logger.warning("openpyxl hatası, CSV fallback: %s", e)
            return ExportService.export_to_csv(data, str(path.with_suffix(".csv")))

    @staticmethod
    def export_to_pdf(analysis_data: Dict[str, Any], filepath: str) -> str:
        """Tercih listesi raporunu PDF dosyası olarak kaydeder."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

            doc = SimpleDocTemplate(str(path), pagesize=A4)
            styles = getSampleStyleSheet()
            story = []

            title_style = ParagraphStyle("TitleStyle", parent=styles["Heading1"], fontSize=18, leading=22)
            story.append(Paragraph(f"YKS 2026 Tercih Listesi Raporu — {analysis_data.get('title', 'Tercih Listem')}", title_style))
            story.append(Spacer(1, 12))

            info_text = f"Ogrenci Sıralaması: {analysis_data.get('candidate_rank', 0):,d} | Toplam Tercih: {analysis_data.get('total_items', 0)} | Ort. Tahmin: {analysis_data.get('avg_predicted_rank', 0):,d}"
            story.append(Paragraph(info_text, styles["Normal"]))
            story.append(Spacer(1, 12))

            # Table Data
            table_data = [["Sıra", "Universite", "Bolum", "Sehir", "2025 Sira", "2026 Tahmin", "Ihtimal"]]
            for item in analysis_data.get("items", []):
                table_data.append([
                    str(item["position"]),
                    str(item["universite_adi"])[:20],
                    str(item["birim_grup_adi"])[:20],
                    str(item["il_adi"]),
                    f"{item['lag1_taban_siralama']:.0f}",
                    f"{item['pred_2026_siralama']:.0f}",
                    str(item["admission_probability"])
                ])

            t = Table(table_data)
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.navy),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            story.append(t)

            doc.build(story)
            return str(path.absolute())
        except Exception as e:
            logger.warning("ReportLab PDF hatası, Markdown fallback kullanılıyor: %s", e)
            return ExportService.export_to_markdown(analysis_data, str(path.with_suffix(".md")))
