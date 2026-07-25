"""
Tercih Listesi Strateji ve Kabul İhtimali Analiz Motoru.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from db.repository import PreferenceListRepository
from services.search_service import SearchService

logger = logging.getLogger(__name__)


class PreferenceService:
    """Tercih Listesi yönetim ve strateji analiz servisi."""

    @staticmethod
    def calculate_admission_probability(candidate_rank: int, pred_rank: float) -> Tuple[str, str, str]:
        """
        Öğrencinin YKS sıralaması ile 2026 tahmini arasındaki ilişkiye göre:
        - (Kabul İhtimali, Risk Seviyesi, Açıklama) döndürür.
        """
        if candidate_rank <= 0 or pred_rank <= 0:
            return "Bilinmiyor", "NOT_DEFINED", "Geçersiz sıralama verisi."

        diff = pred_rank - candidate_rank  # Tahmin 200k, öğrenci 180k -> diff = +20k (kolay girer)

        if diff >= candidate_rank * 0.35:
            return "Çok Yüksek", "GARANTİ", "Sıralamanız bu programın tahmininin çok üzerinde. Yerleşme ihtimali son derece yüksek."
        elif diff >= candidate_rank * 0.10:
            return "Yüksek", "GÜVENLİ", "Sıralamanız tahmin edilen taban sıralamanın rahatlıkla üzerinde."
        elif diff >= -candidate_rank * 0.10:
            return "Orta", "İDEAL/HEDEF", "Sıralamanız tahmin edilen taban sıralamaya başabaş yakınlıkta. İdeal hedef tercih."
        elif diff >= -candidate_rank * 0.25:
            return "Düşük", "SÜRPRİZ", "Sıralamanız tahmin edilen taban sıralamanın bir miktar altında. Sürpriz/üst tercih."
        else:
            return "Çok Düşük", "YÜKSEK RİSK", "Sıralamanız bu programın tahmininden belirgin şekilde düşük. Yerleşme şansı az."

    @classmethod
    def analyze_preference_list(cls, list_id: int, candidate_rank: Optional[int] = None) -> Dict[str, Any]:
        """
        Bir tercih listesinin tüm satırlarını ve genel liste stratejisini analiz eder.
        """
        plist = PreferenceListRepository.get_list_by_id(list_id)
        if not plist:
            return {"error": "Tercih listesi bulunamadı."}

        items = PreferenceListRepository.get_list_items(list_id)
        c_rank = candidate_rank or plist.target_rank or 150000

        evaluated_items = []
        safe_count = 0
        balanced_count = 0
        risky_count = 0
        dream_count = 0
        state_count = 0
        foundation_count = 0
        cities = {}

        total_predicted_rank = 0.0

        for item in items:
            prog = SearchService.get_program_by_code(item["kilavuz_kodu"])
            if not prog:
                continue

            lag1_rank = float(prog.get("lag1_taban_siralama", 0.0) or 0.0)
            
            # Tahmin simülasyon hesabı (Model S / Model M / Baseline mantığı)
            trend = float(prog.get("siralama_trend", 0.0) or 0.0)
            pred_rank = lag1_rank + trend * 0.3 if lag1_rank > 0 else lag1_rank

            pct_change = ((pred_rank - lag1_rank) / lag1_rank * 100) if lag1_rank > 0 else 0.0
            prob, risk, exp = cls.calculate_admission_probability(c_rank, pred_rank)

            if risk in ["GARANTİ"]:
                safe_count += 1
            elif risk in ["GÜVENLİ"]:
                safe_count += 1
            elif risk in ["İDEAL/HEDEF"]:
                balanced_count += 1
            elif risk in ["SÜRPRİZ"]:
                risky_count += 1
            else:
                dream_count += 1

            u_turu = str(prog.get("universite_turu", "DEVLET")).upper()
            if "VAKIF" in u_turu:
                foundation_count += 1
            else:
                state_count += 1

            il = str(prog.get("il_adi", "Diğer"))
            cities[il] = cities.get(il, 0) + 1
            total_predicted_rank += pred_rank

            evaluated_items.append({
                "position": item["position"],
                "kilavuz_kodu": item["kilavuz_kodu"],
                "universite_adi": prog.get("universite_adi", "Bilinmiyor"),
                "birim_grup_adi": prog.get("birim_grup_adi", "Bilinmiyor"),
                "il_adi": il,
                "puan_turu": prog.get("puan_turu", "EA"),
                "genel_kontenjan": prog.get("lag1_genel_kontenjan", 0.0),
                "lag1_taban_siralama": lag1_rank,
                "pred_2026_siralama": int(round(pred_rank)),
                "predicted_change_pct": round(pct_change, 1),
                "admission_probability": prob,
                "risk_level": risk,
                "explanation": exp,
                "notes": item.get("notes", ""),
            })

        total_items = len(evaluated_items)
        avg_rank = (total_predicted_rank / total_items) if total_items > 0 else 0.0

        # Strateji Öneri Algoritması
        recommendations = []
        if total_items == 0:
            recommendations.append("Listenizde henüz hiç tercih bulunmamaktadır. Arama ekranından program ekleyebilirsiniz.")
        else:
            if safe_count == 0:
                recommendations.append("⚠️ LISTENIZDE GARANTI TERCIH YOK: Açıkta kalma riskini önlemek için listenizin son sıralarına 2-3 garanti tercih eklemeniz önerilir.")
            if risky_count + dream_count > total_items * 0.6:
                recommendations.append("⚠️ YÜKSEK RİSKLİ LİSTE: Listenizin %60'ından fazlası sürpriz ve yüksek riskli tercihlerden oluşuyor.")
            if safe_count >= 2 and balanced_count >= 2:
                recommendations.append("✅ DENGELİ STRATEJİ: Listeniz garanti, güvenli ve ideal hedefler arasında dengeli dağılmış.")

        return {
            "list_id": plist.id,
            "title": plist.title,
            "candidate_rank": c_rank,
            "total_items": total_items,
            "safe_count": safe_count,
            "balanced_count": balanced_count,
            "risky_count": risky_count,
            "dream_count": dream_count,
            "state_count": state_count,
            "foundation_count": foundation_count,
            "avg_predicted_rank": int(round(avg_rank)),
            "city_distribution": cities,
            "recommendations": recommendations,
            "items": evaluated_items,
        }
