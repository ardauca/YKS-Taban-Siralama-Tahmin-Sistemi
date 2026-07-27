"""
Tercih Listesi Strateji ve Kabul İhtimali Analiz Motoru.
Artık gerçek 2026 ML simülasyon tahmini ve güven aralığı kullanılır.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from db.repository import PreferenceListRepository
from services.search_service import SearchService

logger = logging.getLogger(__name__)


class PreferenceService:
    """Tercih Listesi yönetim ve strateji analiz servisi."""

    @staticmethod
    def calculate_admission_probability(
        candidate_rank: int,
        pred_rank: float,
        pred_lower: float = 0.0,
        pred_upper: float = 0.0,
    ) -> Tuple[str, str, str]:
        """
        Öğrencinin YKS sıralaması ile 2026 ML tahmini ve güven aralığı
        kullanılarak kabul olasılığı ve risk seviyesi hesaplanır.

        Args:
            candidate_rank : Öğrencinin YKS başarı sıralaması
            pred_rank      : CatBoost nokta tahmini (2026)
            pred_lower     : %80 güven aralığı alt sınırı
            pred_upper     : %80 güven aralığı üst sınırı

        Returns:
            (Kabul İhtimali etiketi, Risk Seviyesi kodu, Açıklama)
        """
        if candidate_rank <= 0 or pred_rank <= 0:
            return "Bilinmiyor", "NOT_DEFINED", "Geçersiz sıralama verisi."

        # Nokta tahminine göre fark (pozitif = öğrenci daha iyi sıralamada)
        diff = pred_rank - candidate_rank

        # Güven aralığı yoksa nokta tahminden hesapla
        if pred_lower <= 0:
            pred_lower = pred_rank * 0.80
        if pred_upper <= 0:
            pred_upper = pred_rank * 1.25

        # En iyimser senaryo (üst sınır): bu bile altında kalıyorsak garanti
        # En kötümser senaryo (alt sınır): bu bile üstündeyse çok riskli
        if candidate_rank < pred_lower:
            # Öğrenci, en iyimser senaryonun bile üstünde → GARANTİ
            return (
                "Çok Yüksek ✅",
                "GARANTİ",
                f"Sıralamanız ({candidate_rank:,}) ML tahmininin alt sınırından "
                f"({pred_lower:,.0f}) bile iyi. Yerleşme çok yüksek ihtimal.",
            )
        elif candidate_rank < pred_rank:
            # Nokta tahmin üstünde → GÜVENLİ
            return (
                "Yüksek 🟢",
                "GÜVENLİ",
                f"Sıralamanız ({candidate_rank:,}), 2026 tahmininin ({pred_rank:,.0f}) üstünde.",
            )
        elif candidate_rank < pred_upper * 1.05:
            # Nokta tahmin ile üst sınır arasında → İDEAL/HEDEF
            return (
                "Orta 🟡",
                "İDEAL/HEDEF",
                f"Sıralamanız ({candidate_rank:,}) 2026 tahmininin ({pred_rank:,.0f}) "
                "yakınında. Gerçek olasılık %40-60 civarı.",
            )
        elif candidate_rank < pred_upper * 1.35:
            # Üst sınırı hafifçe aşıyor → SÜRPRİZ
            return (
                "Düşük 🟠",
                "SÜRPRİZ",
                f"Sıralamanız ({candidate_rank:,}) 2026 güven aralığının "
                f"({pred_upper:,.0f}) üstünde. Sürpriz tercih.",
            )
        else:
            return (
                "Çok Düşük 🔴",
                "YÜKSEK RİSK",
                f"Sıralamanız ({candidate_rank:,}) tahmininden "
                f"({pred_rank:,.0f}) belirgin şekilde düşük.",
            )

    @classmethod
    def analyze_preference_list(
        cls,
        list_id: int,
        candidate_rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Bir tercih listesinin tüm satırlarını ve genel stratejiyi analiz eder.
        Gerçek 2026 ML tahminlerini ve güven aralıklarını kullanır.
        """
        plist = PreferenceListRepository.get_list_by_id(list_id)
        if not plist:
            return {"error": "Tercih listesi bulunamadı."}

        items = PreferenceListRepository.get_list_items(list_id)
        c_rank = candidate_rank or plist.target_rank or 150000

        evaluated_items: List[Dict[str, Any]] = []
        safe_count = balanced_count = risky_count = dream_count = 0
        state_count = foundation_count = 0
        cities: Dict[str, int] = {}
        total_pred_rank = 0.0

        for item in items:
            prog = SearchService.get_program_by_code(item["kilavuz_kodu"])
            if not prog:
                continue

            lag1_rank = float(prog.get("lag1_taban_siralama") or 0.0)

            # ── Gerçek ML Tahminini Kullan ──────────────────────────────────
            pred_rank = float(prog.get("pred_2026") or 0)
            pred_lower = float(prog.get("pred_lower") or 0)
            pred_upper = float(prog.get("pred_upper") or 0)

            # Fallback: ML tahmini yoksa basit heuristic
            if pred_rank <= 0 and lag1_rank > 0:
                trend = float(prog.get("siralama_trend") or 0.0)
                pred_rank = max(1.0, lag1_rank + trend * 0.3)
                pred_lower = pred_rank * 0.80
                pred_upper = pred_rank * 1.25

            pct_change = ((pred_rank - lag1_rank) / lag1_rank * 100) if lag1_rank > 0 else 0.0

            prob, risk, exp = cls.calculate_admission_probability(
                c_rank, pred_rank, pred_lower, pred_upper
            )

            # Sayaçlar
            if risk == "GARANTİ":
                safe_count += 1
            elif risk == "GÜVENLİ":
                safe_count += 1
            elif risk == "İDEAL/HEDEF":
                balanced_count += 1
            elif risk == "SÜRPRİZ":
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
            total_pred_rank += pred_rank

            evaluated_items.append({
                "position": item["position"],
                "kilavuz_kodu": item["kilavuz_kodu"],
                "universite_adi": prog.get("universite_adi", "Bilinmiyor"),
                "birim_grup_adi": prog.get("birim_grup_adi") or prog.get("birim_adi", "Bilinmiyor"),
                "il_adi": il,
                "puan_turu": prog.get("puan_turu", "EA"),
                "universite_turu": u_turu,
                "burs_orani": prog.get("burs_orani", "-"),
                "genel_kontenjan": prog.get("lag1_genel_kontenjan", 0.0),
                "lag1_taban_siralama": lag1_rank,
                "pred_2026_siralama": int(round(pred_rank)),
                "pred_lower": int(round(pred_lower)),
                "pred_upper": int(round(pred_upper)),
                "predicted_change_pct": round(pct_change, 1),
                "admission_probability": prob,
                "risk_level": risk,
                "explanation": exp,
                "risk_renk": prog.get("risk_renk", "🟡 STABIL"),
                "notes": item.get("notes", ""),
            })

        total_items = len(evaluated_items)
        avg_rank = (total_pred_rank / total_items) if total_items > 0 else 0.0

        # ── Strateji Öneri Algoritması ─────────────────────────────────────────
        recommendations: List[str] = []
        if total_items == 0:
            recommendations.append(
                "Listenizde henüz tercih bulunmuyor. Arama ekranından (F) program ekleyebilirsiniz."
            )
        else:
            if safe_count == 0:
                recommendations.append(
                    "⚠️ GARANTİ TERCİH YOK: Listenizin son sıralarına 2-3 garanti tercih ekleyin."
                )
            if risky_count + dream_count > total_items * 0.6:
                recommendations.append(
                    f"⚠️ YÜKSEK RİSKLİ LİSTE: {risky_count + dream_count} / {total_items} tercih riskli."
                )
            if safe_count >= 2 and balanced_count >= 2:
                recommendations.append(
                    "✅ DENGELİ STRATEJİ: Listeniz garanti, güvenli ve ideal hedefler arasında dengeli."
                )
            if foundation_count > state_count and total_items >= 4:
                recommendations.append(
                    f"💡 VAKIF AĞIRLIKLI: {foundation_count} vakıf / {state_count} devlet. "
                    "Devlet alternatiflerinizi de değerlendirin."
                )
            top_cities = sorted(cities.items(), key=lambda x: x[1], reverse=True)
            if top_cities and top_cities[0][1] > total_items * 0.7:
                recommendations.append(
                    f"📍 ŞEHİR RİSKİ: Tercihlerinizin %70'i {top_cities[0][0]}'da. Farklı şehirler deneyin."
                )

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
