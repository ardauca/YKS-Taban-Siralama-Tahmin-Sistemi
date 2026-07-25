"""
Veri Erişim Katmanı (Repository Pattern) — SQLite CRUD İşlemleri.
"""
from __future__ import annotations

from typing import List, Optional
from db.database import SessionLocal, init_db
from db.models import FavoriteProgram, PreferenceList, PreferenceListItem, SearchHistory

# DB tablolarının hazır olduğundan emin ol
init_db()


class FavoriteRepository:
    """Favori program CRUD işlemleri."""

    @staticmethod
    def add_favorite(kilavuz_kodu: int, universite_adi: str, birim_grup_adi: str, puan_turu: str, il_adi: str = "") -> FavoriteProgram:
        db = SessionLocal()
        try:
            existing = db.query(FavoriteProgram).filter(FavoriteProgram.kilavuz_kodu == kilavuz_kodu).first()
            if existing:
                return existing
            fav = FavoriteProgram(
                kilavuz_kodu=kilavuz_kodu,
                universite_adi=universite_adi,
                birim_grup_adi=birim_grup_adi,
                puan_turu=puan_turu,
                il_adi=il_adi,
            )
            db.add(fav)
            db.commit()
            db.refresh(fav)
            return fav
        finally:
            db.close()

    @staticmethod
    def remove_favorite(kilavuz_kodu: int) -> bool:
        db = SessionLocal()
        try:
            fav = db.query(FavoriteProgram).filter(FavoriteProgram.kilavuz_kodu == kilavuz_kodu).first()
            if fav:
                db.delete(fav)
                db.commit()
                return True
            return False
        finally:
            db.close()

    @staticmethod
    def is_favorite(kilavuz_kodu: int) -> bool:
        db = SessionLocal()
        try:
            return db.query(FavoriteProgram).filter(FavoriteProgram.kilavuz_kodu == kilavuz_kodu).first() is not None
        finally:
            db.close()

    @staticmethod
    def get_all_favorites() -> List[dict]:
        db = SessionLocal()
        try:
            favs = db.query(FavoriteProgram).order_by(FavoriteProgram.created_at.desc()).all()
            return [f.to_dict() for f in favs]
        finally:
            db.close()


class PreferenceListRepository:
    """Tercih Listeleri CRUD işlemleri."""

    @staticmethod
    def create_list(title: str, target_rank: Optional[int] = None, point_type: str = "EA", notes: str = "") -> PreferenceList:
        db = SessionLocal()
        try:
            plist = PreferenceList(title=title, target_rank=target_rank, point_type=point_type, notes=notes)
            db.add(plist)
            db.commit()
            db.refresh(plist)
            return plist
        finally:
            db.close()

    @staticmethod
    def get_all_lists() -> List[dict]:
        db = SessionLocal()
        try:
            lists = db.query(PreferenceList).order_by(PreferenceList.updated_at.desc()).all()
            return [l.to_dict() for l in lists]
        finally:
            db.close()

    @staticmethod
    def get_list_by_id(list_id: int) -> Optional[PreferenceList]:
        db = SessionLocal()
        try:
            return db.query(PreferenceList).filter(PreferenceList.id == list_id).first()
        finally:
            db.close()

    @staticmethod
    def delete_list(list_id: int) -> bool:
        db = SessionLocal()
        try:
            plist = db.query(PreferenceList).filter(PreferenceList.id == list_id).first()
            if plist:
                db.delete(plist)
                db.commit()
                return True
            return False
        finally:
            db.close()

    @staticmethod
    def add_item_to_list(list_id: int, kilavuz_kodu: int, notes: str = "") -> Optional[PreferenceListItem]:
        db = SessionLocal()
        try:
            plist = db.query(PreferenceList).filter(PreferenceList.id == list_id).first()
            if not plist:
                return None
            
            # Zaten eklendiyse tekrar ekleme
            existing = db.query(PreferenceListItem).filter(
                PreferenceListItem.list_id == list_id,
                PreferenceListItem.kilavuz_kodu == kilavuz_kodu
            ).first()
            if existing:
                return existing

            max_pos = db.query(PreferenceListItem).filter(PreferenceListItem.list_id == list_id).count()
            item = PreferenceListItem(
                list_id=list_id,
                kilavuz_kodu=kilavuz_kodu,
                position=max_pos + 1,
                notes=notes
            )
            db.add(item)
            db.commit()
            db.refresh(item)
            return item
        finally:
            db.close()

    @staticmethod
    def remove_item_from_list(list_id: int, kilavuz_kodu: int) -> bool:
        db = SessionLocal()
        try:
            item = db.query(PreferenceListItem).filter(
                PreferenceListItem.list_id == list_id,
                PreferenceListItem.kilavuz_kodu == kilavuz_kodu
            ).first()
            if item:
                db.delete(item)
                db.commit()

                # Re-index positions
                items = db.query(PreferenceListItem).filter(PreferenceListItem.list_id == list_id).order_by(PreferenceListItem.position).all()
                for idx, it in enumerate(items, start=1):
                    it.position = idx
                db.commit()
                return True
            return False
        finally:
            db.close()

    @staticmethod
    def move_item(list_id: int, kilavuz_kodu: int, direction: str) -> bool:
        """Item'ı yukarı ('up') veya aşağı ('down') taşır."""
        db = SessionLocal()
        try:
            items = db.query(PreferenceListItem).filter(PreferenceListItem.list_id == list_id).order_by(PreferenceListItem.position).all()
            target_idx = None
            for idx, it in enumerate(items):
                if it.kilavuz_kodu == kilavuz_kodu:
                    target_idx = idx
                    break
            
            if target_idx is None:
                return False

            if direction == "up" and target_idx > 0:
                items[target_idx].position, items[target_idx - 1].position = items[target_idx - 1].position, items[target_idx].position
                db.commit()
                return True
            elif direction == "down" and target_idx < len(items) - 1:
                items[target_idx].position, items[target_idx + 1].position = items[target_idx + 1].position, items[target_idx].position
                db.commit()
                return True
            return False
        finally:
            db.close()

    @staticmethod
    def update_item_note(list_id: int, kilavuz_kodu: int, note: str) -> bool:
        db = SessionLocal()
        try:
            item = db.query(PreferenceListItem).filter(
                PreferenceListItem.list_id == list_id,
                PreferenceListItem.kilavuz_kodu == kilavuz_kodu
            ).first()
            if item:
                item.notes = note
                db.commit()
                return True
            return False
        finally:
            db.close()

    @staticmethod
    def get_list_items(list_id: int) -> List[dict]:
        db = SessionLocal()
        try:
            items = db.query(PreferenceListItem).filter(PreferenceListItem.list_id == list_id).order_by(PreferenceListItem.position).all()
            return [it.to_dict() for it in items]
        finally:
            db.close()


class SearchHistoryRepository:
    """Arama ve Tahmin Geçmişi CRUD işlemleri."""

    @staticmethod
    def add_history(query_type: str, query_summary: str) -> SearchHistory:
        db = SessionLocal()
        try:
            sh = SearchHistory(query_type=query_type, query_summary=query_summary)
            db.add(sh)
            db.commit()
            db.refresh(sh)
            return sh
        finally:
            db.close()

    @staticmethod
    def get_recent_history(limit: int = 10) -> List[dict]:
        db = SessionLocal()
        try:
            hist = db.query(SearchHistory).order_by(SearchHistory.created_at.desc()).limit(limit).all()
            return [{"id": h.id, "type": h.query_type, "summary": h.query_summary, "created_at": h.created_at.strftime("%H:%M:%S")} for h in hist]
        finally:
            db.close()
