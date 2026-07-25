"""
SQLAlchemy ORM Modelleri — YKS Tahmin Sistemi Kalıcılık Katmanı.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class FavoriteProgram(Base):
    """Favorilere eklenen üniversite programları."""

    __tablename__ = "favorite_programs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    kilavuz_kodu = Column(Integer, unique=True, nullable=False, index=True)
    universite_adi = Column(String(255), nullable=False)
    birim_grup_adi = Column(String(255), nullable=False)
    puan_turu = Column(String(10), nullable=False)
    il_adi = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kilavuz_kodu": self.kilavuz_kodu,
            "universite_adi": self.universite_adi,
            "birim_grup_adi": self.birim_grup_adi,
            "puan_turu": self.puan_turu,
            "il_adi": self.il_adi,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }


class PreferenceList(Base):
    """Kullanıcının oluşturduğu tercih listesi (ör. 'Garanti Tercihlerim 2026')."""

    __tablename__ = "preference_lists"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    target_rank = Column(Integer, nullable=True)  # Öğrencinin YKS sıralaması (ör. 185000)
    point_type = Column(String(10), nullable=True, default="EA")  # SAY, EA, SÖZ, DİL
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    items = relationship("PreferenceListItem", back_populates="preference_list", cascade="all, delete-orphan", order_by="PreferenceListItem.position")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "target_rank": self.target_rank,
            "point_type": self.point_type,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
            "item_count": len(self.items) if self.items else 0,
        }


class PreferenceListItem(Base):
    """Tercih listesi içindeki her bir tercih satırı."""

    __tablename__ = "preference_list_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    list_id = Column(Integer, ForeignKey("preference_lists.id", ondelete="CASCADE"), nullable=False, index=True)
    kilavuz_kodu = Column(Integer, nullable=False)
    position = Column(Integer, nullable=False)  # Tercih Sırası (1, 2, 3...)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    preference_list = relationship("PreferenceList", back_populates="items")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "list_id": self.list_id,
            "kilavuz_kodu": self.kilavuz_kodu,
            "position": self.position,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }


class SearchHistory(Base):
    """Kullanıcının yaptığı son aramalar ve tahmin sorguları."""

    __tablename__ = "search_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_type = Column(String(50), nullable=False)  # 'SEARCH', 'PREDICT', 'COMPARE'
    query_summary = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
