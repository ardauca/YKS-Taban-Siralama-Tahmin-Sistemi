"""
SQLAlchemy Veritabanı Bağlantı ve Oturum Yönetimi.
"""
from __future__ import annotations

from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from db.models import Base

DB_PATH = Path(__file__).parent.parent / "yks_tahmin.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = scoped_session(
    sessionmaker(autocommit=False, autoflush=False, bind=engine)
)


def init_db():
    """Veritabanı tablolarını oluşturur."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Oturum döndüren jeneratör."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
