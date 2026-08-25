"""Compatibility facade for the modular database package."""

from core.database import Base, SessionLocal, engine, get_db

__all__ = ["Base", "SessionLocal", "engine", "get_db"]
