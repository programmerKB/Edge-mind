"""SQLAlchemy engine, declarative base, and session factory."""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from edgemind.infrastructure.config import settings


# ``pool_pre_ping`` cheaply replaces stale connections after a database restart
# instead of failing the first user request that receives one from the pool.
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
Base = declarative_base()
