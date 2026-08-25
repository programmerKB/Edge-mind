"""SQLAlchemy engine, session factory, and FastAPI session dependency."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from core.config import settings


# ``pool_pre_ping`` cheaply replaces stale connections after a database restart
# instead of failing the first user request that receives one from the pool.
engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Yield one request-scoped session and always return it to the pool."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
