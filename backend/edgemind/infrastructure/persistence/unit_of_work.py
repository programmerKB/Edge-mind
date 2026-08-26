"""SQLAlchemy transaction boundary implementing the application UoW port."""

from __future__ import annotations

from sqlalchemy.orm import Session, sessionmaker

from edgemind.infrastructure.persistence.model_repository import (
    SqlAlchemyForecastModelRepository,
)
from edgemind.infrastructure.persistence.sensor_repository import (
    SqlAlchemySensorRepository,
)


class SqlAlchemyUnitOfWork:
    """Own one session and expose repository adapters for a single use case."""

    def __init__(self, session_factory: sessionmaker):
        """Store the factory used to open one session per use case."""
        self._session_factory = session_factory
        self._session: Session | None = None

    def __enter__(self) -> "SqlAlchemyUnitOfWork":
        """Open a session and construct repositories sharing its transaction."""
        self._session = self._session_factory()
        self.sensors = SqlAlchemySensorRepository(self._session)
        self.forecast_models = SqlAlchemyForecastModelRepository(self._session)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Rollback failed work and always return the connection to the pool."""
        assert self._session is not None
        try:
            if exc_type is not None:
                self._session.rollback()
        finally:
            self._session.close()
            self._session = None

    def commit(self) -> None:
        """Commit all repository changes in this unit of work."""
        assert self._session is not None
        self._session.commit()

    def rollback(self) -> None:
        """Explicitly rollback the current transaction."""
        assert self._session is not None
        self._session.rollback()
