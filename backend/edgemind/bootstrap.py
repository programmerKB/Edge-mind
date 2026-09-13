"""Composition root wiring concrete adapters to application-layer ports."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from edgemind.application.agent import AgentService
from edgemind.application.diagnostics import DiagnosticToolService
from edgemind.application.forecasts import ForecastService
from edgemind.application.ports import UnitOfWork
from edgemind.application.ridge_forecasts import RidgeForecastService
from edgemind.application.sensors import SensorService
from edgemind.infrastructure.ai.gemini import GeminiModelGateway
from edgemind.infrastructure.config import settings
from edgemind.infrastructure.persistence.database import SessionLocal
from edgemind.infrastructure.persistence.unit_of_work import SqlAlchemyUnitOfWork
from edgemind.infrastructure.reporting.context import ReportContext
from edgemind.infrastructure.reporting.gateway import FilesystemReportGateway
from edgemind.infrastructure.runtime import create_lifespan
from edgemind.presentation.api.router import create_api_router


@dataclass(frozen=True, slots=True)
class ApplicationContainer:
    """Concrete dependency graph shared by routes and background Agent tools."""

    uow_factory: Callable[[], UnitOfWork]
    reports: FilesystemReportGateway
    forecasts: ForecastService
    ridge_forecasts: RidgeForecastService
    sensors: SensorService
    tools: DiagnosticToolService
    model: GeminiModelGateway
    agent: AgentService


def create_container() -> ApplicationContainer:
    """Instantiate each application service and infrastructure adapter once."""
    uow_factory = partial(SqlAlchemyUnitOfWork, SessionLocal)
    reports = FilesystemReportGateway(ReportContext.from_settings(settings))
    forecasts = ForecastService(reports)
    ridge_forecasts = RidgeForecastService(reports)
    sensors = SensorService()
    tools = DiagnosticToolService(uow_factory, forecasts, ridge_forecasts, sensors)
    model = GeminiModelGateway(settings)
    agent = AgentService(model, tools, reports)
    return ApplicationContainer(
        uow_factory=uow_factory,
        reports=reports,
        forecasts=forecasts,
        ridge_forecasts=ridge_forecasts,
        sensors=sensors,
        tools=tools,
        model=model,
        agent=agent,
    )


def create_app(container: ApplicationContainer | None = None) -> FastAPI:
    """Build the FastAPI application around an explicit dependency graph."""
    dependencies = container or create_container()
    app = FastAPI(
        title="邊緣設備診斷 Agent API",
        lifespan=create_lifespan(dependencies.agent, dependencies.reports),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=settings.cors_origins != ("*",),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(create_api_router(dependencies))
    app.state.container = dependencies
    return app
