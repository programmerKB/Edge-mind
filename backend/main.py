"""FastAPI composition root.

The entry point contains wiring only: infrastructure, middleware, and route
packages are assembled here while business behavior remains in services.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.lifecycle import application_lifespan
from routers.api import api_router


app = FastAPI(
    title="邊緣設備診斷 Agent API",
    lifespan=application_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=settings.cors_origins != ("*",),
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router)
