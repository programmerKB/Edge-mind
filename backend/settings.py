"""Compatibility exports for code that still imports top-level settings.

New modules should import :mod:`core.config`; this facade prevents breaking
existing deployments while the project adopts package-based imports.
"""

from core.config import AGENT_SYSTEM_INSTRUCTION, settings


DATABASE_URL = settings.database_url
MODEL_ID = settings.model_id
AGENT_RESPONSE_TIMEOUT_SECONDS = settings.agent_response_timeout_seconds
SEED_DEMO_DATA = settings.seed_demo_data
CORS_ORIGINS = settings.cors_origins

__all__ = [
    "AGENT_RESPONSE_TIMEOUT_SECONDS",
    "AGENT_SYSTEM_INSTRUCTION",
    "CORS_ORIGINS",
    "DATABASE_URL",
    "MODEL_ID",
    "SEED_DEMO_DATA",
]
