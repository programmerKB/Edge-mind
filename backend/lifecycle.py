"""Compatibility facade for application lifecycle hooks."""

from core.lifecycle import application_lifespan, initialize_database

__all__ = ["application_lifespan", "initialize_database"]
