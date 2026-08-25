"""Compatibility facade for Agent database tools."""

from services.agent_tools import get_motor_status, get_temperature_forecast

__all__ = ["get_motor_status", "get_temperature_forecast"]
