"""Guard retained API routes and deployment readiness after Lab removal."""

from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from edgemind.presentation.api.router import create_api_router


class ApiRoutesTests(unittest.TestCase):
    def setUp(self):
        self.sensors = Mock()
        self.sensors.list_device_summaries.return_value = []

        @contextmanager
        def uow_factory():
            yield SimpleNamespace(sensors=self.sensors)

        app = FastAPI()
        app.include_router(create_api_router(SimpleNamespace(
            uow_factory=uow_factory,
            sensors=Mock(),
            forecasts=Mock(),
            reports=Mock(),
            agent=Mock(),
        )))
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_lab_routes_are_unavailable_and_absent_from_api_documentation(self):
        for method, path in (
            ("GET", "/api/ridge-lab/config"),
            ("POST", "/api/ridge-lab/experiments"),
            ("GET", "/api/ridge-lab/experiments/saved-id"),
            ("POST", "/api/ridge-lab/forecasts"),
        ):
            with self.subTest(method=method, path=path):
                self.assertEqual(self.client.request(method, path).status_code, 404)
        paths = self.client.get("/openapi.json").json()["paths"]
        self.assertFalse(any(path.startswith("/api/ridge-lab") for path in paths))
        for path in (
            "/api/chat",
            "/api/chat_utf8",
            "/api/sensor-readings",
            "/api/predictions/train/{motor_id}",
            "/api/predictions/temperature/{motor_id}",
            "/api/performance/summary",
            "/api/report-artifacts/{artifact_path}",
        ):
            with self.subTest(path=path):
                self.assertIn(path, paths)

    def test_health_checks_storage_and_accepts_an_empty_database(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.sensors.list_device_summaries.assert_called_once_with()

    def test_health_returns_unavailable_when_storage_query_fails(self):
        self.sensors.list_device_summaries.side_effect = RuntimeError("database unavailable")
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "資料服務尚未就緒"})
