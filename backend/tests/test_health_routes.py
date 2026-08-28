"""Deployment health endpoint tests."""

import unittest

from fastapi import HTTPException

from edgemind.presentation.api.routes.health import create_router


class FakeSensors:
    def list_device_summaries(self):
        return []


class FakeUnitOfWork:
    def __enter__(self):
        self.sensors = FakeSensors()
        return self

    def __exit__(self, *_args):
        return None


class FailingUnitOfWork:
    def __enter__(self):
        raise RuntimeError("database unavailable")

    def __exit__(self, *_args):
        return None


class HealthRouteTests(unittest.TestCase):
    @staticmethod
    def endpoint(router, path):
        return next(route.endpoint for route in router.routes if route.path == path)

    def test_liveness_and_readiness(self):
        router = create_router(FakeUnitOfWork)

        self.assertEqual(
            self.endpoint(router, "/api/health/live")(),
            {"status": "ok"},
        )
        self.assertEqual(
            self.endpoint(router, "/api/health/ready")(),
            {"status": "ready", "database": "reachable"},
        )

    def test_readiness_fails_closed_without_leaking_exception_details(self):
        endpoint = self.endpoint(
            create_router(FailingUnitOfWork),
            "/api/health/ready",
        )

        with self.assertRaises(HTTPException) as context:
            endpoint()
        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(context.exception.detail, "database is not ready")


if __name__ == "__main__":
    unittest.main()
