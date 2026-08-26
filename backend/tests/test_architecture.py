"""Guard EdgeMind's dependency direction against accidental layer coupling."""

import ast
from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "edgemind"


def imported_modules(path: Path) -> set[str]:
    """Return absolute import targets declared by one Python module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


class ArchitectureTests(unittest.TestCase):
    """Enforce inward-only dependencies and removal of legacy facades."""

    def assert_layer_avoids(self, layer: str, forbidden: tuple[str, ...]):
        """Fail with file and import details when a layer crosses outward."""
        violations = []
        for path in (PACKAGE_ROOT / layer).rglob("*.py"):
            for module in imported_modules(path):
                if module.startswith(forbidden):
                    violations.append(f"{path.relative_to(PACKAGE_ROOT)} -> {module}")
        self.assertEqual(violations, [], "\n".join(violations))

    def test_domain_has_no_outward_dependencies(self):
        self.assert_layer_avoids(
            "domain",
            (
                "edgemind.application",
                "edgemind.infrastructure",
                "edgemind.presentation",
                "fastapi",
                "sqlalchemy",
                "google",
            ),
        )

    def test_application_depends_only_on_domain_and_ports(self):
        self.assert_layer_avoids(
            "application",
            (
                "edgemind.infrastructure",
                "edgemind.presentation",
                "fastapi",
                "sqlalchemy",
                "google",
            ),
        )

    def test_presentation_does_not_reach_into_infrastructure(self):
        self.assert_layer_avoids(
            "presentation",
            ("edgemind.infrastructure", "sqlalchemy", "google"),
        )

    def test_infrastructure_does_not_import_presentation(self):
        self.assert_layer_avoids(
            "infrastructure",
            ("edgemind.presentation", "fastapi"),
        )

    def test_legacy_top_level_facades_are_removed(self):
        backend_root = PACKAGE_ROOT.parent
        legacy = (
            "database.py",
            "forecasting.py",
            "intent_routing.py",
            "lifecycle.py",
            "performance.py",
            "report_artifacts.py",
            "reporting.py",
            "settings.py",
            "tools.py",
        )
        remaining = [name for name in legacy if (backend_root / name).exists()]
        self.assertEqual(remaining, [])
