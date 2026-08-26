"""ASGI entry point; all dependency wiring lives in ``edgemind.bootstrap``."""

from edgemind.bootstrap import create_app


app = create_app()
