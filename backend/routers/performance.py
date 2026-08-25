"""Read-only endpoint for aggregate inference performance."""

import json

from fastapi import APIRouter

from reports.config import REPORT_ROOT


router = APIRouter(prefix="/api/performance", tags=["performance"])


@router.get("/summary")
def get_performance_summary():
    """Return aggregate statistics for all finalized inference runs."""
    summary_path = REPORT_ROOT / "performance" / "performance_summary.json"
    if not summary_path.is_file():
        return {"run_count": 0, "metrics": {}, "files": {}}
    return json.loads(summary_path.read_text(encoding="utf-8"))
