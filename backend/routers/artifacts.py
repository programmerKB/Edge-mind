"""Read-only HTTP access to allow-listed generated report charts."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from reports.artifacts import resolve_chart_artifact


router = APIRouter(prefix="/api/report-artifacts", tags=["report-artifacts"])


@router.get("/{artifact_path:path}")
def get_report_artifact(artifact_path: str):
    """Serve only allow-listed SVG files below completed run directories."""
    chart_path = resolve_chart_artifact(artifact_path)
    if chart_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="找不到報表圖表",
        )
    return FileResponse(
        chart_path,
        media_type="image/svg+xml",
        headers={"Cache-Control": "private, max-age=3600"},
    )
