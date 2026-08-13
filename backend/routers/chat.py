from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from schemas import ChatRequest
from services.agent_service import stream_agent_response


router = APIRouter(prefix="/api", tags=["agent"])


def _stream(request: ChatRequest, *, ensure_ascii: bool) -> StreamingResponse:
    return StreamingResponse(
        stream_agent_response(request.message, ensure_ascii=ensure_ascii),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def chat_with_agent(request: ChatRequest):
    return _stream(request, ensure_ascii=True)


@router.post("/chat_utf8")
async def chat_with_agent_utf8(request: ChatRequest):
    return _stream(request, ensure_ascii=False)
