"""SSE endpoints adapting transport-neutral Agent events to HTTP."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from edgemind.application.agent import AgentService
from edgemind.presentation.schemas import ChatRequest
from edgemind.presentation.sse import encode_sse_event


def create_router(agent: AgentService) -> APIRouter:
    """Create chat routes bound to the Agent application service."""
    router = APIRouter(prefix="/api", tags=["agent"])

    def stream(request: ChatRequest, *, ensure_ascii: bool) -> StreamingResponse:
        """Adapt application events to a non-buffered SSE response."""

        async def encoded_events():
            """Encode each transport-neutral Agent event as an SSE frame."""
            async for event in agent.stream(
                request.message,
                model_name=request.model_name,
            ):
                yield encode_sse_event(event, ensure_ascii)

        return StreamingResponse(
            encoded_events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @router.post("/chat")
    async def chat_with_agent(request: ChatRequest):
        """Stream an ASCII-escaped response for legacy clients."""
        return stream(request, ensure_ascii=True)

    @router.post("/chat_utf8")
    async def chat_with_agent_utf8(request: ChatRequest):
        """Stream native UTF-8 events for the React client."""
        return stream(request, ensure_ascii=False)

    return router
