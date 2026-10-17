import asyncio
import uuid
from collections.abc import AsyncIterable
from contextlib import aclosing

from fastapi import FastAPI, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from scilitrag.api.health import dependency_checks
from scilitrag.api.pipeline import run_query
from scilitrag.api.telemetry import RequestLog, request_id_var, setup_app_log, setup_request_log
from scilitrag.components import Components, get_components
from scilitrag.config import Settings, load_settings
from scilitrag.contracts import CONTRACT_VERSION, QueryRequest


class RequestIdMiddleware:
    """Gives every request an id and returns it as X-Request-ID.

    The id is in request.state.request_id and in request_id_var, so every log line written
    while the request runs (also in worker threads and the pipeline task) carries it.

    Plain ASGI on purpose: BaseHTTPMiddleware interferes with streaming and disconnects.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [*message.get("headers", []), (b"x-request-id", request_id.encode())]
                message = {**message, "headers": headers}
            await send(message)

        token = request_id_var.set(request_id)
        try:
            await self.app(scope, receive, send_with_id)
        finally:
            request_id_var.reset(token)


def create_app(components: Components | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    components = components or get_components(settings)
    setup_app_log()
    setup_request_log(settings.log_dir)
    app = FastAPI(title="SciLitRAG")
    app.add_middleware(RequestIdMiddleware)

    @app.get("/health")
    def health() -> dict:
        checks = dependency_checks(settings)
        unhealthy = any(value not in ("ok", "skipped") for value in checks.values())
        return {
            "status": "degraded" if unhealthy else "ok",
            "checks": checks,
            "use_mocks": settings.use_mocks,
            "real_components": sorted(settings.real_components),
            "contract_version": CONTRACT_VERSION,
        }

    @app.post("/query", response_class=EventSourceResponse)
    async def query(body: QueryRequest, request: Request) -> AsyncIterable[ServerSentEvent]:
        log = RequestLog(query_chars=len(body.query), request_id=request.state.request_id)
        events = run_query(body.query, components, settings.request_timeout_s)
        try:
            async with aclosing(events):
                async for name, event in events:
                    log.observe(event)
                    yield ServerSentEvent(data=event, event=name)
        except (asyncio.CancelledError, GeneratorExit):
            # Client disconnected: cancelled in an await, or closed at the yield.
            if log.status == "running":
                log.status = "cancelled"
            raise
        finally:
            log.write()

    return app
