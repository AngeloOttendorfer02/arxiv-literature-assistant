from collections.abc import AsyncIterable

from fastapi import FastAPI
from fastapi.sse import EventSourceResponse, ServerSentEvent

from scilitrag.api.pipeline import run_query
from scilitrag.components import Components, get_components
from scilitrag.config import Settings, load_settings
from scilitrag.contracts import CONTRACT_VERSION, QueryRequest


def create_app(components: Components | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    components = components or get_components(settings)
    app = FastAPI(title="SciLitRAG")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "use_mocks": settings.use_mocks,
            "real_components": sorted(settings.real_components),
            "contract_version": CONTRACT_VERSION,
        }

    @app.post("/query", response_class=EventSourceResponse)
    async def query(request: QueryRequest) -> AsyncIterable[ServerSentEvent]:
        async for name, event in run_query(request.query, components):
            yield ServerSentEvent(data=event, event=name)

    return app
