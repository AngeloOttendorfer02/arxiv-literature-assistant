import asyncio
from dataclasses import replace

from scilitrag.api.pipeline import run_query
from scilitrag.components import mock_components


def test_disconnect_finishes_provider_cleanup_before_returning():
    closed = []

    async def generate(plan, evidence):
        try:
            while True:
                yield "Supported result [1]. "
                await asyncio.sleep(0)
        finally:
            closed.append(True)

    async def scenario():
        stream = run_query("retrieval", replace(mock_components(0), generate=generate))
        async for name, _ in stream:
            if name == "sentence":
                break
        await stream.aclose()
        assert closed == [True]

    asyncio.run(scenario())


def test_slow_consumer_applies_backpressure_to_provider():
    emitted = []

    async def generate(plan, evidence):
        for index in range(1000):
            emitted.append(index)
            yield "Supported result [1]. "

    async def scenario():
        stream = run_query("retrieval", replace(mock_components(0), generate=generate))
        await anext(stream)
        for _ in range(200):
            await asyncio.sleep(0)
        assert len(emitted) < 100
        await stream.aclose()

    asyncio.run(scenario())
