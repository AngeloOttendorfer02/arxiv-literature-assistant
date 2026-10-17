from collections.abc import AsyncIterator

from scilitrag.contracts import Evidence, Plan


def generate(plan: Plan, evidence: list[Evidence]) -> AsyncIterator[str]:
    raise NotImplementedError("generate is not implemented yet (WP2); use the mock via USE_MOCKS=1")
