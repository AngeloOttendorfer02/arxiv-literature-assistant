import re
from collections.abc import AsyncIterator
from contextlib import aclosing

from scilitrag.config import load_settings
from scilitrag.contracts import Evidence, Plan
from scilitrag.rag.evidence.context import build_context
from scilitrag.rag.evidence.labels import ABSTRACT_ONLY_SECTION, abstract_notice
from scilitrag.rag.llm.client import ChatClient


def _is_plain_author_ranking(plan: Plan) -> bool:
    """Only optimize a standalone ranking request whose topic matches the plan."""
    if plan.scenario != "S3" or not plan.query or not plan.topic:
        return False
    match = re.fullmatch(
        r"\s*(?:rank|list|show|find)\s+(?:the\s+)?"
        r"(?:(?:up to|top)\s+\d+\s+)?most\s+cited\s+authors\s+"
        r"(?:for|on|about|in)\s+(.+?)\s*[?.]?\s*",
        plan.query,
        flags=re.IGNORECASE,
    )
    return bool(match and match.group(1).strip().casefold() == plan.topic.strip().casefold())


async def generate_with(
    plan: Plan, evidence: list[Evidence], client, settings=None
) -> AsyncIterator[str]:
    if not evidence:
        return
    # Frame sentences from code go out with the model's first token, so they do not count as TTFT.
    frame = ""
    if plan.scenario == "S3":
        for item in evidence:
            if item.type == "sql":
                frame += (
                    "The ranking counts one credit per author for each indexed citing-paper "
                    f"and cited-paper pair [{item.label}]. "
                )
                if _is_plain_author_ranking(plan):
                    yield frame
                    return
    settings = settings or getattr(client, "settings", None)
    packet = build_context(
        plan,
        evidence,
        char_budget=settings.context_char_budget if settings else 16000,
        item_limit=settings.evidence_item_char_limit if settings else 2400,
    )
    if not packet.included_labels:
        if frame:
            yield frame
        return
    included = set(packet.included_labels)
    abstract_only = [
        item.label
        for item in evidence
        if item.label in included and item.type == "paper" and item.section == ABSTRACT_ONLY_SECTION
    ]
    if abstract_only:
        frame += abstract_notice(abstract_only) + " "
    async with aclosing(client.stream(packet.messages)) as stream:
        async for token in stream:
            if frame:
                yield frame
                frame = ""
            yield token
    if frame:
        yield frame


def build_component(settings):
    return lambda plan, evidence: generate_with(plan, evidence, ChatClient(settings), settings)


async def generate(plan: Plan, evidence: list[Evidence]) -> AsyncIterator[str]:
    async with aclosing(build_component(load_settings())(plan, evidence)) as stream:
        async for token in stream:
            yield token
