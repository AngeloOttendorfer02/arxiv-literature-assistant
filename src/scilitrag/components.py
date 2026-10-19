import importlib
from dataclasses import dataclass

from scilitrag.config import ConfigError, Settings, load_settings
from scilitrag.contracts import (
    CheckInput,
    CheckSentence,
    Embed,
    Generate,
    GetSqlFacts,
    MaskPii,
    PlanQuery,
    RenderCitations,
    Retrieve,
)
from scilitrag.mocks import data as mock_data
from scilitrag.mocks import guardrails as mock_guardrails
from scilitrag.mocks import rag as mock_rag

# Imported lazily, so mock mode never loads model dependencies.
REAL = {
    "check_input": "scilitrag.guardrails.input",
    "mask_pii": "scilitrag.guardrails.pii",
    "embed": "scilitrag.data.embedding",
    "plan": "scilitrag.rag.planner",
    "retrieve": "scilitrag.rag.retrieval",
    "sql_facts": "scilitrag.rag.sql_facts",
    "generate": "scilitrag.rag.generation",
    "check_sentence": "scilitrag.guardrails.validation",
    "render_citations": "scilitrag.rag.citations",
}


@dataclass(frozen=True)
class Components:
    check_input: CheckInput
    mask_pii: MaskPii
    embed: Embed
    plan: PlanQuery
    retrieve: Retrieve
    sql_facts: GetSqlFacts
    generate: Generate
    check_sentence: CheckSentence
    render_citations: RenderCitations


def mock_components(token_delay: float = 0.02) -> Components:
    return Components(
        check_input=mock_guardrails.check_input,
        mask_pii=mock_guardrails.mask_pii,
        embed=mock_data.embed,
        plan=mock_rag.plan,
        retrieve=mock_rag.retrieve,
        sql_facts=mock_rag.sql_facts,
        generate=mock_rag.Generate(token_delay),
        check_sentence=mock_guardrails.check_sentence,
        render_citations=mock_rag.render_citations,
    )


def get_components(settings: Settings | None = None, token_delay: float = 0.02) -> Components:
    """Select components lazily and pass one settings snapshot to supported factories."""
    settings = settings or load_settings()
    unknown = sorted(settings.real_components - REAL.keys())
    if unknown:
        raise ConfigError(f"Unknown REAL_COMPONENTS: {', '.join(unknown)}")

    mocks = mock_components(token_delay)
    selected = {name: getattr(mocks, name) for name in REAL}
    for name in REAL.keys() if not settings.use_mocks else settings.real_components:
        module = importlib.import_module(REAL[name])
        # Settings-aware components use a factory. Modules without one expose
        # the component callable directly.
        factory = getattr(module, "build_component", None)
        selected[name] = factory(settings) if factory is not None else getattr(module, name)
    return Components(**selected)
