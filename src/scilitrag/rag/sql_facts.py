"""SQL facts adapter; the pipeline supplies its exact retrieved evidence when available."""

from scilitrag.config import load_settings
from scilitrag.contracts import Evidence, Plan, SqlFacts
from scilitrag.rag.scenarios.authors import author_facts
from scilitrag.rag.search.backend import default_backend
from scilitrag.rag.search.database import DatabaseReader


class SqlFactsComponent:
    def __init__(self, settings):
        self.settings = settings

    def with_evidence(self, plan: Plan, evidence: list[Evidence]) -> list[SqlFacts]:
        if plan.scenario != "S3":
            return []
        return author_facts(plan, evidence, DatabaseReader(self.settings))

    def __call__(self, plan: Plan) -> list[SqlFacts]:
        if plan.scenario != "S3":
            return []
        # Standalone contract entry point. The API uses with_evidence instead,
        # so SQL ranks exactly the seeds from the already completed retrieval.
        from scilitrag.rag.retrieval import retrieve_with

        backend = default_backend(self.settings)
        return author_facts(
            plan,
            retrieve_with(plan, backend),
            backend.reader,
            dataset_revision=backend.metadata.dataset_revision,
        )


def build_component(settings):
    return SqlFactsComponent(settings)


def sql_facts(plan: Plan) -> list[SqlFacts]:
    return build_component(load_settings())(plan)
