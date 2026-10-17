from scilitrag.contracts import Plan, SqlFacts


def sql_facts(plan: Plan) -> list[SqlFacts]:
    raise NotImplementedError(
        "sql_facts is not implemented yet (WP2); use the mock via USE_MOCKS=1"
    )
