"""Check schema readiness or the complete database search handoff."""

import argparse
import json
from dataclasses import asdict

from scilitrag.config import load_settings
from scilitrag.rag.database import DatabaseReader
from scilitrag.rag.database_contract import validate_database
from scilitrag.rag.errors import RagUserError
from scilitrag.rag.search_readiness import validate_search_indexes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-search", action="store_true")
    args = parser.parse_args()
    settings = load_settings()
    reader = DatabaseReader(settings)
    try:
        metadata = validate_database(
            reader, settings.embed_model, require_search=args.require_search
        )
        if args.require_search:
            validate_search_indexes(reader)
    except RagUserError as error:
        print(f"Database not ready: {error}")
        return 2
    print(
        json.dumps(
            {"mode": "search" if args.require_search else "schema", **asdict(metadata)}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
