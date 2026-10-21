"""Download the LitSearch dataset (princeton-nlp/LitSearch) into DATA_DIR.

DATA_DIR comes from .env in the project root. The Hub cache is under
DATA_DIR/hf_home, prepared datasets under DATA_DIR/hf_datasets.

Local execution (all three configs, or only the ones given):
    uv run --group data python scripts/download_data.py
    uv run --group data python scripts/download_data.py query corpus_clean

"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

DATASET = "princeton-nlp/LitSearch"
CONFIGS = ["query", "corpus_clean", "corpus_s2orc"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_dirs() -> Path:
    load_dotenv(PROJECT_ROOT / ".env")
    data_dir = os.getenv("DATA_DIR")
    if not data_dir:
        sys.exit("DATA_DIR is missing. Create a .env file using .env.example as a template.")
    data_dir = Path(data_dir)
    (data_dir / "hf_home").mkdir(parents=True, exist_ok=True)
    (data_dir / "hf_datasets").mkdir(parents=True, exist_ok=True)

    # Set these before importing datasets so it uses the configured cache.
    os.environ["HF_HOME"] = str(data_dir / "hf_home")
    os.environ["HF_DATASETS_CACHE"] = str(data_dir / "hf_datasets")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return data_dir


def main() -> None:
    data_dir = setup_dirs()
    configs = sys.argv[1:] or CONFIGS
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"Unknown config(s): {unknown}. Allowed: {CONFIGS}")

    from datasets import load_dataset

    print(f"DATA_DIR: {data_dir}")
    for config in configs:
        print(f"\n--- {DATASET} / {config} ---")
        ds = load_dataset(
            DATASET,
            config,
            revision="9573fb284a1026c998df47024b888a163f0f0e25",
            cache_dir=str(data_dir / "hf_datasets"),
        )
        for split, part in ds.items():
            print(f"{split}: {part.num_rows} rows, columns: {part.column_names}")


if __name__ == "__main__":
    main()
