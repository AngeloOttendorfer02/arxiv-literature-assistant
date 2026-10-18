"""Download the LitSearch dataset (princeton-nlp/LitSearch) into DATA_DIR.

DATA_DIR comes from .env in the project root. The Hub cache is under
DATA_DIR/hf_home, prepared datasets under DATA_DIR/hf_datasets.

Usage (all three configs, or only the ones given):
    uv run --group data python scripts/download_data.py
    uv run --group data python scripts/download_data.py query corpus_clean

Windows Smart App Control blocks the unsigned PyArrow DLLs ("DLL load failed ...
application control policy"). Then run it in a Linux container from the project
root (PowerShell, Docker Desktop running). Replace <DATA_DIR> with the Windows
path from .env, outside OneDrive (e.g. C:/Users/<name>/scai_data):
    docker run --rm -it -v "${PWD}:/app" -v "<DATA_DIR>:/data" -v scilitrag-uv-cache:/root/.cache/uv -w /app -e DATA_DIR=/data -e UV_PROJECT_ENVIRONMENT=/opt/venv -e UV_LINK_MODE=copy ghcr.io/astral-sh/uv:python3.13-bookworm-slim uv run --locked --group data python scripts/download_data.py

If a download stops (e.g. 401 from cas-server.xethub.hf.co or a dropped
connection), rerun the same command. Finished files are reused from the cache,
so do not delete DATA_DIR between runs.
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
