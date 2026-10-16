"""Laedt den LitSearch-Datensatz (princeton-nlp/LitSearch) nach DATA_DIR.

DATA_DIR wird aus der .env im Projektordner gelesen. Der Hugging-Face-Cache
liegt unter DATA_DIR/hf_home, die Datensaetze unter DATA_DIR/hf_datasets.

Aufruf:
    python scripts/download_data.py                  # alle drei Configs
    python scripts/download_data.py query corpus_clean
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
        sys.exit("DATA_DIR fehlt. Lege eine .env an (Vorlage: .env.example).")
    data_dir = Path(data_dir)
    (data_dir / "hf_home").mkdir(parents=True, exist_ok=True)
    (data_dir / "hf_datasets").mkdir(parents=True, exist_ok=True)

    # Muss vor dem Import von `datasets` gesetzt sein.
    os.environ["HF_HOME"] = str(data_dir / "hf_home")
    os.environ["HF_DATASETS_CACHE"] = str(data_dir / "hf_datasets")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return data_dir


def main() -> None:
    data_dir = setup_dirs()
    configs = sys.argv[1:] or CONFIGS
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        sys.exit(f"Unbekannte Config(s): {unknown}. Erlaubt: {CONFIGS}")

    from datasets import load_dataset

    print(f"DATA_DIR: {data_dir}")
    for config in configs:
        print(f"\n--- {DATASET} / {config} ---")
        ds = load_dataset(DATASET, config, cache_dir=str(data_dir / "hf_datasets"))
        for split, part in ds.items():
            print(f"{split}: {part.num_rows} Zeilen, Spalten: {part.column_names}")


if __name__ == "__main__":
    main()
