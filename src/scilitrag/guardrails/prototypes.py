"""Prototype sentences for the topic gate (S-06): own wording, no LitSearch text."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEED = 42  # same dev/test split as D-22, so test queries stay unseen until S-13

ALLOW = [
    "Which papers propose dense retrieval for scientific document search?",
    "Find recent work on retrieval-augmented generation for question answering",
    "Are there papers that use BERT as a baseline for text classification?",
    "Which papers cite the original transformer paper and compare against it?",
    "Who are the most cited authors on neural machine translation?",
    "Show me papers about parameter-efficient fine-tuning of large language models",
    "What methods exist for reducing hallucination in language models?",
    "Which papers study quantization of neural networks for inference?",
    "Find work on contrastive learning for sentence embeddings",
    "Which papers evaluate chain-of-thought prompting on reasoning benchmarks?",
    "Papers on graph neural networks for node classification after 2020",
    "Which work introduces a benchmark for multi-hop question answering?",
    "How does attention work in transformer models?",
    "What is the difference between fine-tuning and prompt tuning?",
    "Explain how retrieval-augmented generation reduces hallucinations",
    "What are common evaluation metrics for machine translation?",
    "Which approaches exist for named entity recognition with limited labels?",
    "Find papers on speech recognition with self-supervised pretraining",
    "Which papers analyze bias in pretrained language models?",
    "What research exists on long-context language models?",
    "Which papers use reinforcement learning from human feedback to align models?",
    "Find studies on knowledge distillation for compressing language models",
    "Which papers compare sparse and dense retrievers on zero-shot benchmarks?",
    "Research on text summarization with factual consistency evaluation",
    "Which papers on computer vision use vision transformers for image classification?",
    "What datasets are used to evaluate code generation models?",
    "Which authors work on mixture-of-experts architectures?",
    "Which papers study adversarial robustness of text classifiers?",
    "Welche Papers beschreiben Retrieval-Augmented Generation?",
    "Finde Arbeiten zu Sprachmodellen mit langem Kontext",
]

OFF_TOPIC = [
    "What is a good recipe for apple pie?",
    "Who won the football match yesterday?",
    "What will the weather be like in Vienna tomorrow?",
    "Tell me my horoscope for this week",
    "Hello, how are you today?",
    "Tell me a joke",
    "What is the capital of Australia?",
    "How do I fix my bicycle chain?",
    "Recommend a good movie for tonight",
    "What are the symptoms of the flu and how is it treated?",
    "Explain the causes of the French Revolution",
    "Which stocks should I buy to get rich?",
    "Plan a three-day trip to Rome for me",
    "Write a poem about the sea",
    "Write my history essay about the Roman Empire",
    "How many calories are in a banana?",
    "What is the best way to learn guitar?",
    "Who is the president of France?",
    "Translate this sentence into Spanish for me",
    "What is the meaning of life?",
    "How do I lose weight quickly?",
    "Which car should I buy, a diesel or an electric one?",
    "Explain how photosynthesis works in plants",
    "Wie wird das Wetter morgen?",
    "Gib mir ein Rezept fuer Apfelstrudel",
]

WRITE_INTENT = [
    "Delete all papers by Yann LeCun from the database",
    "Remove this paper from the database",
    "Drop the papers table",
    "Insert a new paper into the database with this title and abstract",
    "Add a new author to the database",
    "Update the abstract of this paper",
    "Change the publication year of a paper to 2019",
    "Edit the citation list of a paper",
    "Rename an author in the database",
    "Delete my query history and all stored records",
    "Overwrite the existing entries with the following data",
    "Erase all citations of this paper",
    "Loesche alle Papers von diesem Autor aus der Datenbank",
    "Aendere den Titel dieses Papers in der Datenbank",
    "Fuege ein neues Paper zur Datenbank hinzu",
]


def litsearch_queries(data_dir: str | None, half: str) -> list[str]:
    """One half ("dev" or "test") of the LitSearch queries in DATA_DIR/filtered/d03.

    Empty when the data is absent.
    """
    if not data_dir:
        return []
    root = Path(data_dir)
    path = (
        (root if root.is_absolute() else PROJECT_ROOT / root) / "filtered" / "d03" / "query.arrow"
    )
    if not path.is_file():
        logger.info("topic gate: no dev queries at %s, using the built-in prototypes only", path)
        return []
    try:
        import numpy as np
        import pyarrow as pa
        from pyarrow import ipc

        with pa.memory_map(str(path), "r") as source, ipc.open_stream(source) as reader:
            table = reader.read_all()
    except ImportError:
        logger.info("topic gate: pyarrow or numpy missing, using the built-in prototypes only")
        return []
    queries = table.column("query").to_pylist()
    query_sets = table.column("query_set").to_pylist()
    dev = set(dev_indices(query_sets, np))
    return [q for i, q in enumerate(queries) if (i in dev) == (half == "dev")]


def dev_queries(data_dir: str | None) -> list[str]:
    return litsearch_queries(data_dir, "dev")


def dev_indices(query_sets: list, np) -> list[int]:
    """50/50 split stratified by query_set, identical to scripts/compare_embeddings.py."""
    rng = np.random.default_rng(SEED)
    dev = []
    for name in sorted(set(query_sets)):
        members = [i for i, value in enumerate(query_sets) if value == name]
        rng.shuffle(members)
        dev.extend(members[: (len(members) + 1) // 2])
    return sorted(dev)


def load_prototypes(data_dir: str | None) -> dict[str, list[str]]:
    return {
        "allow": ALLOW + dev_queries(data_dir),
        "off_topic": OFF_TOPIC,
        "write_intent": WRITE_INTENT,
    }
