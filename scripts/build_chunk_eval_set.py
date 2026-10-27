"""D-19 chunk evaluation set (O16): synthetic questions about full-text passages with source spans.

Usage (local Ollama, OpenAI-compatible at LLM_BASE_URL):
    uv run --locked --group data --group rag python scripts/build_chunk_eval_set.py

Passages are sampled from the D-12 papers, cleaned exactly like the database build (D-10, D-11),
independent of chunk boundaries. Output: eval/private/chunk_eval_set.json (ignored, contains
dataset text). An existing output is resumed, so finished questions are not generated twice.
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from filter_data import batches
from fulltext import clean_sections
from select_fixture import data_directory

from scilitrag.guardrails.sentences import split_sentences

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "eval" / "private" / "chunk_eval_set.json"
SEED = 1789
MODEL = "gpt-oss:20b"
MIN_WORDS, MAX_WORDS = 15, 60
MAX_COPIED_RUN = 5
MAX_JACCARD = 0.5
SKIP_SECTIONS = re.compile(r"(?i)acknowledg|funding")
CAPTION = re.compile(r"(?i)^(?:fig(?:ure)?|table)\.?\s*\d")
SELF_REFERENCE = re.compile(r"(?i)\b(?:this|the) (?:paper|passage|study|work|text|section)\b")
WORD = re.compile(r"[^\W_]+")

PROMPT = """You write evaluation questions for a passage search engine over scientific papers.

Paper title: {title}
Passage:
{passage}

Write one question that a researcher could type into the search engine and that this passage
answers. Rules:
- Name the specific method, dataset, task or finding, so the question can identify this passage
  among thousands of papers.
- Do not state the answer in the question.
- Paraphrase: never copy more than 4 consecutive words from the passage.
- Do not write "this paper", "the authors", "we" or "the passage".
Set usable to false if the passage holds no self-contained fact, e.g. only a reference,
a table or formula fragment, or a transition sentence."""

SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "usable": {"type": "boolean"}},
    "required": ["question", "usable"],
    "additionalProperties": False,
}


def words(text):
    return WORD.findall(text.casefold())


def sentence_spans(text):
    """Character spans of the S-11 sentences of a section text."""
    sentences, rest = split_sentences(text + "\n")
    spans, position = [], 0
    for sentence in [*sentences, rest.strip()]:
        start = text.find(sentence, position) if sentence else -1
        if start < 0:
            continue
        spans.append((start, start + len(sentence)))
        position = start + len(sentence)
    return spans


def usable_sentence(text):
    if CAPTION.match(text) or "<EMAIL>" in text or len(words(text)) < 8:
        return False
    visible = [c for c in text if not c.isspace()]
    return sum(c.isalpha() for c in visible) >= 0.7 * len(visible)


def pick_passage(sections, rng):
    """One random passage of 1 or 2 consecutive usable sentences: (section_index, start, end)."""
    candidates = []
    for index, (title, text) in enumerate(sections):
        if title and SKIP_SECTIONS.search(title):
            continue
        spans = sentence_spans(text)
        for i, (start, end) in enumerate(spans):
            if not usable_sentence(text[start:end]):
                continue
            for last in (i, i + 1):
                if last >= len(spans) or not usable_sentence(text[spans[last][0] : spans[last][1]]):
                    continue
                count = len(words(text[start : spans[last][1]]))
                if MIN_WORDS <= count <= MAX_WORDS:
                    candidates.append((index, start, spans[last][1]))
    return rng.choice(candidates) if candidates else None


def longest_copied_run(question, passage):
    """Longest run of consecutive words shared by question and passage."""
    q, p = words(question), words(passage)
    best, previous = 0, [0] * (len(p) + 1)
    for a in q:
        current = [0] * (len(p) + 1)
        for j, b in enumerate(p, 1):
            if a == b:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def jaccard(question, passage):
    a, b = set(words(question)), set(words(passage))
    return len(a & b) / len(a | b) if a | b else 1.0


def rejection(question, usable, passage):
    """Reason to drop a generated question, or None."""
    if not usable:
        return "not_usable"
    if len(words(question)) < 5:
        return "too_short"
    if SELF_REFERENCE.search(question):
        return "self_reference"
    if longest_copied_run(question, passage) > MAX_COPIED_RUN:
        return "copied"
    if jaccard(question, passage) > MAX_JACCARD:
        return "lexical_overlap"
    return None


def ask(client, base_url, model, title, passage):
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json={
            "model": model,
            "temperature": 0,
            "seed": SEED,
            "messages": [{"role": "user", "content": PROMPT.format(title=title, passage=passage)}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "strict": True, "schema": SCHEMA},
            },
        },
    )
    response.raise_for_status()
    answer = json.loads(response.json()["choices"][0]["message"]["content"])
    return answer["question"].strip(), bool(answer["usable"])


def selected_papers(data_dir):
    """Cleaned sections and titles of the D-12 papers, as stored in the chunks table."""
    selection = json.loads(
        (data_dir / "fulltext" / "v1" / "selection.json").read_text(encoding="utf-8")
    )
    ids, folder = set(selection["corpusids"]), data_dir / "filtered" / "d03"
    titles = {}
    for batch in batches([folder / "corpus_clean.arrow"]):
        for row in batch.select(["corpusid", "title"]).to_pylist():
            if row["corpusid"] in ids:
                titles[row["corpusid"]] = row["title"] or ""
    sections = {}
    for batch in batches([folder / "corpus_s2orc.arrow"]):
        for row in batch.select(["corpusid", "content"]).to_pylist():
            if row["corpusid"] in ids:
                sections[row["corpusid"]] = clean_sections(row)[0]
    if sections.keys() != ids:
        raise ValueError("Some D-12 papers have no S2ORC full text.")
    return selection, titles, sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=int, default=100, help="Accepted questions to keep.")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    import httpx

    # Windows consoles cannot print every character of the generated questions.
    sys.stdout.reconfigure(errors="replace")
    data_dir = data_directory()
    if not (base_url := os.environ.get("LLM_BASE_URL")):
        parser.error("Set LLM_BASE_URL in .env.")
    selection, titles, sections = selected_papers(data_dir)
    prompt_sha = hashlib.sha256(PROMPT.encode()).hexdigest()
    previous = {}
    if args.output.exists():
        old = json.loads(args.output.read_text(encoding="utf-8"))
        if (old["model"], old["prompt_sha256"], old["seed"]) == (args.model, prompt_sha, SEED):
            previous = {item["corpus_id"]: item for item in old["items"]}
    papers = sorted(sections)
    random.Random(SEED).shuffle(papers)
    rng = random.Random(SEED)
    items = []
    started = time.perf_counter()
    with httpx.Client(timeout=600) as client:
        for corpus_id in papers:
            if sum(item["rejected"] is None for item in items) >= args.questions:
                break
            # Draw for every paper, so resumed runs keep the same passages.
            picked = pick_passage(sections[corpus_id], rng)
            if picked is None:
                continue
            index, start, end = picked
            title, text = sections[corpus_id][index]
            passage = text[start:end]
            if corpus_id in previous and previous[corpus_id]["passage"] == passage:
                items.append(previous[corpus_id])
                continue
            question, usable = ask(client, base_url, args.model, titles[corpus_id], passage)
            item = {
                "id": f"q{len(items):03d}",
                "corpus_id": corpus_id,
                "section_index": index,
                "section": title,
                "start": start,
                "end": end,
                "passage": passage,
                "question": question,
                "rejected": rejection(question, usable, passage),
                "review": None,
                "review_note": "",
            }
            items.append(item)
            kept = sum(i["rejected"] is None for i in items)
            print(f"{len(items):4d} {kept:4d} kept  {item['rejected'] or ''}  {question[:90]}")
            write(args, selection, prompt_sha, items)
    write(args, selection, prompt_sha, items)
    reasons = {}
    for item in items:
        reasons[item["rejected"] or "kept"] = reasons.get(item["rejected"] or "kept", 0) + 1
    print(f"{args.output}: {reasons}, {time.perf_counter() - started:.0f}s")


def write(args, selection, prompt_sha, items):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "task": "D-19",
        "model": args.model,
        "prompt_sha256": prompt_sha,
        "prompt": PROMPT,
        "seed": SEED,
        "selection_sha256": hashlib.sha256(json.dumps(selection["corpusids"]).encode()).hexdigest(),
        "filters": {
            "min_words": MIN_WORDS,
            "max_words": MAX_WORDS,
            "max_copied_run": MAX_COPIED_RUN,
            "max_jaccard": MAX_JACCARD,
        },
        "items": items,
    }
    args.output.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
