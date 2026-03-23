"""
Named Entity Recognition & Relation Extraction
===============================================
Loads crawled JSONL pages, applies spaCy NER, extracts
PERSON/ORG/GPE/DATE/WORK_OF_ART entities and SVO triples.

Outputs:
    data/entities.csv  — entity, type, source_url, count
    data/triples.csv   — subject, predicate, object, source_url

Usage:
    python src/ie/ner_extractor.py
    python src/ie/ner_extractor.py --input data/crawled_pages.jsonl --output-dir data/
"""

import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import pandas as pd
import spacy
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT = Path("data/crawled_pages.jsonl")
DEFAULT_OUTPUT_DIR = Path("data")
ENTITY_TYPES = {"PERSON", "ORG", "GPE", "DATE", "WORK_OF_ART"}

# Maximum document length for spaCy (characters)
MAX_DOC_LENGTH = 100_000

# SVO dependency relation labels
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj"}
OBJECT_DEPS = {"dobj", "attr", "pobj", "iobj"}

# ---------------------------------------------------------------------------
# Load spaCy model (try transformer, fall back to sm)
# ---------------------------------------------------------------------------


def load_spacy_model() -> spacy.language.Language:
    """Load the best available spaCy model for English NER."""
    models = ["en_core_web_trf", "en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
    for model_name in models:
        try:
            nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
            return nlp
        except OSError:
            logger.debug(f"Model {model_name} not available, trying next...")
    raise RuntimeError(
        "No spaCy English model found. Run: python -m spacy download en_core_web_sm"
    )


# ---------------------------------------------------------------------------
# JSONL reader
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> Iterator[dict]:
    """Lazily read a JSONL file line by line."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_no}: JSON parse error: {e}")


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


def normalize_entity(text: str) -> str:
    """Clean and normalize entity text."""
    # Remove leading/trailing whitespace and punctuation
    text = text.strip().strip(".,;:()[]{}\"'")
    # Collapse internal whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def extract_entities(doc: spacy.tokens.Doc, url: str) -> list[dict]:
    """
    Extract named entities from a spaCy Doc.
    Returns list of {entity, type, source_url} dicts.
    """
    entities = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ not in ENTITY_TYPES:
            continue
        norm = normalize_entity(ent.text)
        if not norm or len(norm) < 2:
            continue
        key = (norm.lower(), ent.label_)
        if key in seen:
            continue
        seen.add(key)
        entities.append({
            "entity": norm,
            "type": ent.label_,
            "source_url": url,
        })
    return entities


# ---------------------------------------------------------------------------
# SVO Triple extraction via dependency parsing
# ---------------------------------------------------------------------------


def extract_svo_triples(doc: spacy.tokens.Doc, url: str) -> list[dict]:
    """
    Extract Subject-Verb-Object triples using dependency parsing.
    Focuses on relations between named entities.
    """
    triples = []

    # Build a quick entity lookup: token index → entity label
    token_to_ent: dict[int, str] = {}
    for ent in doc.ents:
        if ent.label_ in ENTITY_TYPES:
            for token in ent:
                token_to_ent[token.i] = ent.label_

    for token in doc:
        # Look for verbs as the predicate
        if token.pos_ not in ("VERB", "AUX"):
            continue

        verb = token.lemma_.lower()
        if verb in ("be", "have", "do", "say", "make", "go", "get"):
            # Skip auxiliary / light verbs unless they carry meaning
            if verb not in ("work", "win", "receive", "publish", "found", "join", "lead"):
                pass  # We still process them but filter below

        subjects = []
        objects = []

        for child in token.children:
            if child.dep_ in SUBJECT_DEPS:
                # Get the full noun phrase for subject
                span = _get_entity_span(child, doc, token_to_ent)
                if span:
                    subjects.append(span)
            elif child.dep_ in OBJECT_DEPS:
                span = _get_entity_span(child, doc, token_to_ent)
                if span:
                    objects.append(span)

        # Emit triples where both subject and object are named entities
        for subj_text, subj_type in subjects:
            for obj_text, obj_type in objects:
                if subj_text.lower() == obj_text.lower():
                    continue
                # Prefer triples where at least one end is a PERSON
                if "PERSON" not in (subj_type, obj_type):
                    continue
                triples.append({
                    "subject": normalize_entity(subj_text),
                    "predicate": verb,
                    "object": normalize_entity(obj_text),
                    "source_url": url,
                })

    return triples


def _get_entity_span(token: spacy.tokens.Token, doc: spacy.tokens.Doc, token_to_ent: dict) -> tuple | None:
    """
    Get the entity span for a token. Returns (text, entity_type) or None.
    Walks up to the entity root to get the full entity text.
    """
    # Check if the token itself is part of an entity
    if token.i in token_to_ent:
        # Find the full entity span this token belongs to
        for ent in doc.ents:
            if any(t.i == token.i for t in ent):
                return (ent.text, ent.label_)

    # Check if any child of the token is an entity (for compound nouns etc.)
    for child in token.children:
        if child.dep_ in ("compound", "poss", "amod") and child.i in token_to_ent:
            for ent in doc.ents:
                if any(t.i == child.i for t in ent):
                    return (ent.text, ent.label_)

    return None


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------


def run_extraction(
    input_path: Path,
    output_dir: Path,
    batch_size: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the full NER + triple extraction pipeline.
    Returns (entities_df, triples_df).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    nlp = load_spacy_model()

    # Accumulate results
    all_entities: list[dict] = []       # {entity, type, source_url}
    all_triples: list[dict] = []        # {subject, predicate, object, source_url}

    pages = list(read_jsonl(input_path))
    logger.info(f"Processing {len(pages)} pages from {input_path}")

    # Process in batches for memory efficiency
    texts_urls = [(p.get("text", "")[:MAX_DOC_LENGTH], p.get("url", "")) for p in pages]

    for start in tqdm(range(0, len(texts_urls), batch_size), desc="NER + SVO"):
        batch = texts_urls[start : start + batch_size]
        texts = [t for t, _ in batch]
        urls = [u for _, u in batch]

        docs = list(nlp.pipe(texts, batch_size=batch_size))
        for doc, url in zip(docs, urls):
            all_entities.extend(extract_entities(doc, url))
            all_triples.extend(extract_svo_triples(doc, url))

    # -----------------------------------------------------------------------
    # Aggregate entities: count occurrences across documents
    # -----------------------------------------------------------------------
    entity_counter: dict[tuple, dict] = defaultdict(lambda: {"count": 0, "urls": []})
    for e in all_entities:
        key = (e["entity"].lower(), e["type"])
        entity_counter[key]["count"] += 1
        entity_counter[key]["urls"].append(e["source_url"])
        entity_counter[key]["entity"] = e["entity"]
        entity_counter[key]["type"] = e["type"]

    entities_rows = []
    for (_, etype), data in entity_counter.items():
        # Canonical form: most common capitalization (just use first seen)
        entities_rows.append({
            "entity": data["entity"],
            "type": data["type"],
            "source_url": data["urls"][0],
            "count": data["count"],
        })

    entities_df = pd.DataFrame(entities_rows)
    if not entities_df.empty:
        entities_df = entities_df.sort_values(["count", "entity"], ascending=[False, True])
        entities_df = entities_df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # De-duplicate triples
    # -----------------------------------------------------------------------
    triples_df = pd.DataFrame(all_triples)
    if not triples_df.empty:
        triples_df = triples_df.drop_duplicates(
            subset=["subject", "predicate", "object"]
        ).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Save to CSV
    # -----------------------------------------------------------------------
    entities_out = output_dir / "entities.csv"
    triples_out = output_dir / "triples.csv"

    entities_df.to_csv(entities_out, index=False, encoding="utf-8")
    triples_df.to_csv(triples_out, index=False, encoding="utf-8")

    logger.info(f"Entities saved: {entities_out} ({len(entities_df)} rows)")
    logger.info(f"Triples saved:  {triples_out} ({len(triples_df)} rows)")

    # -----------------------------------------------------------------------
    # Print summary statistics
    # -----------------------------------------------------------------------
    print("\n=== Extraction Summary ===")
    print(f"Pages processed  : {len(pages)}")
    print(f"Total entities   : {len(entities_df)}")
    print(f"Total triples    : {len(triples_df)}")

    if not entities_df.empty:
        print("\nTop entities by type:")
        for etype in ENTITY_TYPES:
            subset = entities_df[entities_df["type"] == etype].head(5)
            if not subset.empty:
                names = ", ".join(subset["entity"].tolist())
                print(f"  {etype}: {names}")

    if not triples_df.empty:
        print(f"\nSample triples:")
        for _, row in triples_df.head(5).iterrows():
            print(f"  ({row['subject']}) --[{row['predicate']}]--> ({row['object']})")

    return entities_df, triples_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="NER + Relation Extraction for AI Researchers KG")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSONL file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="spaCy batch size (default: 10)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(
            f"Input file not found: {args.input}\n"
            "Run the crawler first: python src/crawl/crawler.py"
        )
        return

    run_extraction(
        input_path=args.input,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
