"""
KGE Dataset Preparation
========================
Loads the expanded (or sample) KG, cleans triples,
creates entity2id / relation2id mappings, and splits into
train / valid / test sets for Knowledge Graph Embedding models.

Outputs (in data/ folder):
    entity2id.txt
    relation2id.txt
    train.txt
    valid.txt
    test.txt

Usage:
    python src/kge/kge_prep.py
"""

import logging
import random
from collections import Counter
from pathlib import Path
from typing import Optional

from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import OWL, RDF, RDFS, XSD

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

TRAIN_RATIO = 0.80
VALID_RATIO = 0.10
TEST_RATIO  = 0.10
RANDOM_SEED = 42

# Predicates to exclude (ontology/schema triples, not facts)
EXCLUDED_PREDICATES = {
    str(RDF.type),
    str(RDFS.label),
    str(RDFS.comment),
    str(RDFS.subClassOf),
    str(RDFS.domain),
    str(RDFS.range),
    str(OWL.sameAs),
    str(OWL.equivalentProperty),
    str(OWL.inverseOf),
    str(OWL.imports),
    str(OWL.Ontology),
    "http://www.w3.org/2002/07/owl#SymmetricProperty",
    "http://www.w3.org/2000/01/rdf-schema#isDefinedBy",
}

# Minimum entity frequency to be included (avoids extreme sparsity)
MIN_ENTITY_FREQ = 1


# ---------------------------------------------------------------------------
# Sample KG fallback (used if no KG files are available)
# ---------------------------------------------------------------------------

SAMPLE_TRIPLES = [
    # Researchers → worksAt
    ("Yann_LeCun",      "worksAt",          "Meta_AI"),
    ("Geoffrey_Hinton", "worksAt",          "Google_DeepMind"),
    ("Yoshua_Bengio",   "worksAt",          "Mila"),
    ("Andrew_Ng",       "worksAt",          "Stanford_University"),
    ("Fei-Fei_Li",      "worksAt",          "Stanford_University"),
    ("Ian_Goodfellow",  "worksAt",          "Apple"),
    ("Ilya_Sutskever",  "worksAt",          "OpenAI"),
    ("Demis_Hassabis",  "worksAt",          "Google_DeepMind"),
    ("Sam_Altman",      "worksAt",          "OpenAI"),
    ("Pieter_Abbeel",   "worksAt",          "UC_Berkeley"),
    ("Stuart_Russell",  "worksAt",          "UC_Berkeley"),
    ("Peter_Norvig",    "worksAt",          "Google"),
    ("Oriol_Vinyals",   "worksAt",          "Google_DeepMind"),
    ("David_Silver",    "worksAt",          "Google_DeepMind"),
    ("Alex_Krizhevsky", "worksAt",          "Google"),
    ("Sergey_Levine",   "worksAt",          "UC_Berkeley"),
    ("Chelsea_Finn",    "worksAt",          "Stanford_University"),
    ("Kyunghyun_Cho",   "worksAt",          "New_York_University"),
    ("Hugo_Larochelle", "worksAt",          "Google_Brain"),
    ("Judea_Pearl",     "worksAt",          "UCLA"),
    ("Jürgen_Schmidhuber", "worksAt",       "IDSIA"),
    ("Bernhard_Schölkopf", "worksAt",       "MPI_Tübingen"),
    ("Zoubin_Ghahramani",  "worksAt",       "Cambridge_University"),
    ("Michael_I_Jordan",   "worksAt",       "UC_Berkeley"),
    ("Greg_Brockman",      "worksAt",       "OpenAI"),
    # Researchers → wonAward
    ("Yann_LeCun",      "wonAward",         "Turing_Award"),
    ("Geoffrey_Hinton", "wonAward",         "Turing_Award"),
    ("Yoshua_Bengio",   "wonAward",         "Turing_Award"),
    ("Judea_Pearl",     "wonAward",         "Turing_Award"),
    ("Andrew_Ng",       "wonAward",         "MIT_TR35_Award"),
    ("Fei-Fei_Li",      "wonAward",         "ACM_Fellows_Award"),
    ("Ian_Goodfellow",  "wonAward",         "ICML_Best_Paper_2014"),
    ("Demis_Hassabis",  "wonAward",         "CSTB_Award"),
    ("David_Silver",    "wonAward",         "NeurIPS_Best_Paper"),
    # Researchers → studiedAt
    ("Yann_LeCun",      "studiedAt",        "Pierre_Marie_Curie_University"),
    ("Geoffrey_Hinton", "studiedAt",        "Cambridge_University"),
    ("Yoshua_Bengio",   "studiedAt",        "McGill_University"),
    ("Andrew_Ng",       "studiedAt",        "UC_Berkeley"),
    ("Fei-Fei_Li",      "studiedAt",        "Caltech"),
    ("Ian_Goodfellow",  "studiedAt",        "Stanford_University"),
    ("Ilya_Sutskever",  "studiedAt",        "University_of_Toronto"),
    ("Sam_Altman",      "studiedAt",        "Stanford_University"),
    # Researchers → supervisedBy
    ("Ilya_Sutskever",  "supervisedBy",     "Geoffrey_Hinton"),
    ("Chelsea_Finn",    "supervisedBy",     "Sergey_Levine"),
    ("Sergey_Levine",   "supervisedBy",     "Pieter_Abbeel"),
    ("Kyunghyun_Cho",   "supervisedBy",     "Yoshua_Bengio"),
    ("Hugo_Larochelle", "supervisedBy",     "Yoshua_Bengio"),
    # Researchers → collaboratesWith
    ("Yann_LeCun",      "collaboratesWith", "Geoffrey_Hinton"),
    ("Yann_LeCun",      "collaboratesWith", "Yoshua_Bengio"),
    ("Geoffrey_Hinton", "collaboratesWith", "Yoshua_Bengio"),
    ("Ilya_Sutskever",  "collaboratesWith", "Geoffrey_Hinton"),
    ("Ian_Goodfellow",  "collaboratesWith", "Yoshua_Bengio"),
    ("David_Silver",    "collaboratesWith", "Demis_Hassabis"),
    ("Oriol_Vinyals",   "collaboratesWith", "Demis_Hassabis"),
    # Researchers → authorOf
    ("Yann_LeCun",      "authorOf",         "LeNet"),
    ("Yann_LeCun",      "authorOf",         "Convolutional_Neural_Networks"),
    ("Geoffrey_Hinton", "authorOf",         "Backpropagation_Paper"),
    ("Geoffrey_Hinton", "authorOf",         "AlexNet"),
    ("Ian_Goodfellow",  "authorOf",         "GAN_Paper"),
    ("Yoshua_Bengio",   "authorOf",         "Attention_Mechanism_Paper"),
    ("Kyunghyun_Cho",   "authorOf",         "GRU_Paper"),
    ("David_Silver",    "authorOf",         "AlphaGo_Paper"),
    ("David_Silver",    "authorOf",         "AlphaZero_Paper"),
    ("Andrew_Ng",       "authorOf",         "Deep_Learning_Book"),
    # Organizations → locatedIn
    ("Meta_AI",              "locatedIn",   "USA"),
    ("Google_DeepMind",      "locatedIn",   "UK"),
    ("Google_DeepMind",      "locatedIn",   "USA"),
    ("OpenAI",               "locatedIn",   "USA"),
    ("Mila",                 "locatedIn",   "Canada"),
    ("Stanford_University",  "locatedIn",   "USA"),
    ("UC_Berkeley",          "locatedIn",   "USA"),
    ("Cambridge_University", "locatedIn",   "UK"),
    ("IDSIA",                "locatedIn",   "Switzerland"),
    ("MPI_Tübingen",         "locatedIn",   "Germany"),
    ("New_York_University",  "locatedIn",   "USA"),
    ("UCLA",                 "locatedIn",   "USA"),
    ("McGill_University",    "locatedIn",   "Canada"),
    ("University_of_Toronto","locatedIn",   "Canada"),
    ("Google_Brain",         "locatedIn",   "USA"),
    ("Google",               "locatedIn",   "USA"),
    ("Apple",                "locatedIn",   "USA"),
    # Researchers → bornIn
    ("Yann_LeCun",      "bornIn",           "France"),
    ("Geoffrey_Hinton", "bornIn",           "UK"),
    ("Yoshua_Bengio",   "bornIn",           "France"),
    ("Andrew_Ng",       "bornIn",           "UK"),
    ("Jürgen_Schmidhuber", "bornIn",        "Germany"),
    ("Demis_Hassabis",  "bornIn",           "UK"),
    ("Ilya_Sutskever",  "bornIn",           "Russia"),
    ("Oriol_Vinyals",   "bornIn",           "Spain"),
    ("Sam_Altman",      "bornIn",           "USA"),
    ("Judea_Pearl",     "bornIn",           "Israel"),
    # Organizations → memberOf
    ("Google_DeepMind", "memberOf",         "Alphabet"),
    ("Google_Brain",    "memberOf",         "Google"),
    ("Meta_AI",         "memberOf",         "Meta_Platforms"),
    ("Mila",            "memberOf",         "CIFAR_Network"),
]


# ---------------------------------------------------------------------------
# Load KG from file
# ---------------------------------------------------------------------------


def load_kg(path: Path) -> list[tuple[str, str, str]]:
    """
    Load an RDF graph and return fact triples as (subject, predicate, object).
    Filters out ontology/schema triples.
    """
    logger.info(f"Loading KG from: {path}")
    g = Graph()
    fmt = "ntriples" if str(path).endswith(".nt") else "turtle"
    g.parse(str(path), format=fmt)
    logger.info(f"Loaded {len(g)} raw triples")

    triples = []
    for s, p, o in g:
        # Skip blank nodes
        if isinstance(s, BNode) or isinstance(o, BNode):
            continue
        # Skip ontology predicates
        if str(p) in EXCLUDED_PREDICATES:
            continue
        # Skip literal objects (keep only entity-entity triples for KGE)
        if isinstance(o, Literal):
            # Allow if it's a meaningful data property like hIndex
            continue
        # Use local names (fragment or last path segment)
        s_id = _get_local_name(str(s))
        p_id = _get_local_name(str(p))
        o_id = _get_local_name(str(o))
        if s_id and p_id and o_id:
            triples.append((s_id, p_id, o_id))

    logger.info(f"Filtered to {len(triples)} entity-entity triples")
    return triples


def _get_local_name(uri: str) -> str:
    """Extract local name from URI (fragment or last path segment)."""
    if "#" in uri:
        return uri.split("#")[-1]
    if "/" in uri:
        return uri.split("/")[-1]
    return uri


# ---------------------------------------------------------------------------
# Clean and validate triples
# ---------------------------------------------------------------------------


def clean_triples(triples: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Remove malformed, duplicate, or self-referential triples."""
    seen = set()
    cleaned = []
    for s, p, o in triples:
        # Skip empty strings
        if not s or not p or not o:
            continue
        # Skip self-loops (trivial)
        if s == o:
            continue
        # Skip very long URIs (likely malformed)
        if len(s) > 200 or len(p) > 200 or len(o) > 200:
            continue
        # Skip blank node remnants
        if s.startswith("N") and len(s) == 32:  # typical blank node ID
            continue
        key = (s, p, o)
        if key not in seen:
            seen.add(key)
            cleaned.append((s, p, o))
    logger.info(f"After cleaning: {len(cleaned)} triples")
    return cleaned


# ---------------------------------------------------------------------------
# Create entity/relation mappings
# ---------------------------------------------------------------------------


def create_mappings(
    triples: list[tuple[str, str, str]],
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Create entity2id and relation2id dictionaries.
    Returns (entity2id, relation2id).
    """
    entity_counter: Counter = Counter()
    relation_counter: Counter = Counter()

    for s, p, o in triples:
        entity_counter[s] += 1
        entity_counter[o] += 1
        relation_counter[p] += 1

    # Filter rare entities
    entities = sorted(
        [e for e, c in entity_counter.items() if c >= MIN_ENTITY_FREQ]
    )
    relations = sorted(relation_counter.keys())

    entity2id = {e: i for i, e in enumerate(entities)}
    relation2id = {r: i for i, r in enumerate(relations)}

    logger.info(f"Entities: {len(entity2id)}, Relations: {len(relation2id)}")
    return entity2id, relation2id


# ---------------------------------------------------------------------------
# Train/valid/test split
# ---------------------------------------------------------------------------


def split_triples(
    triples: list[tuple[str, str, str]],
    entity2id: dict[str, int],
    relation2id: dict[str, int],
    seed: int = RANDOM_SEED,
) -> tuple[list, list, list]:
    """
    Split triples into train/valid/test.
    Ensures no entity appears ONLY in valid/test (cold-start prevention).
    """
    random.seed(seed)

    # Only keep triples where both entities and relation are in our vocab
    valid_triples = [
        (s, p, o) for s, p, o in triples
        if s in entity2id and p in relation2id and o in entity2id
    ]

    if not valid_triples:
        logger.warning("No valid triples after vocabulary filtering!")
        return [], [], []

    random.shuffle(valid_triples)
    n = len(valid_triples)
    n_train = max(1, int(n * TRAIN_RATIO))
    n_valid = max(1, int(n * VALID_RATIO))

    # Entities seen in training set
    train_set = valid_triples[:n_train]
    train_entities = set()
    for s, _, o in train_set:
        train_entities.add(s)
        train_entities.add(o)

    # For valid/test, only include triples where both entities are in train
    remaining = valid_triples[n_train:]
    safe_remaining = [(s, p, o) for s, p, o in remaining if s in train_entities and o in train_entities]

    n_valid_actual = min(n_valid, len(safe_remaining) // 2)
    valid_set = safe_remaining[:n_valid_actual]
    test_set = safe_remaining[n_valid_actual:]

    logger.info(f"Split: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)}")
    return train_set, valid_set, test_set


# ---------------------------------------------------------------------------
# Save files
# ---------------------------------------------------------------------------


def save_mapping(mapping: dict[str, int], path: Path):
    """Save id mapping file in PyKEEN/OpenKE format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{len(mapping)}\n")
        for name, idx in sorted(mapping.items(), key=lambda x: x[1]):
            f.write(f"{name}\t{idx}\n")
    logger.info(f"Saved: {path} ({len(mapping)} entries)")


def save_triples(
    triples: list[tuple[str, str, str]],
    path: Path,
    entity2id: dict[str, int],
    relation2id: dict[str, int],
):
    """Save triple file in tab-separated (head_id, relation_id, tail_id) format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s, p, o in triples:
            f.write(f"{entity2id[s]}\t{relation2id[p]}\t{entity2id[o]}\n")
    logger.info(f"Saved: {path} ({len(triples)} triples)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Try to load an existing KG
    kg_candidates = [
        Path("kg_artifacts/expanded_kg.nt"),
        Path("kg_artifacts/initial_kg.nt"),
        Path("kg_artifacts/sample_kg.ttl"),
    ]

    triples: list[tuple[str, str, str]] = []
    for kg_path in kg_candidates:
        if kg_path.exists():
            logger.info(f"Loading KG: {kg_path}")
            try:
                triples = load_kg(kg_path)
                if triples:
                    break
            except Exception as e:
                logger.warning(f"Failed to load {kg_path}: {e}")

    # Fallback: use sample triples
    if not triples:
        logger.info("No KG found or empty. Using built-in sample triples.")
        triples = SAMPLE_TRIPLES

    # Clean triples
    triples = clean_triples(triples)

    if not triples:
        logger.error("No triples available after cleaning. Exiting.")
        return

    # Supplement with sample triples if KG is very small
    if len(triples) < 50:
        logger.info(f"KG has only {len(triples)} triples. Supplementing with sample data.")
        existing_set = set(triples)
        for t in SAMPLE_TRIPLES:
            if t not in existing_set:
                triples.append(t)
        triples = clean_triples(triples)

    # Create entity/relation mappings
    entity2id, relation2id = create_mappings(triples)

    # Split
    train_set, valid_set, test_set = split_triples(triples, entity2id, relation2id)

    # Save everything
    save_mapping(entity2id, data_dir / "entity2id.txt")
    save_mapping(relation2id, data_dir / "relation2id.txt")
    save_triples(train_set, data_dir / "train.txt", entity2id, relation2id)
    save_triples(valid_set, data_dir / "valid.txt", entity2id, relation2id)
    save_triples(test_set, data_dir / "test.txt", entity2id, relation2id)

    # Also save human-readable versions
    with open(data_dir / "train_readable.txt", "w", encoding="utf-8") as f:
        for s, p, o in train_set:
            f.write(f"{s}\t{p}\t{o}\n")
    with open(data_dir / "valid_readable.txt", "w", encoding="utf-8") as f:
        for s, p, o in valid_set:
            f.write(f"{s}\t{p}\t{o}\n")
    with open(data_dir / "test_readable.txt", "w", encoding="utf-8") as f:
        for s, p, o in test_set:
            f.write(f"{s}\t{p}\t{o}\n")

    # Print statistics
    print("\n=== KGE Dataset Statistics ===")
    print(f"Total triples  : {len(triples)}")
    print(f"Entities       : {len(entity2id)}")
    print(f"Relations      : {len(relation2id)}")
    print(f"Train          : {len(train_set)} ({100*len(train_set)/len(triples):.1f}%)")
    print(f"Valid          : {len(valid_set)} ({100*len(valid_set)/len(triples):.1f}%)")
    print(f"Test           : {len(test_set)} ({100*len(test_set)/len(triples):.1f}%)")
    print(f"\nRelation distribution:")
    from collections import Counter
    rel_counts = Counter(p for _, p, _ in triples)
    for rel, count in rel_counts.most_common():
        print(f"  {rel}: {count}")


if __name__ == "__main__":
    main()
