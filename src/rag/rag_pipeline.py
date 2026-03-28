"""
RAG Pipeline: Natural Language → SPARQL → Answer
==================================================
Loads an RDF Knowledge Graph, builds a schema summary,
uses a local Ollama LLM to generate SPARQL from natural language,
executes the SPARQL, and generates a final answer.

Features:
    - Schema-aware SPARQL generation
    - Self-repair loop (up to 3 retries on SPARQL errors)
    - Baseline comparison mode (LLM without KG)
    - Interactive CLI mode
    - Structured result formatting

Usage:
    python src/rag/rag_pipeline.py --interactive
    python src/rag/rag_pipeline.py --question "Who works at DeepMind?"
    python src/rag/rag_pipeline.py --question "..." --mode baseline
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import requests
from rdflib import Graph, URIRef
from rdflib.namespace import OWL, RDF, RDFS
# rdflib 7.x: ResultException moved — use base Exception as fallback
try:
    from rdflib.plugins.sparql.exceptions import ResultException
except ImportError:
    ResultException = Exception

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

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:1b"   # Change to "mistral" or "llama3.1:8b" for better quality
OLLAMA_TIMEOUT = 300           # seconds

MAX_REPAIR_ATTEMPTS = 3
MAX_RESULTS_DISPLAY = 20

# KG file search order
KG_CANDIDATES = [
    Path("kg_artifacts/expanded_kg.nt"),
    Path("kg_artifacts/initial_kg.nt"),
    Path("kg_artifacts/sample_kg.ttl"),
]

# Namespace prefix used in the KG
EX_PREFIX = "https://aikg.example.org/"
EX_NS_SHORT = "ex"


# ---------------------------------------------------------------------------
# KG Loading
# ---------------------------------------------------------------------------


def load_graph(kg_path=None) -> Graph:
    """
    Load the RDF Knowledge Graph.
    Tries KG_CANDIDATES if no path provided.
    Accepts a string or Path object for kg_path.
    """
    if kg_path is not None and not isinstance(kg_path, Path):
        kg_path = Path(kg_path)
    candidates = [kg_path] if kg_path else KG_CANDIDATES
    g = Graph()
    g.bind("ex", EX_PREFIX)
    g.bind("owl", "http://www.w3.org/2002/07/owl#")
    g.bind("rdfs", "http://www.w3.org/2000/01/rdf-schema#")
    g.bind("wd", "http://www.wikidata.org/entity/")

    for path in candidates:
        if path and path.exists():
            fmt = "ntriples" if str(path).endswith(".nt") else "turtle"
            try:
                g.parse(str(path), format=fmt)
                logger.info(f"Loaded KG: {path} ({len(g)} triples)")
                return g
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    logger.warning("No KG file found. Using empty graph.")
    return g


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------


def build_schema_summary(g: Graph, max_predicates: int = 80, max_classes: int = 40, max_sample_triples: int = 20) -> str:
    """
    Build a schema summary string for use as LLM context.
    Includes: prefixes, classes, predicates, sample triples.
    """
    lines = []

    # -----------------------------------------------------------------------
    # Prefixes
    # -----------------------------------------------------------------------
    lines.append("=== KNOWLEDGE GRAPH SCHEMA ===\n")
    lines.append("## Namespace Prefixes")
    lines.append(f"PREFIX ex: <{EX_PREFIX}>")
    lines.append("PREFIX owl: <http://www.w3.org/2002/07/owl#>")
    lines.append("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>")
    lines.append("PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>")
    lines.append("PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>")
    lines.append("PREFIX wd: <http://www.wikidata.org/entity/>")
    lines.append("")

    # -----------------------------------------------------------------------
    # Classes (from rdf:type declarations)
    # -----------------------------------------------------------------------
    classes = set()
    for s, _, o in g.triples((None, RDF.type, None)):
        if str(o).startswith(EX_PREFIX):
            cls_name = str(o).split("/")[-1].split("#")[-1]
            classes.add(cls_name)

    # Also from rdfs:subClassOf
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        if str(s).startswith(EX_PREFIX):
            classes.add(str(s).split("/")[-1])
        if str(o).startswith(EX_PREFIX):
            classes.add(str(o).split("/")[-1])

    if classes:
        lines.append("## Classes (ex:ClassName)")
        for cls in sorted(classes)[:max_classes]:
            lines.append(f"  ex:{cls}")
        lines.append("")

    # -----------------------------------------------------------------------
    # Predicates (from actual triples)
    # -----------------------------------------------------------------------
    predicates = {}
    for s, p, o in g:
        p_str = str(p)
        if p_str not in predicates:
            predicates[p_str] = 0
        predicates[p_str] += 1

    # Focus on our EX predicates
    ex_preds = {p: c for p, c in predicates.items() if p.startswith(EX_PREFIX)}
    other_preds = {p: c for p, c in predicates.items() if not p.startswith(EX_PREFIX)}

    lines.append("## Object/Data Properties (predicate: usage count)")

    # Custom EX predicates first
    for pred, count in sorted(ex_preds.items(), key=lambda x: -x[1])[:max_predicates // 2]:
        short = pred.split("/")[-1].split("#")[-1]
        lines.append(f"  ex:{short}  (used {count} times)")

    # Standard predicates
    for pred, count in sorted(other_preds.items(), key=lambda x: -x[1])[:max_predicates // 2]:
        if "rdf-syntax" in pred or "rdf/type" in pred:
            lines.append(f"  rdf:type  (used {count} times)")
        elif "2000/01/rdf-schema#label" in pred:
            lines.append(f"  rdfs:label  (used {count} times)")
        elif "owl#sameAs" in pred:
            lines.append(f"  owl:sameAs  (used {count} times)")
    lines.append("")

    # -----------------------------------------------------------------------
    # Sample triples showing how data looks
    # -----------------------------------------------------------------------
    lines.append("## Sample Triples (subject, predicate, object)")

    ex_triples = [
        (s, p, o) for s, p, o in g
        if str(s).startswith(EX_PREFIX) and str(p).startswith(EX_PREFIX)
    ]

    shown = 0
    for s, p, o in ex_triples[:max_sample_triples * 3]:
        if shown >= max_sample_triples:
            break
        s_short = "ex:" + str(s).split("/")[-1]
        p_short = "ex:" + str(p).split("/")[-1]
        if str(o).startswith(EX_PREFIX):
            o_short = "ex:" + str(o).split("/")[-1]
        elif str(o).startswith("http://www.wikidata.org"):
            o_short = "wd:" + str(o).split("/")[-1]
        else:
            o_short = f'"{str(o)}"'
        lines.append(f"  ({s_short}, {p_short}, {o_short})")
        shown += 1
    lines.append("")

    # -----------------------------------------------------------------------
    # Important named entities: label → URI mapping (crucial for the LLM)
    # -----------------------------------------------------------------------
    lines.append("## IMPORTANT: Named Entity URI Lookup")
    lines.append("  Use these EXACT URIs when referring to these entities in SPARQL:")
    lines.append("")

    # Collect entities by type with their labels
    entity_types = {
        "Award":       "ex:Award",
        "Researcher":  "ex:Researcher",
        "University":  "ex:University",
        "Company":     "ex:Company",
        "ResearchLab": "ex:ResearchLab",
        "Country":     "ex:Country",
    }

    for type_local, type_label in entity_types.items():
        type_uri = URIRef(EX_PREFIX + type_local)
        entities = []
        for s in g.subjects(RDF.type, type_uri):
            s_str = str(s)
            if not s_str.startswith(EX_PREFIX):
                continue
            # Get a human-readable label
            label = None
            for lbl in g.objects(s, RDFS.label):
                label = str(lbl)
                break
            if not label:
                for lbl in g.objects(s, URIRef(EX_PREFIX + "name")):
                    label = str(lbl)
                    break
            if label:
                short_uri = "ex:" + s_str.split("/")[-1]
                entities.append((label, short_uri))

        if entities:
            lines.append(f"  ### {type_local}s")
            # Sort and show up to 15 per type
            for label, uri in sorted(entities, key=lambda x: x[0])[:15]:
                lines.append(f'    "{label}"  ->  {uri}')
            lines.append("")

    lines.append("  RULE: Never invent URIs like ex:Turing_Award or ex:Geoffrey_Hinton.")
    lines.append("        Always look up the entity above and use the exact URI shown.")
    lines.append("        For labels not listed, use rdfs:label filter:")
    lines.append("        FILTER(CONTAINS(LCASE(STR(?label)), \"keyword\"))")
    lines.append("")

    # -----------------------------------------------------------------------
    # Summary counts
    # -----------------------------------------------------------------------
    n_subjects = len(set(g.subjects()))
    lines.append("## KG Statistics")
    lines.append(f"  Total triples : {len(g)}")
    lines.append(f"  Unique subjects: {n_subjects}")
    lines.append(f"  Unique predicates: {len(predicates)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ollama API
# ---------------------------------------------------------------------------


def check_ollama_available() -> bool:
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def get_available_models() -> list[str]:
    """Get list of available Ollama models."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def call_ollama(prompt: str, model: str = OLLAMA_MODEL, system: str = "") -> str:
    """
    Call the Ollama REST API and return the generated text.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,    # Low temperature for deterministic SPARQL
            "top_p": 0.9,
            "num_predict": 500,
        },
    }
    if system:
        payload["system"] = system

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=OLLAMA_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Is Ollama running? Start it with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Ollama request timed out after {OLLAMA_TIMEOUT}s")
    except Exception as e:
        raise RuntimeError(f"Ollama API error: {e}")


# ---------------------------------------------------------------------------
# SPARQL generation prompt
# ---------------------------------------------------------------------------

SPARQL_SYSTEM_PROMPT = """You are an expert SPARQL query generator for a Knowledge Graph about AI researchers.
Your task is to convert natural language questions into valid SPARQL 1.1 queries.

IMPORTANT RULES:
1. Always use the PREFIX declarations provided in the schema
2. Use ONLY predicates and classes that exist in the schema
3. Return ONLY the SPARQL query, no explanation, no markdown, no backticks
4. Use OPTIONAL for properties that might not exist
5. Always add LIMIT 20 to avoid very large results
6. For name lookups, use FILTER(CONTAINS(LCASE(?label), LCASE("name"))) or rdfs:label
7. The main namespace is ex: = <https://aikg.example.org/>
8. Entity URIs look like: ex:Yann_LeCun, ex:Stanford_University, ex:Turing_Award
9. Use rdfs:label or ex:name to get human-readable names

Example valid queries:

Q: Who works at DeepMind?
A:
PREFIX ex: <https://aikg.example.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?researcher ?name WHERE {
  ?researcher a ex:Researcher ;
              ex:worksAt ex:DeepMind ;
              ex:name ?name .
}
LIMIT 20

Q: Which researchers won the Turing Award?
A:
PREFIX ex: <https://aikg.example.org/>
SELECT ?researcher ?name WHERE {
  ?researcher a ex:Researcher ;
              ex:wonAward ex:Turing_Award ;
              ex:name ?name .
}
LIMIT 20
"""


PREFIX_BLOCK = """PREFIX ex: <https://aikg.example.org/>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>"""


# ---------------------------------------------------------------------------
# Problem 4 helpers
# ---------------------------------------------------------------------------

def find_entity_uri(g: Graph, name: str) -> Optional[str]:
    """
    Problem 4 helper: find the canonical URI for a named entity.
    1. Try ex:Name_With_Underscores directly (must have at least one triple)
    2. Search by rdfs:label or ex:name (case-insensitive)
    3. Return the URI that has the most triples (most connected = canonical)
    Returns a SPARQL-ready string like "ex:Q185667" or None.
    """
    ex_ns = EX_PREFIX
    candidates: list[tuple[str, int]] = []

    # Step 1: try the direct underscore-joined form — prefer it over QIDs
    direct_local = name.strip().title().replace(" ", "_")
    direct_uri = URIRef(ex_ns + direct_local)
    direct_count = sum(1 for _ in g.triples((direct_uri, None, None))) + \
                   sum(1 for _ in g.triples((None, None, direct_uri)))
    if direct_count > 0:
        # Prefer named URI immediately — avoid QID noise
        return f"ex:{direct_local}"

    # Step 2: label / name search
    name_lower = name.lower()
    for s, _, o in g.triples((None, RDFS.label, None)):
        s_str = str(s)
        if s_str.startswith(ex_ns) and str(o).lower() == name_lower:
            cnt = sum(1 for _ in g.triples((s, None, None))) + \
                  sum(1 for _ in g.triples((None, None, s)))
            candidates.append((s_str, cnt))
    for s, _, o in g.triples((None, URIRef(ex_ns + "name"), None)):
        s_str = str(s)
        if s_str.startswith(ex_ns) and str(o).lower() == name_lower:
            cnt = sum(1 for _ in g.triples((s, None, None))) + \
                  sum(1 for _ in g.triples((None, None, s)))
            candidates.append((s_str, cnt))

    if not candidates:
        return None

    # Return the most-connected URI
    best_uri = max(candidates, key=lambda t: t[1])[0]
    local = best_uri.split("/")[-1]
    return f"ex:{local}"


def try_template_sparql(question: str, g: Optional[Graph] = None) -> Optional[str]:
    """
    Problem 4 fix: try to match the question against known patterns and
    return a validated SPARQL template.  Returns None if no template matches.

    When graph `g` is provided, entity names are resolved to real URIs via
    find_entity_uri(); otherwise a label FILTER is used as fallback.
    """
    q = question.lower().strip().rstrip("?")

    # ------------------------------------------------------------------
    # Helper: resolve an entity name to a URI token or FILTER clause
    # ------------------------------------------------------------------
    def _uri_or_filter(name: str, var: str) -> tuple[str, str]:
        """
        Returns (uri_token, extra_triples).
        If a real URI is found, uri_token = "ex:Q12345" and extra_triples = "".
        Otherwise uri_token = var and extra_triples has a FILTER line.
        """
        if g is not None:
            uri = find_entity_uri(g, name)
            if uri:
                return uri, ""
        # Fallback to label filter
        name_lower = name.lower()
        extra = (
            f'  ?{var}_lbl_res rdfs:label ?{var}_lbl .\n'
            f'  FILTER(CONTAINS(LCASE(STR(?{var}_lbl)), "{name_lower}"))\n'
        )
        return f"?{var}", extra

    # ------------------------------------------------------------------
    # Pattern: "who won the X award" / "which researchers won X"
    # Also catches bare "turing award", "nobel prize" queries
    # ------------------------------------------------------------------
    # Strategy: extract the award keyword robustly
    award_kw = None

    # Try: "won/received the <X> award" or "won/received <X>"
    m = re.search(
        r'(?:won|received?|got)\s+(?:the\s+)?(.+?)(?:\s+award\b.*)?$',
        q,
    )
    if m and re.search(r'(?:who|which\s+researcher)', q):
        raw = m.group(1).strip()
        # Strip trailing " award" if present
        raw = re.sub(r'\s+award$', '', raw).strip()
        if raw:
            award_kw = raw

    # Bare award-name pattern (e.g. "turing award winners", "turing award")
    if not award_kw:
        m2 = re.search(
            r'(turing award|turing|nobel\s+prize|nobel|fields medal|acm\s+prize|acm turing)',
            q,
        )
        if m2:
            award_kw = m2.group(1).strip()

    if award_kw:
        # Try to resolve award URI
        award_uri = None
        if g is not None:
            for candidate in [
                award_kw.title(),
                award_kw.replace(" ", "_").title(),
                award_kw.replace(" award", "").title(),
            ]:
                award_uri = find_entity_uri(g, candidate)
                if award_uri:
                    break
        if award_uri:
            return f"""{PREFIX_BLOCK}
SELECT DISTINCT ?researcher ?name WHERE {{
  ?researcher a ex:Researcher ;
              ex:wonAward {award_uri} .
  OPTIONAL {{ ?researcher rdfs:label ?name }}
}} LIMIT 20"""
        else:
            return f"""{PREFIX_BLOCK}
SELECT DISTINCT ?researcher ?name WHERE {{
  ?researcher a ex:Researcher ;
              ex:wonAward ?award .
  ?award rdfs:label ?awardLabel .
  OPTIONAL {{ ?researcher rdfs:label ?name }}
  FILTER(CONTAINS(LCASE(STR(?awardLabel)), "{award_kw}"))
}} LIMIT 20"""

    # ------------------------------------------------------------------
    # Pattern: "where does X work" / "where is X based"
    # ------------------------------------------------------------------
    m = re.search(r'where (?:does|is)\s+(.+?)\s+(?:work|employed|based)', q)
    if m:
        name_kw = m.group(1).strip()
        person_uri, extra = _uri_or_filter(name_kw, "researcher")
        if extra:
            return f"""{PREFIX_BLOCK}
SELECT ?org ?orgName WHERE {{
  ?researcher rdfs:label ?rlabel .
  FILTER(CONTAINS(LCASE(STR(?rlabel)), "{name_kw.lower()}"))
  ?researcher ex:worksAt ?org .
  OPTIONAL {{ ?org rdfs:label ?orgName }}
}} LIMIT 10"""
        else:
            return f"""{PREFIX_BLOCK}
SELECT ?org ?orgName WHERE {{
  {person_uri} ex:worksAt ?org .
  OPTIONAL {{ ?org rdfs:label ?orgName }}
}} LIMIT 10"""

    # ------------------------------------------------------------------
    # Pattern: "awards from/at organization" (e.g. "awards did researchers from Stanford receive")
    # Must be checked BEFORE the individual-person award pattern to avoid mis-matching.
    # ------------------------------------------------------------------
    m_org_award = re.search(
        r'awards?.{0,20}(?:researchers?\s+)?(?:from|at|in)\s+(.+?)\s+(?:receive|win|get|earn)',
        q,
    )
    if not m_org_award:
        m_org_award = re.search(r'awards?\s+(?:from|at|in)\s+(.+)$', q)
    if m_org_award:
        org_kw = m_org_award.group(1).strip()
        org_uri, _ = _uri_or_filter(org_kw, "org")
        if org_uri.startswith("ex:"):
            return f"""{PREFIX_BLOCK}
SELECT DISTINCT ?researcher ?name ?award ?awardName WHERE {{
  ?researcher a ex:Researcher ;
              ex:worksAt {org_uri} ;
              ex:wonAward ?award .
  OPTIONAL {{ ?researcher rdfs:label ?name }}
  OPTIONAL {{ ?award rdfs:label ?awardName }}
}} LIMIT 20"""
        else:
            return f"""{PREFIX_BLOCK}
SELECT DISTINCT ?researcher ?name ?award ?awardName WHERE {{
  ?researcher a ex:Researcher ;
              ex:wonAward ?award .
  ?org rdfs:label ?orgLabel .
  FILTER(CONTAINS(LCASE(STR(?orgLabel)), "{org_kw.lower()}"))
  ?researcher ex:worksAt ?org .
  OPTIONAL {{ ?researcher rdfs:label ?name }}
  OPTIONAL {{ ?award rdfs:label ?awardName }}
}} LIMIT 20"""

    # ------------------------------------------------------------------
    # Pattern: "what awards did X win/receive"
    # ------------------------------------------------------------------
    # Require 'did' explicitly to prevent lazy (.+?) from under-matching
    m = re.search(r'(?:what|which)\s+awards?\s+did\s+(.+?)\s+(?:win|receive|get|earn)', q)
    if not m:
        # Also catch: "awards of X" / "awards received by X"
        m = re.search(r'awards?\s+(?:received?\s+by|of|won\s+by)\s+(.+)$', q)
    if m:
        name_kw = m.group(1).strip()
        person_uri, _ = _uri_or_filter(name_kw, "researcher")
        # Always use label-filter UNION to cover cases where the KG has split entities
        # (e.g. ex:Yoshua_Bengio has no wonAward but ex:Q3572699 with label "Yoshua Bengio" does).
        # The label filter catches all duplicates; adding the direct URI is harmless.
        name_lower = name_kw.lower()
        if person_uri.startswith("ex:"):
            return f"""{PREFIX_BLOCK}
SELECT DISTINCT ?award ?awardName WHERE {{
  {{
    {person_uri} ex:wonAward ?award .
  }} UNION {{
    ?researcher rdfs:label ?rlabel .
    FILTER(CONTAINS(LCASE(STR(?rlabel)), "{name_lower}"))
    ?researcher ex:wonAward ?award .
  }}
  OPTIONAL {{ ?award rdfs:label ?awardName }}
}} LIMIT 30"""
        else:
            return f"""{PREFIX_BLOCK}
SELECT DISTINCT ?award ?awardName WHERE {{
  ?researcher rdfs:label ?rlabel .
  FILTER(CONTAINS(LCASE(STR(?rlabel)), "{name_lower}"))
  ?researcher ex:wonAward ?award .
  OPTIONAL {{ ?award rdfs:label ?awardName }}
}} LIMIT 30"""

    # ------------------------------------------------------------------
    # Pattern: "who works at X" / "researchers at X"
    # ------------------------------------------------------------------
    m = re.search(r'(?:who\s+)?works?\s+at\s+(.+)$', q)
    if not m:
        m = re.search(r'researchers?\s+at\s+(.+)$', q)
    if not m:
        m = re.search(r'staff\s+(?:at|of)\s+(.+)$', q)
    if m:
        org_kw = m.group(1).strip()
        org_uri, _ = _uri_or_filter(org_kw, "org")
        if org_uri.startswith("ex:"):
            return f"""{PREFIX_BLOCK}
SELECT ?researcher ?name WHERE {{
  ?researcher a ex:Researcher ;
              ex:worksAt {org_uri} .
  OPTIONAL {{ ?researcher rdfs:label ?name }}
}} LIMIT 20"""
        else:
            return f"""{PREFIX_BLOCK}
SELECT ?researcher ?name WHERE {{
  ?researcher a ex:Researcher ;
              ex:worksAt ?org .
  ?org rdfs:label ?orgLabel .
  OPTIONAL {{ ?researcher rdfs:label ?name }}
  FILTER(CONTAINS(LCASE(STR(?orgLabel)), "{org_kw.lower()}"))
}} LIMIT 20"""

    # ------------------------------------------------------------------
    # Pattern: "where is X from" / "nationality of X" / "X born in"
    # ------------------------------------------------------------------
    m = re.search(r'where is (.+?) from|nationality of (.+)|(.+?) born in', q)
    if m:
        name_kw = next(g for g in m.groups() if g).strip()
        person_uri, _ = _uri_or_filter(name_kw, "person")
        if person_uri.startswith("ex:"):
            return f"""{PREFIX_BLOCK}
SELECT ?place ?placename WHERE {{
  {{ {person_uri} ex:bornIn ?place }}
  UNION
  {{ {person_uri} ex:nationality ?place }}
  OPTIONAL {{ ?place rdfs:label ?placename }}
}} LIMIT 5"""
        else:
            return f"""{PREFIX_BLOCK}
SELECT ?place ?placename WHERE {{
  ?person rdfs:label ?rlabel .
  FILTER(CONTAINS(LCASE(STR(?rlabel)), "{name_kw.lower()}"))
  {{ ?person ex:bornIn ?place }} UNION {{ ?person ex:nationality ?place }}
  OPTIONAL {{ ?place rdfs:label ?placename }}
}} LIMIT 5"""

    # ------------------------------------------------------------------
    # Pattern: "who did X supervise" / "students of X"
    # ------------------------------------------------------------------
    m = re.search(r'who did (.+?) supervise|students? of (.+)', q)
    if m:
        name_kw = next(grp for grp in m.groups() if grp).strip()
        person_uri, _ = _uri_or_filter(name_kw, "supervisor")
        if person_uri.startswith("ex:"):
            return f"""{PREFIX_BLOCK}
SELECT ?student ?name WHERE {{
  ?student ex:supervisedBy {person_uri} .
  OPTIONAL {{ ?student rdfs:label ?name }}
}} LIMIT 20"""
        else:
            return f"""{PREFIX_BLOCK}
SELECT ?student ?name WHERE {{
  ?supervisor rdfs:label ?slabel .
  FILTER(CONTAINS(LCASE(STR(?slabel)), "{name_kw.lower()}"))
  ?student ex:supervisedBy ?supervisor .
  OPTIONAL {{ ?student rdfs:label ?name }}
}} LIMIT 20"""

    # ------------------------------------------------------------------
    # Pattern: "who supervised X" / "X's advisor"
    # ------------------------------------------------------------------
    m = re.search(r'who\s+(?:supervised|advised|was\s+the\s+(?:advisor|supervisor)\s+of)\s+(.+)$', q)
    if not m:
        m = re.search(r"(.+?)(?:'s)?\s+(?:advisor|supervisor|doctoral)", q)
    if m:
        name_kw = m.group(1).strip()
        person_uri, _ = _uri_or_filter(name_kw, "student")
        if person_uri.startswith("ex:"):
            return f"""{PREFIX_BLOCK}
SELECT ?supervisor ?name WHERE {{
  {person_uri} ex:supervisedBy ?supervisor .
  OPTIONAL {{ ?supervisor rdfs:label ?name }}
}} LIMIT 5"""
        else:
            return f"""{PREFIX_BLOCK}
SELECT ?supervisor ?name WHERE {{
  ?researcher rdfs:label ?rlabel .
  FILTER(CONTAINS(LCASE(STR(?rlabel)), "{name_kw.lower()}"))
  ?researcher ex:supervisedBy ?supervisor .
  OPTIONAL {{ ?supervisor rdfs:label ?name }}
}} LIMIT 5"""

    # ------------------------------------------------------------------
    # Pattern: "researchers in X" / "who works in X"
    # ------------------------------------------------------------------
    m = re.search(r'(?:which\s+)?researchers?\s+(?:work(?:ing)?\s+)?in\s+(.+)$', q)
    if not m:
        m = re.search(r'(?:who\s+)?(?:works?|based)\s+in\s+(.+)$', q)
    if m:
        # Strip leading "the " and normalize aliases
        country_kw = re.sub(r'^the\s+', '', m.group(1).strip())
        _country_aliases = {
            "usa": "United States",
            "us": "United States",
            "u.s.": "United States",
            "u.s.a.": "United States",
            "uk": "United Kingdom",
            "u.k.": "United Kingdom",
        }
        country_kw = _country_aliases.get(country_kw.lower(), country_kw)
        country_uri, _ = _uri_or_filter(country_kw, "country")
        if country_uri.startswith("ex:"):
            return f"""{PREFIX_BLOCK}
SELECT DISTINCT ?researcher ?name WHERE {{
  ?researcher a ex:Researcher .
  OPTIONAL {{ ?researcher rdfs:label ?name }}
  {{
    ?researcher ex:worksAt ?org .
    ?org ex:locatedIn {country_uri} .
  }} UNION {{
    ?researcher ex:nationality {country_uri} .
  }} UNION {{
    ?researcher ex:locatedIn {country_uri} .
  }}
}} LIMIT 20"""
        else:
            country_title = country_kw.title().replace(" ", "_")
            return f"""{PREFIX_BLOCK}
SELECT DISTINCT ?researcher ?name WHERE {{
  ?researcher a ex:Researcher .
  OPTIONAL {{ ?researcher rdfs:label ?name }}
  {{
    ?researcher ex:worksAt ?org .
    ?org ex:locatedIn ex:{country_title} .
  }} UNION {{
    ?researcher ex:nationality ex:{country_title} .
  }} UNION {{
    ?researcher ex:locatedIn ex:{country_title} .
  }}
}} LIMIT 20"""

    # ------------------------------------------------------------------
    # Pattern: "what papers/publications did X write"
    # ------------------------------------------------------------------
    # Bug fix: require 'did' explicitly — using optional (?:did\s+)? with lazy (.+?)
    # allows .{0,20} to consume "did <name>" leaving (.+?) to match a single char.
    # "what papers/publications did X write/publish/author"
    m = re.search(
        r'(?:what|which)\s+(?:papers?|publications?|articles?|books?).{0,10}'
        r'did\s+(.+?)\s+(?:write|author|publish|co-author)',
        q,
    )
    if not m:
        # "what did X publish/write" or "X famous papers" or "papers by X"
        m2 = re.search(r'what did\s+(.+?)\s+(?:publish|write|author)', q)
        m3 = re.search(r'(?:what are\s+(?:the\s+)?(.+?)(?:\'s|s)?\s+(?:famous|notable|key|main|best)\s*papers?)', q)
        m4 = re.search(r'papers?\s+(?:by|from|of)\s+(.+)', q)
        if m2:
            name_kw = m2.group(1).strip()
        elif m3:
            name_kw = m3.group(1).strip()
        elif m4:
            name_kw = m4.group(1).strip().rstrip('?.')
        else:
            name_kw = None
    else:
        name_kw = m.group(1).strip()

    if name_kw and len(name_kw) > 2:
        person_uri, _ = _uri_or_filter(name_kw, "researcher")
        if person_uri.startswith("ex:"):
            return f"""{PREFIX_BLOCK}
SELECT ?paper ?paperName WHERE {{
  {person_uri} ex:authorOf ?paper .
  OPTIONAL {{ ?paper rdfs:label ?paperName }}
}} LIMIT 20"""
        else:
            return f"""{PREFIX_BLOCK}
SELECT ?paper ?paperName WHERE {{
  ?researcher rdfs:label ?rlabel .
  FILTER(CONTAINS(LCASE(STR(?rlabel)), "{name_kw.lower()}"))
  ?researcher ex:authorOf ?paper .
  OPTIONAL {{ ?paper rdfs:label ?paperName }}
}} LIMIT 20"""

    # ------------------------------------------------------------------
    # Pattern: "who collaborated with X" / "X's collaborators"
    # ------------------------------------------------------------------
    m = re.search(r'(?:who\s+)?collaborated?\s+with\s+(.+)$', q)
    if not m:
        m = re.search(r'collaborators?\s+of\s+(.+)$', q)
    if m:
        name_kw = m.group(1).strip()
        person_uri, _ = _uri_or_filter(name_kw, "researcher")
        if person_uri.startswith("ex:"):
            return f"""{PREFIX_BLOCK}
SELECT DISTINCT ?collaborator ?name WHERE {{
  {{
    {person_uri} ex:collaboratesWith ?collaborator .
  }} UNION {{
    ?collaborator ex:collaboratesWith {person_uri} .
  }}
  OPTIONAL {{ ?collaborator rdfs:label ?name }}
}} LIMIT 20"""
        else:
            return f"""{PREFIX_BLOCK}
SELECT DISTINCT ?collaborator ?name WHERE {{
  ?researcher rdfs:label ?rlabel .
  FILTER(CONTAINS(LCASE(STR(?rlabel)), "{name_kw.lower()}"))
  {{
    ?researcher ex:collaboratesWith ?collaborator .
  }} UNION {{
    ?collaborator ex:collaboratesWith ?researcher .
  }}
  OPTIONAL {{ ?collaborator rdfs:label ?name }}
}} LIMIT 20"""

    # ------------------------------------------------------------------
    # Pattern: "organizations/universities in X"
    # ------------------------------------------------------------------
    m = re.search(
        r'(?:what|which)\s+(?:organizations?|universities|companies|labs?).{0,20}'
        r'(?:in|located\s+in)\s+(.+)$',
        q,
    )
    if m:
        # Strip leading "the " from country keyword (e.g. "the USA" -> "USA")
        country_kw = re.sub(r'^the\s+', '', m.group(1).strip())
        # Normalize common country name aliases
        _country_aliases = {
            "usa": "United States",
            "us": "United States",
            "u.s.": "United States",
            "u.s.a.": "United States",
            "uk": "United Kingdom",
            "u.k.": "United Kingdom",
        }
        country_kw_norm = _country_aliases.get(country_kw.lower(), country_kw)
        country_uri, _ = _uri_or_filter(country_kw_norm, "country")
        if country_uri.startswith("ex:"):
            return f"""{PREFIX_BLOCK}
SELECT DISTINCT ?org ?name WHERE {{
  ?org ex:locatedIn {country_uri} .
  OPTIONAL {{ ?org rdfs:label ?name }}
}} LIMIT 20"""
        else:
            country_title = country_kw_norm.title().replace(" ", "_")
            return f"""{PREFIX_BLOCK}
SELECT DISTINCT ?org ?name WHERE {{
  ?org ex:locatedIn ex:{country_title} .
  OPTIONAL {{ ?org rdfs:label ?name }}
}} LIMIT 20"""

    return None  # No template matched


def generate_sparql(
    question: str,
    schema: str,
    model: str = OLLAMA_MODEL,
    g: Optional[Graph] = None,
) -> str:
    """
    Generate a SPARQL query from a natural language question.
    First tries template matching (Problem 4), then falls back to the LLM.
    Pass `g` to enable graph-aware entity resolution in templates.
    """
    # Try template first — faster and more reliable for common patterns
    template_sparql = try_template_sparql(question, g=g)
    if template_sparql:
        logger.info(f"[TEMPLATE MATCH] Using pre-built SPARQL template for: {question[:60]}")
        return template_sparql

    # Fall back to LLM
    prompt = f"""Given this Knowledge Graph schema:

{schema}

Generate a SPARQL query to answer this question:
"{question}"

Return ONLY the SPARQL query, starting with PREFIX declarations."""

    raw = call_ollama(prompt, model=model, system=SPARQL_SYSTEM_PROMPT)
    sparql = extract_sparql_from_response(raw)
    return sparql


def extract_sparql_from_response(text: str) -> str:
    """Extract the SPARQL query from an LLM response, removing markdown etc."""
    # Remove markdown code blocks
    text = re.sub(r"```sparql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)

    # Find the actual query starting with PREFIX or SELECT
    lines = text.split("\n")
    query_lines = []
    in_query = False

    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("PREFIX") or stripped.upper().startswith("SELECT") or stripped.upper().startswith("ASK") or stripped.upper().startswith("CONSTRUCT"):
            in_query = True
        if in_query:
            query_lines.append(line)

    if query_lines:
        return "\n".join(query_lines).strip()

    # Fallback: return the whole text cleaned up
    return text.strip()


# ---------------------------------------------------------------------------
# Problem 1 — normalize wrong predicates the LLM tends to hallucinate
# ---------------------------------------------------------------------------

# Mapping of wrong predicate local-names → correct local-names (ex: prefix assumed)
_PRED_NORM_MAP: dict[str, str] = {
    # authorOf variants
    "wrote":              "authorOf",
    "authored":           "authorOf",
    "published":          "authorOf",
    "hasPublication":     "authorOf",
    "isAuthorOf":         "authorOf",
    "publishedBy":        "authorOf",
    # worksAt variants
    "worksIn":            "worksAt",
    "workAt":             "worksAt",
    "workedAt":           "worksAt",
    "employedAt":         "worksAt",
    "affiliatedWith":     "worksAt",
    "hasEmployer":        "worksAt",
    # wonAward variants
    "received":           "wonAward",
    "hasAward":           "wonAward",
    "receivedAward":      "wonAward",
    "awardedWith":        "wonAward",
    "won":                "wonAward",
    # locatedIn variants
    "locatedAt":          "locatedIn",
    "basedIn":            "locatedIn",
    "headquarteredIn":    "locatedIn",
    "located":            "locatedIn",
    # memberOf / collaboratesWith (no triples for collaboratesWith)
    "collaboratesWith":   "memberOf",
    # studiedAt variants
    "graduatedFrom":      "studiedAt",
    "attendedUniversity": "studiedAt",
    "educated":           "studiedAt",
    "educatedAt":         "studiedAt",
    # bornIn variants
    "born":               "bornIn",
    "birthPlace":         "bornIn",
    "birthCountry":       "bornIn",
    # supervisedBy — keep as-is but normalise common aliases
    "advisedBy":          "supervisedBy",
    "doctoralAdvisor":    "supervisedBy",
    # nationality variants
    "hasNationality":     "nationality",
    "citizenOf":          "nationality",
    "isNationalOf":       "nationality",
    # hasField variants
    "researchField":      "hasField",
    "fieldOf":            "hasField",
    "specializes":        "hasField",
}

# Keep the old name around for the predicate-synonym section used in resolve_entity_uris
PREDICATE_SYNONYMS = _PRED_NORM_MAP


def normalize_sparql_predicates(sparql_query: str) -> str:
    """
    Problem 1 fix: replace known wrong ex: predicates with correct ones.
    Applied BEFORE execution so the graph query has a chance to find results.
    Uses whole-word matching so ex:won is NOT confused with ex:wonAward.
    """
    for wrong, correct in _PRED_NORM_MAP.items():
        if wrong == correct:
            continue
        pattern = re.compile(r'\bex:' + re.escape(wrong) + r'\b')
        if pattern.search(sparql_query):
            sparql_query = pattern.sub(f"ex:{correct}", sparql_query)
            logger.info(f"  [PRED-NORM] ex:{wrong}  ->  ex:{correct}")
    return sparql_query


# ---------------------------------------------------------------------------
# Problem 2 — fix common SPARQL syntax errors produced by small LLMs
# ---------------------------------------------------------------------------


def fix_sparql_syntax(sparql_query: str) -> str:
    """
    Problem 2 fix: repair common LLM syntax errors in SPARQL before parsing.
    - Concatenated prefix tokens: ex:worksInEx: → ex:worksIn ex:
    - Missing space before OPTIONAL / FILTER / LIMIT / UNION / BIND
    - LIMIT/OFFSET accidentally placed inside the WHERE { } block
    """
    # Fix concatenated prefix: ex:SomePredicateEx:Something
    # e.g. ex:worksInEx:France  →  ex:worksIn ex:France
    sparql_query = re.sub(
        r'(ex:[A-Za-z][A-Za-z0-9_]*)(Ex:|ex:|Ex :|ex :)',
        lambda m: m.group(1) + ' ex:',
        sparql_query,
    )

    # Fix missing space before keywords that must start on their own
    for kw in ('OPTIONAL', 'FILTER', 'UNION', 'BIND', 'MINUS', 'VALUES', 'SERVICE'):
        sparql_query = re.sub(r'(?<=[^\s{])\s*(' + kw + r'\b)', r' \1', sparql_query)

    # Fix LIMIT/OFFSET mistakenly placed inside WHERE { … }
    # Detect pattern:  LIMIT \d+ } (possibly followed by more content)
    # Move LIMIT clause after the closing brace
    def _move_limit_out(m_text: str) -> str:
        # Extract all LIMIT/OFFSET lines that appear before the final "}"
        limit_pattern = re.compile(r'\n?\s*(LIMIT\s+\d+|OFFSET\s+\d+)\s*\n', re.IGNORECASE)
        limits_found = limit_pattern.findall(m_text)
        if not limits_found:
            return m_text
        cleaned = limit_pattern.sub('\n', m_text)
        return cleaned + '\n' + '\n'.join(l.strip() for l in limits_found)

    # Only rewrite if LIMIT appears inside WHERE block (before the final "}")
    where_match = re.search(r'(WHERE\s*\{)(.*?)(\}\s*)$', sparql_query, re.DOTALL | re.IGNORECASE)
    if where_match:
        where_body = where_match.group(2)
        if re.search(r'\bLIMIT\b', where_body, re.IGNORECASE):
            fixed_body = _move_limit_out(where_body)
            sparql_query = (
                sparql_query[:where_match.start(2)]
                + fixed_body
                + sparql_query[where_match.end(2):]
            )

    return sparql_query


# ---------------------------------------------------------------------------
# Entity URI resolver — fix hallucinated URIs before execution
# ---------------------------------------------------------------------------

def resolve_entity_uris(g: Graph, sparql_query: str) -> str:
    """
    Problem 3 fix: predicate-aware URI resolution.

    For each ex:LocalName that appears as an OBJECT in the query:
      1. Find the predicate that governs it in the query (e.g. ex:wonAward).
      2. Check if ?s <predicate> ex:LocalName has any results in the graph.
      3. If not → follow owl:sameAs to find which URI *is* used as object of
         that predicate, OR find via rdfs:label / ex:name match.
      4. Replace the URI with the working one.

    Also falls back to general object-URI coverage check and label matching
    for URIs whose governing predicate cannot be determined.
    """
    ex_ns = EX_PREFIX

    # -----------------------------------------------------------------------
    # Pre-build lookup tables
    # -----------------------------------------------------------------------

    # Build label -> URI lookup (lowercase key, preserve longest match)
    label_to_uri: dict[str, str] = {}
    for s, _, o in g.triples((None, RDFS.label, None)):
        s_str = str(s)
        if s_str.startswith(ex_ns):
            label_to_uri[str(o).lower()] = s_str
    for s, _, o in g.triples((None, URIRef(ex_ns + "name"), None)):
        s_str = str(s)
        if s_str.startswith(ex_ns):
            label_to_uri[str(o).lower()] = s_str

    # Build set of URIs that appear as objects of ex: predicates
    real_object_uris: set[str] = set()
    # Also: predicate -> set of objects (for predicate-specific checks)
    pred_to_objects: dict[str, set[str]] = {}
    for s, p, o in g:
        p_str = str(p)
        if p_str.startswith(ex_ns):
            o_str = str(o)
            real_object_uris.add(o_str)
            pred_to_objects.setdefault(p_str, set()).add(o_str)

    # Build sets of valid predicates and class names — never touch these
    valid_predicates: set[str] = set()
    for s, p, o in g:
        valid_predicates.add(str(p).split("/")[-1].split("#")[-1])

    valid_classes: set[str] = set()
    for s, p, o in g.triples((None, RDF.type, None)):
        valid_classes.add(str(o).split("/")[-1].split("#")[-1])

    # -----------------------------------------------------------------------
    # Parse query to identify (predicate, object) pairs
    # -----------------------------------------------------------------------
    # We look for patterns like:  ex:somePredicate ex:SomeObject
    # or                          ex:somePredicate <full-uri>
    pred_obj_pattern = re.compile(
        r'\bex:([a-z][A-Za-z0-9_]*)\s+ex:([A-Z][A-Za-z0-9_]*|[A-Za-z0-9_]*_[A-Za-z0-9_]*)',
    )
    # Map local-name-of-object -> predicate full URI
    object_to_predicate: dict[str, str] = {}
    for pred_local, obj_local in pred_obj_pattern.findall(sparql_query):
        pred_full = ex_ns + pred_local
        object_to_predicate[obj_local] = pred_full

    # -----------------------------------------------------------------------
    # Find all ex:LocalName tokens in the query
    # -----------------------------------------------------------------------
    token_pattern = re.compile(r'\bex:([A-Za-z0-9_\-]+)')
    matches = token_pattern.findall(sparql_query)

    replacements: dict[str, str] = {}
    for local_name in set(matches):
        is_predicate = local_name in valid_predicates
        is_class = local_name in valid_classes
        looks_like_entity = "_" in local_name or (local_name[0].isupper() and not is_class)
        if is_predicate or not looks_like_entity:
            continue

        full_uri = URIRef(ex_ns + local_name)
        full_uri_str = str(full_uri)
        token = f"ex:{local_name}"

        # Determine the governing predicate for this object (if known)
        governing_pred = object_to_predicate.get(local_name)
        pred_objects = pred_to_objects.get(governing_pred, set()) if governing_pred else set()

        # Check 1: is this URI already used as object of the specific predicate?
        if governing_pred:
            if full_uri_str in pred_objects:
                continue  # Already correct for this predicate
        else:
            # No specific predicate known — fall back to any ex: object check
            if full_uri_str in real_object_uris:
                continue

        # --- Attempt to find the canonical URI ---

        # Strategy A: if we have a governing predicate, look for sameAs targets
        # that ARE in pred_objects
        if governing_pred:
            # Direct owl:sameAs from this URI
            for target in g.objects(full_uri, OWL.sameAs):
                target_str = str(target)
                if target_str in pred_objects:
                    correct_local = target_str.split("/")[-1]
                    replacements[token] = f"ex:{correct_local}"
                    break
                # Bug fix: sameAs may point to a Wikidata URI (wd:Qxxx), not ex:.
                # In that case, find which ex: URI also points to the same wd: target
                # and is in pred_objects (2-hop: ex:A sameAs wd:X, ex:B sameAs wd:X, ex:B in pred_objects).
                elif not target_str.startswith(ex_ns):
                    for ex_equiv in g.subjects(OWL.sameAs, URIRef(target_str)):
                        ex_equiv_str = str(ex_equiv)
                        if ex_equiv_str.startswith(ex_ns) and ex_equiv_str in pred_objects:
                            correct_local = ex_equiv_str.split("/")[-1]
                            replacements[token] = f"ex:{correct_local}"
                            break
                    if token in replacements:
                        break

            if token not in replacements:
                # Also check reverse: something sameAs this URI and in pred_objects
                for subj in g.subjects(OWL.sameAs, full_uri):
                    subj_str = str(subj)
                    if subj_str in pred_objects:
                        correct_local = subj_str.split("/")[-1]
                        replacements[token] = f"ex:{correct_local}"
                        break

            if token not in replacements:
                # Strategy B: label match → find URI → check if in pred_objects
                candidate_label = local_name.replace("_", " ").lower()
                matched_uri = label_to_uri.get(candidate_label)
                if matched_uri and matched_uri in pred_objects:
                    correct_local = matched_uri.split("/")[-1]
                    replacements[token] = f"ex:{correct_local}"

            if token not in replacements:
                # Strategy C: among all objects of that predicate, find one
                # whose label best matches local_name
                candidate_label = local_name.replace("_", " ").lower()
                for obj_uri in pred_objects:
                    obj_ref = URIRef(obj_uri)
                    for lbl in g.objects(obj_ref, RDFS.label):
                        if str(lbl).lower() == candidate_label:
                            correct_local = obj_uri.split("/")[-1]
                            replacements[token] = f"ex:{correct_local}"
                            break
                    if token in replacements:
                        break
                    for lbl in g.objects(obj_ref, URIRef(ex_ns + "name")):
                        if str(lbl).lower() == candidate_label:
                            correct_local = obj_uri.split("/")[-1]
                            replacements[token] = f"ex:{correct_local}"
                            break
                    if token in replacements:
                        break
        else:
            # No governing predicate — use original strategies
            # Strategy 1: follow owl:sameAs chain (including 2-hop via Wikidata)
            for target in g.objects(full_uri, OWL.sameAs):
                target_str = str(target)
                if target_str in real_object_uris:
                    correct_local = target_str.split("/")[-1]
                    replacements[token] = f"ex:{correct_local}"
                    break
                # 2-hop: sameAs -> wd: -> find ex: alias that IS in real_object_uris
                elif not target_str.startswith(ex_ns):
                    for ex_equiv in g.subjects(OWL.sameAs, URIRef(target_str)):
                        ex_equiv_str = str(ex_equiv)
                        if ex_equiv_str.startswith(ex_ns) and ex_equiv_str in real_object_uris:
                            correct_local = ex_equiv_str.split("/")[-1]
                            replacements[token] = f"ex:{correct_local}"
                            break
                    if token in replacements:
                        break
            if token in replacements:
                continue

            # Strategy 2: exact label match
            candidate_label = local_name.replace("_", " ").lower()
            if candidate_label in label_to_uri:
                correct_uri = label_to_uri[candidate_label]
                correct_local = correct_uri.split("/")[-1]
                replacements[token] = f"ex:{correct_local}"
                continue

            # Strategy 3: partial label match
            for lbl, uri in label_to_uri.items():
                if candidate_label in lbl or lbl in candidate_label:
                    correct_local = uri.split("/")[-1]
                    replacements[token] = f"ex:{correct_local}"
                    break

    # Apply replacements
    for wrong, correct in replacements.items():
        if wrong != correct:
            sparql_query = sparql_query.replace(wrong, correct)
            logger.info(f"  [URI fix] {wrong}  ->  {correct}")

    return sparql_query


# ---------------------------------------------------------------------------
# SPARQL execution
# ---------------------------------------------------------------------------


def execute_sparql(g: Graph, sparql_query: str) -> tuple[list[dict], str]:
    """
    Execute a SPARQL query on the graph.
    Applies Problem 2, 1, and 3 fixes in order before executing.
    Returns (results_list, error_message).
    """
    def resolve_label(g: Graph, uri_str: str) -> str:
        """Return human-readable label for a URI, falling back to local name."""
        if not uri_str.startswith("http"):
            return uri_str  # already a literal
        uri = URIRef(uri_str)
        for lbl in g.objects(uri, RDFS.label):
            return str(lbl)
        for lbl in g.objects(uri, URIRef(EX_PREFIX + "name")):
            return str(lbl)
        # fallback: extract local name and clean underscores
        local = uri_str.split("/")[-1].split("#")[-1]
        if local.startswith("Q") and local[1:].isdigit():
            return local  # keep raw QID if no label found
        return local.replace("_", " ")

    # --- Pre-processing pipeline (Problems 1, 2, 3) ---
    sparql_query = fix_sparql_syntax(sparql_query)           # Problem 2
    sparql_query = normalize_sparql_predicates(sparql_query)  # Problem 1
    sparql_query = resolve_entity_uris(g, sparql_query)       # Problem 3

    try:
        results = g.query(sparql_query)
        rows = []
        for row in results:
            row_dict = {}
            for var in results.vars:
                val = row.get(var)
                if val is not None:
                    raw = str(val)
                    row_dict[str(var)] = resolve_label(g, raw)
            rows.append(row_dict)
        return rows, ""
    except Exception as e:
        error_msg = str(e)
        # Clean up error message
        error_msg = error_msg[:500]  # Truncate very long errors
        return [], error_msg


# ---------------------------------------------------------------------------
# Problem 5 — enrich QID results with human-readable labels
# ---------------------------------------------------------------------------


def enrich_results(g: Graph, results: list[dict]) -> list[dict]:
    """
    Problem 5 fix: replace raw QID values (e.g. "Q92823") in result rows
    with their rdfs:label or ex:name if available in the graph.
    """
    ex_ns = EX_PREFIX
    qid_re = re.compile(r'^Q\d+$')

    enriched = []
    for row in results:
        new_row = {}
        for key, val in row.items():
            if isinstance(val, str) and qid_re.match(val):
                uri = URIRef(ex_ns + val)
                label = None
                for lbl in g.objects(uri, RDFS.label):
                    label = str(lbl)
                    break
                if not label:
                    for lbl in g.objects(uri, URIRef(ex_ns + "name")):
                        label = str(lbl)
                        break
                new_row[key] = label if label else val
            else:
                new_row[key] = val
        enriched.append(new_row)
    return enriched


# ---------------------------------------------------------------------------
# SPARQL self-repair
# ---------------------------------------------------------------------------

REPAIR_SYSTEM_PROMPT = """You are an expert SPARQL debugger.
Fix the SPARQL query that produced an error.
Return ONLY the corrected SPARQL query, no explanation."""


def repair_sparql(
    question: str,
    broken_sparql: str,
    error_message: str,
    schema: str,
    model: str = OLLAMA_MODEL,
) -> str:
    """
    Ask the LLM to fix a broken SPARQL query.
    """
    prompt = f"""The following SPARQL query produced an error when executed against a KG about AI researchers.

SCHEMA:
{schema[:2000]}

QUESTION: {question}

BROKEN SPARQL:
{broken_sparql}

ERROR MESSAGE:
{error_message}

Fix the SPARQL query. Common fixes:
- Use correct prefixes (ex: for https://aikg.example.org/)
- Check property names match the schema
- Fix syntax errors (missing brackets, dots, semicolons)
- Use OPTIONAL for optional properties
- Add LIMIT 20

Return ONLY the corrected SPARQL query:"""

    raw = call_ollama(prompt, model=model, system=REPAIR_SYSTEM_PROMPT)
    return extract_sparql_from_response(raw)


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about AI researchers.
Given SPARQL query results from a Knowledge Graph, provide a clear, concise answer.
If the results are empty, say so and provide what general knowledge you have.
Keep answers brief and factual."""


def generate_answer(
    question: str,
    sparql_query: str,
    results: list[dict],
    model: str = OLLAMA_MODEL,
) -> str:
    """
    Generate a natural language answer from SPARQL results.
    """
    if not results:
        results_str = "No results found in the Knowledge Graph."
    else:
        results_str = json.dumps(results[:MAX_RESULTS_DISPLAY], indent=2, ensure_ascii=False)

    prompt = f"""Question: {question}

SPARQL Query Used:
{sparql_query}

Knowledge Graph Results:
{results_str}

Provide a clear, concise answer based on these results. If results are empty, say the information was not found in the KG."""

    return call_ollama(prompt, model=model, system=ANSWER_SYSTEM_PROMPT)


def generate_baseline_answer(question: str, model: str = OLLAMA_MODEL) -> str:
    """
    Generate an answer using only the LLM's parametric knowledge (no KG).
    """
    prompt = f"""Answer this question about AI researchers based on your knowledge:

{question}

Provide a factual, concise answer."""

    system = "You are a knowledgeable assistant about artificial intelligence researchers, their work, and institutions."
    return call_ollama(prompt, model=model, system=system)


# ---------------------------------------------------------------------------
# Main RAG pipeline
# ---------------------------------------------------------------------------


class RAGPipeline:
    """
    Complete RAG pipeline: question → SPARQL → answer.
    """

    def __init__(self, kg_path: Optional[Path] = None, model: str = OLLAMA_MODEL):
        self.model = model
        self.graph = load_graph(kg_path)
        self.schema = build_schema_summary(self.graph, max_predicates=30, max_classes=20, max_sample_triples=10)
        logger.info(f"RAG Pipeline initialized. KG has {len(self.graph)} triples.")
        logger.info(f"Using LLM: {self.model}")

    def query(self, question: str, mode: str = "rag") -> dict:
        """
        Process a question and return structured response.

        Args:
            question: Natural language question
            mode: "rag" (uses KG) or "baseline" (LLM only)

        Returns:
            dict with keys: answer, sparql_query, results, repair_attempts, mode
        """
        if mode == "baseline":
            answer = generate_baseline_answer(question, model=self.model)
            return {
                "answer": answer,
                "sparql_query": None,
                "results": [],
                "repair_attempts": 0,
                "mode": "baseline",
                "error": None,
            }

        # --- RAG mode ---
        sparql_query = ""
        results = []
        repair_attempts = 0
        last_error = ""

        # Step 1: Generate initial SPARQL
        # Pass the graph so template matching can resolve entity URIs directly
        try:
            sparql_query = generate_sparql(
                question, self.schema, model=self.model, g=self.graph
            )
            logger.info(f"Generated SPARQL:\n{sparql_query}")
        except RuntimeError as e:
            return {
                "answer": f"LLM unavailable: {e}",
                "sparql_query": "",
                "results": [],
                "repair_attempts": 0,
                "mode": "rag",
                "error": str(e),
            }

        # Note: execute_sparql now applies fix_sparql_syntax, normalize_sparql_predicates,
        # and resolve_entity_uris internally (Problems 1, 2, 3).
        # We keep an explicit pre-resolve here for the initial query so the
        # logged query reflects fixes before any repair attempts.
        sparql_query = fix_sparql_syntax(sparql_query)
        sparql_query = normalize_sparql_predicates(sparql_query)
        sparql_query = resolve_entity_uris(self.graph, sparql_query)

        # Step 2: Execute with self-repair loop
        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            results, error = execute_sparql(self.graph, sparql_query)

            if not error:
                break  # Success

            if attempt < MAX_REPAIR_ATTEMPTS:
                logger.info(f"SPARQL error (attempt {attempt + 1}): {error}")
                repair_attempts += 1
                try:
                    sparql_query = repair_sparql(
                        question, sparql_query, error, self.schema, model=self.model
                    )
                    logger.info(f"Repaired SPARQL:\n{sparql_query}")
                except RuntimeError as e:
                    last_error = str(e)
                    break
            else:
                last_error = error

        # Step 2b: Enrich results — replace QIDs with human-readable labels (Problem 5)
        results = enrich_results(self.graph, results)

        # Step 3: Generate natural language answer
        try:
            answer = generate_answer(question, sparql_query, results, model=self.model)
        except RuntimeError as e:
            answer = f"Could not generate answer: {e}"
            if results:
                answer += f"\n\nRaw results: {json.dumps(results[:5], indent=2)}"

        return {
            "answer": answer,
            "sparql_query": sparql_query,
            "results": results,
            "repair_attempts": repair_attempts,
            "mode": "rag",
            "error": last_error if last_error and not results else None,
        }

    def answer_question(self, question: str, mode: str = "rag") -> dict:
        """
        Alias for query() — added for compatibility with the test command.
        """
        return self.query(question, mode=mode)

    def get_stats(self) -> dict:
        """Return KG statistics."""
        subjects = set(self.graph.subjects())
        predicates = set(self.graph.predicates())
        objects = set(self.graph.objects())
        entities = subjects | {o for o in objects if str(o).startswith(EX_PREFIX)}

        return {
            "total_triples": len(self.graph),
            "unique_entities": len(entities),
            "unique_relations": len(predicates),
            "unique_subjects": len(subjects),
        }

    def get_schema(self) -> str:
        """Return schema summary."""
        return self.schema


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def interactive_loop(pipeline: RAGPipeline, mode: str = "rag"):
    """Run an interactive CLI loop."""
    print("\n" + "=" * 60)
    print("AI Knowledge Graph RAG System — Interactive Mode")
    print(f"Mode: {mode.upper()} | Model: {pipeline.model}")
    print(f"KG: {len(pipeline.graph)} triples")
    print("Type 'quit' to exit, 'mode rag' or 'mode baseline' to switch")
    print("=" * 60 + "\n")

    current_mode = mode

    while True:
        try:
            question = input(f"[{current_mode.upper()}] Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue
        if question.lower() == "quit":
            break
        if question.lower().startswith("mode "):
            new_mode = question.split()[1].lower()
            if new_mode in ("rag", "baseline"):
                current_mode = new_mode
                print(f"Switched to {current_mode.upper()} mode.")
            else:
                print("Unknown mode. Use 'mode rag' or 'mode baseline'.")
            continue

        print("\nProcessing...\n")
        result = pipeline.query(question, mode=current_mode)

        if result.get("sparql_query"):
            print("--- SPARQL Query ---")
            print(result["sparql_query"])
            print()

        if result.get("repair_attempts", 0) > 0:
            print(f"(Repaired {result['repair_attempts']} time(s))")
            print()

        if result.get("results"):
            print("--- KG Results ---")
            for row in result["results"][:10]:
                print(" | ".join(f"{k}: {v}" for k, v in row.items()))
            if len(result["results"]) > 10:
                print(f"  ... and {len(result['results']) - 10} more")
            print()

        print("--- Answer ---")
        print(result["answer"])
        print()


def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline for AI Researchers KG")
    parser.add_argument("--question", type=str, help="Single question to answer")
    parser.add_argument("--mode", choices=["rag", "baseline"], default="rag")
    parser.add_argument("--interactive", action="store_true", help="Run interactive CLI")
    parser.add_argument("--kg", type=Path, help="Path to KG file (optional)")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL)
    args = parser.parse_args()

    # Check Ollama
    if not check_ollama_available():
        print(f"WARNING: Ollama not available at {OLLAMA_BASE_URL}")
        print("Start Ollama with: ollama serve")
        print("Then pull a model: ollama pull llama3.2:1b")
        if not args.interactive:
            sys.exit(1)

    # Initialize pipeline
    pipeline = RAGPipeline(kg_path=args.kg, model=args.model)

    if args.interactive:
        interactive_loop(pipeline, mode=args.mode)
    elif args.question:
        result = pipeline.query(args.question, mode=args.mode)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
