"""
Knowledge Graph Expander via Wikidata SPARQL
=============================================
Reads the alignment file to find Wikidata QIDs for our entities,
then runs 1-hop and 2-hop SPARQL queries to enrich the KG.

Output:
    kg_artifacts/expanded_kg.nt

Usage:
    python src/kg/kg_expander.py
"""

import logging
import time
from pathlib import Path
from typing import Iterator

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD
from SPARQLWrapper import JSON, SPARQLWrapper

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
# Namespaces
# ---------------------------------------------------------------------------

EX = Namespace("https://aikg.example.org/")
WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
SCHEMA = Namespace("https://schema.org/")

# ---------------------------------------------------------------------------
# Wikidata endpoint config
# ---------------------------------------------------------------------------

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
REQUEST_DELAY = 1.5   # seconds between requests

# Wikidata property → our EX property mapping
WD_PROP_MAP = {
    "P108":  EX.worksAt,           # employer
    "P17":   EX.locatedIn,         # country
    "P166":  EX.wonAward,          # award received
    "P800":  EX.authorOf,          # notable work
    "P19":   EX.bornIn,            # place of birth
    "P69":   EX.studiedAt,         # educated at
    "P184":  EX.supervisedBy,      # doctoral advisor
    "P463":  EX.memberOf,          # member of
    "P569":  EX.birthDate,         # date of birth
    "P856":  EX.homepage,          # official website
    "P1929": EX.hIndex,            # h-index
    "P6889": EX.citationCount,     # total citations
    "P1066": EX.supervisedBy,      # student of (doctoral)
    "P27":   EX.nationality,       # country of citizenship
}

# Classes to assign based on Wikidata instance-of
WD_CLASS_MAP = {
    "Q5":        EX.Researcher,     # human → assume researcher if in our alignment
    "Q3918":     EX.University,     # university
    "Q38723":    EX.University,     # higher educational institution
    "Q875538":   EX.University,     # public university
    "Q4287745":  EX.ResearchLab,    # research institute
    "Q31855":    EX.ResearchLab,    # research institute
    "Q4830453":  EX.Company,        # business
    "Q6881511":  EX.Company,        # enterprise
    "Q185167":   EX.Award,          # award
    "Q1980247":  EX.Award,          # prize
    "Q13442814": EX.Publication,    # scholarly article
    "Q571":      EX.Publication,    # book
}


# ---------------------------------------------------------------------------
# SPARQL helper
# ---------------------------------------------------------------------------


def get_sparql_client() -> SPARQLWrapper:
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    sparql.addCustomHttpHeader(
        "User-Agent", "AcademicKGExpander/1.0 (student project; educational)"
    )
    sparql.setReturnFormat(JSON)
    return sparql


def run_query(sparql: SPARQLWrapper, query: str) -> list[dict]:
    """Run a SPARQL query and return bindings list."""
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        return results["results"]["bindings"]
    except Exception as e:
        logger.warning(f"SPARQL query error: {e}")
        return []


# ---------------------------------------------------------------------------
# 1-hop expansion: get all direct properties of a Wikidata entity
# ---------------------------------------------------------------------------


def expand_1hop(sparql: SPARQLWrapper, qid: str) -> list[dict]:
    """
    Fetch 1-hop triples from Wikidata for entity with given QID.
    Only fetches properties we care about (WD_PROP_MAP keys).
    """
    props = " ".join(f"wdt:P{p[1:]}" for p in WD_PROP_MAP if p.startswith("P"))
    # Build VALUES clause for properties
    prop_values = " ".join(f"wdt:{pid}" for pid in WD_PROP_MAP)

    query = f"""
    SELECT ?prop ?propLabel ?value ?valueLabel WHERE {{
      VALUES ?prop {{ {prop_values} }}
      wd:{qid} ?prop ?value .
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "en,fr".
      }}
    }}
    LIMIT 50
    """
    return run_query(sparql, query)


# ---------------------------------------------------------------------------
# 2-hop expansion: researcher → organization → location
# ---------------------------------------------------------------------------


def expand_2hop_org_location(sparql: SPARQLWrapper, qid: str) -> list[dict]:
    """
    2-hop: researcher → worksAt → organization → locatedIn → country
    """
    query = f"""
    SELECT ?org ?orgLabel ?country ?countryLabel WHERE {{
      wd:{qid} wdt:P108 ?org .
      ?org wdt:P17 ?country .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 10
    """
    return run_query(sparql, query)


def expand_2hop_award_field(sparql: SPARQLWrapper, qid: str) -> list[dict]:
    """
    2-hop: researcher → wonAward → award → field
    """
    query = f"""
    SELECT ?award ?awardLabel ?field ?fieldLabel WHERE {{
      wd:{qid} wdt:P166 ?award .
      OPTIONAL {{ ?award wdt:P101 ?field. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 10
    """
    return run_query(sparql, query)


def expand_collaborators(sparql: SPARQLWrapper, qid: str) -> list[dict]:
    """
    Find researchers who share the same employer as this researcher.
    (Approximate 'collaboratesWith' through shared org)
    """
    query = f"""
    SELECT ?collab ?collabLabel WHERE {{
      wd:{qid} wdt:P108 ?org .
      ?collab wdt:P108 ?org .
      ?collab wdt:P31 wd:Q5 .
      ?collab wdt:P101 wd:Q11660 .  # field: AI
      FILTER(?collab != wd:{qid})
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 5
    """
    return run_query(sparql, query)


# ---------------------------------------------------------------------------
# Convert Wikidata results to our EX namespace
# ---------------------------------------------------------------------------


def uri_to_ex(wikidata_uri: str) -> URIRef:
    """Convert a Wikidata URI to our EX namespace URI."""
    qid = wikidata_uri.split("/")[-1]
    return EX[qid]


def wikidata_binding_to_triples(
    subject_ex: URIRef,
    bindings: list[dict],
) -> Iterator[tuple]:
    """
    Convert Wikidata SPARQL result bindings to RDF triples in our namespace.
    Yields (subject, predicate, object) tuples.
    """
    for b in bindings:
        prop_uri_str = b.get("prop", {}).get("value", "")
        value_data = b.get("value", {})
        value_label = b.get("valueLabel", {}).get("value", "")

        if not prop_uri_str or not value_data:
            continue

        # Map Wikidata property to our EX property
        pid = prop_uri_str.split("/")[-1]
        ex_prop = WD_PROP_MAP.get(pid)
        if not ex_prop:
            continue

        value_type = value_data.get("type", "")
        value_str = value_data.get("value", "")

        if value_type == "uri":
            # Entity reference
            value_qid = value_str.split("/")[-1]
            if value_qid.startswith("Q"):
                obj_uri = EX[value_qid]
                yield (subject_ex, ex_prop, obj_uri)
                # Also add a label
                if value_label:
                    yield (obj_uri, RDFS.label, Literal(value_label))
                    yield (obj_uri, EX.name, Literal(value_label))
                # Add owl:sameAs back to Wikidata
                yield (obj_uri, OWL.sameAs, WD[value_qid])
        elif value_type == "literal":
            # Datatype literal
            dtype = value_data.get("datatype", "")
            if dtype == str(XSD.date) or "date" in dtype:
                yield (subject_ex, ex_prop, Literal(value_str, datatype=XSD.date))
            elif dtype == str(XSD.integer) or dtype == str(XSD.decimal):
                try:
                    yield (subject_ex, ex_prop, Literal(int(value_str), datatype=XSD.integer))
                except ValueError:
                    yield (subject_ex, ex_prop, Literal(value_str))
            else:
                yield (subject_ex, ex_prop, Literal(value_str))


# ---------------------------------------------------------------------------
# Main expansion logic
# ---------------------------------------------------------------------------


def expand_kg(
    initial_kg_path: Path,
    alignment_path: Path,
    output_path: Path,
    max_entities: int = 30,
) -> Graph:
    """
    Load the initial KG and alignment, expand via Wikidata SPARQL,
    merge and save the expanded KG.
    """
    # Load initial KG
    logger.info(f"Loading initial KG: {initial_kg_path}")
    g = Graph()
    g.bind("ex", EX)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("wd", WD)

    if initial_kg_path.exists():
        fmt = "ntriples" if str(initial_kg_path).endswith(".nt") else "turtle"
        g.parse(str(initial_kg_path), format=fmt)
        logger.info(f"Loaded {len(g)} triples from initial KG")
    else:
        logger.warning(f"Initial KG not found: {initial_kg_path}")

    # Load alignment
    logger.info(f"Loading alignment: {alignment_path}")
    alignment_g = Graph()
    if alignment_path.exists():
        alignment_g.parse(str(alignment_path), format="turtle")
        logger.info(f"Loaded {len(alignment_g)} alignment triples")
    else:
        logger.warning(f"Alignment file not found: {alignment_path}")

    # Merge alignment into main graph
    for triple in alignment_g:
        g.add(triple)

    # -----------------------------------------------------------------------
    # Collect entity → Wikidata QID mappings from the graph
    # -----------------------------------------------------------------------
    entity_qid_map: dict[URIRef, str] = {}
    for s, _, o in g.triples((None, OWL.sameAs, None)):
        if str(o).startswith(str(WD)):
            qid = str(o).split("/")[-1]
            if qid.startswith("Q"):
                entity_qid_map[s] = qid

    logger.info(f"Found {len(entity_qid_map)} entities with Wikidata QIDs")

    # -----------------------------------------------------------------------
    # Expand via SPARQL (limit to max_entities to be polite)
    # -----------------------------------------------------------------------
    sparql = get_sparql_client()
    entities_to_expand = list(entity_qid_map.items())[:max_entities]

    new_triples = 0
    for entity_uri, qid in entities_to_expand:
        label = str(list(g.objects(entity_uri, RDFS.label)) or [qid])[0]
        logger.info(f"Expanding: {label} (wd:{qid})")

        # 1-hop expansion
        time.sleep(REQUEST_DELAY)
        bindings_1hop = expand_1hop(sparql, qid)
        for triple in wikidata_binding_to_triples(entity_uri, bindings_1hop):
            if triple not in g:
                g.add(triple)
                new_triples += 1

        # Only do 2-hop for researchers (not orgs/awards)
        entity_types = list(g.objects(entity_uri, RDF.type))
        if EX.Researcher in entity_types:
            # 2-hop: org → country
            time.sleep(REQUEST_DELAY)
            bindings_2hop = expand_2hop_org_location(sparql, qid)
            for b in bindings_2hop:
                org_str = b.get("org", {}).get("value", "")
                country_str = b.get("country", {}).get("value", "")
                country_label = b.get("countryLabel", {}).get("value", "")
                if org_str and country_str:
                    org_uri = EX[org_str.split("/")[-1]]
                    country_uri = EX[country_str.split("/")[-1]]
                    g.add((org_uri, EX.locatedIn, country_uri))
                    if country_label:
                        g.add((country_uri, RDFS.label, Literal(country_label)))
                        g.add((country_uri, RDF.type, EX.Country))
                    new_triples += 2

            # 2-hop: award details
            time.sleep(REQUEST_DELAY)
            bindings_award = expand_2hop_award_field(sparql, qid)
            for b in bindings_award:
                award_str = b.get("award", {}).get("value", "")
                award_label = b.get("awardLabel", {}).get("value", "")
                if award_str:
                    award_qid = award_str.split("/")[-1]
                    award_uri = EX[award_qid]
                    g.add((entity_uri, EX.wonAward, award_uri))
                    if award_label:
                        g.add((award_uri, RDFS.label, Literal(award_label)))
                        g.add((award_uri, RDF.type, EX.Award))
                    new_triples += 2

        logger.info(f"  Added triples (running total: {new_triples})")

    # -----------------------------------------------------------------------
    # Remove duplicate triples (rdflib handles this as a set)
    # -----------------------------------------------------------------------
    logger.info("Finalizing graph (removing blank nodes, cleaning URIs)...")

    # Filter out blank nodes as subjects/objects for cleanliness
    final_g = Graph()
    final_g.bind("ex", EX)
    final_g.bind("owl", OWL)
    final_g.bind("rdfs", RDFS)
    final_g.bind("wd", WD)

    for s, p, o in g:
        # Skip blank nodes
        from rdflib import BNode
        if isinstance(s, BNode) or isinstance(o, BNode):
            continue
        final_g.add((s, p, o))

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_g.serialize(destination=str(output_path), format="ntriples")

    # Print statistics
    subjects = set(final_g.subjects())
    predicates = set(final_g.predicates())
    print(f"\n=== Expanded KG Statistics ===")
    print(f"Total triples   : {len(final_g)}")
    print(f"Unique entities : {len(subjects)}")
    print(f"Unique relations: {len(predicates)}")
    print(f"New triples added via expansion: {new_triples}")
    print(f"\nOutput: {output_path}")

    return final_g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    initial_kg = Path("kg_artifacts/initial_kg.nt")
    alignment = Path("kg_artifacts/alignment.ttl")
    output = Path("kg_artifacts/expanded_kg.nt")

    # Fallback: if initial KG doesn't exist, use sample KG
    if not initial_kg.exists():
        sample = Path("kg_artifacts/sample_kg.ttl")
        if sample.exists():
            logger.info(f"Initial KG not found, using sample KG: {sample}")
            initial_kg = sample
        else:
            logger.warning("No KG found. Run kg_builder.py first.")

    if not alignment.exists():
        logger.info("Alignment not found. Run entity_aligner.py first. Using pre-built mappings.")
        # Run alignment inline
        from src.kg.entity_aligner import build_alignment_graph
        align_g = build_alignment_graph(initial_kg if initial_kg.exists() else None)
        alignment.parent.mkdir(parents=True, exist_ok=True)
        align_g.serialize(str(alignment), format="turtle")

    expand_kg(
        initial_kg_path=initial_kg,
        alignment_path=alignment,
        output_path=output,
        max_entities=25,  # Limit to avoid hammering Wikidata
    )


if __name__ == "__main__":
    main()
