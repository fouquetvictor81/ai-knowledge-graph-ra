"""
Bulk Knowledge Graph Expander — Target: 50 000 – 200 000 triples
=================================================================
Runs a series of focused SPARQL queries against the Wikidata public
endpoint.  Each query targets one relation type and returns up to
5 000 rows.  Results are converted to our EX namespace and merged
into the existing expanded_kg.nt.

Strategy (15 bulk queries):
  1.  AI researchers  → employer
  2.  AI researchers  → nationality
  3.  AI researchers  → education
  4.  AI researchers  → award
  5.  AI researchers  → doctoral advisor
  6.  AI researchers  → birth place
  7.  AI researchers  → field of work
  8.  CS awards       → given by / for field
  9.  AI companies    → location / founded
  10. Universities    → location / country
  11. Publications    → author / venue
  12. Conferences     → location / field
  13. Research labs   → parent org / location
  14. Countries       → capital / continent
  15. Turing Award recipients (high-quality seed)

Usage:
    python src/kg/kg_bulk_expander.py
"""

import logging
import time
from pathlib import Path

from rdflib import BNode, Graph, Literal, Namespace, URIRef
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
EX   = Namespace("https://aikg.example.org/")
WD   = Namespace("http://www.wikidata.org/entity/")
WDT  = Namespace("http://www.wikidata.org/prop/direct/")
SCHEMA = Namespace("https://schema.org/")

# ---------------------------------------------------------------------------
# Wikidata AI-related field QIDs (used in VALUES clauses)
# ---------------------------------------------------------------------------
AI_FIELDS = " ".join([
    "wd:Q11660",    # artificial intelligence
    "wd:Q2539",     # machine learning
    "wd:Q30070477", # deep learning
    "wd:Q1137323",  # natural language processing
    "wd:Q1662673",  # computer vision
    "wd:Q206855",   # reinforcement learning
    "wd:Q170076",   # artificial neural network
    "wd:Q8513",     # data mining
    "wd:Q170417",   # robotics
    "wd:Q2878974",  # knowledge representation
    "wd:Q1128474",  # information retrieval
    "wd:Q7189713",  # knowledge graph
    "wd:Q2539",     # machine learning (repeated for weight)
    "wd:Q483247",   # computer science
    "wd:Q9143",     # programming language theory
])

# ---------------------------------------------------------------------------
# All bulk SPARQL queries
# ---------------------------------------------------------------------------
BULK_QUERIES = {

    # ── 1. Researchers → employer ─────────────────────────────────────────
    "researcher_employer": (
        "researcher", "worksAt", "employer",
        f"""
        SELECT DISTINCT ?researcher ?researcherLabel ?employer ?employerLabel WHERE {{
          ?researcher wdt:P31 wd:Q5 ;
                      wdt:P101 ?field ;
                      wdt:P108 ?employer .
          VALUES ?field {{ {AI_FIELDS} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT 5000
        """
    ),

    # ── 2. Researchers → nationality ──────────────────────────────────────
    "researcher_nationality": (
        "researcher", "nationality", "country",
        f"""
        SELECT DISTINCT ?researcher ?researcherLabel ?country ?countryLabel WHERE {{
          ?researcher wdt:P31 wd:Q5 ;
                      wdt:P101 ?field ;
                      wdt:P27  ?country .
          VALUES ?field {{ {AI_FIELDS} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT 5000
        """
    ),

    # ── 3. Researchers → education ────────────────────────────────────────
    "researcher_education": (
        "researcher", "studiedAt", "university",
        f"""
        SELECT DISTINCT ?researcher ?researcherLabel ?university ?universityLabel WHERE {{
          ?researcher wdt:P31 wd:Q5 ;
                      wdt:P101 ?field ;
                      wdt:P69  ?university .
          VALUES ?field {{ {AI_FIELDS} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT 5000
        """
    ),

    # ── 4. Researchers → award ────────────────────────────────────────────
    "researcher_award": (
        "researcher", "wonAward", "award",
        f"""
        SELECT DISTINCT ?researcher ?researcherLabel ?award ?awardLabel WHERE {{
          ?researcher wdt:P31 wd:Q5 ;
                      wdt:P101 ?field ;
                      wdt:P166 ?award .
          VALUES ?field {{ {AI_FIELDS} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT 5000
        """
    ),

    # ── 5. Researchers → doctoral advisor ─────────────────────────────────
    "researcher_advisor": (
        "researcher", "supervisedBy", "advisor",
        f"""
        SELECT DISTINCT ?researcher ?researcherLabel ?advisor ?advisorLabel WHERE {{
          ?researcher wdt:P31 wd:Q5 ;
                      wdt:P101 ?field ;
                      wdt:P184 ?advisor .
          VALUES ?field {{ {AI_FIELDS} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT 3000
        """
    ),

    # ── 6. Researchers → birth place ──────────────────────────────────────
    "researcher_birthplace": (
        "researcher", "bornIn", "place",
        f"""
        SELECT DISTINCT ?researcher ?researcherLabel ?place ?placeLabel WHERE {{
          ?researcher wdt:P31 wd:Q5 ;
                      wdt:P101 ?field ;
                      wdt:P19  ?place .
          VALUES ?field {{ {AI_FIELDS} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT 4000
        """
    ),

    # ── 7. Researchers → field of work ────────────────────────────────────
    "researcher_field": (
        "researcher", "hasField", "field",
        f"""
        SELECT DISTINCT ?researcher ?researcherLabel ?field ?fieldLabel WHERE {{
          ?researcher wdt:P31 wd:Q5 ;
                      wdt:P101 ?field .
          VALUES ?field {{ {AI_FIELDS} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT 5000
        """
    ),

    # ── 8. CS Awards → given-by organisation ─────────────────────────────
    "award_givenby": (
        "award", "givenBy", "org",
        """
        SELECT DISTINCT ?award ?awardLabel ?org ?orgLabel WHERE {
          VALUES ?awardcat { wd:Q185167 wd:Q1980247 wd:Q19020 }
          ?award wdt:P31 ?awardcat ;
                 wdt:P1027 ?org .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 2000
        """
    ),

    # ── 9. AI companies → headquarters location ───────────────────────────
    "company_location": (
        "company", "locatedIn", "place",
        f"""
        SELECT DISTINCT ?company ?companyLabel ?place ?placeLabel WHERE {{
          ?company wdt:P31/wdt:P279* wd:Q4830453 ;
                   wdt:P452 ?industry ;
                   wdt:P159 ?place .
          VALUES ?industry {{ wd:Q11660 wd:Q2539 wd:Q483247 }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
        }}
        LIMIT 3000
        """
    ),

    # ── 10. Universities → country ────────────────────────────────────────
    "university_country": (
        "university", "locatedIn", "country",
        """
        SELECT DISTINCT ?university ?universityLabel ?country ?countryLabel WHERE {
          VALUES ?utype { wd:Q3918 wd:Q38723 wd:Q875538 wd:Q902104 }
          ?university wdt:P31 ?utype ;
                      wdt:P17 ?country .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 5000
        """
    ),

    # ── 11. Publications → author (AI venue papers) ───────────────────────
    "publication_author": (
        "publication", "authoredBy", "researcher",
        """
        SELECT DISTINCT ?pub ?pubLabel ?author ?authorLabel WHERE {
          VALUES ?venue {
            wd:Q1112324   # NeurIPS
            wd:Q1130538   # ICML
            wd:Q115570561 # ICLR
            wd:Q97650813  # CVPR
            wd:Q6046043   # ACL
            wd:Q15707572  # EMNLP
          }
          ?pub wdt:P1433 ?venue ;
               wdt:P50   ?author .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 5000
        """
    ),

    # ── 12. Publications → venue ──────────────────────────────────────────
    "publication_venue": (
        "publication", "publishedIn", "venue",
        """
        SELECT DISTINCT ?pub ?pubLabel ?venue ?venueLabel WHERE {
          VALUES ?venue {
            wd:Q1112324 wd:Q1130538 wd:Q115570561
            wd:Q97650813 wd:Q6046043 wd:Q15707572
          }
          ?pub wdt:P1433 ?venue .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 5000
        """
    ),

    # ── 13. Research labs → parent organisation ───────────────────────────
    "lab_parent": (
        "lab", "partOf", "org",
        """
        SELECT DISTINCT ?lab ?labLabel ?parent ?parentLabel WHERE {
          VALUES ?ltype { wd:Q4287745 wd:Q31855 wd:Q178706 }
          ?lab wdt:P31 ?ltype ;
               wdt:P749 ?parent .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 3000
        """
    ),

    # ── 14. Countries → continent ─────────────────────────────────────────
    "country_continent": (
        "country", "locatedIn", "continent",
        """
        SELECT DISTINCT ?country ?countryLabel ?continent ?continentLabel WHERE {
          ?country wdt:P31 wd:Q6256 ;
                   wdt:P30 ?continent .
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 300
        """
    ),

    # ── 15. Turing Award recipients (gold seed) ───────────────────────────
    "turing_recipients": (
        "researcher", "wonAward", "award",
        """
        SELECT DISTINCT ?researcher ?researcherLabel ?employer ?employerLabel
                        ?country ?countryLabel ?field ?fieldLabel WHERE {
          ?researcher wdt:P166 wd:Q185667 .  # Turing Award
          OPTIONAL { ?researcher wdt:P108 ?employer }
          OPTIONAL { ?researcher wdt:P27  ?country }
          OPTIONAL { ?researcher wdt:P101 ?field }
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
        }
        LIMIT 500
        """
    ),
}

# ---------------------------------------------------------------------------
# Predicate name → EX URI mapping
# ---------------------------------------------------------------------------
PRED_MAP = {
    "worksAt":     EX.worksAt,
    "nationality": EX.nationality,
    "studiedAt":   EX.studiedAt,
    "wonAward":    EX.wonAward,
    "supervisedBy":EX.supervisedBy,
    "bornIn":      EX.bornIn,
    "hasField":    EX.hasField,
    "givenBy":     EX.givenBy,
    "locatedIn":   EX.locatedIn,
    "authoredBy":  EX.authoredBy,
    "publishedIn": EX.publishedIn,
    "partOf":      EX.partOf,
}

# Entity type inference from role name
ROLE_TYPE_MAP = {
    "researcher":  EX.Researcher,
    "employer":    EX.Organization,
    "university":  EX.University,
    "award":       EX.Award,
    "advisor":     EX.Researcher,
    "place":       EX.Place,
    "country":     EX.Country,
    "field":       EX.ResearchField,
    "org":         EX.Organization,
    "company":     EX.Company,
    "publication": EX.Publication,
    "author":      EX.Researcher,
    "venue":       EX.Venue,
    "lab":         EX.ResearchLab,
    "parent":      EX.Organization,
    "continent":   EX.Continent,
}

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
DELAY = 2.0   # seconds between queries (be polite to Wikidata)


# ---------------------------------------------------------------------------
# SPARQL client
# ---------------------------------------------------------------------------

def make_sparql() -> SPARQLWrapper:
    s = SPARQLWrapper(WIKIDATA_ENDPOINT)
    s.addCustomHttpHeader(
        "User-Agent",
        "StudentKGBulkExpander/1.0 (academic project; educational use only)"
    )
    s.setReturnFormat(JSON)
    return s


def run_query(sparql: SPARQLWrapper, query: str, name: str) -> list[dict]:
    sparql.setQuery(query)
    try:
        res = sparql.query().convert()
        rows = res["results"]["bindings"]
        logger.info(f"  [{name}] → {len(rows)} rows")
        return rows
    except Exception as e:
        logger.warning(f"  [{name}] FAILED: {e}")
        return []


# ---------------------------------------------------------------------------
# Convert rows → RDF triples
# ---------------------------------------------------------------------------

def slug(uri: str) -> str:
    """Extract last path segment, replace spaces/hyphens."""
    part = uri.rstrip("/").split("/")[-1]
    return part.replace(" ", "_").replace("-", "_")


def qid(uri: str) -> str | None:
    """Return Wikidata QID if URI is a Wikidata entity, else None."""
    s = slug(uri)
    return s if s.startswith("Q") else None


def rows_to_triples(
    rows: list[dict],
    subj_var: str,
    pred_name: str,
    obj_var: str,
    g: Graph,
) -> int:
    """Convert SPARQL result rows to RDF triples added to g."""
    pred_uri = PRED_MAP.get(pred_name, EX[pred_name])
    subj_type = ROLE_TYPE_MAP.get(subj_var)
    obj_type  = ROLE_TYPE_MAP.get(obj_var)

    added = 0
    subj_label_var = subj_var + "Label"
    obj_label_var  = obj_var  + "Label"

    for row in rows:
        subj_data = row.get(subj_var, {})
        obj_data  = row.get(obj_var,  {})

        if not subj_data or not obj_data:
            continue

        subj_val = subj_data.get("value", "")
        obj_val  = obj_data.get("value",  "")

        if not subj_val or not obj_val:
            continue

        # Build subject URI
        subj_qid = qid(subj_val)
        if subj_qid:
            subj_uri = EX[subj_qid]
            g.add((subj_uri, OWL.sameAs, WD[subj_qid]))
        else:
            continue   # skip non-entity subjects

        # Build object URI or literal
        obj_qid = qid(obj_val)
        if obj_data.get("type") == "uri" and obj_qid:
            obj_uri = EX[obj_qid]
            g.add((subj_uri, pred_uri, obj_uri))
            g.add((obj_uri,  OWL.sameAs, WD[obj_qid]))
            added += 2

            # Type assertions
            if obj_type:
                g.add((obj_uri, RDF.type, obj_type))
                added += 1

            # Label for object
            obj_label = row.get(obj_label_var, {}).get("value", "")
            if obj_label:
                g.add((obj_uri, RDFS.label,  Literal(obj_label, lang="en")))
                g.add((obj_uri, EX.name,     Literal(obj_label)))
                added += 2
        elif obj_data.get("type") == "literal":
            lit_val  = obj_val
            dtype    = obj_data.get("datatype", "")
            if "date" in dtype:
                g.add((subj_uri, pred_uri, Literal(lit_val, datatype=XSD.date)))
            elif "integer" in dtype:
                try:
                    g.add((subj_uri, pred_uri, Literal(int(lit_val), datatype=XSD.integer)))
                except ValueError:
                    g.add((subj_uri, pred_uri, Literal(lit_val)))
            else:
                g.add((subj_uri, pred_uri, Literal(lit_val)))
            added += 1
        else:
            continue

        # Type for subject
        if subj_type:
            g.add((subj_uri, RDF.type, subj_type))
            added += 1

        # Label for subject
        subj_label = row.get(subj_label_var, {}).get("value", "")
        if subj_label:
            g.add((subj_uri, RDFS.label, Literal(subj_label, lang="en")))
            g.add((subj_uri, EX.name,    Literal(subj_label)))
            added += 2

    return added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    output_path  = Path("kg_artifacts/expanded_kg.nt")
    existing_path = Path("kg_artifacts/expanded_kg.nt")
    sample_path   = Path("kg_artifacts/sample_kg.ttl")

    # Load existing graph
    g = Graph()
    g.bind("ex",   EX)
    g.bind("owl",  OWL)
    g.bind("rdfs", RDFS)
    g.bind("wd",   WD)

    if existing_path.exists():
        logger.info(f"Loading existing KG: {existing_path}")
        g.parse(str(existing_path), format="ntriples")
        logger.info(f"  Loaded {len(g)} existing triples")
    elif sample_path.exists():
        logger.info(f"Loading sample KG: {sample_path}")
        g.parse(str(sample_path), format="turtle")
        logger.info(f"  Loaded {len(g)} sample triples")

    start_count = len(g)
    sparql = make_sparql()

    # Run all bulk queries
    for query_name, (subj_var, pred_name, obj_var, query_str) in BULK_QUERIES.items():
        logger.info(f"\n>>> Running: {query_name}")
        time.sleep(DELAY)
        rows = run_query(sparql, query_str.strip(), query_name)
        if rows:
            added = rows_to_triples(rows, subj_var, pred_name, obj_var, g)
            logger.info(f"  Added {added} triples — running total: {len(g)}")
        else:
            logger.warning(f"  No results for {query_name}, skipping.")

        # Save checkpoint every 3 queries
        if list(BULK_QUERIES.keys()).index(query_name) % 3 == 2:
            logger.info(f"  Checkpoint save → {output_path} ({len(g)} triples)")
            _save(g, output_path)

    # Final save
    _save(g, output_path)

    # ── Stats ──────────────────────────────────────────────────────────────
    subjects    = set(s for s, _, _ in g if not isinstance(s, BNode))
    predicates  = set(p for _, p, _ in g)
    objects_ent = set(o for _, _, o in g if isinstance(o, URIRef))

    print("\n" + "=" * 60)
    print("  EXPANDED KG STATISTICS")
    print("=" * 60)
    print(f"  Total triples   : {len(g):>10,}")
    print(f"  Unique subjects : {len(subjects):>10,}")
    print(f"  Unique predicates: {len(predicates):>9,}")
    print(f"  Unique obj URIs : {len(objects_ent):>10,}")
    print(f"  New triples added: {len(g) - start_count:>9,}")
    print(f"  Output file     : {output_path}")
    print("=" * 60)

    if len(g) < 50_000:
        print(f"\n⚠️  Only {len(g):,} triples — target is 50,000+")
        print("   Wikidata may have rate-limited some queries.")
        print("   Re-run the script to add more triples.")
    else:
        print(f"\n✅ Target reached: {len(g):,} triples (≥ 50,000)")


def _save(g: Graph, path: Path):
    """Save graph to N-Triples, filtering blank nodes."""
    clean = Graph()
    clean.bind("ex",   EX)
    clean.bind("owl",  OWL)
    clean.bind("rdfs", RDFS)
    clean.bind("wd",   WD)
    for s, p, o in g:
        if isinstance(s, BNode) or isinstance(o, BNode):
            continue
        clean.add((s, p, o))
    path.parent.mkdir(parents=True, exist_ok=True)
    clean.serialize(destination=str(path), format="ntriples")
    logger.info(f"  Saved {len(clean):,} triples → {path}")


if __name__ == "__main__":
    main()
