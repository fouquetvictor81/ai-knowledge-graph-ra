"""
Entity Alignment with Wikidata
================================
For each researcher/organization in our KG, queries the Wikidata SPARQL
endpoint to find matching entities and adds owl:sameAs links.

Also maps our object properties to Wikidata properties.

Output:
    kg_artifacts/alignment.ttl

Usage:
    python src/kg/entity_aligner.py
"""

import logging
import time
from pathlib import Path
from typing import Optional

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
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")

# ---------------------------------------------------------------------------
# Wikidata SPARQL endpoint
# ---------------------------------------------------------------------------

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
REQUEST_DELAY = 2.0  # Be polite to Wikidata

# ---------------------------------------------------------------------------
# Known researcher → Wikidata QID (pre-populated to avoid excessive queries)
# ---------------------------------------------------------------------------

KNOWN_ALIGNMENTS = {
    # Researchers
    "Yann_LeCun":          ("Q1330151", 0.99),
    "Geoffrey_Hinton":     ("Q92743",   0.99),
    "Yoshua_Bengio":       ("Q55683",   0.99),
    "Andrew_Ng":           ("Q2678895", 0.99),
    "Fei-Fei_Li":          ("Q57389",   0.99),
    "Jurgen_Schmidhuber":  ("Q92744",   0.99),
    "Ian_Goodfellow":      ("Q28140906", 0.99),
    "Ilya_Sutskever":      ("Q28485575", 0.99),
    "Demis_Hassabis":      ("Q562141",  0.99),
    "Oriol_Vinyals":       ("Q55432168", 0.95),
    "Sam_Altman":          ("Q27543895", 0.99),
    "Greg_Brockman":       ("Q28485580", 0.95),
    "Pieter_Abbeel":       ("Q48402538", 0.95),
    "Judea_Pearl":         ("Q92722",   0.99),
    "Stuart_Russell":      ("Q92748",   0.99),
    "Peter_Norvig":        ("Q1969226", 0.99),
    "Michael_I_Jordan":    ("Q92745",   0.99),
    "Bernhard_Scholkopf":  ("Q92739",   0.99),
    "Zoubin_Ghahramani":   ("Q92734",   0.99),
    "David_Silver":        ("Q29049366", 0.95),
    "Alex_Krizhevsky":     ("Q28485551", 0.95),
    "Hugo_Larochelle":     ("Q55683011", 0.90),
    "Kyunghyun_Cho":       ("Q29575183", 0.90),
    "Sergey_Levine":       ("Q51601869", 0.90),
    "Chelsea_Finn":        ("Q61051882", 0.90),
    "Nando_de_Freitas":    ("Q7404",    0.90),
    # Organizations
    "MetaAI":              ("Q95075",   0.99),
    "DeepMind":            ("Q15733006", 0.99),
    "OpenAI":              ("Q21708200", 0.99),
    "Mila":                ("Q17083398", 0.99),
    "Stanford":            ("Q41506",   0.99),
    "MIT":                 ("Q49108",   0.99),
    "CMU":                 ("Q190080",  0.99),
    "UC_Berkeley":         ("Q168719",  0.99),
    "Google":              ("Q95",      0.99),
    "Apple":               ("Q312",     0.99),
    # Awards
    "Turing_Award":        ("Q185667",  0.99),
    # Publications
    "AlexNet":             ("Q15733308", 0.95),
    "Generative_adversarial_network": ("Q21246376", 0.90),
    "BERT":                ("Q59667175", 0.90),
    "Attention_is_All_You_Need": ("Q59670178", 0.90),
    "AlphaGo":             ("Q20983027", 0.99),
}

# ---------------------------------------------------------------------------
# Property equivalence mappings (our property → Wikidata property)
# ---------------------------------------------------------------------------

PROPERTY_ALIGNMENTS = {
    EX.worksAt:          WDT["P108"],   # employer
    EX.locatedIn:        WDT["P17"],    # country
    EX.wonAward:         WDT["P166"],   # award received
    EX.authorOf:         WDT["P800"],   # notable work
    EX.bornIn:           WDT["P19"],    # place of birth
    EX.studiedAt:        WDT["P69"],    # educated at
    EX.supervisedBy:     WDT["P184"],   # doctoral advisor
    EX.collaboratesWith: WDT["P1066"],  # student/collaborator (approximate)
    EX.memberOf:         WDT["P463"],   # member of
    EX.name:             WDT["P1705"],  # native label
    EX.birthDate:        WDT["P569"],   # date of birth
    EX.hIndex:           WDT["P1929"],  # h-index
    EX.citationCount:    WDT["P6889"],  # total citations
    EX.homepage:         WDT["P856"],   # official website
}


# ---------------------------------------------------------------------------
# Wikidata SPARQL querier
# ---------------------------------------------------------------------------


def query_wikidata_for_person(name: str) -> Optional[str]:
    """
    Query Wikidata SPARQL for a researcher by name.
    Returns Wikidata QID string (e.g., "Q1330151") or None.
    """
    sparql = SPARQLWrapper(WIKIDATA_SPARQL)
    sparql.addCustomHttpHeader("User-Agent", "AcademicAligner/1.0 (student project; educational)")

    query = f"""
    SELECT ?item ?itemLabel WHERE {{
      ?item wdt:P31 wd:Q5 .
      ?item rdfs:label "{name}"@en .
      ?item wdt:P101 wd:Q11660 .  # field of work: artificial intelligence
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 3
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        if bindings:
            uri = bindings[0]["item"]["value"]
            qid = uri.split("/")[-1]
            return qid
    except Exception as e:
        logger.warning(f"Wikidata query failed for '{name}': {e}")

    return None


# ---------------------------------------------------------------------------
# Build alignment graph
# ---------------------------------------------------------------------------


def build_alignment_graph(kg_path: Optional[Path] = None) -> Graph:
    """
    Build the alignment RDF graph with owl:sameAs and owl:equivalentProperty triples.
    """
    g = Graph()
    g.bind("ex", EX)
    g.bind("owl", OWL)
    g.bind("wd", WD)
    g.bind("wdt", WDT)
    g.bind("rdfs", RDFS)
    g.bind("skos", SKOS)
    g.bind("xsd", XSD)

    # -----------------------------------------------------------------------
    # Add known entity alignments (owl:sameAs) with confidence annotations
    # -----------------------------------------------------------------------
    logger.info("Adding entity alignments (owl:sameAs)...")
    for local_name, (qid, confidence) in KNOWN_ALIGNMENTS.items():
        local_uri = EX[local_name]
        wd_uri = WD[qid]

        # owl:sameAs triple
        g.add((local_uri, OWL.sameAs, wd_uri))

        # Reified statement with confidence score
        reif_uri = EX[f"alignment_{local_name}_{qid}"]
        g.add((reif_uri, RDF.type, RDF.Statement))
        g.add((reif_uri, RDF.subject, local_uri))
        g.add((reif_uri, RDF.predicate, OWL.sameAs))
        g.add((reif_uri, RDF.object, wd_uri))
        g.add((reif_uri, EX.confidence, Literal(confidence, datatype=XSD.decimal)))
        g.add((reif_uri, RDFS.comment, Literal(
            f"Alignment between {local_name} and Wikidata entity {qid} "
            f"with confidence {confidence}"
        )))

    # -----------------------------------------------------------------------
    # Add equivalentProperty mappings
    # -----------------------------------------------------------------------
    logger.info("Adding property equivalences (owl:equivalentProperty)...")
    for our_prop, wd_prop in PROPERTY_ALIGNMENTS.items():
        g.add((our_prop, OWL.equivalentProperty, wd_prop))
        g.add((our_prop, SKOS.closeMatch, wd_prop))

    # -----------------------------------------------------------------------
    # Try to query Wikidata for additional alignments from the KG
    # (only if a KG file is provided and we haven't pre-filled the alignment)
    # -----------------------------------------------------------------------
    if kg_path and kg_path.exists():
        logger.info(f"Checking KG for additional entities to align: {kg_path}")
        kg = Graph()
        if str(kg_path).endswith(".nt"):
            kg.parse(str(kg_path), format="ntriples")
        else:
            kg.parse(str(kg_path), format="turtle")

        # Get all PERSON entities not already aligned
        already_aligned = {EX[local] for local in KNOWN_ALIGNMENTS}
        all_researchers = set(kg.subjects(RDF.type, EX.Researcher))
        unaligned = all_researchers - already_aligned

        logger.info(f"Found {len(unaligned)} potentially unaligned researchers")

        for entity_uri in list(unaligned)[:10]:  # Limit Wikidata queries
            # Get the label
            label_triples = list(kg.objects(entity_uri, RDFS.label))
            if not label_triples:
                continue
            label = str(label_triples[0])

            logger.info(f"Querying Wikidata for: {label}")
            time.sleep(REQUEST_DELAY)

            qid = query_wikidata_for_person(label)
            if qid:
                wd_uri = WD[qid]
                g.add((entity_uri, OWL.sameAs, wd_uri))
                confidence = 0.80  # Lower confidence for auto-aligned
                local_name = str(entity_uri).split("/")[-1]
                reif_uri = EX[f"alignment_{local_name}_{qid}_auto"]
                g.add((reif_uri, RDF.type, RDF.Statement))
                g.add((reif_uri, RDF.subject, entity_uri))
                g.add((reif_uri, RDF.predicate, OWL.sameAs))
                g.add((reif_uri, RDF.object, wd_uri))
                g.add((reif_uri, EX.confidence, Literal(confidence, datatype=XSD.decimal)))
                g.add((reif_uri, RDFS.comment, Literal(f"Auto-aligned via SPARQL query")))
                logger.info(f"  Aligned: {label} → {qid}")
            else:
                logger.info(f"  No Wikidata match found for: {label}")

    logger.info(f"Alignment graph built: {len(g)} triples")
    return g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    output_path = Path("kg_artifacts/alignment.ttl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try to load the initial KG for additional entity discovery
    kg_path = Path("kg_artifacts/initial_kg.nt")
    if not kg_path.exists():
        logger.warning(f"Initial KG not found at {kg_path}. Run kg_builder.py first.")
        kg_path = None

    logger.info("Building alignment graph...")
    alignment_g = build_alignment_graph(kg_path)

    alignment_g.serialize(destination=str(output_path), format="turtle")
    logger.info(f"Alignment saved: {output_path}")

    # Summary
    same_as_count = len(list(alignment_g.triples((None, OWL.sameAs, None))))
    equiv_prop_count = len(list(alignment_g.triples((None, OWL.equivalentProperty, None))))
    print(f"\n=== Alignment Summary ===")
    print(f"owl:sameAs links         : {same_as_count}")
    print(f"owl:equivalentProperty   : {equiv_prop_count}")
    print(f"Total alignment triples  : {len(alignment_g)}")


if __name__ == "__main__":
    main()
