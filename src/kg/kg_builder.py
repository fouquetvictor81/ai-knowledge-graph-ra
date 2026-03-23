"""
RDF Knowledge Graph Builder
============================
Builds an RDF Knowledge Graph from entity/triple CSVs and the sample KG,
using a custom ontology namespace for the AI Researchers domain.

Outputs:
    kg_artifacts/ontology.ttl    — OWL ontology
    kg_artifacts/initial_kg.nt   — N-Triples KG

Usage:
    python src/kg/kg_builder.py
"""

import csv
import logging
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
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
# Namespaces
# ---------------------------------------------------------------------------

EX = Namespace("https://aikg.example.org/")
WD = Namespace("http://www.wikidata.org/entity/")
SCHEMA = Namespace("https://schema.org/")

# ---------------------------------------------------------------------------
# Known researcher → Wikidata QID mappings
# ---------------------------------------------------------------------------

RESEARCHER_WIKIDATA = {
    "Yann LeCun": "Q1330151",
    "Geoffrey Hinton": "Q92743",
    "Yoshua Bengio": "Q55683",
    "Andrew Ng": "Q2678895",
    "Fei-Fei Li": "Q57389",
    "Jürgen Schmidhuber": "Q92744",
    "Ian Goodfellow": "Q28140906",
    "Ilya Sutskever": "Q28485575",
    "Demis Hassabis": "Q562141",
    "Oriol Vinyals": "Q55432168",
    "Sam Altman": "Q27543895",
    "Pieter Abbeel": "Q48402538",
    "Judea Pearl": "Q92722",
    "Stuart Russell": "Q92748",
    "Peter Norvig": "Q1969226",
    "Michael I. Jordan": "Q92745",
    "Bernhard Schölkopf": "Q92739",
    "Zoubin Ghahramani": "Q92734",
    "David Silver": "Q29049366",
    "Alex Krizhevsky": "Q28485551",
    "Hugo Larochelle": "Q55683011",
    "Kyunghyun Cho": "Q29575183",
    "Greg Brockman": "Q28485580",
    "Sergey Levine": "Q51601869",
    "Chelsea Finn": "Q61051882",
    "Nando de Freitas": "Q7404",
}

# Known organization → Wikidata QID mappings
ORG_WIKIDATA = {
    "Meta AI": "Q95075",
    "DeepMind": "Q15733006",
    "Google DeepMind": "Q15733006",
    "OpenAI": "Q21708200",
    "Mila": "Q17083398",
    "Stanford University": "Q41506",
    "MIT": "Q49108",
    "Carnegie Mellon University": "Q190080",
    "University of California, Berkeley": "Q168719",
    "University of Toronto": "Q180865",
    "Google Brain": "Q4849038",
    "Microsoft Research": "Q1575037",
    "New York University": "Q49115",
    "Montreal Institute for Learning Algorithms": "Q17083398",
}


# ---------------------------------------------------------------------------
# Helper: URI from label
# ---------------------------------------------------------------------------


def label_to_uri(label: str, namespace: Namespace) -> URIRef:
    """Convert a text label to a URI-safe identifier."""
    safe = (
        label.strip()
        .replace(" ", "_")
        .replace(",", "")
        .replace(".", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace("'", "")
        .replace('"', "")
        .replace("ü", "u")
        .replace("é", "e")
        .replace("ö", "o")
        .replace("ä", "a")
        .replace("ñ", "n")
        .replace("ş", "s")
        .replace("ı", "i")
    )
    return namespace[safe]


# ---------------------------------------------------------------------------
# Build the ontology graph
# ---------------------------------------------------------------------------


def build_ontology() -> Graph:
    """
    Build the OWL ontology for the AI Researchers domain.
    Returns an rdflib Graph containing all ontology axioms.
    """
    g = Graph()
    g.bind("ex", EX)
    g.bind("owl", OWL)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)
    g.bind("schema", SCHEMA)
    g.bind("wd", WD)

    # -----------------------------------------------------------------------
    # Ontology declaration
    # -----------------------------------------------------------------------
    onto_uri = EX["AIResearchersOntology"]
    g.add((onto_uri, RDF.type, OWL.Ontology))
    g.add((onto_uri, RDFS.label, Literal("AI Researchers Knowledge Graph Ontology")))
    g.add((onto_uri, RDFS.comment, Literal(
        "Ontology for representing scientific researchers in Artificial Intelligence, "
        "their affiliations, awards, publications and collaborations."
    )))

    # -----------------------------------------------------------------------
    # Classes
    # -----------------------------------------------------------------------
    classes = {
        "Researcher": "A person who conducts scientific research in Artificial Intelligence.",
        "Organization": "An institution, company or research group.",
        "University": "An academic institution offering higher education.",
        "ResearchLab": "A research laboratory or institute.",
        "Company": "A commercial company involved in AI research.",
        "Country": "A nation or sovereign state.",
        "Award": "A prize or honor given for achievements.",
        "Publication": "A scientific paper, book or technical report.",
        "Conference": "An academic conference or symposium.",
        "Dataset": "A collection of data used for training or evaluation.",
    }

    for cls_name, comment in classes.items():
        cls_uri = EX[cls_name]
        g.add((cls_uri, RDF.type, OWL.Class))
        g.add((cls_uri, RDFS.label, Literal(cls_name)))
        g.add((cls_uri, RDFS.comment, Literal(comment)))

    # Subclass relationships
    g.add((EX.University, RDFS.subClassOf, EX.Organization))
    g.add((EX.ResearchLab, RDFS.subClassOf, EX.Organization))
    g.add((EX.Company, RDFS.subClassOf, EX.Organization))

    # -----------------------------------------------------------------------
    # Object Properties
    # -----------------------------------------------------------------------
    obj_props = [
        ("worksAt",        "The organization where a researcher is currently employed.",
         EX.Researcher, EX.Organization),
        ("locatedIn",      "The country or city where an organization is located.",
         EX.Organization, EX.Country),
        ("wonAward",       "An award received by a researcher.",
         EX.Researcher, EX.Award),
        ("authorOf",       "A publication authored by a researcher.",
         EX.Researcher, EX.Publication),
        ("collaboratesWith", "A researcher who collaborates with another researcher.",
         EX.Researcher, EX.Researcher),
        ("bornIn",         "The country or city where a researcher was born.",
         EX.Researcher, EX.Country),
        ("studiedAt",      "An educational institution where a researcher studied.",
         EX.Researcher, EX.University),
        ("supervisedBy",   "The academic supervisor of a researcher.",
         EX.Researcher, EX.Researcher),
        ("memberOf",       "A group or organization a researcher is a member of.",
         EX.Researcher, EX.Organization),
        ("affiliatedWith", "An organization with which a researcher is affiliated.",
         EX.Researcher, EX.Organization),
        ("publishedAt",    "The conference or venue where a work was published.",
         EX.Publication, EX.Conference),
        ("fieldOfStudy",   "A research field the researcher works in.",
         EX.Researcher, None),
    ]

    for prop_name, comment, domain, range_ in obj_props:
        prop_uri = EX[prop_name]
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))
        g.add((prop_uri, RDFS.label, Literal(prop_name)))
        g.add((prop_uri, RDFS.comment, Literal(comment)))
        if domain:
            g.add((prop_uri, RDFS.domain, domain))
        if range_:
            g.add((prop_uri, RDFS.range, range_))

    # Inverse properties
    g.add((EX.supervisedBy, OWL.inverseOf, EX.supervised))
    g.add((EX.collaboratesWith, RDF.type, OWL.SymmetricProperty))

    # -----------------------------------------------------------------------
    # Data Properties
    # -----------------------------------------------------------------------
    data_props = [
        ("name",          "Full name of the researcher or organization.", XSD.string),
        ("birthDate",     "Date of birth.", XSD.date),
        ("hIndex",        "h-index bibliometric indicator.", XSD.integer),
        ("citationCount", "Total number of citations.", XSD.integer),
        ("email",         "Email address.", XSD.string),
        ("homepage",      "Personal or organizational homepage URL.", XSD.anyURI),
        ("nationality",   "Nationality of the researcher.", XSD.string),
        ("alias",         "Alternative name or abbreviation.", XSD.string),
    ]

    for prop_name, comment, dtype in data_props:
        prop_uri = EX[prop_name]
        g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
        g.add((prop_uri, RDFS.label, Literal(prop_name)))
        g.add((prop_uri, RDFS.comment, Literal(comment)))
        g.add((prop_uri, RDFS.range, dtype))

    logger.info(f"Ontology built: {len(g)} triples")
    return g


# ---------------------------------------------------------------------------
# Build the initial KG from CSV data + sample data
# ---------------------------------------------------------------------------


def build_kg_from_csv(entities_csv: Path, triples_csv: Path) -> Graph:
    """
    Build an RDF graph from extracted entity/triple CSVs.
    """
    g = Graph()
    g.bind("ex", EX)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("wd", WD)

    entity_uri_map: dict[str, URIRef] = {}

    # -----------------------------------------------------------------------
    # Load entities
    # -----------------------------------------------------------------------
    if entities_csv.exists():
        with open(entities_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entity = row.get("entity", "").strip()
                etype = row.get("type", "").strip()
                if not entity or not etype:
                    continue

                uri = label_to_uri(entity, EX)
                entity_uri_map[entity.lower()] = uri

                # Map spaCy entity type to ontology class
                rdf_class = {
                    "PERSON": EX.Researcher,
                    "ORG": EX.Organization,
                    "GPE": EX.Country,
                    "DATE": None,
                    "WORK_OF_ART": EX.Publication,
                }.get(etype)

                if rdf_class:
                    g.add((uri, RDF.type, rdf_class))
                    g.add((uri, EX.name, Literal(entity)))
                    g.add((uri, RDFS.label, Literal(entity)))

                    # Add Wikidata sameAs for known researchers
                    if etype == "PERSON" and entity in RESEARCHER_WIKIDATA:
                        wd_uri = WD[RESEARCHER_WIKIDATA[entity]]
                        g.add((uri, OWL.sameAs, wd_uri))
                    elif etype == "ORG" and entity in ORG_WIKIDATA:
                        wd_uri = WD[ORG_WIKIDATA[entity]]
                        g.add((uri, OWL.sameAs, wd_uri))

        logger.info(f"Loaded {len(entity_uri_map)} entities from {entities_csv}")
    else:
        logger.warning(f"Entities CSV not found: {entities_csv}")

    # -----------------------------------------------------------------------
    # Load triples
    # -----------------------------------------------------------------------
    triples_added = 0
    if triples_csv.exists():
        predicate_map = {
            "work": EX.worksAt,
            "win": EX.wonAward,
            "receive": EX.wonAward,
            "publish": EX.authorOf,
            "found": EX.memberOf,
            "join": EX.memberOf,
            "lead": EX.worksAt,
            "supervise": EX.supervisedBy,
            "collaborate": EX.collaboratesWith,
            "study": EX.studiedAt,
            "teach": EX.worksAt,
            "develop": EX.authorOf,
            "create": EX.authorOf,
            "invent": EX.authorOf,
            "propose": EX.authorOf,
            "introduce": EX.authorOf,
        }

        with open(triples_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                subj_text = row.get("subject", "").strip()
                pred_text = row.get("predicate", "").strip()
                obj_text = row.get("object", "").strip()

                if not subj_text or not pred_text or not obj_text:
                    continue

                subj_uri = entity_uri_map.get(subj_text.lower()) or label_to_uri(subj_text, EX)
                obj_uri = entity_uri_map.get(obj_text.lower()) or label_to_uri(obj_text, EX)
                pred_uri = predicate_map.get(pred_text, EX[pred_text])

                g.add((subj_uri, pred_uri, obj_uri))
                triples_added += 1

        logger.info(f"Added {triples_added} triples from {triples_csv}")
    else:
        logger.warning(f"Triples CSV not found: {triples_csv}")

    return g


def add_hardcoded_researchers(g: Graph) -> Graph:
    """
    Add a core set of hardcoded researcher facts to ensure the KG
    has useful data even if the crawler/NER hasn't been run.
    """
    # Core researchers with their organizations
    core_data = [
        ("Yann LeCun",      EX.Researcher, EX.MetaAI,    "Q1330151"),
        ("Geoffrey Hinton", EX.Researcher, EX.Google,    "Q92743"),
        ("Yoshua Bengio",   EX.Researcher, EX.Mila,      "Q55683"),
        ("Andrew Ng",       EX.Researcher, EX.Stanford,  "Q2678895"),
        ("Fei-Fei Li",      EX.Researcher, EX.Stanford,  "Q57389"),
        ("Ian Goodfellow",  EX.Researcher, EX.Apple,     "Q28140906"),
        ("Ilya Sutskever",  EX.Researcher, EX.OpenAI,    "Q28485575"),
        ("Demis Hassabis",  EX.Researcher, EX.DeepMind,  "Q562141"),
        ("Sam Altman",      EX.Researcher, EX.OpenAI,    "Q27543895"),
        ("Pieter Abbeel",   EX.Researcher, EX.UC_Berkeley, "Q48402538"),
    ]

    for name, cls, org, qid in core_data:
        uri = label_to_uri(name, EX)
        g.add((uri, RDF.type, cls))
        g.add((uri, EX.name, Literal(name)))
        g.add((uri, RDFS.label, Literal(name)))
        g.add((uri, EX.worksAt, org))
        g.add((uri, OWL.sameAs, WD[qid]))

    # Organizations
    orgs = [
        (EX.MetaAI, EX.Company, "Meta AI", "Q95075"),
        (EX.Google, EX.Company, "Google", "Q95"),
        (EX.DeepMind, EX.Company, "Google DeepMind", "Q15733006"),
        (EX.OpenAI, EX.Company, "OpenAI", "Q21708200"),
        (EX.Mila, EX.ResearchLab, "Mila - Quebec AI Institute", "Q17083398"),
        (EX.Stanford, EX.University, "Stanford University", "Q41506"),
        (EX.MIT, EX.University, "MIT", "Q49108"),
        (EX.CMU, EX.University, "Carnegie Mellon University", "Q190080"),
        (EX.UC_Berkeley, EX.University, "UC Berkeley", "Q168719"),
        (EX.Apple, EX.Company, "Apple", "Q312"),
    ]

    for uri, cls, name, qid in orgs:
        g.add((uri, RDF.type, cls))
        g.add((uri, EX.name, Literal(name)))
        g.add((uri, RDFS.label, Literal(name)))
        g.add((uri, OWL.sameAs, WD[qid]))

    logger.info("Added hardcoded researcher/org data to KG")
    return g


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    entities_csv = Path("data/entities.csv")
    triples_csv = Path("data/triples.csv")
    output_dir = Path("kg_artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Build and save ontology
    # -----------------------------------------------------------------------
    logger.info("Building ontology...")
    onto_g = build_ontology()
    onto_path = output_dir / "ontology.ttl"
    onto_g.serialize(destination=str(onto_path), format="turtle")
    logger.info(f"Ontology saved: {onto_path} ({len(onto_g)} triples)")

    # -----------------------------------------------------------------------
    # Build KG
    # -----------------------------------------------------------------------
    logger.info("Building initial knowledge graph...")
    kg = Graph()
    kg.bind("ex", EX)
    kg.bind("owl", OWL)
    kg.bind("rdfs", RDFS)
    kg.bind("wd", WD)

    # Add ontology import reference
    kg.add((EX["AIResearchersKG"], RDF.type, OWL.Ontology))
    kg.add((EX["AIResearchersKG"], OWL.imports, EX["AIResearchersOntology"]))

    # Load from CSVs (if available)
    csv_graph = build_kg_from_csv(entities_csv, triples_csv)
    for triple in csv_graph:
        kg.add(triple)

    # Always add hardcoded core data
    add_hardcoded_researchers(kg)

    # -----------------------------------------------------------------------
    # Save initial KG
    # -----------------------------------------------------------------------
    kg_path = output_dir / "initial_kg.nt"
    kg.serialize(destination=str(kg_path), format="ntriples")
    logger.info(f"Initial KG saved: {kg_path}")

    # Print statistics
    print("\n=== KG Statistics ===")
    print(f"Total triples  : {len(kg)}")
    print(f"Researchers    : {len(list(kg.subjects(RDF.type, EX.Researcher)))}")
    print(f"Organizations  : {len(list(kg.subjects(RDF.type, EX.Organization))) + len(list(kg.subjects(RDF.type, EX.University))) + len(list(kg.subjects(RDF.type, EX.Company)))}")
    print(f"Publications   : {len(list(kg.subjects(RDF.type, EX.Publication)))}")
    print(f"sameAs links   : {len(list(kg.triples((None, OWL.sameAs, None))))}")


if __name__ == "__main__":
    main()
