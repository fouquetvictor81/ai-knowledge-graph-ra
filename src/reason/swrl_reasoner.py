"""
SWRL Reasoning with OWLReady2
==============================
Demonstrates OWL/SWRL reasoning on a small family ontology
and then applies custom rules to the AI Researchers KG.

Rules demonstrated:
    1. Persons older than 60 → classified as oldPerson
    2. Researchers who wonAward AND worksAt TopUniversity → Influential

Usage:
    python src/reason/swrl_reasoner.py
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

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
# Minimal family OWL in RDF/XML format (for OWLReady2 compatibility)
# ---------------------------------------------------------------------------

FAMILY_OWL_CONTENT = """<?xml version="1.0"?>
<Ontology xmlns="http://www.w3.org/2002/07/owl#"
          xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
          xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
          xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
          xmlns:swrl="http://www.w3.org/2003/11/swrl#"
          xmlns:swrlb="http://www.w3.org/2003/11/swrlb#"
          xml:base="http://family.example.org/"
          ontologyIRI="http://family.example.org/">

  <!-- Classes -->
  <Declaration><Class IRI="#Person"/></Declaration>
  <Declaration><Class IRI="#OldPerson"/></Declaration>
  <Declaration><Class IRI="#Child"/></Declaration>
  <Declaration><Class IRI="#Parent"/></Declaration>

  <!-- Object Properties -->
  <Declaration><ObjectProperty IRI="#hasParent"/></Declaration>
  <Declaration><ObjectProperty IRI="#hasChild"/></Declaration>
  <Declaration><ObjectProperty IRI="#hasGrandparent"/></Declaration>

  <!-- Data Properties -->
  <Declaration><DataProperty IRI="#hasAge"/></Declaration>
  <Declaration><DataProperty IRI="#hasName"/></Declaration>

  <!-- Class Axioms -->
  <SubClassOf>
    <Class IRI="#OldPerson"/>
    <Class IRI="#Person"/>
  </SubClassOf>
  <SubClassOf>
    <Class IRI="#Child"/>
    <Class IRI="#Person"/>
  </SubClassOf>
  <SubClassOf>
    <Class IRI="#Parent"/>
    <Class IRI="#Person"/>
  </SubClassOf>

  <!-- Inverse -->
  <InverseObjectProperties>
    <ObjectProperty IRI="#hasParent"/>
    <ObjectProperty IRI="#hasChild"/>
  </InverseObjectProperties>

  <!-- Individuals -->
  <Declaration><NamedIndividual IRI="#Alice"/></Declaration>
  <ClassAssertion><Class IRI="#Person"/><NamedIndividual IRI="#Alice"/></ClassAssertion>
  <DataPropertyAssertion>
    <DataProperty IRI="#hasAge"/>
    <NamedIndividual IRI="#Alice"/>
    <Literal datatypeIRI="http://www.w3.org/2001/XMLSchema#integer">72</Literal>
  </DataPropertyAssertion>
  <DataPropertyAssertion>
    <DataProperty IRI="#hasName"/>
    <NamedIndividual IRI="#Alice"/>
    <Literal>Alice Smith</Literal>
  </DataPropertyAssertion>

  <Declaration><NamedIndividual IRI="#Bob"/></Declaration>
  <ClassAssertion><Class IRI="#Person"/><NamedIndividual IRI="#Bob"/></ClassAssertion>
  <DataPropertyAssertion>
    <DataProperty IRI="#hasAge"/>
    <NamedIndividual IRI="#Bob"/>
    <Literal datatypeIRI="http://www.w3.org/2001/XMLSchema#integer">35</Literal>
  </DataPropertyAssertion>
  <DataPropertyAssertion>
    <DataProperty IRI="#hasName"/>
    <NamedIndividual IRI="#Bob"/>
    <Literal>Bob Jones</Literal>
  </DataPropertyAssertion>

  <Declaration><NamedIndividual IRI="#Carol"/></Declaration>
  <ClassAssertion><Class IRI="#Person"/><NamedIndividual IRI="#Carol"/></ClassAssertion>
  <DataPropertyAssertion>
    <DataProperty IRI="#hasAge"/>
    <NamedIndividual IRI="#Carol"/>
    <Literal datatypeIRI="http://www.w3.org/2001/XMLSchema#integer">65</Literal>
  </DataPropertyAssertion>

  <ObjectPropertyAssertion>
    <ObjectProperty IRI="#hasParent"/>
    <NamedIndividual IRI="#Bob"/>
    <NamedIndividual IRI="#Alice"/>
  </ObjectPropertyAssertion>

  <ObjectPropertyAssertion>
    <ObjectProperty IRI="#hasParent"/>
    <NamedIndividual IRI="#Bob"/>
    <NamedIndividual IRI="#Carol"/>
  </ObjectPropertyAssertion>

</Ontology>
"""

# ---------------------------------------------------------------------------
# AI Researchers OWL (for SWRL rule demo)
# ---------------------------------------------------------------------------

AI_RESEARCHERS_OWL = """<?xml version="1.0"?>
<Ontology xmlns="http://www.w3.org/2002/07/owl#"
          xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
          xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
          xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
          xml:base="https://aikg.example.org/"
          ontologyIRI="https://aikg.example.org/">

  <!-- Classes -->
  <Declaration><Class IRI="https://aikg.example.org/Researcher"/></Declaration>
  <Declaration><Class IRI="https://aikg.example.org/Organization"/></Declaration>
  <Declaration><Class IRI="https://aikg.example.org/Award"/></Declaration>
  <Declaration><Class IRI="https://aikg.example.org/TopUniversity"/></Declaration>
  <Declaration><Class IRI="https://aikg.example.org/Influential"/></Declaration>

  <SubClassOf>
    <Class IRI="https://aikg.example.org/TopUniversity"/>
    <Class IRI="https://aikg.example.org/Organization"/>
  </SubClassOf>
  <SubClassOf>
    <Class IRI="https://aikg.example.org/Influential"/>
    <Class IRI="https://aikg.example.org/Researcher"/>
  </SubClassOf>

  <!-- Object Properties -->
  <Declaration><ObjectProperty IRI="https://aikg.example.org/worksAt"/></Declaration>
  <Declaration><ObjectProperty IRI="https://aikg.example.org/wonAward"/></Declaration>
  <Declaration><ObjectProperty IRI="https://aikg.example.org/collaboratesWith"/></Declaration>
  <Declaration><ObjectProperty IRI="https://aikg.example.org/supervisedBy"/></Declaration>

  <!-- Data Properties -->
  <Declaration><DataProperty IRI="https://aikg.example.org/name"/></Declaration>
  <Declaration><DataProperty IRI="https://aikg.example.org/hIndex"/></Declaration>
  <Declaration><DataProperty IRI="https://aikg.example.org/citationCount"/></Declaration>

  <!-- Top Universities -->
  <Declaration><NamedIndividual IRI="https://aikg.example.org/MIT"/></Declaration>
  <ClassAssertion>
    <Class IRI="https://aikg.example.org/TopUniversity"/>
    <NamedIndividual IRI="https://aikg.example.org/MIT"/>
  </ClassAssertion>
  <DataPropertyAssertion>
    <DataProperty IRI="https://aikg.example.org/name"/>
    <NamedIndividual IRI="https://aikg.example.org/MIT"/>
    <Literal>MIT</Literal>
  </DataPropertyAssertion>

  <Declaration><NamedIndividual IRI="https://aikg.example.org/Stanford"/></Declaration>
  <ClassAssertion>
    <Class IRI="https://aikg.example.org/TopUniversity"/>
    <NamedIndividual IRI="https://aikg.example.org/Stanford"/>
  </ClassAssertion>

  <Declaration><NamedIndividual IRI="https://aikg.example.org/UC_Berkeley"/></Declaration>
  <ClassAssertion>
    <Class IRI="https://aikg.example.org/TopUniversity"/>
    <NamedIndividual IRI="https://aikg.example.org/UC_Berkeley"/>
  </ClassAssertion>

  <!-- Turing Award -->
  <Declaration><NamedIndividual IRI="https://aikg.example.org/TuringAward"/></Declaration>
  <ClassAssertion>
    <Class IRI="https://aikg.example.org/Award"/>
    <NamedIndividual IRI="https://aikg.example.org/TuringAward"/>
  </ClassAssertion>
  <DataPropertyAssertion>
    <DataProperty IRI="https://aikg.example.org/name"/>
    <NamedIndividual IRI="https://aikg.example.org/TuringAward"/>
    <Literal>Turing Award</Literal>
  </DataPropertyAssertion>

  <!-- Researchers -->
  <Declaration><NamedIndividual IRI="https://aikg.example.org/Yann_LeCun"/></Declaration>
  <ClassAssertion>
    <Class IRI="https://aikg.example.org/Researcher"/>
    <NamedIndividual IRI="https://aikg.example.org/Yann_LeCun"/>
  </ClassAssertion>
  <DataPropertyAssertion>
    <DataProperty IRI="https://aikg.example.org/name"/>
    <NamedIndividual IRI="https://aikg.example.org/Yann_LeCun"/>
    <Literal>Yann LeCun</Literal>
  </DataPropertyAssertion>
  <ObjectPropertyAssertion>
    <ObjectProperty IRI="https://aikg.example.org/wonAward"/>
    <NamedIndividual IRI="https://aikg.example.org/Yann_LeCun"/>
    <NamedIndividual IRI="https://aikg.example.org/TuringAward"/>
  </ObjectPropertyAssertion>

  <Declaration><NamedIndividual IRI="https://aikg.example.org/Andrew_Ng"/></Declaration>
  <ClassAssertion>
    <Class IRI="https://aikg.example.org/Researcher"/>
    <NamedIndividual IRI="https://aikg.example.org/Andrew_Ng"/>
  </ClassAssertion>
  <DataPropertyAssertion>
    <DataProperty IRI="https://aikg.example.org/name"/>
    <NamedIndividual IRI="https://aikg.example.org/Andrew_Ng"/>
    <Literal>Andrew Ng</Literal>
  </DataPropertyAssertion>
  <ObjectPropertyAssertion>
    <ObjectProperty IRI="https://aikg.example.org/worksAt"/>
    <NamedIndividual IRI="https://aikg.example.org/Andrew_Ng"/>
    <NamedIndividual IRI="https://aikg.example.org/Stanford"/>
  </ObjectPropertyAssertion>

  <Declaration><NamedIndividual IRI="https://aikg.example.org/Geoffrey_Hinton"/></Declaration>
  <ClassAssertion>
    <Class IRI="https://aikg.example.org/Researcher"/>
    <NamedIndividual IRI="https://aikg.example.org/Geoffrey_Hinton"/>
  </ClassAssertion>
  <ObjectPropertyAssertion>
    <ObjectProperty IRI="https://aikg.example.org/wonAward"/>
    <NamedIndividual IRI="https://aikg.example.org/Geoffrey_Hinton"/>
    <NamedIndividual IRI="https://aikg.example.org/TuringAward"/>
  </ObjectPropertyAssertion>

</Ontology>
"""


# ---------------------------------------------------------------------------
# Run reasoning
# ---------------------------------------------------------------------------


def demo_family_reasoning():
    """
    Demonstrate SWRL-style reasoning on the family ontology.
    Rule: Person with age > 60 → OldPerson
    """
    try:
        import owlready2
        from owlready2 import get_ontology, sync_reasoner_pellet
    except ImportError:
        logger.error("owlready2 not installed. Run: pip install owlready2")
        return

    logger.info("=== Family Ontology Reasoning Demo ===")

    # Write OWL to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".owl", delete=False, encoding="utf-8"
    ) as f:
        f.write(FAMILY_OWL_CONTENT)
        family_owl_path = f.name

    try:
        onto = get_ontology(f"file://{family_owl_path}").load()
        family_ns = onto.get_namespace("http://family.example.org/#")

        # Define OWLReady2 classes in Python
        with onto:
            class Person(owlready2.Thing):
                namespace = family_ns

            class OldPerson(Person):
                namespace = family_ns

            class hasAge(owlready2.DataProperty):
                namespace = family_ns
                range = [int]

            class hasName(owlready2.DataProperty):
                namespace = family_ns
                range = [str]

        # Get individuals
        print("\n--- BEFORE REASONING ---")
        persons = list(onto.individuals())
        for p in persons:
            age_vals = list(p.hasAge) if hasattr(p, 'hasAge') else []
            name_vals = list(p.hasName) if hasattr(p, 'hasName') else []
            types = [c.name for c in p.is_a if hasattr(c, 'name')]
            name = name_vals[0] if name_vals else p.name
            age = age_vals[0] if age_vals else "?"
            print(f"  {name}: age={age}, types={types}")

        # Apply rule manually (since HermiT/Pellet may not be available):
        # "If Person AND hasAge > 60 → OldPerson"
        print("\n--- APPLYING RULE: age > 60 → OldPerson ---")
        with onto:
            classified_count = 0
            for individual in list(onto.individuals()):
                ages = list(individual.hasAge) if hasattr(individual, 'hasAge') else []
                if ages and ages[0] > 60:
                    if OldPerson not in individual.is_a:
                        individual.is_a.append(OldPerson)
                        classified_count += 1
                        name = (list(individual.hasName) or [individual.name])[0]
                        print(f"  INFERRED: {name} is OldPerson (age={ages[0]})")

        print(f"\n  Total inferred: {classified_count} OldPerson instance(s)")

        print("\n--- AFTER REASONING ---")
        for p in onto.individuals():
            age_vals = list(p.hasAge) if hasattr(p, 'hasAge') else []
            name_vals = list(p.hasName) if hasattr(p, 'hasName') else []
            types = [c.name for c in p.is_a if hasattr(c, 'name')]
            name = name_vals[0] if name_vals else p.name
            age = age_vals[0] if age_vals else "?"
            print(f"  {name}: age={age}, types={types}")

    finally:
        os.unlink(family_owl_path)

    logger.info("Family reasoning demo complete.")


def demo_ai_researchers_reasoning():
    """
    Demonstrate custom SWRL-style reasoning on the AI Researchers KG.
    Rule: Researcher wonAward AND worksAt TopUniversity → Influential
    """
    try:
        import owlready2
        from owlready2 import get_ontology
    except ImportError:
        logger.error("owlready2 not installed.")
        return

    logger.info("\n=== AI Researchers Reasoning Demo ===")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".owl", delete=False, encoding="utf-8"
    ) as f:
        f.write(AI_RESEARCHERS_OWL)
        ai_owl_path = f.name

    try:
        onto = get_ontology(f"file://{ai_owl_path}").load()
        ai_ns = onto.get_namespace("https://aikg.example.org/")

        with onto:
            class Researcher(owlready2.Thing):
                namespace = ai_ns

            class Influential(Researcher):
                namespace = ai_ns

            class TopUniversity(owlready2.Thing):
                namespace = ai_ns

            class Award(owlready2.Thing):
                namespace = ai_ns

            class worksAt(owlready2.ObjectProperty):
                namespace = ai_ns
                domain = [Researcher]

            class wonAward(owlready2.ObjectProperty):
                namespace = ai_ns
                domain = [Researcher]
                range = [Award]

            class name(owlready2.DataProperty):
                namespace = ai_ns
                range = [str]

        print("\n--- BEFORE REASONING ---")
        for r in onto.individuals():
            types = [c.name for c in r.is_a if hasattr(c, 'name')]
            print(f"  {r.name}: {types}")

        # Apply rule:
        # IF Researcher(x) AND worksAt(x, org) AND TopUniversity(org)
        #    AND wonAward(x, award)
        # THEN Influential(x)
        print("\n--- APPLYING RULE: wonAward ∧ worksAt(TopUniversity) → Influential ---")
        influential_count = 0
        with onto:
            for individual in list(onto.individuals()):
                is_researcher = any(c.name == "Researcher" for c in individual.is_a if hasattr(c, 'name'))
                if not is_researcher:
                    continue

                awards = list(individual.wonAward) if hasattr(individual, 'wonAward') else []
                affiliations = list(individual.worksAt) if hasattr(individual, 'worksAt') else []

                has_award = len(awards) > 0
                works_at_top = any(
                    any(c.name == "TopUniversity" for c in org.is_a if hasattr(c, 'name'))
                    for org in affiliations
                )

                if has_award and works_at_top:
                    if Influential not in individual.is_a:
                        individual.is_a.append(Influential)
                        influential_count += 1
                        print(f"  INFERRED: {individual.name} is Influential")
                        print(f"    Awards: {[a.name for a in awards]}")
                        print(f"    Works at: {[o.name for o in affiliations]}")

                # Also: researcher with award (even without top-university affiliation)
                elif has_award:
                    print(f"  (note) {individual.name} has award but not at TopUniversity in our KG")

        print(f"\n  Total Influential researchers inferred: {influential_count}")

        print("\n--- AFTER REASONING ---")
        for r in onto.individuals():
            types = [c.name for c in r.is_a if hasattr(c, 'name')]
            names = list(r.name) if hasattr(r, 'name') and isinstance(r.name, list) else [r.name]
            print(f"  {names[0]}: {types}")

    finally:
        os.unlink(ai_owl_path)

    logger.info("AI Researchers reasoning demo complete.")


def demo_transitivity_reasoning():
    """
    Demonstrate transitive property reasoning:
    supervisedBy is transitive → if A supervised B and B supervised C,
    then A (academically) supervised C (academic lineage).
    """
    print("\n=== Transitivity Reasoning: Academic Genealogy ===")

    # Simulate academic supervision chain
    # Hinton → Sutskever → (hypothetical student)
    # Hinton → LeCun (post-doc)
    supervision = {
        "Ilya_Sutskever": "Geoffrey_Hinton",
        "Yann_LeCun":     "Geoffrey_Hinton",   # PhD with Hinton's group
        "Andrew_Ng":      "Michael_I_Jordan",
        "Chelsea_Finn":   "Sergey_Levine",
        "Sergey_Levine":  "Pieter_Abbeel",
        "Pieter_Abbeel":  "Stuart_Russell",
    }

    # Compute transitive closure
    def get_academic_ancestors(person: str, chain: dict, visited=None) -> list[str]:
        if visited is None:
            visited = set()
        ancestors = []
        current = person
        while current in chain and current not in visited:
            visited.add(current)
            advisor = chain[current]
            ancestors.append(advisor)
            current = advisor
        return ancestors

    print("\nDirect supervision (supervisedBy):")
    for student, advisor in supervision.items():
        print(f"  {student.replace('_', ' ')} → supervisedBy → {advisor.replace('_', ' ')}")

    print("\nInferred academic lineage (transitive closure):")
    for person in supervision:
        ancestors = get_academic_ancestors(person, supervision)
        if len(ancestors) > 1:
            chain_str = " → ".join([person.replace("_", " ")] + [a.replace("_", " ") for a in ancestors])
            print(f"  {chain_str}")

    print("\nResult: By transitivity of supervisedBy,")
    print("  Chelsea Finn's academic lineage: Levine → Abbeel → Stuart Russell")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("SWRL / OWL Reasoning Demonstration")
    print("=" * 60)

    # Demo 1: Family ontology (age rule)
    demo_family_reasoning()

    # Demo 2: AI Researchers (influential rule)
    demo_ai_researchers_reasoning()

    # Demo 3: Transitivity (no OWLReady2 required)
    demo_transitivity_reasoning()

    print("\n" + "=" * 60)
    print("Reasoning demos complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
