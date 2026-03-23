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
from rdflib import Graph
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
OLLAMA_TIMEOUT = 120           # seconds

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


def load_graph(kg_path: Optional[Path] = None) -> Graph:
    """
    Load the RDF Knowledge Graph.
    Tries KG_CANDIDATES if no path provided.
    """
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


def generate_sparql(question: str, schema: str, model: str = OLLAMA_MODEL) -> str:
    """
    Generate a SPARQL query from a natural language question using Ollama.
    """
    prompt = f"""Given this Knowledge Graph schema:

{schema}

Generate a SPARQL query to answer this question:
"{question}"

Return ONLY the SPARQL query, starting with PREFIX declarations."""

    raw = call_ollama(prompt, model=model, system=SPARQL_SYSTEM_PROMPT)

    # Extract SPARQL from response (strip any markdown code blocks)
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
# SPARQL execution
# ---------------------------------------------------------------------------


def execute_sparql(g: Graph, sparql_query: str) -> tuple[list[dict], str]:
    """
    Execute a SPARQL query on the graph.
    Returns (results_list, error_message).
    """
    try:
        results = g.query(sparql_query)
        rows = []
        for row in results:
            row_dict = {}
            for var in results.vars:
                val = row.get(var)
                if val is not None:
                    row_dict[str(var)] = str(val)
            rows.append(row_dict)
        return rows, ""
    except Exception as e:
        error_msg = str(e)
        # Clean up error message
        error_msg = error_msg[:500]  # Truncate very long errors
        return [], error_msg


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
        self.schema = build_schema_summary(self.graph)
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
        try:
            sparql_query = generate_sparql(question, self.schema, model=self.model)
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
