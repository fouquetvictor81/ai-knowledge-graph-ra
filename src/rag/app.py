"""
Flask Web Application for AI Knowledge Graph RAG System
========================================================
Serves a beautiful web UI and REST API for querying the KG
using natural language via a local Ollama LLM.

Endpoints:
    GET  /                  → Web UI
    POST /api/query         → RAG or baseline query
    GET  /api/stats         → KG statistics
    GET  /api/schema        → Schema summary
    GET  /api/history       → Query history
    GET  /api/health        → Health check

Usage:
    python src/rag/app.py
    python src/rag/app.py --port 5001 --host 0.0.0.0
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.rag_pipeline import (
    OLLAMA_MODEL,
    RAGPipeline,
    check_ollama_available,
    get_available_models,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
CORS(app)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

pipeline: Optional[RAGPipeline] = None
query_history: list[dict] = []
MAX_HISTORY = 100

# ---------------------------------------------------------------------------
# Initialize pipeline
# ---------------------------------------------------------------------------


def init_pipeline(kg_path: Optional[Path] = None, model: str = OLLAMA_MODEL):
    global pipeline
    logger.info("Initializing RAG pipeline...")
    try:
        pipeline = RAGPipeline(kg_path=kg_path, model=model)
        logger.info(f"Pipeline ready. KG: {len(pipeline.graph)} triples")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        pipeline = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the main web UI."""
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Health check endpoint."""
    ollama_ok = check_ollama_available()
    available_models = get_available_models() if ollama_ok else []

    status = {
        "status": "ok" if pipeline else "degraded",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "pipeline_loaded": pipeline is not None,
        "ollama_available": ollama_ok,
        "available_models": available_models,
        "current_model": pipeline.model if pipeline else None,
        "kg_triples": len(pipeline.graph) if pipeline else 0,
    }
    code = 200 if pipeline else 503
    return jsonify(status), code


@app.route("/api/stats")
def stats():
    """Return Knowledge Graph statistics."""
    if not pipeline:
        return jsonify({"error": "Pipeline not initialized"}), 503

    try:
        kg_stats = pipeline.get_stats()
        return jsonify({
            "success": True,
            "stats": kg_stats,
            "query_count": len(query_history),
        })
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/schema")
def schema():
    """Return the KG schema summary."""
    if not pipeline:
        return jsonify({"error": "Pipeline not initialized"}), 503

    try:
        schema_text = pipeline.get_schema()
        return jsonify({
            "success": True,
            "schema": schema_text,
        })
    except Exception as e:
        logger.error(f"Schema error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/history")
def history():
    """Return recent query history."""
    limit = request.args.get("limit", 20, type=int)
    return jsonify({
        "success": True,
        "history": query_history[-limit:][::-1],  # Most recent first
        "total": len(query_history),
    })


@app.route("/api/query", methods=["POST"])
def query():
    """
    Process a natural language query.

    Request body (JSON):
        {
            "question": "Which AI researchers won the Turing Award?",
            "mode": "rag"  // or "baseline"
        }

    Response (JSON):
        {
            "success": true,
            "answer": "...",
            "sparql_query": "SELECT ...",
            "results": [...],
            "repair_attempts": 0,
            "mode": "rag",
            "elapsed_ms": 1234,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    """
    if not pipeline:
        return jsonify({
            "success": False,
            "error": "Pipeline not initialized. Check server logs."
        }), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Request body must be JSON"}), 400

    question = data.get("question", "").strip()
    mode = data.get("mode", "rag").lower()

    if not question:
        return jsonify({"success": False, "error": "Question cannot be empty"}), 400

    if mode not in ("rag", "baseline"):
        return jsonify({"success": False, "error": "mode must be 'rag' or 'baseline'"}), 400

    logger.info(f"Query [{mode}]: {question}")
    start = time.time()

    try:
        result = pipeline.query(question, mode=mode)
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500

    elapsed_ms = int((time.time() - start) * 1000)
    timestamp = datetime.utcnow().isoformat() + "Z"

    # Build response
    response = {
        "success": True,
        "question": question,
        "answer": result.get("answer", ""),
        "sparql_query": result.get("sparql_query", ""),
        "results": result.get("results", [])[:MAX_HISTORY],
        "repair_attempts": result.get("repair_attempts", 0),
        "mode": mode,
        "elapsed_ms": elapsed_ms,
        "timestamp": timestamp,
        "error": result.get("error"),
    }

    # Store in history
    history_entry = {
        "question": question,
        "mode": mode,
        "answer": result.get("answer", "")[:300],  # Truncate for storage
        "sparql_query": result.get("sparql_query", ""),
        "result_count": len(result.get("results", [])),
        "repair_attempts": result.get("repair_attempts", 0),
        "elapsed_ms": elapsed_ms,
        "timestamp": timestamp,
    }
    query_history.append(history_entry)

    # Trim history
    if len(query_history) > MAX_HISTORY:
        query_history.pop(0)

    logger.info(f"Query complete in {elapsed_ms}ms. Results: {len(result.get('results', []))}")
    return jsonify(response)


@app.route("/api/example-questions")
def example_questions():
    """Return example questions for the UI."""
    examples = [
        {
            "question": "Which AI researchers won the Turing Award?",
            "category": "Awards",
            "icon": "🏆",
        },
        {
            "question": "Where does Yann LeCun work?",
            "category": "Affiliation",
            "icon": "🏢",
        },
        {
            "question": "Who collaborated with Geoffrey Hinton?",
            "category": "Collaboration",
            "icon": "🤝",
        },
        {
            "question": "What awards did researchers from Stanford receive?",
            "category": "Awards",
            "icon": "🌟",
        },
        {
            "question": "Which researchers work in France?",
            "category": "Location",
            "icon": "🌍",
        },
        {
            "question": "Who supervised Ilya Sutskever?",
            "category": "Academic",
            "icon": "🎓",
        },
        {
            "question": "What papers did Yoshua Bengio author?",
            "category": "Publications",
            "icon": "📄",
        },
        {
            "question": "Which organizations are located in the USA?",
            "category": "Location",
            "icon": "🗺️",
        },
    ]
    return jsonify({"success": True, "examples": examples})


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "status": 404}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error", "status": 500}), 500


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="AI KG RAG Flask Web App")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    parser.add_argument("--kg", type=Path, help="Path to KG file")
    parser.add_argument("--model", default=OLLAMA_MODEL, help=f"Ollama model (default: {OLLAMA_MODEL})")
    args = parser.parse_args()

    # Check Ollama
    if check_ollama_available():
        models = get_available_models()
        logger.info(f"Ollama available. Models: {models}")
        if args.model not in models and models:
            logger.warning(f"Model '{args.model}' not found. Available: {models}")
            logger.warning(f"Run: ollama pull {args.model}")
    else:
        logger.warning(
            f"Ollama not available at {OLLAMA_MODEL}. "
            "RAG queries will fail. Start Ollama: ollama serve"
        )

    # Initialize pipeline (do this before first request)
    init_pipeline(kg_path=args.kg, model=args.model)

    print(f"\n{'='*50}")
    print(f"  AI Knowledge Graph RAG System")
    print(f"{'='*50}")
    print(f"  URL:   http://{args.host}:{args.port}")
    print(f"  Model: {args.model}")
    print(f"  KG:    {len(pipeline.graph) if pipeline else 0} triples")
    print(f"{'='*50}\n")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        use_reloader=False,  # Avoid double initialization
    )


if __name__ == "__main__":
    main()
