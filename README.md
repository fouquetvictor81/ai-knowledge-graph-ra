# AI Researchers Knowledge Graph + RAG Pipeline

A complete Web Mining & Semantic Web project that builds a **Knowledge Graph of Scientific Researchers in AI** and exposes it through a **Retrieval-Augmented Generation (RAG)** interface powered by a local LLM (Ollama).

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Hardware Requirements](#hardware-requirements)
4. [Installation](#installation)
5. [Ollama Setup](#ollama-setup)
6. [Running Each Module](#running-each-module)
7. [Web UI](#web-ui)
8. [Project Structure](#project-structure)
9. [Pipeline Diagram](#pipeline-diagram)

---

## Project Overview

This project implements a full **Knowledge Graph construction and querying pipeline**:

| Phase | Module | Description |
|-------|--------|-------------|
| 1. Web Crawling | `src/crawl/crawler.py` | Crawls Wikipedia, arXiv, and research lab pages about AI researchers |
| 2. Information Extraction | `src/ie/ner_extractor.py` | Extracts named entities and SVO triples using spaCy |
| 3. KG Construction | `src/kg/kg_builder.py` | Builds an RDF Knowledge Graph with custom ontology |
| 4. Entity Alignment | `src/kg/entity_aligner.py` | Aligns entities with Wikidata via SPARQL |
| 5. KG Expansion | `src/kg/kg_expander.py` | Expands KG using Wikidata 1-hop/2-hop queries |
| 6. SWRL Reasoning | `src/reason/swrl_reasoner.py` | Applies OWL/SWRL rules to infer new facts |
| 7. KGE Preparation | `src/kge/kge_prep.py` | Prepares train/valid/test splits for embedding |
| 8. KGE Training | `src/kge/kge_train.py` | Trains TransE and RotatE models with PyKEEN |
| 9. RAG Pipeline | `src/rag/rag_pipeline.py` | Natural language → SPARQL → Answer via local LLM |
| 10. Web App | `src/rag/app.py` | Beautiful Flask web UI to explore the KG |

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│         RAG Pipeline                │
│  ┌─────────────┐  ┌───────────────┐ │
│  │  KG Schema  │  │  Ollama LLM   │ │
│  │  Context    │→ │  (local)      │ │
│  └─────────────┘  └───────┬───────┘ │
│                           │         │
│              SPARQL Query Generated │
│                           │         │
│                    ┌──────▼──────┐  │
│                    │  RDF Graph  │  │
│                    │  (rdflib)   │  │
│                    └──────┬──────┘  │
│                           │         │
│              Results Retrieved      │
│                           │         │
│                    ┌──────▼──────┐  │
│                    │  LLM Answer │  │
│                    │  Generator  │  │
│                    └─────────────┘  │
└─────────────────────────────────────┘
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU | None (CPU) | NVIDIA 8GB VRAM (for LLM) |
| Storage | 5 GB | 20 GB |
| CPU | 4 cores | 8+ cores |
| OS | Windows 10 / Ubuntu 20.04 / macOS 12 | Same |

**Note:** The KGE training (TransE/RotatE) benefits greatly from a GPU. The Ollama LLM (llama3.2:1b or mistral) can run on CPU but will be slow without a GPU.

---

## Installation

### 1. Clone / Download the Project

```bash
cd "C:\Users\victo\Documents\DIA3\Web\Projet final"
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
# Optional (better NER, requires more RAM):
# python -m spacy download en_core_web_trf
```

### 5. Create `__init__.py` files

```bash
touch src/__init__.py src/crawl/__init__.py src/ie/__init__.py
touch src/kg/__init__.py src/reason/__init__.py src/kge/__init__.py src/rag/__init__.py
```

---

## Ollama Setup

Ollama runs a local LLM server that the RAG pipeline queries via REST API.

### Install Ollama

- **Windows/macOS**: Download from https://ollama.com/download
- **Linux**:
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  ```

### Pull a Model

```bash
# Lightweight (fast, ~2GB):
ollama pull llama3.2:1b

# Better quality (~4GB):
ollama pull mistral

# Best quality for SPARQL generation (~8GB):
ollama pull llama3.1:8b
```

### Start Ollama Server

```bash
ollama serve
```

The server listens at `http://localhost:11434`. Verify:
```bash
curl http://localhost:11434/api/tags
```

### Configure Model in RAG Pipeline

Edit `src/rag/rag_pipeline.py`, change `OLLAMA_MODEL`:
```python
OLLAMA_MODEL = "llama3.2:1b"   # or "mistral", "llama3.1:8b"
```

---

## Running Each Module

Run all commands from the project root:
```bash
cd "C:\Users\victo\Documents\DIA3\Web\Projet final"
```

### Step 1: Web Crawling

```bash
python src/crawl/crawler.py
```
Output: `data/crawled_pages.jsonl`

Options:
```bash
python src/crawl/crawler.py --output data/my_crawl.jsonl --max-pages 100 --delay 2.0
```

### Step 2: Named Entity Recognition & Triple Extraction

```bash
python src/ie/ner_extractor.py --input data/crawled_pages.jsonl
```
Output:
- `data/entities.csv`
- `data/triples.csv`

### Step 3: Build Initial Knowledge Graph

```bash
python src/kg/kg_builder.py
```
Output:
- `kg_artifacts/ontology.ttl`
- `kg_artifacts/initial_kg.nt`

### Step 4: Entity Alignment with Wikidata

```bash
python src/kg/entity_aligner.py
```
Output: `kg_artifacts/alignment.ttl`

*Note: Requires internet connection to query Wikidata SPARQL endpoint.*

### Step 5: Expand Knowledge Graph

```bash
python src/kg/kg_expander.py
```
Output: `kg_artifacts/expanded_kg.nt`

### Step 6: SWRL Reasoning

```bash
python src/reason/swrl_reasoner.py
```
Demonstrates OWL/SWRL inference on the ontology.

### Step 7: Prepare KGE Dataset

```bash
python src/kge/kge_prep.py
```
Output:
- `data/entity2id.txt`
- `data/relation2id.txt`
- `data/train.txt`
- `data/valid.txt`
- `data/test.txt`

### Step 8: Train Knowledge Graph Embeddings

```bash
python src/kge/kge_train.py
```
Output:
- `models/transe_model/`
- `models/rotate_model/`
- `results/kge_comparison.png`
- `results/tsne_embeddings.png`

*Warning: This may take 10-60 minutes depending on dataset size and hardware.*

### Step 9: Launch the RAG Web Application

```bash
python src/rag/app.py
```
Then open: **http://localhost:5000**

The app loads `kg_artifacts/sample_kg.ttl` (or `expanded_kg.nt` if available) automatically.

### Quick Demo (CLI RAG)

```bash
python src/rag/rag_pipeline.py --interactive
```

---

## Web UI

The Flask web application provides a beautiful interface to:
- Ask natural language questions about AI researchers
- Compare RAG mode (uses KG) vs Baseline mode (LLM only)
- View generated SPARQL queries
- Explore KG statistics and schema
- Browse query history

**Prerequisites:** Ollama must be running with at least one model pulled.

---

## Project Structure

```
Projet final/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── crawl/
│   │   └── crawler.py          # Web crawler
│   ├── ie/
│   │   └── ner_extractor.py    # NER + triple extraction
│   ├── kg/
│   │   ├── kg_builder.py       # RDF KG construction
│   │   ├── entity_aligner.py   # Wikidata alignment
│   │   └── kg_expander.py      # KG expansion via SPARQL
│   ├── reason/
│   │   └── swrl_reasoner.py    # OWL/SWRL reasoning
│   ├── kge/
│   │   ├── kge_prep.py         # KGE data preparation
│   │   └── kge_train.py        # TransE/RotatE training
│   └── rag/
│       ├── rag_pipeline.py     # RAG pipeline core
│       ├── app.py              # Flask web app
│       ├── templates/
│       │   └── index.html      # Web UI
│       └── static/
│           └── style.css       # Custom styles
├── kg_artifacts/
│   ├── ontology.ttl            # OWL ontology
│   ├── sample_kg.ttl           # Sample knowledge graph
│   ├── alignment.ttl           # Wikidata alignment
│   ├── initial_kg.nt           # Generated by kg_builder
│   └── expanded_kg.nt          # Generated by kg_expander
├── data/
│   ├── README.md               # Data folder docs
│   ├── crawled_pages.jsonl     # Crawler output
│   ├── entities.csv            # NER output
│   ├── triples.csv             # Triple extraction output
│   ├── entity2id.txt           # KGE entity mapping
│   ├── relation2id.txt         # KGE relation mapping
│   ├── train.txt               # KGE training set
│   ├── valid.txt               # KGE validation set
│   └── test.txt                # KGE test set
├── models/                     # Trained KGE models
└── results/                    # Plots and visualizations
```

---

## Pipeline Diagram

```
[Web Pages] → [Crawler] → [JSONL]
                               ↓
                        [NER Extractor] → [entities.csv, triples.csv]
                               ↓
                        [KG Builder] → [initial_kg.nt]
                               ↓
                       [Entity Aligner] → [alignment.ttl]
                               ↓
                        [KG Expander] → [expanded_kg.nt]
                               ↓
                   ┌──────────┤
                   ↓          ↓
            [SWRL Reasoner] [KGE Prep] → [train/valid/test]
                                ↓
                         [KGE Train] → [TransE, RotatE models]
                   ↓
            [RAG Pipeline] ← [expanded_kg.nt]
                   ↓
              [Flask App] → http://localhost:5000
```

---

## Authors

- Student project for Web Mining & Semantics course
- Domain: Scientific Researchers in Artificial Intelligence
- Technologies: Python, RDFLib, spaCy, PyKEEN, Ollama, Flask
