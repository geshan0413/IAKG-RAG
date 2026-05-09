# IAKG-RAG

**Intent-Aware Knowledge Graph Augmented Retrieval-Augmented Generation for Maritime Fire Safety Compliance Checking**

This repository contains the code, data, and experimental results for the paper *"IAKG-RAG: Intent-Aware Knowledge Graph Augmented RAG for Maritime Fire Safety Compliance Checking"*.

## Overview

IAKG-RAG is a framework that combines domain-specific knowledge graph retrieval with intent-aware query reformulation to automate maritime fire safety compliance checking against SOLAS Chapter II-2 and the FSS Code.

### Key Features

- **Knowledge Graph Retrieval**: Structured traversal over a domain ontology (1,461 clauses, 6,694 semantic relations) to retrieve contextually complete regulatory clauses
- **Intent Recognition & Retrieval Routing (IRIR)**: Three-tier fault-tolerant intent classifier that identifies query intent (violation / compliance / comparative) and reformulates queries for targeted retrieval
- **Three-way Fusion Retrieval**: Weighted integration of graph traversal, dense vector retrieval (BGE-base-zh), and BM25 sparse retrieval
- **Structured Reasoning Prompt**: Four-step compliance judgment workflow (clause filtering → requirement extraction → exemption checking → final verdict)

## Repository Structure

```
IAKG-RAG/
├── README.md                          # This file
├── src/                               # Core system code
│   ├── compliance_rag_v7.py           # Main IAKG-RAG pipeline (v7)
│   ├── graph_retriever.py             # Knowledge graph traversal & entity extraction
│   ├── embed_clauses.py               # Dense vector retrieval (FAISS + BGE)
│   ├── build_kg_v3.py                 # Knowledge graph construction from CSV
│   └── baselines.py                   # Baseline retrieval methods
├── data/
│   ├── test_set_v4.json               # 300-case test set (annotated)
│   ├── prompt_template.txt            # Structured compliance judgment prompt
│   └── kg/                            # Knowledge graph CSV files
│       ├── nodes_Clause.csv           # 1,461 clause nodes
│       ├── nodes_Equipment.csv        # 70 equipment nodes
│       ├── nodes_Space.csv            # 23 space nodes
│       ├── nodes_ShipType.csv         # 13 ship type nodes
│       ├── nodes_Regulation.csv       # 2 regulation source nodes
│       └── relationships.csv          # 6,694 semantic relations
├── results/                           # Experiment results (JSON)
│   ├── E1_pure_llm.json               # E1: Pure LLM (no retrieval)
│   ├── E2_bm25_rag.json               # E2: BM25 + LLM
│   ├── E3_naive_rag.json              # E3: Dense vector RAG
│   ├── E4_hybrid_rag.json             # E4: BM25 + Dense hybrid
│   ├── E5_iakg_rag.json               # E5: IAKG-RAG (full system)
│   ├── E6_iakg_rag_deepseek.json      # E6: IAKG-RAG with DeepSeek (cross-model)
│   ├── A1_no_prompt.json              # Ablation: w/o structured prompt
│   ├── A2_no_irir.json                # Ablation: w/o IRIR module
│   └── A3_no_graph.json               # Ablation: w/o knowledge graph
├── scripts/                           # Experiment runner scripts
│   ├── run_all_experiments.py         # Run all E1-E5 experiments
│   ├── run_pure_llm_baseline.py       # E1: Pure LLM
│   ├── run_bm25_baseline.py           # E2: BM25 baseline
│   ├── run_naive_rag_baseline.py      # E3: Naive RAG
│   ├── run_hybrid_rag_baseline.py     # E4: Hybrid RAG
│   ├── run_cross_model.py             # E6: Cross-model (DeepSeek)
│   ├── run_ablation.py                # A1/A3 ablation
│   ├── run_ablation_no_irir.py        # A2 ablation
│   └── compute_ragas.py               # RAGAS metric computation
└── figures/                           # (placeholder for paper figures)
```

## Experimental Results

### Main Experiments (E1–E6)

| Model | Description | Acc(%) | P(%) | R(%) | F1(%) | HR@5(%) | MRR |
|:------|:------------|-------:|-----:|-----:|------:|--------:|----:|
| E1 | Pure LLM | 74.7 | 79.1 | 80.0 | 79.6 | N/A | N/A |
| E2 | BM25 + LLM | 84.0 | 83.4 | 92.4 | 87.7 | 80.3 | 0.667 |
| E3 | Naive RAG | 82.3 | 80.8 | 93.5 | 86.7 | 70.7 | 0.572 |
| E4 | Hybrid RAG | 81.0 | 78.2 | 93.1 | 85.0 | 60.0 | 0.495 |
| **E5** | **IAKG-RAG (Ours)** | **93.7** | **96.5** | **92.8** | **94.6** | **73.0** | **0.569** |
| E6 | IAKG-RAG (DeepSeek) | 91.0 | 96.4 | 88.3 | 92.2 | 68.7 | 0.502 |

### Ablation Experiments (A1–A3)

| Model | Removed Module | Acc(%) | P(%) | R(%) | F1(%) | HR@5(%) | MRR | ΔAcc(pp) |
|:------|:---------------|-------:|-----:|-----:|------:|--------:|----:|---------:|
| **E5** | **(Full System)** | **93.7** | **96.5** | **92.8** | **94.6** | **73.0** | **0.569** | — |
| A1 | w/o Structured Prompt | 86.0 | 83.5 | 95.6 | 89.1 | 69.3 | 0.514 | −7.7 |
| A2 | w/o IRIR | 89.0 | 95.1 | 86.1 | 90.4 | 51.3 | 0.388 | −4.7 |
| A3 | w/o Knowledge Graph | 81.7 | 83.2 | 87.8 | 85.4 | 53.3 | 0.430 | −12.0 |

> **Note**: P, R, F1 are computed with "non-compliant" as the positive class. HR@5 and MRR use hierarchical clause ID matching (parent-child prefix matching).

## Knowledge Graph

The knowledge graph is constructed from two regulatory instruments:

| Source | Clauses | Description |
|:-------|--------:|:------------|
| SOLAS Chapter II-2 | 971 | International Convention for the Safety of Life at Sea — Fire Protection, Detection and Extinction |
| FSS Code | 490 | International Code for Fire Safety Systems |
| **Total** | **1,461** | |

### Ontology

- **Nodes** (5 types, 1,569 total): Clause, Equipment (70), Space (23), ShipType (13), Regulation (2)
- **Edges** (6 types, 6,694 total): MENTIONS_SPACE, MENTIONS_EQUIPMENT, HAS_CLAUSE, HAS_SUBCLAUSE, APPLIES_TO, REFERENCES

## Setup

### Prerequisites

- Python 3.10+
- Neo4j 5.x (for knowledge graph storage)
- CUDA-compatible GPU (for embedding model)

### Dependencies

```bash
pip install neo4j openai sentence-transformers faiss-cpu rank-bm25 ragas
```

### Configuration

Update API keys in `src/compliance_rag_v7.py`:

```python
API_KEY = "your-api-key"
BASE_URL = "https://api.xiaomimimo.com/v1"
MODEL_NAME = "mimo-v2-flash"
```

### Knowledge Graph Import

```bash
# 1. Start Neo4j
# 2. Import CSV data
python src/build_kg_v3.py
```

### Run Experiments

```bash
# Run all main experiments (E1-E5)
python scripts/run_all_experiments.py

# Run ablation experiments
python scripts/run_ablation.py
python scripts/run_ablation_no_irir.py

# Cross-model validation (DeepSeek)
python scripts/run_cross_model.py
```

## Result Format

Each result JSON file follows this structure, preserving all fields for full reproducibility:

```json
{
  "system": "E5",
  "test_set": "test_set_v4.json",
  "summary": {
    "total": 300,
    "correct": 281,
    "accuracy": 0.937
  },
  "details": [
    {
      "id": "TC_001",
      "question": "一艘2018年建造的客船...",
      "full_answer": "### 第一步：相关条款筛选...",
      "predicted_compliance": "不合规",
      "predicted_refs": ["SOLAS II-2/9.2.5.1.1", ...],
      "retrieved_clauses": [
        {
          "clause_id": "SOLAS II-2/9.2.5",
          "title": "...",
          "content": "条款全文...",
          "graph_score": 1.0,
          "vector_score": 0.646,
          "final_score": 1.07,
          "cross_hit": true
        }
      ],
      "retrieved_contexts": ["条款1全文...", "条款2全文..."],
      "expected_compliance": "不合规",
      "expected_refs": ["FSS Code/9.2.5.1.5", ...],
      "reference": "不合规，依据：...",
      "reference_contexts": ["GT条款全文..."],
      "accuracy_correct": true,
      "llm_time_ms": 2100,
      "intent": "negative"
    }
  ]
}
```

**Key fields for reproducibility:**
- `question` / `full_answer`: Complete input/output for each case
- `retrieved_clauses` / `retrieved_contexts`: Full retrieval results with scores
- `reference` / `reference_contexts`: Ground truth clauses for evaluation

## Citation

If you find this work useful, please cite:

```bibtex
@article{hu2026iakgrag,
  title={IAKG-RAG: Intent-Aware Knowledge Graph Augmented Retrieval-Augmented Generation for Maritime Fire Safety Compliance Checking},
  author={Hu, Hao and others},
  year={2026}
}
```

## License

This project is released for academic research purposes.

## Acknowledgments

- Regulatory texts: IMO SOLAS Chapter II-2 and FSS Code
- Base LLM: Xiaomi MiMo-v2-flash
- Embedding model: BAAI/bge-base-zh-v1.5
