# 🧠 Advanced RAG Techniques — Playbook & Toolkit

A hands-on collection of **Retrieval-Augmented Generation (RAG)** patterns that solve real-world failure modes of naive RAG systems. Each module is self-contained, well-documented, and ready to run.

---

## Problem

Naive RAG (embed → retrieve → generate) breaks in production:

| Failure Mode | Symptom |
|---|---|
| **Pure vector search misses exact matches** | Product codes (`SKU-7742X`), error codes (`E_CONN_REFUSED`), acronyms (`WCAG`), names → zero results |
| **Poorly phrased queries** | Users ask one way, documents are written another way → relevant chunks are never retrieved |
| **Irrelevant content dilutes context** | Retrieved chunks are 90% filler; LLM gets "lost in the middle" and hallucinates |
| **Chunk size conflict** | Small chunks = better search; large chunks = better LLM context. You can't have both — or can you? |
| **Uncontrolled API costs** | No pre-flight budget → every oversized prompt costs money. No tracking → can't attribute costs per user |

---

## Solution — Six Modules

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                               RAG Playbook                                          │
├──────────────┬──────────────┬──────────────┬─────────────┬──────────┬───────────────┤
│  hybridSearch│ optimizeRAG  │ tokenTracking│  costOpt.   │scalingRAG│ securityLayer │
│              │              │ &Limiting    │             │          │               │
│  Vector +    │ Multi-Query  │ Pre-flight   │ Dim. Reduce │ HNSW     │ Prompt Inject │
│  BM25        │ Contextual   │ budget       │ Quantization│ Indexing │ PII/Secrets   │
│  (keywords)  │ Compression  │ Per-user     │ Batching    │ M & ef   │ LLM Guard     │
│              │ Parent Doc   │ tracking     │ Caching     │ tuning   │ Rate Limiter  │
│              │ Ensemble     │ Cost ceiling │ Right-size  │          │ Audit Logging │
└──────────────┴──────────────┴──────────────┴─────────────┴──────────┴───────────────┘
```

### 1. [`hybridSearch/`](hybridSearch/README.md) — Hybrid Search (Vector + BM25)

Combines **OpenAI embeddings** (semantic meaning) with **BM25** (keyword/term-frequency) via weighted Reciprocal Rank Fusion.

```
Query ──┬─► Vector Retriever ──┐
         │                      ├──► EnsembleRetriever (RRF) ──► Results
         └─► BM25 Retriever ────┘
```

**[→ Go to module](hybridSearch/README.md)**

### 2. [`optimizeRAG/`](optimizeRAG/README.md) — Advanced RAG Pipeline

Four techniques composed into an end-to-end pipeline:

| # | Technique | Benefit |
|---|-----------|---------|
| 1 | **Multi-Query Retriever** | LLM rewrites query into 3 variants → catches poor phrasing |
| 2 | **Hybrid Ensemble (BM25 + Vector)** | Handles both semantics and exact keywords |
| 3 | **Parent Document Retriever** | Small child chunks for search, large parent chunks for LLM context |
| 4 | **Contextual Compression** | Strips irrelevant filler from chunks → saves tokens, reduces hallucination |

**Pipeline flow:**
```
Query → MultiQuery → Hybrid Ensemble (BM25 + ParentDoc Vector) → Compress → LLM
```

**[→ Go to module](optimizeRAG/README.md)**

### 3. [`costOptimization/`](costOptimization/README.md) — Vector Search Cost Optimization

Five strategies to cut embedding & vector DB costs 60–90% without sacrificing retrieval quality:

| # | Technique | Impact |
|---|-----------|--------|
| 1 | **Dimensionality Reduction** | Truncate 1536→256–512 dims via Matryoshka embeddings |
| 2 | **Quantization** | Compress float32→int8 for ~75% memory savings |
| 3 | **Batching** | Batch API calls, upserts, and searches to reduce overhead |
| 4 | **Caching** | Exact-match (Redis), semantic (vector), and LLM prompt caching |
| 5 | **Right-Sizing** | Cap dimensions, payload size, HNSW params, TTLs |

**[→ Go to module](costOptimization/README.md)**

### 4. [`scalingRAG/`](scalingRAG/README.md) — HNSW Indexing & Scaling

Deep-dive into **HNSW** (Hierarchical Navigable Small World) — the ANN algorithm powering production vector databases:

| Concept | What it controls |
|---------|-----------------|
| **M** (max connections) | Memory vs. recall trade-off |
| **ef** (search effort) | Speed vs. accuracy trade-off |
| **ef_construction** | Build quality vs. build time |

Covers configuration in Weaviate, ChromaDB, and notes on Pinecone.

**[→ Go to module](scalingRAG/README.md)**

### 5. [`tokenTrackingandLimiting/`](tokenTrackingandLimiting/README.md) — Token Budgeting

Two classes for production cost control:

| Class | What it does |
|-------|-------------|
| `TokenBudget` | Local token counting (tiktoken), max-token enforcement, aggregate stats |
| `BudgetedLLM` | Pre-flight rejection before any API call + exact usage recording from response metadata |

**[→ Go to module](tokenTrackingandLimiting/README.md)**

### 6. [`securityLayer/`](securityLayer/README.md) — LLM Security & PII Pipeline

Production-grade guardrails for LLM applications: input sanitization, PII/secrets detection, LLM-as-guard classification, output validation, rate limiting, and audit logging — composed into one `SecurePipeline`.

| # | Layer | What it does |
|---|-------|-------------|
| 1 | **Rate Limiter** | Token-bucket per user/session — rejects over-quota requests before any processing |
| 2 | **Input Sanitizer** | Regex prompt-injection screen (fast, local, zero API cost) |
| 3 | **PII/Secrets Mask** | Redacts emails, SSNs, credit cards, API keys, tokens before reaching the LLM |
| 4 | **LLM Guard** | Semantic classifier (protected by circuit breaker — fails closed by default) |
| 5 | **Output Validator** | Re-checks LLM output for leaked PII/secrets before returning to user |
| 6 | **Audit Logger** | Structured events at every stage for incident response and debugging |

**[→ Go to module](securityLayer/README.md)**

---

## Quick Start

```bash
# 1. Clone
git clone <repo-url>
cd RAG

# 2. Install dependencies
pip install langchain langchain-core langchain-openai langchain-chroma langchain-community langchain-text-splitters tiktoken python-dotenv

# 3. Set your OpenAI key
echo "OPENAI_API_KEY=sk-..." > .env

# 4. Run any module
python hybridSearch/hybridSearch.py
python optimizeRAG/completePipeline.py
python optimizeRAG/parentDocumentRetriever.py
python costOptimization/optimize.py
python tokenTrackingandLimiting/TokenTracking
python securityLayer/security_pipeline.py
```

---

## Technologies

`Python 3` · `LangChain` · `OpenAI` (`gpt-4o-mini`, `text-embedding-3-small`) · `ChromaDB` · `BM25` · `tiktoken` · `python-dotenv`
