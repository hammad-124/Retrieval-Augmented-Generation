# HNSW Indexing for RAG

Notes on how HNSW (Hierarchical Navigable Small World) indexing works, why RAG systems at scale rely on it, and how to tune it.

## Table of Contents

- [The Problem](#the-problem)
- [What HNSW Is](#what-hnsw-is)
- [Two Key Parameters](#two-key-parameters)
- [The Core Trade-off](#the-core-trade-off)
- [Pinecone Is Different](#important-note-pinecone-is-different)

---

## The Problem

Most vector databases (Pinecone, pgvector, Chroma, Weaviate, Milvus, Qdrant) need a fast way to find the "nearest" vectors to a query vector.

The naive approach — **brute-force search** — compares the query vector against *every* stored vector. This is `O(n)`:

- Fine for 10K chunks.
- Unusable for 10M+ chunks, since you'd be running millions of cosine similarity computations per query.

**HNSW** solves this by building a graph structure that gets you to roughly `O(log n)` search time. The trade-off: it's *approximate* nearest neighbor (ANN) search, not exact — you give up a small amount of accuracy for a massive speed gain.

---

## What HNSW Is

HNSW is the indexing algorithm most production vector DBs use under the hood for approximate nearest neighbor search. It matters because it directly determines your RAG system's **retrieval latency**, **recall**, and **memory footprint** at scale.

---

## Two Key Parameters

### 1. `M` — Max Connections per Node

How many edges (connections) each node in the graph is allowed to have.

| Value | Effect |
|---|---|
| Low (8–16) | Smaller index, faster builds, lower accuracy |
| High (32–64) | Larger index, slower builds, higher accuracy |

### 2. `ef` — Search Effort

How many candidate nodes are explored during search. Comes in two flavors:

- **`efConstruction`** — effort used while *building* the graph
- **`efSearch`** — effort used at *query time*

| Value | Effect |
|---|---|
| Low (32–64) | Faster search, lower accuracy |
| High (200+) | Slower search, higher accuracy |

---

## The Core Trade-off

When scaling RAG to millions or billions of vectors, you run into a classic **"pick two out of three"** tension between:

- **Accuracy** (Recall)
- **Memory** (RAM)
- **Speed** (Latency)

- Want **higher accuracy**? → Increase both `M` and `ef` — but you pay in memory and speed.
- Want **faster queries**? → Decrease `efSearch` — but you pay in accuracy.

### Tuning Cheat Sheet

| Tuning Goal | `M` | `efConstruction` | `efSearch` | Impact |
|---|---|---|---|---|
| 🎯 Higher Accuracy | 🟢 Increase | 🟢 Increase | 🟢 Increase | ❌ High memory overhead & slower queries |
| ⚡ Faster Query Speed | ⚪ No effect | ⚪ No effect | 🔴 Decrease | ❌ Lower accuracy (may miss nearest neighbors) |
| 💾 Lower Memory Footprint | 🔴 Decrease | 🔴 Decrease | ⚪ No effect | ❌ Lower accuracy (sparser graph, more local minima) |

---

## Important Note: Pinecone Is Different

You **cannot** directly tune HNSW parameters (`M`, `ef`) in Pinecone.

Instead, Pinecone splits your vector data into immutable units called **"slabs"** and automatically selects the indexing algorithm based on data size — the tuning knobs above apply to self-hosted/configurable systems like pgvector, Qdrant, Weaviate, and Milvus, not to Pinecone.