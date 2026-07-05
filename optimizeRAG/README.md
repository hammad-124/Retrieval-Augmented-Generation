# Advanced RAG Techniques

## Overview

While **Naive RAG** simply takes a user query, embeds it, grabs the most similar text chunks, and hands them to an LLM, it often fails in production. Chunks might contain irrelevant noise, queries might be phrased poorly, or a vector search might completely miss exact keyword matches.

## Solution

**Advanced RAG** introduces specialized layers before and after retrieval to solve these issues. Here is a breakdown of the four core patterns implemented in the codebase.

---

## 1. Multi-Query Retriever (Optimizing the Input)

**The Problem:** Users don't always write perfect queries. If their phrasing doesn't closely match the way the document was written, semantic vector search can fail to find the right chunk.

**The Solution:** The `MultiQueryRetriever` uses an LLM to take a single user query and rewrite it into multiple variations from different perspectives.

- It runs a vector search for each generated variation.
- It merges all the results together and removes duplicates.

See implementation: [`completePipeline.py`](completePipeline.py) (lines 105-114)

```python
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=hybrid_ensemble_retriever,
    llm=llm_worker
)
```

**How it works:** If a user asks "What tools can I use to build AI applications?", the LLM might generate:
- "Frameworks for AI development"
- "Libraries for building LLM apps"

This ensures a much higher chance of hitting relevant documents.

---

## 2. Contextual Compression (Cleaning the Output)

**The Problem:** Even if you find the right document chunk, only 10% of it might actually answer the user's question. Passing massive chunks full of filler text to the LLM wastes tokens, increases costs, and can cause the model to miss the actual answer (the "lost in the middle" phenomenon).

**The Solution:** `ContextualCompressionRetriever` acts as a post-retrieval filter. It uses an LLM element (`LLMChainExtractor`) to scan the retrieved documents and extract only the specific sentences or fragments relevant to the query, discarding the rest.

See implementation: [`completePipeline.py`](completePipeline.py) (lines 117-128)

```python
compressor = LLMChainExtractor.from_llm(llm_worker)

final_advanced_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_query_retriever
)
```

Instead of passing a 1,000-character paragraph to your final generation chain, it compresses it down to the exact 100 characters that matter.

---

## 3. Ensemble / Hybrid Search (Combining Vectors & Keywords)

**The Problem:** Vector search (embeddings) is great at capturing meaning and intent, but terrible at finding exact matches like product serial numbers, specific error codes, or unique IDs (e.g., "pgvector", "ACID").

**The Solution:** Hybrid search combines two different paradigms:

- **BM25 Retriever:** Traditional sparse keyword matching (like a mini Google search for exact words).
- **Semantic Vector Retriever:** Dense embedding matching (for concepts and meaning).

The `EnsembleRetriever` merges their outputs using an algorithm called **Reciprocal Rank Fusion (RRF)** and allows you to weight which retriever you trust more.

See implementation: [`completePipeline.py`](completePipeline.py) (lines 86-101)

```python
hybrid_ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, parent_doc_retriever],
    weights=[0.3, 0.7]  # 30% importance to explicit keywords, 70% to vector semantics
)
```

---

## 4. Parent Document Retriever (Splitting the Difference)

**The Problem:** There is a fundamental conflict when chunking documents for RAG:

- **Small chunks** are better for vector search because they have focused embeddings without diluted meaning.
- **Large chunks** are better for generation because the LLM needs full surrounding context to answer accurately.

**The Solution:** The `ParentDocumentRetriever` solves this by decoupling _what you search_ from _what you pass to the LLM_.

- It splits a document into large **Parent** chunks, and splits those further into tiny **Child** chunks.
- The tiny **Child** chunks are embedded and saved into the Vector Store for highly precise searching.
- The large **Parent** chunks are kept in a separate Document Store (`InMemoryStore`).

See dedicated implementation: [`parentDocumentRetriever.py`](parentDocumentRetriever.py)
See integrated usage: [`completePipeline.py`](completePipeline.py) (lines 60-82)

```python
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,   # Tiny chunks (e.g., 200 chars)
    parent_splitter=parent_splitter, # Large context chunks (e.g., 800 chars)
)
```

**How it works:** When a search matches a small Child chunk, the retriever instantly looks up its associated full Parent chunk and hands that larger piece of context to the LLM.

---

## Complete Pipeline

All four techniques are composed together in [`completePipeline.py`](completePipeline.py) into a single end-to-end RAG chain:

```
Input Query
    |
    v
MultiQueryRetriever  (expands query into multiple variations)
    |
    v
EnsembleRetriever    (hybrid BM25 + vector search)
    |
    v
ContextualCompressionRetriever  (extracts only relevant content)
    |
    v
LLM (final answer generation)
```

### Pipeline Flow

1. **Input:** You ask: "What distinct advantages does LangGraph provide over basic linear chains?"
2. **Expansion:** `MultiQueryRetriever` creates 3 versions of your question.
3. **Retrieval:** All queries hit the hybrid ensemble (BM25 + vector search via ParentDocumentRetriever), gathering a broad net of documents.
4. **Filtering:** `ContextualCompressionRetriever` strips out all irrelevant filler text from those documents.
5. **Generation:** The final LLM gets a hyper-focused, noise-free context packet, allowing it to generate an incredibly accurate answer.

Run the complete pipeline:

```bash
python completePipeline.py
```
