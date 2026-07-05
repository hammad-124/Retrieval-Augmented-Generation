# RAG / Vector Search Cost Optimization — Production Playbook

**Context:** You're running embeddings + a vector DB (Qdrant here) in production and costs are creeping up — OpenAI embedding calls, Qdrant memory/node size, and query latency under load.

The core idea behind every technique below is the same: **you're paying for precision you don't need, at a granularity you don't need, more often than you need.** Fix those three things and cost drops 60-90% without hurting retrieval quality.

---

## 1. Dimensionality Reduction

### Simple explanation
An embedding is just a list of numbers representing meaning. OpenAI's `text-embedding-3-large` gives you 1536 numbers by default. But most of the "meaning" that matters for search is packed into the first few hundred dimensions — the rest is diminishing returns, like a JPEG at 100% vs 95% quality. You can't tell the difference visually, but the file is way smaller.

OpenAI's newer embedding models let you **truncate the vector directly at generation time** using the `dimensions` parameter — it's not just chopping off the array, the model is trained so the first N dimensions are the most information-dense (Matryoshka representation learning).

### Real scenario
You have 2M product descriptions indexed for a semantic search feature. At 1536 dims × 4 bytes (float32) = ~6KB per vector. That's 12GB just for raw vectors, before Qdrant's internal overhead (HNSW graph links, payloads, etc.) which typically **doubles** that. Your Qdrant cluster needs bigger, more expensive nodes just to hold the index in RAM.

Drop to 512 dims → 2KB per vector → ~4GB raw. Same recall for 95%+ of queries, because most real-world queries don't need the full semantic resolution — they need "is this roughly about the same thing."

### Code (see [`optimize.py`](optimize.py))

### When to use it
| Use case | Recommended dims |
|---|---|
| General semantic search, chatbots, FAQ retrieval | 256–512 |
| Legal/medical document retrieval (nuance matters) | 1024–1536 |
| Deduplication / "roughly similar" clustering | 128–256 |

### When NOT to use it
Don't reduce dimensions for domains where subtle semantic distinctions matter (legal contract clauses, medical symptom matching) — you can lose the exact nuance that differentiates two close-but-different meanings. Always A/B test recall@k before and after on a labeled eval set. Don't guess.

---

## 2. Quantization

### Simple explanation
Even after reducing dimensions, each number is still stored as a `float32` — a highly precise decimal (like `0.043782910...`). For similarity search, you don't need that precision. Qdrant can compress each float down to an `int8` (a single byte, values -128 to 127) or even binary (1 bit). This is quantization: trading a tiny bit of numerical precision for a massive memory win.

Think of it like this: float32 is measuring distance with a laser micrometer. int8 is measuring with a ruler. For "which products are semantically similar," a ruler is more than accurate enough.

### Real scenario
Your 4GB (post dimension-reduction) index still needs to live in RAM for fast HNSW search. With `int8` scalar quantization, Qdrant stores the compressed vectors for search and optionally keeps float32 only on disk for rescoring the top candidates. Result: **~75% additional memory reduction**, meaning that 4GB index becomes ~1GB. You go from needing a 32GB RAM node to an 8GB one — that's often a 4x cost difference on managed Qdrant Cloud or your own infra.

### Code (see [`optimize.py`](optimize.py))

### Types of quantization (pick based on tradeoff)
| Type | Memory savings | Accuracy loss | Best for |
|---|---|---|---|
| **Scalar (int8)** | ~75% | Minimal (<1-2% recall drop) | Default choice for most production systems |
| **Binary (1-bit)** | ~97% | Noticeable (needs rescoring) | Massive scale (100M+ vectors), cost-critical |
| **Product Quantization (PQ)** | ~90-95% | Moderate | Very large indexes where binary is too lossy |

### Production tip
Always pair quantization with **rescoring** — search fast using compressed vectors to get top 100-200 candidates, then rescore just those against the original float32 vectors. You get the speed/memory win with almost none of the accuracy loss.

See [`optimize.py`](optimize.py) for the full example.

---

## 3. Batching Requests

### Simple explanation
Every API call — to OpenAI, to Qdrant — has fixed overhead: network round trip, TLS handshake, auth, serialization. If you embed 1,000 texts one at a time, you pay that overhead 1,000 times. If you batch them into groups, you pay it once (or a handful of times) and let the provider process the array internally, which is dramatically more efficient on their end too.

### Real scenario
You're backfilling embeddings for 50,000 existing support tickets overnight. Looping one call at a time: ~50,000 HTTP round trips, each with ~150-300ms latency → **2-4 hours**, and you'll hit rate limits and get throttled/retried, extending it further. Batch 100 at a time → **500 calls** → same job finishes in **10-15 minutes**, with far fewer rate-limit errors (which themselves cost you retries = money).

### Code (see [`optimize.py`](optimize.py))

Same principle applies on the Qdrant side — batch your upserts and your searches:

See [`optimize.py`](optimize.py) for the full example.

### When to use it
- **Bulk indexing / backfills** — always batch, no exception.
- **Real-time user queries** — batch only if you can tolerate a small collection window (e.g., collect requests for 20-50ms and dispatch together). Good for high-QPS systems, bad if a single user's latency matters more than throughput.

---

## 4. Caching — the highest-leverage technique

### Simple explanation
Caching intercepts a request **before** it reaches the expensive part of your system (the embedding API call, or the vector search itself). If you've seen this exact (or semantically similar) query before, skip the computation entirely and serve the stored answer. This is the technique with the best ROI because real-world query traffic is heavily repetitive — a small number of queries account for a large fraction of your traffic (Pareto distribution).

### Real scenario
You run a customer support chatbot. Analysis of your logs shows "how do I reset my password" and its 40 near-identical phrasings account for 15% of daily query volume. Without caching, every single one triggers an OpenAI embedding call + a Qdrant search + often an LLM generation call. With caching, the first one costs you the full pipeline; the next 10,000 that day cost you a Redis `GET` — sub-millisecond, essentially free.

### Types of caching (use the right one for the right layer)

| Type | What it matches | Speed | Use case |
|---|---|---|---|
| **Exact-match caching** | Identical string, byte-for-byte | ~1ms | Repeated identical queries (autocomplete, common FAQs) |
| **Semantic caching** | Similar *meaning*, different wording | ~10-30ms | "reset password" vs "forgot my password" — catches paraphrases |
| **Embedding caching** | Cache the vector itself, not the answer | ~1ms | Same document re-embedded during re-indexing/updates |
| **Prompt/response caching (LLM-side)** | Cache LLM completions for repeated prompts/context | Provider-side, near-instant | RAG pipelines where the same context + question recurs |

#### 4a. Exact-match caching (simplest, do this first) — see [`optimize.py`](optimize.py)

#### 4b. Semantic caching (catches paraphrases exact-match misses)
This is the more powerful version — it embeds the incoming query, then checks if a *similar enough* query already has a cached answer, using the vector DB itself (or a small dedicated cache collection) as the similarity check.

See [`optimize.py`](optimize.py) for the full implementation.

**Important production caveat:** semantic caching is a correctness risk if your threshold is too loose — "cancel my subscription" and "cancel my order" might score 0.93 similarity but need completely different answers. Always test the threshold against a labeled set of query pairs that *should* and *shouldn't* match before shipping.

#### 4c. LLM prompt caching (if your RAG pipeline calls an LLM after retrieval)
Anthropic and OpenAI both support caching large repeated context blocks (like your system prompt or a large retrieved document set) so you're not re-billed full input tokens every call.

See [`optimize.py`](optimize.py) for the full example.

### Cache invalidation — the part people forget
Caching without expiration is how you serve stale/wrong answers forever. Always set:
- **TTL** (`setex` in Redis) so entries expire naturally — 1-7 days is typical for support/FAQ content.
- **Explicit invalidation** when the underlying source data changes (e.g., a product's price updates → purge related cache keys).

---

## 5. Right-Sizing Infrastructure

### Simple explanation
This one isn't a single trick — it's a discipline: don't provision for a worst-case guess, provision for what your actual traffic and data need, and put hard caps everywhere so nothing grows unbounded.

### Real scenario
A team provisions a 64GB Qdrant node "to be safe" for a collection that, post dimension-reduction and quantization, only needs 6GB. That's paying for 10x the capacity you use — often the single biggest line item on the cloud bill, bigger than the API costs it was meant to hedge against.

### Concrete right-sizing moves

See [`optimize.py`](optimize.py) for the code.

### The right-sizing checklist for production
- [ ] Load-test with **real** traffic patterns, not synthetic uniform load, before picking node size.
- [ ] Set a monitoring alert on memory/CPU at 70% utilization — that's your signal to right-size, not 95%.
- [ ] Review payload sizes quarterly — payloads silently grow as devs add "just one more field."
- [ ] Set TTLs on every cache and ephemeral collection — nothing lives forever by default.

---

## Putting it together: cost impact summary

| Technique | Typical savings | Effort to implement | Risk if misapplied |
|---|---|---|---|
| Dimensionality reduction | ~66% index size | Low | Recall drop on nuanced domains |
| Quantization (int8) | ~75% additional memory | Low-Medium | Minor recall drop (mitigated by rescoring) |
| Batching | Latency ↓, throughput ↑, fewer rate-limit retries | Low | None if done correctly |
| Caching (exact + semantic) | Up to 90%+ on repeat traffic | Medium | Stale/wrong answers if TTL & threshold not tuned |
| Right-sizing | Often the single biggest line-item cut | Medium (needs monitoring discipline) | Under-provisioning causes outages under load |

**Rule of thumb for rollout order:** dimensionality reduction and batching first (safe, mechanical, no risk to correctness) → quantization with rescoring next (measure recall before/after) → caching last (needs the most tuning and monitoring, but has the highest ceiling on savings once your traffic patterns are understood).