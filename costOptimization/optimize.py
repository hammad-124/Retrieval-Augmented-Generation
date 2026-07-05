# ============================================================
# 1. Dimensionality Reduction
# ============================================================

from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, dims: int = 512):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        dimensions=dims  # <-- truncates at generation, not after
    )
    return response.data[0].embedding

# Full precision (expensive, rarely needed)
vec_full = get_embedding("wireless noise cancelling headphones", dims=1536)

# Production default (cheap, near-identical recall for most use cases)
vec_prod = get_embedding("wireless noise cancelling headphones", dims=512)


# ============================================================
# 2. Quantization — Collection Creation
# ============================================================

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, ScalarQuantization, ScalarQuantizationConfig, ScalarType
)

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="products",
    vectors_config=VectorParams(
        size=512,                     # matches the reduced dims from step 1
        distance=Distance.COSINE,
    ),
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(
            type=ScalarType.INT8,
            quantile=0.99,             # clips outlier extremes for better compression
            always_ram=True,           # keep compressed vectors in RAM for speed
        )
    ),
)


# ============================================================
# 2b. Quantization — Rescoring
# ============================================================

from qdrant_client.models import SearchParams, QuantizationSearchParams

results = client.query_points(
    collection_name="products",
    query=query_vector,
    limit=10,
    search_params=SearchParams(
        quantization=QuantizationSearchParams(
            rescore=True,        # <-- this is the key line
            oversampling=2.0,    # fetch 2x candidates before rescoring
        )
    ),
)


# ============================================================
# 3. Batching Requests — Embedding
# ============================================================

import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def embed_batch(texts: list[str], dims: int = 512):
    # OpenAI's embeddings endpoint natively accepts an array of inputs
    response = await client.embeddings.create(
        model="text-embedding-3-large",
        input=texts,          # <-- array, not a single string
        dimensions=dims
    )
    return [item.embedding for item in response.data]

async def process_all(all_texts: list[str], batch_size: int = 100):
    all_vectors = []
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i:i + batch_size]
        vectors = await embed_batch(batch)
        all_vectors.extend(vectors)
    return all_vectors


# ============================================================
# 3b. Batching Requests — Upsert & Search
# ============================================================

from qdrant_client.models import PointStruct

# BAD: one upsert per point = 50,000 network calls
# for point in points:
#     client.upsert(collection_name="products", points=[point])

# GOOD: batch upsert
points = [
    PointStruct(id=i, vector=vec, payload={"text": txt})
    for i, (vec, txt) in enumerate(zip(all_vectors, all_texts))
]
for i in range(0, len(points), 500):
    client.upsert(collection_name="products", points=points[i:i+500])

# GOOD: batch search (search multiple queries in one network call)
results = client.query_batch_points(
    collection_name="products",
    requests=[query_1, query_2, query_3],  # instead of 3 separate calls
)


# ============================================================
# 4a. Exact-Match Caching
# ============================================================

import redis
import hashlib
import json

r = redis.Redis(host="localhost", port=6379, decode_responses=True)

def cache_key(text: str) -> str:
    return f"embed:{hashlib.sha256(text.encode()).hexdigest()}"

def get_embedding_cached(text: str, dims: int = 512):
    key = cache_key(text)
    cached = r.get(key)
    if cached:
        return json.loads(cached)   # cache HIT — zero compute cost

    vector = get_embedding(text, dims)   # cache MISS — pay the real cost
    r.setex(key, 60 * 60 * 24 * 7, json.dumps(vector))  # TTL: 7 days
    return vector


# ============================================================
# 4b. Semantic Caching
# ============================================================

SIMILARITY_THRESHOLD = 0.95  # tune this — higher = stricter match

def get_answer_with_semantic_cache(query: str):
    query_vec = get_embedding_cached(query)  # embedding cache from 4a

    # Check the semantic cache collection first
    cache_hits = client.query_points(
        collection_name="semantic_cache",
        query=query_vec,
        limit=1,
        score_threshold=SIMILARITY_THRESHOLD,
    )

    if cache_hits.points:
        return cache_hits.points[0].payload["answer"]   # near-zero cost

    # Cache miss: run the real pipeline (search + LLM generation)
    answer = run_full_rag_pipeline(query, query_vec)

    # Store it for next time
    client.upsert(
        collection_name="semantic_cache",
        points=[PointStruct(
            id=hash(query) & 0x7FFFFFFF,
            vector=query_vec,
            payload={"query": query, "answer": answer},
        )],
    )
    return answer


# ============================================================
# 4c. LLM Prompt Caching
# ============================================================

# Anthropic prompt caching example
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": large_static_context,   # e.g. your retrieved docs / instructions
            "cache_control": {"type": "ephemeral"}  # <-- cached, billed once
        }
    ],
    messages=[{"role": "user", "content": user_query}]
)


# ============================================================
# 5. Right-Sizing Infrastructure
# ============================================================

# 1. Cap embedding dimensions at the source (from step 1) — don't let
#    different services in your stack request full-precision vectors "just in case."
DEFAULT_EMBEDDING_DIMS = 512

# 2. Cap payload size stored alongside vectors — don't dump entire documents
#    into Qdrant payloads when you only need an ID + short snippet.
payload = {
    "doc_id": doc.id,
    "snippet": doc.text[:500],   # not the full 10,000-word document
}

# 3. Set TTLs everywhere data expires naturally — sessions, caches, temp indexes.
r.setex(f"session:{user_id}", 60 * 30, session_data)   # 30 min TTL

# 4. Set explicit resource limits so a bug can't silently balloon your bill.
client.create_collection(
    collection_name="products",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    hnsw_config=HnswConfigDiff(
        m=16,                  # lower = less memory, slightly less recall
        ef_construct=100,      # tune down from default 200 if recall allows
    ),
)
