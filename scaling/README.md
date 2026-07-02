The problem:
Brute-force search (compare query vector against every stored vector) is O(n). Fine for 10K chunks, unusable for 10M+ chunks — you'd be doing millions of cosine similarity computations per query. HNSW gets you to roughly O(log n) search time by trading a small amount of accuracy (approximate, not exact nearest neighbors) for massive speed gains.




HNSW (Hierarchical Navigable Small World)

HNSW is the indexing algorithm that most production vector DBs (Pinecone, Weaviate, Qdrant, Milvus, pgvector) use under the hood for approximate nearest neighbor (ANN) search. Understanding it matters because it directly determines your RAG system's retrieval latency, recall, and memory footprint at scale.