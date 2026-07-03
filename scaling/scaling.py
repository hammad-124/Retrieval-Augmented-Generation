# Weaviate (Python client v4)

import weaviate.classes.config as wc

client.collections.create(
    name="Documents",
    vector_index_config=wc.Configure.VectorIndex.hnsw(
        ef_construction=64,
        max_connections=16,   # this is "M"
        ef=100,               # default query-time ef
    ),
)

# ef can also be overridden per-query
response = collection.query.near_vector(
    near_vector=query_embedding,
    limit=10,
    # dynamic ef can be tuned via ef / ef_dynamic settings at collection level
)



# chromadb 

import chromadb

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.create_collection(
    name="documents",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:M": 16,
        "hnsw:construction_ef": 64,
        "hnsw:search_ef": 100,  # query-time ef
    },
)



