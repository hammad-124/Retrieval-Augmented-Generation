import os
import logging
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.retrievers import BM25Retriever
from langchain_classic.storage import InMemoryStore

# The Advanced Retrievers
from langchain_classic.retrievers import (
    ParentDocumentRetriever,
    EnsembleRetriever,
    ContextualCompressionRetriever
)
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

# Core Schema & LCEL Utilities
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# Enable LangChain Multi-Query logs so we can watch it rewrite questions in our terminal
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# =====================================================================
# STEP 1: PREPARE DATA AND CHUNKING STRATEGIES
# =====================================================================

raw_documents = [
    Document(
        page_content="""
        # SECTION A: ARCHITECTURE DESCRIPTION
        The enterprise system architecture runs inside AWS EKS container instances. We orchestrate microservices via Kubernetes. 
        Database layers rely on a high-availability PostgreSQL engine paired with Redis layers for low-latency session caching.
        
        # SECTION B: AI ENGINE AND ORCHESTRATION
        LangGraph is built for stateful, multi-actor LLM applications. Unlike standard linear chains, LangGraph natively 
        supports execution cycles, loops, and graph-based routing mechanics. This allows autonomous systems to recursively 
        correct runtime execution paths, maintain state variables across workflows, and leverage complex agent patterns.
        
        # SECTION C: TELEMETRY AND METRICS
        Systems emit structured JSON logs downstream into Datadog. Monitoring targets alert when p99 latency passes 500ms bounds.
        """,
        metadata={"source": "system_blueprint_v4.md"}
    )
]

# Splitters for Parent-Child Strategy
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=25)


# =====================================================================
# STEP 2: INITIALIZE LAYER 1 - PARENT DOCUMENT RETRIEVER (CONCEPT 4)
# =====================================================================

# Store A: Only holds numerical vector math for the tiny child cuts
vectorstore = Chroma(
    collection_name="master_advanced_rag",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)

# Store B: Only holds raw, readable parent text strings inside server memory
docstore = InMemoryStore()

# Forge the Parent-Child Retriever link
parent_doc_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Populating our database architecture using the two splitters simultaneously
parent_doc_retriever.add_documents(raw_documents)


# =====================================================================
# STEP 3: INITIALIZE LAYER 2 - ENSEMBLE/HYBRID RETRIEVER (CONCEPT 3)
# =====================================================================

# Extract the full parent documents out to seed the keyword engine
all_parent_docs = list(docstore.yield_keys())
parent_documents_list = [docstore.mget([k])[0] for k in all_parent_docs]

# Create sparse keyword search (BM25)
bm25_retriever = BM25Retriever.from_documents(parent_documents_list)
bm25_retriever.k = 2

# Wrap our Parent-Child system and Keyword search together
hybrid_ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, parent_doc_retriever],
    weights=[0.3, 0.7]  # 30% importance to explicit keywords, 70% to vector semantics
)


# =====================================================================
# STEP 4: INITIALIZE LAYER 3 - MULTI-QUERY EXPANSION (CONCEPT 1)
# =====================================================================

llm_worker = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# MultiQuery intercepts the query first, multiplies it, and fires them into our Hybrid engine
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=hybrid_ensemble_retriever,
    llm=llm_worker
)


# =====================================================================
# STEP 5: INITIALIZE LAYER 4 - CONTEXTUAL COMPRESSION (CONCEPT 2)
# =====================================================================

# Create the extractor that knows how to skim documents for specific answers
compressor = LLMChainExtractor.from_llm(llm_worker)

# Put the final wrapper around everything. This sits at the end of the extraction tunnel.
final_advanced_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=multi_query_retriever
)


# =====================================================================
# STEP 6: ASSEMBLE THE COMPLETE RAG CHAIN
# =====================================================================

prompt_template = ChatPromptTemplate.from_template("""
You are a senior system engineer. Answer the following question accurately using only the provided context blocks.

Context:
{context}

Question: {question}
Answer:""")

# LCEL Pipeline Execution Map
rag_pipeline = (
    {"context": final_advanced_retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm_worker
    | StrOutputParser()
)

if __name__ == "__main__":
    user_query = "What distinct advantages does LangGraph provide over basic linear chains?"
    print(f"\n[USER INPUT]: {user_query}")
    
    # Run the query through our multi-layered retriever structure
    final_output = rag_pipeline.invoke(user_query)
    
    print("\n" + "="*50)
    print(f"[FINAL RESPONSIVE ANSWER]:\n{final_output}")
    print("="*50)