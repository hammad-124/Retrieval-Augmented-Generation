import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.storage import InMemoryStore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# =====================================================================
# PHASE 1: INITIALIZE STORAGE SYSTEMS
# =====================================================================

# 1. The Vector Store (Only indexes and searches tiny CHILD chunks)
vectorstore = Chroma(
    collection_name="parent_document_retrieval_demo",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
)

# 2. The Docstore (A key-value store holding the large PARENT raw text)
docstore = InMemoryStore()


# =====================================================================
# PHASE 2: DEFINE CHUNKING STRATEGIES
# =====================================================================

# Parent chunks: Large enough to keep context intact for the LLM
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Child chunks: Small and precise for hyper-accurate vector search
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=25)


# =====================================================================
# PHASE 3: CREATE THE PARENT DOCUMENT RETRIEVER
# =====================================================================

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)


# =====================================================================
# PHASE 4: INGEST DATA
# =====================================================================

# Imagine a long manual page where LangGraph details are buried deep inside Chapter 2
knowledge_base = [
    Document(
        page_content="""
        # CHAPTER 1: BASIC ORCHESTRATION
        LangChain provides foundational abstractions for simple chains. It connects LLMs to prompts easily.
        
        # CHAPTER 2: ADVANCED STATE MANAGEMENT
        LangGraph is built for stateful, multi-actor LLM applications. Unlike standard linear chains, 
        LangGraph supports cycles, loops, and graph-based execution. This allows agents to correct 
        their errors, loop back to previous steps, and maintain persistent state over complex agent workflows.
        
        # CHAPTER 3: MONITORING PRODUCTION
        LangSmith helps you trace and debug your complex agents by tracking token usage and execution latency.
        """,
        metadata={"source": "agent_architecture_guide.md"}
    )
]

# Under the hood, this cuts the text into parents, saves them to docstore, 
# then cuts them into children, embeds them, and saves them to vectorstore.
retriever.add_documents(knowledge_base)


# =====================================================================
# PHASE 5: ASSEMBLE THE RAG PIPELINE
# =====================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = ChatPromptTemplate.from_template("""
Answer the user's question using only the provided context.

Context:
{context}

Question: {question}
Answer:""")

# LCEL Pipeline: The retriever will swap out matched child chunks for full parent chunks automatically
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Execute the query
if __name__ == "__main__":
    query = "Why should I use LangGraph instead of standard linear chains?"
    print(f"\nUser Query: {query}\n")
    
    response = rag_chain.invoke(query)
    print(f"LLM Response:\n{response}")