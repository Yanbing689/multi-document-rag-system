# ============================================================
# multi-rag.py — Multi-Document RAG System
# Based on Aman XAI's Guided AI Software Projects
# Run: python3.11 multi-rag.py
# Requirements: ollama must be running with llama3 model
# ============================================================

# ── Imports ─────────────────────────────────────────────────
import os
import sys

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ── Configuration ────────────────────────────────────────────
FOLDER_PATH = "/Users/yanbingjiang/Desktop/guided-ai-software-projects/multi-document-rag-system/data"
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.2"   # or "mistral, "llama2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Embedding Function (loaded once at startup) ───────────────
print("🔧 Loading embedding model...")
embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
print("✅ Embedding model ready!")

# ── Step 1: Load Documents ───────────────────────────────────
def load_documents(folder_path: str):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist")

    documents = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            print(f"📄 Loading: {filename}")
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                print(f"❌ Error loading {filename}: {e}")

    print(f"✅ Total pages loaded: {len(documents)}")
    return documents

# ── Step 2: Split Text into Chunks ───────────────────────────
def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  Created {len(chunks)} chunks")
    return chunks

# ── Step 3: Create Vector Store ──────────────────────────────
def create_vector_store(chunks):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_DB_PATH,
        collection_name=COLLECTION_NAME
    )
    print("✅ Vector database created and saved!")
    return vector_store

# ── Step 4: Format Retrieved Docs ────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Step 5: Query the RAG System ─────────────────────────────
def query_rag_system(query_text, vector_store):
    llm = ChatOllama(model=LLM_MODEL)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.
        Answer ONLY using the context below.
        If the answer is not present, say "I don't know."

        Context:
        {context}

        Question:
        {question}
        """
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(query_text)

# ── Main Entry Point ─────────────────────────────────────────
def main():
    if not os.path.exists(CHROMA_DB_PATH):
        print("📦 No vector DB found. Creating one...")
        docs = load_documents(FOLDER_PATH)
        chunks = split_text(docs)
        vector_store = create_vector_store(chunks)
    else:
        print("📦 Loading existing vector DB...")
        vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedding_function,
            collection_name=COLLECTION_NAME
        )
        print("✅ Vector database loaded!")

    print("\n🚀 RAG System ready! Ask me anything about your documents.")
    print("   (Type 'exit' to quit)\n")

    while True:
        query = input("❓ Ask a question: ")
        if query.lower().strip() == "exit":
            print("👋 Goodbye!")
            break

        print("🤔 Thinking...")
        try:
            answer = query_rag_system(query, vector_store)
            print(f"\n🧠 Answer:\n{answer}\n")
            print("-" * 60)
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()
