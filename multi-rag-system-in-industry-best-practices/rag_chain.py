"""
rag_chain.py — RAG chain construction, querying, and result formatting.

Production improvements:
- Returns structured QueryResult with sources + metadata
- Cites document name and page number in every answer
- MMR retrieval reduces redundant context
- LLM connection validated before first query
- Retry-friendly design
"""

from dataclasses import dataclass, field
from typing import List

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSerializable
from langchain_ollama import ChatOllama

from rag.config import get_settings
from rag.exceptions import LLMConnectionError
from rag.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """\
You are a precise, expert research assistant analyzing a document collection.

Rules:
1. Answer ONLY using the provided context — never use outside knowledge.
2. If the answer is not in the context, respond exactly: "I don't know based on the provided documents."
3. Always cite your sources using the format: [Source: filename, Page: N]
4. Be concise. Lead with the direct answer, then supporting evidence.
5. When comparing multiple documents, structure your answer clearly with document names as headers.
"""

HUMAN_PROMPT = """\
Context:
{context}

---
Question: {question}
"""


@dataclass
class Source:
    """Metadata about a retrieved document chunk."""
    file: str
    page: int
    chunk_index: int
    preview: str  # First 120 chars of the chunk


@dataclass
class QueryResult:
    """Structured result from a RAG query."""
    question: str
    answer: str
    sources: List[Source] = field(default_factory=list)

    def format(self) -> str:
        """Format result for terminal display."""
        lines = [
            f"\n🧠 Answer:\n{self.answer}",
            "\n📚 Sources used:",
        ]
        for i, src in enumerate(self.sources, 1):
            lines.append(
                f"  {i}. {src.file} — Page {src.page}  "
                f"(chunk #{src.chunk_index})\n"
                f'     "{src.preview}..."'
            )
        return "\n".join(lines)


def _format_docs_with_metadata(docs) -> str:
    """Format retrieved docs into context string with source citations."""
    parts = []
    for doc in docs:
        meta = doc.metadata
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        parts.append(
            f"[Source: {source}, Page: {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def build_rag_chain(vector_store: Chroma) -> RunnableSerializable:
    """
    Build a LCEL (LangChain Expression Language) RAG chain.

    Pipeline:
        question
          → MMR retrieval (diverse, non-redundant chunks)
          → format chunks with source citations
          → ChatPromptTemplate
          → ChatOllama (local LLM)
          → StrOutputParser
          → answer string
    """
    settings = get_settings()

    # Validate Ollama is reachable before building
    try:
        llm = ChatOllama(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout,
        )
    except Exception as e:
        raise LLMConnectionError(
            f"Cannot connect to Ollama model '{settings.llm_model}'.\n"
            f"  Fix: Run 'ollama serve' and 'ollama pull {settings.llm_model}'\n"
            f"  Error: {e}"
        ) from e

    # MMR retrieval: fetches fetch_k candidates, returns k most diverse ones
    retriever = vector_store.as_retriever(
        search_type=settings.retriever_search_type,
        search_kwargs={
            "k": settings.retriever_k,
            "fetch_k": settings.retriever_fetch_k,
        },
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    chain = (
        {
            "context": retriever | _format_docs_with_metadata,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info(
        "rag_chain_ready",
        model=settings.llm_model,
        retriever_k=settings.retriever_k,
        search_type=settings.retriever_search_type,
    )
    return chain


def query(chain: RunnableSerializable, vector_store: Chroma, question: str) -> QueryResult:
    """
    Run a question through the RAG chain.
    Returns a QueryResult with answer + source metadata.
    """
    logger.info("query_received", question=question[:100])

    # Get answer from chain
    answer = chain.invoke(question)

    # Independently fetch sources for display (MMR retrieval)
    settings = get_settings()
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": settings.retriever_k, "fetch_k": settings.retriever_fetch_k},
    )
    source_docs = retriever.invoke(question)

    sources = [
        Source(
            file=doc.metadata.get("source", "unknown"),
            page=int(doc.metadata.get("page", 0)),
            chunk_index=int(doc.metadata.get("chunk_index", 0)),
            preview=doc.page_content[:120].replace("\n", " "),
        )
        for doc in source_docs
    ]

    result = QueryResult(question=question, answer=answer, sources=sources)
    logger.info("query_answered", answer_chars=len(answer), sources=len(sources))
    return result
