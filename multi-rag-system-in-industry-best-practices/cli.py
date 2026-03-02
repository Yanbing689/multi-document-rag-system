"""
cli.py — Production CLI using Typer + Rich.

Commands:
    rag query "your question"     # One-shot query
    rag chat                      # Interactive Q&A session
    rag build                     # Build/rebuild vector store
    rag info                      # Show system info and config
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from rag.config import get_settings
from rag.exceptions import (
    DataNotFoundError,
    EmbeddingError,
    LLMConnectionError,
    VectorStoreError,
)
from rag.loader import load_documents, split_documents
from rag.logging import get_logger, setup_logging
from rag.rag_chain import build_rag_chain, query
from rag.vectorstore import (
    create_vector_store,
    load_vector_store,
    vector_store_exists,
)

app = typer.Typer(
    name="rag",
    help="🔍 Multi-Document RAG System — Query your PDFs with a local LLM",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()
logger = get_logger(__name__)


def _get_or_build_store(rebuild: bool = False):
    """Shared logic: load existing store or build a new one."""
    settings = get_settings()

    if rebuild or not vector_store_exists():
        if rebuild:
            console.print("[yellow]🔄 Rebuilding vector store...[/yellow]")
        else:
            console.print("[yellow]📦 No vector store found. Building from documents...[/yellow]")

        with console.status("[bold green]Loading PDFs...[/bold green]"):
            docs = load_documents(settings.data_dir)

        with console.status("[bold green]Splitting into chunks...[/bold green]"):
            chunks = split_documents(docs)

        with console.status("[bold green]Embedding and storing...[/bold green]"):
            vector_store = create_vector_store(chunks)

        console.print(f"[green]✅ Vector store built: {len(chunks)} chunks indexed[/green]")
    else:
        with console.status("[bold green]Loading vector store...[/bold green]"):
            vector_store = load_vector_store()
        console.print("[green]✅ Vector store loaded[/green]")

    return vector_store


@app.command()
def build(
    rebuild: bool = typer.Option(
        False, "--rebuild", "-r",
        help="Force rebuild even if vector store already exists"
    ),
    data_dir: Optional[Path] = typer.Option(
        None, "--data-dir", "-d",
        help="Override data directory path"
    ),
):
    """
    [bold]Build[/bold] the vector store from PDFs in the data directory.

    Run this after adding new documents.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    try:
        _get_or_build_store(rebuild=True)
    except DataNotFoundError as e:
        console.print(f"[red]❌ Data error:[/red] {e}")
        raise typer.Exit(1)
    except EmbeddingError as e:
        console.print(f"[red]❌ Embedding error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def query_cmd(
    question: str = typer.Argument(..., help="Question to ask about your documents"),
    rebuild: bool = typer.Option(False, "--rebuild", "-r", help="Rebuild vector store first"),
):
    """
    [bold]Query[/bold] your documents with a single question.

    Example: rag query "What are the top tech jobs in 2025?"
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    try:
        vector_store = _get_or_build_store(rebuild=rebuild)

        with console.status("[bold green]Thinking...[/bold green]"):
            chain = build_rag_chain(vector_store)
            result = query(chain, vector_store, question)

        console.print(
            Panel(
                Markdown(result.answer),
                title="[bold blue]🧠 Answer[/bold blue]",
                border_style="blue",
            )
        )

        # Sources table
        table = Table(title="📚 Sources", show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=3)
        table.add_column("File", style="cyan")
        table.add_column("Page", justify="center")
        table.add_column("Preview", style="dim", max_width=60)

        for i, src in enumerate(result.sources, 1):
            table.add_row(str(i), src.file, str(src.page), f"{src.preview}...")

        console.print(table)

    except DataNotFoundError as e:
        console.print(f"[red]❌ Data error:[/red] {e}")
        raise typer.Exit(1)
    except LLMConnectionError as e:
        console.print(f"[red]❌ LLM error:[/red] {e}")
        raise typer.Exit(1)
    except VectorStoreError as e:
        console.print(f"[red]❌ Vector store error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def chat(
    rebuild: bool = typer.Option(False, "--rebuild", "-r", help="Rebuild vector store first"),
):
    """
    [bold]Chat[/bold] interactively — ask multiple questions in a session.

    Type 'exit' or press Ctrl+C to quit.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    console.print(
        Panel(
            f"[bold]{settings.app_name} v{settings.app_version}[/bold]\n"
            f"Model: [cyan]{settings.llm_model}[/cyan]  •  "
            f"Embeddings: [cyan]{settings.embedding_model.split('/')[-1]}[/cyan]  •  "
            f"Retrieval: [cyan]{settings.retriever_search_type.upper()} k={settings.retriever_k}[/cyan]\n\n"
            "[dim]Type your question and press Enter. Type 'exit' to quit.[/dim]",
            title="🔍 RAG System",
            border_style="green",
        )
    )

    try:
        vector_store = _get_or_build_store(rebuild=rebuild)
        chain = build_rag_chain(vector_store)
    except (DataNotFoundError, EmbeddingError, LLMConnectionError, VectorStoreError) as e:
        console.print(f"[red]❌ Startup error:[/red] {e}")
        raise typer.Exit(1)

    session_count = 0

    while True:
        try:
            question = typer.prompt("\n❓ You")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]👋 Goodbye![/dim]")
            break

        if question.lower().strip() in {"exit", "quit", "q", ":q"}:
            console.print("[dim]👋 Goodbye![/dim]")
            break

        if not question.strip():
            continue

        session_count += 1

        try:
            with console.status("[bold green]Thinking...[/bold green]"):
                result = query(chain, vector_store, question)

            console.print(
                Panel(
                    Markdown(result.answer),
                    title=f"[bold blue]🧠 Answer #{session_count}[/bold blue]",
                    border_style="blue",
                )
            )

            if result.sources:
                source_lines = [
                    f"  [dim]{i}.[/dim] [cyan]{s.file}[/cyan] — Page {s.page}"
                    for i, s in enumerate(result.sources, 1)
                ]
                console.print("[bold]📚 Sources:[/bold]")
                for line in source_lines:
                    console.print(line)

        except LLMConnectionError as e:
            console.print(f"[red]❌ LLM error:[/red] {e}")
        except Exception as e:
            console.print(f"[red]❌ Unexpected error:[/red] {e}")
            logger.exception("unexpected_query_error", error=str(e))


@app.command()
def info():
    """
    Show current [bold]configuration[/bold] and system status.
    """
    settings = get_settings()
    setup_logging(settings.log_level)

    table = Table(title="⚙️  Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    rows = [
        ("App Name", settings.app_name),
        ("Version", settings.app_version),
        ("LLM Model", settings.llm_model),
        ("LLM Temperature", str(settings.llm_temperature)),
        ("Embedding Model", settings.embedding_model),
        ("Data Directory", str(settings.data_dir.resolve())),
        ("ChromaDB Path", str(settings.chroma_db_path.resolve())),
        ("Collection Name", settings.collection_name),
        ("Retriever K", str(settings.retriever_k)),
        ("Search Type", settings.retriever_search_type),
        ("Chunk Size", str(settings.chunk_size)),
        ("Chunk Overlap", str(settings.chunk_overlap)),
        ("Log Level", settings.log_level),
        ("Vector Store Exists", "✅ Yes" if vector_store_exists() else "❌ No — run 'rag build'"),
        ("Data Dir Exists", "✅ Yes" if settings.data_dir.exists() else "❌ No"),
    ]

    for setting, value in rows:
        table.add_row(setting, value)

    console.print(table)


# Allow running as: python -m rag.cli
if __name__ == "__main__":
    app()
