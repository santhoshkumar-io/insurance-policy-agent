"""Command line interface.

The week-1 interface is a CLI, not a browser. A UI cannot tell you whether retrieval is
working and it slows the inner loop; the Next.js dashboard arrives in week 2 (plan.md §3).

    policy ingest samples/policy.pdf --product "Health Guard" --insurer "Acme"
    policy ask "What is the waiting period for pre-existing diseases?"
    policy policies
    policy inspect <document_id> --page 14
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table as RichTable
from sqlalchemy import func, select

from db.models import Chunk, Document, DocumentStatus, Policy
from db.session import get_or_create_dev_tenant, scoped, session_scope
from rag.config import get_settings
from rag.embeddings.openai_embeddings import embed_query
from rag.generation.answer import generate_answer
from rag.ingestion.pipeline import ingest_document
from rag.retrieval.vector import retrieve

app = typer.Typer(add_completion=False, help="Insurance Policy Intelligence AI")
console = Console()


@app.command()
def ingest(
    path: Path = typer.Argument(..., exists=True, readable=True),
    product: str = typer.Option(..., "--product", help="Product name"),
    insurer: str = typer.Option("", "--insurer"),
    version: str = typer.Option("", "--version", help="Policy version identifier"),
) -> None:
    """Ingest a policy PDF: parse, chunk, embed, store."""
    settings = get_settings()
    data = path.read_bytes()
    if not data.startswith(b"%PDF-"):
        console.print(f"[red]{path.name} is not a PDF (missing %PDF- header)[/red]")
        raise typer.Exit(1)

    checksum = hashlib.sha256(data).hexdigest()

    with session_scope() as session:
        tenant = get_or_create_dev_tenant(session)
        existing = session.scalar(
            scoped(select(Document).where(Document.checksum == checksum), Document, tenant.id)
        )
        if existing is not None:
            console.print(
                f"[yellow]Already ingested as {existing.id} "
                f"(status: {existing.status}). Skipping.[/yellow]"
            )
            return

        policy = Policy(
            tenant_id=tenant.id, insurer=insurer, product_name=product, policy_version=version
        )
        session.add(policy)
        session.flush()

        storage_dir = Path(settings.storage_dir) / str(tenant.id)
        storage_dir.mkdir(parents=True, exist_ok=True)
        storage_path = storage_dir / f"{checksum}.pdf"
        storage_path.write_bytes(data)

        document = Document(
            tenant_id=tenant.id,
            policy_id=policy.id,
            filename=path.name,
            storage_path=str(storage_path),
            checksum=checksum,
            byte_size=len(data),
            status=DocumentStatus.PENDING,
        )
        session.add(document)
        session.flush()
        document_id, tenant_id = document.id, tenant.id

    with console.status(f"Ingesting {path.name}..."):
        with session_scope() as session:
            count = ingest_document(session, document_id, tenant_id)

    with session_scope() as session:
        document = session.get(Document, document_id)
        pages = session.execute(
            select(func.min(Chunk.page_from), func.max(Chunk.page_to)).where(
                Chunk.document_id == document_id
            )
        ).one()
        by_type = session.execute(
            select(Chunk.chunk_type, func.count(Chunk.id))
            .where(Chunk.document_id == document_id)
            .group_by(Chunk.chunk_type)
        ).all()
        sections = session.scalars(
            select(Chunk.section_path)
            .where(Chunk.document_id == document_id, Chunk.section_path != "")
            .distinct()
            .limit(8)
        ).all()

        console.print(f"[green]Ingested[/green] {document.filename}")
        console.print(f"  document_id : {document_id}")
        console.print(f"  policy_id   : {document.policy_id}")
        console.print(f"  pages       : {document.page_count} (chunks span {pages[0]}-{pages[1]})")
        console.print(f"  chunks      : {count} ({', '.join(f'{t}={n}' for t, n in by_type)})")
        if sections:
            console.print("  sections    :")
            for section in sections:
                console.print(f"    - {section}")
        else:
            console.print("  sections    : [yellow]none detected (windowed fallback)[/yellow]")


@app.command()
def ask(
    question: str = typer.Argument(...),
    policy: list[uuid.UUID] = typer.Option(None, "--policy", help="Restrict to policy id(s)"),
    top_k: int = typer.Option(None, "--top-k"),
    show_sources: bool = typer.Option(False, "--sources", help="Print the retrieved text"),
) -> None:
    """Ask a question and print the answer with its validated citations."""
    with session_scope() as session:
        tenant = get_or_create_dev_tenant(session)
        vector = embed_query(question)
        chunks = retrieve(
            session,
            tenant_id=tenant.id,
            query_vector=vector,
            policy_ids=list(policy) if policy else None,
            top_k=top_k,
        )
        answer = generate_answer(question, chunks)

    console.rule(f"[bold]{question}")

    if not answer.answerable:
        console.print(f"[yellow]ABSTAINED[/yellow] {answer.text}")
        console.print(f"[dim]Reason: {answer.abstained_reason}[/dim]")
    else:
        console.print(Markdown(answer.text))

    if answer.citations:
        table = RichTable(title="Citations (validated against the retrieved set)")
        table.add_column("ref")
        table.add_column("page")
        table.add_column("type")
        table.add_column("section", overflow="fold")
        table.add_column("score", justify="right")
        for chunk in answer.citations:
            pages = (
                str(chunk.page_from)
                if chunk.page_from == chunk.page_to
                else f"{chunk.page_from}-{chunk.page_to}"
            )
            table.add_row(
                chunk.short_id,
                pages,
                chunk.chunk_type,
                chunk.section_path or "-",
                f"{chunk.score:.3f}",
            )
        console.print(table)

    if show_sources:
        for chunk in chunks:
            console.rule(f"[dim]{chunk.short_id} p{chunk.page_from} {chunk.section_path}")
            console.print(chunk.text[:1200])

    console.print(
        f"[dim]model={answer.model} prompt={answer.prompt_version} "
        f"latency={answer.latency_ms}ms retries={answer.validation_retries} "
        f"retrieved={len(chunks)}[/dim]"
    )


@app.command("policies")
def list_policies() -> None:
    """List policies and their ingestion status."""
    with session_scope() as session:
        tenant = get_or_create_dev_tenant(session)
        rows = session.execute(
            scoped(
                select(
                    Policy.id,
                    Policy.product_name,
                    Policy.insurer,
                    Policy.policy_version,
                    Document.filename,
                    Document.status,
                    Document.page_count,
                    func.count(Chunk.id),
                )
                .join(Document, Document.policy_id == Policy.id, isouter=True)
                .join(Chunk, Chunk.document_id == Document.id, isouter=True)
                .group_by(
                    Policy.id,
                    Policy.product_name,
                    Policy.insurer,
                    Policy.policy_version,
                    Document.filename,
                    Document.status,
                    Document.page_count,
                ),
                Policy,
                tenant.id,
            )
        ).all()

    table = RichTable(title="Policies")
    columns = ("policy_id", "product", "insurer", "version", "file", "status", "pages", "chunks")
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in rows:
        table.add_row(*(str(v) if v is not None else "-" for v in row))
    console.print(table)


@app.command()
def inspect(
    document_id: uuid.UUID = typer.Argument(...),
    page: int = typer.Option(None, "--page", help="Only chunks touching this page"),
    limit: int = typer.Option(10, "--limit"),
) -> None:
    """Print chunks for a document. Use this to check chunk quality by eye."""
    with session_scope() as session:
        tenant = get_or_create_dev_tenant(session)
        stmt = scoped(
            select(Chunk).where(Chunk.document_id == document_id).order_by(Chunk.ordinal),
            Chunk,
            tenant.id,
        )
        if page is not None:
            stmt = stmt.where(Chunk.page_from <= page, Chunk.page_to >= page)
        for chunk in session.scalars(stmt.limit(limit)):
            console.rule(
                f"[bold]#{chunk.ordinal}[/bold] p{chunk.page_from}-{chunk.page_to} "
                f"{chunk.chunk_type} | {chunk.section_path or '(no section)'}"
            )
            console.print(chunk.text[:1500])


if __name__ == "__main__":
    app()
