"""Command line interface.

The CLI is the development interface, not a stopgap: a browser cannot tell you whether
retrieval is working, and `inspect` and `eval` are how chunking and ranking regressions
get caught. The dashboard comes after retrieval settles.

    policy ingest samples/policy.pdf --product "Health Guard" --insurer "Acme"
    policy ask "What is the waiting period for pre-existing diseases?" --mode hybrid
    policy policies
    policy inspect <document_id> --page 14
    policy reindex --all
    policy eval --mode hybrid
    policy experiments
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
from rag.generation.answer import generate_answer
from rag.ingestion.pipeline import ingest_document
from rag.retrieval.search import search
from rag.retrieval.types import RetrievalMode

app = typer.Typer(add_completion=False, help="Insurance Policy Intelligence AI")
console = Console()


# The suites live side by side: the synthetic one is a deterministic fixture that runs
# without the real PDFs (which are gitignored), and the LIC one is the headline benchmark.
# Mixing them would let an easy 4-page corpus inflate the reported numbers.
EVAL_SETS = {
    "acme-synthetic": Path("evals/questions.yaml"),
    "lic-real": Path("evals/questions-lic.yaml"),
}


def _resolve_set(name: str) -> Path:
    if name in EVAL_SETS:
        return EVAL_SETS[name]
    candidate = Path(name)
    if candidate.exists():
        return candidate
    console.print(
        f"[red]Unknown question set {name!r}.[/red] Known: {', '.join(EVAL_SETS)} "
        f"(or pass a path to a YAML file)."
    )
    raise typer.Exit(1)


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
    mode: RetrievalMode = typer.Option(None, "--mode", help="Retrieval strategy"),
    rerank: bool = typer.Option(None, "--rerank/--no-rerank", help="Cross-encoder rerank"),
    section_tiebreak: bool = typer.Option(
        None, "--section-tiebreak/--no-section-tiebreak", help="Section-path tie-breaker"
    ),
    show_sources: bool = typer.Option(False, "--sources", help="Print the retrieved text"),
) -> None:
    """Ask a question and print the answer with its validated citations."""
    with session_scope() as session:
        tenant = get_or_create_dev_tenant(session)
        chunks = search(
            session,
            tenant.id,
            question,
            policy_ids=list(policy) if policy else None,
            top_k=top_k,
            mode=mode,
            rerank=rerank,
            section_tiebreak=section_tiebreak,
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
        table.add_column("found by")
        table.add_column("section", overflow="fold")
        table.add_column("rank score", justify="right")
        table.add_column("cosine", justify="right")
        for chunk in answer.citations:
            table.add_row(
                chunk.short_id,
                chunk.pages_label,
                chunk.chunk_type,
                str(chunk.retrieved_by),
                chunk.section_path or "-",
                f"{chunk.score:.4f}",
                f"{chunk.confidence:.3f}",
            )
        console.print(table)

    if show_sources:
        for chunk in chunks:
            console.rule(
                f"[dim]{chunk.short_id} p{chunk.pages_label} "
                f"({chunk.retrieved_by}, rank {chunk.rank}) {chunk.section_path}"
            )
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


@app.command()
def reindex(
    document_id: uuid.UUID = typer.Argument(None),
    all_documents: bool = typer.Option(False, "--all", help="Re-index every document"),
) -> None:
    """Re-chunk and re-embed from the stored PDF.

    Needed whenever chunking or the embedding model changes. Without it, improving the
    chunker means deleting and re-uploading every document, which is how a corpus ends
    up half-indexed by an old strategy.
    """
    if not all_documents and document_id is None:
        console.print("[red]Pass a document id or --all[/red]")
        raise typer.Exit(1)

    with session_scope() as session:
        tenant = get_or_create_dev_tenant(session)
        stmt = scoped(select(Document.id), Document, tenant.id)
        if not all_documents:
            stmt = stmt.where(Document.id == document_id)
        ids = list(session.scalars(stmt))
        tenant_id = tenant.id

    if not ids:
        console.print("[yellow]Nothing to re-index.[/yellow]")
        return

    failed = 0
    for one in ids:
        try:
            with console.status(f"Re-indexing {one}..."):
                with session_scope() as session:
                    count = ingest_document(session, one, tenant_id)
            console.print(f"[green]ok[/green] {one} -> {count} chunks")
        except Exception as exc:
            failed += 1
            console.print(f"[red]failed[/red] {one}: {exc}")

    console.print(f"Re-indexed {len(ids) - failed}/{len(ids)} document(s).")


@app.command()
def eval(
    question_set: str = typer.Option(
        "acme-synthetic", "--set", help="acme-synthetic | lic-real | path to a YAML file"
    ),
    mode: RetrievalMode = typer.Option(None, "--mode", help="Retrieval strategy"),
    rerank: bool = typer.Option(False, "--rerank", help="Apply the cross-encoder"),
    section_tiebreak: bool = typer.Option(
        False, "--section-tiebreak", help="Apply the section-path tie-breaker"
    ),
    k: int = typer.Option(5, "--k", help="Cut-off for Recall@k and nDCG@k"),
    generate: bool = typer.Option(
        False, "--generate", help="Also call the model, for abstention and token metrics"
    ),
) -> None:
    """Score one retrieval configuration against the frozen question set."""
    from rag.evaluation.dataset import load_eval_set
    from rag.evaluation.runner import Configuration, run_configuration, save_run

    settings = get_settings()
    config = Configuration(
        name=(
            str(mode or settings.retrieval_mode)
            + (" + rerank" if rerank else "")
            + (" + section tie-break" if section_tiebreak else "")
        ),
        mode=RetrievalMode(mode or settings.retrieval_mode),
        rerank=rerank,
        section_tiebreak=section_tiebreak,
    )
    eval_set = load_eval_set(_resolve_set(question_set))

    with console.status(f"Running {config.name} over {len(eval_set.questions)} questions..."):
        with session_scope() as session:
            tenant = get_or_create_dev_tenant(session)
            metrics = run_configuration(
                session, tenant.id, config, k=k, eval_set=eval_set, generate=generate
            )

    _print_runs([metrics], k)
    path = save_run(metrics, config, eval_set.set_name)
    console.print(f"[dim]per-question output: {path}[/dim]")

    unmatched = [r for r in metrics.per_question if r.notes]
    if unmatched:
        console.print(
            f"[yellow]{len(unmatched)} question(s) reference a policy that is not "
            f"ingested; they scored zero. Ingest it or fix the `policy:` label.[/yellow]"
        )
        for result in unmatched[:5]:
            console.print(f"  [dim]{result.question_id}: {result.notes}[/dim]")


@app.command()
def experiments(
    question_set: str = typer.Option(
        "acme-synthetic", "--set", help="acme-synthetic | lic-real | path to a YAML file"
    ),
    k: int = typer.Option(5, "--k"),
    rerank: bool = typer.Option(
        False, "--rerank", help="Add a reranked row (needs the [rerank] extra)"
    ),
    generate: bool = typer.Option(
        False, "--generate", help="Also measure abstention and token cost"
    ),
) -> None:
    """Run the whole matrix and write docs/experiments.md."""
    from rag.evaluation import report
    from rag.evaluation.dataset import load_eval_set
    from rag.evaluation.runner import (
        Configuration,
        default_matrix,
        run_configuration,
        save_run,
    )
    from rag.retrieval.types import RetrievalMode as Mode

    matrix = default_matrix()
    if rerank:
        # Rerank on top of *both* hybrids: the ts_rank_cd pair for continuity with the
        # earlier runs, and the BM25 pair because that is the configuration leading on
        # recall and therefore the one whose ranking is worth improving.
        matrix += [
            Configuration("hybrid RRF(vector + ts_rank_cd) + rerank", Mode.HYBRID, rerank=True),
            Configuration(
                "hybrid RRF(vector + bm25) + rerank", Mode.HYBRID_BM25, rerank=True
            ),
        ]

    eval_set = load_eval_set(_resolve_set(question_set))
    # One embedding per question, shared across every configuration in the matrix.
    vector_cache: dict = {}
    runs = []

    with session_scope() as session:
        tenant = get_or_create_dev_tenant(session)
        for config in matrix:
            with console.status(f"{config.name}..."):
                metrics = run_configuration(
                    session,
                    tenant.id,
                    config,
                    k=k,
                    eval_set=eval_set,
                    generate=generate,
                    vector_cache=vector_cache,
                )
            runs.append(metrics)
            save_run(metrics, config, eval_set.set_name)
            console.print(
                f"  [green]done[/green] {config.name}: "
                f"recall@{k}={metrics.recall_at_k:.3f} mrr={metrics.mrr:.3f}"
            )

    _print_runs(runs, k)
    path = report.write(runs, eval_set, k)
    console.print(f"[green]Wrote[/green] {path}")


def _print_runs(runs: list, k: int) -> None:
    from rag.evaluation.report import COLUMNS

    table = RichTable(title=f"Retrieval experiments (k={k})")
    for _, label in COLUMNS:
        table.add_column(label, overflow="fold")
    for run in runs:
        row = run.as_row()
        table.add_row(*(str(row[key]) for key, _ in COLUMNS))
    console.print(table)


@app.command("eval-draft")
def eval_draft(
    policy_id: uuid.UUID = typer.Argument(...),
    out: Path = typer.Option(None, "--out", help="Write the YAML stub here"),
    limit: int = typer.Option(None, "--limit", help="Only the first N chunks"),
) -> None:
    """Propose candidate benchmark questions from an ingested policy.

    Candidates only. Nothing here is a label until a human has read it against the
    document -- a benchmark that labels itself measures the model's agreement with
    itself, and every number built on it is worthless.
    """
    from rag.evaluation.drafting import draft_for_policy, to_yaml_stub

    with session_scope() as session:
        tenant = get_or_create_dev_tenant(session)
        policy = session.get(Policy, policy_id)
        if policy is None or policy.tenant_id != tenant.id:
            console.print("[red]No such policy in this tenant.[/red]")
            raise typer.Exit(1)
        label = policy.product_name.lower().replace(" ", "-")
        with console.status("Drafting candidates..."):
            candidates = draft_for_policy(
                session, tenant.id, policy_id, limit=limit
            )

    stub = to_yaml_stub(candidates, label)
    if out:
        out.write_text(stub)
        console.print(f"[green]Wrote[/green] {len(candidates)} candidates to {out}")
    else:
        console.print(stub)
    console.print(
        "[yellow]Verify every candidate against the source before using any of "
        "them as a label.[/yellow]"
    )


if __name__ == "__main__":
    app()
