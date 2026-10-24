"""Run one named configuration over the frozen question set.

Two modes:

* retrieval-only — no model call, so retrieval rows are cheap and free of generation
  variance. This is what the experiment matrix uses for Recall@k, MRR and nDCG.
* with generation — adds abstention error rates and token cost, which need the model.

Both go through :func:`rag.retrieval.search.search`, the same entry point the API uses.
A harness with its own retrieval path measures the harness.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Policy
from db.session import scoped
from rag.config import get_settings
from rag.embeddings.openai_embeddings import embed_query
from rag.evaluation.dataset import EvalQuestion, EvalSet, load_eval_set
from rag.evaluation.metrics import (
    QuestionResult,
    RunMetrics,
    aggregate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    section_hit,
)
from rag.retrieval.search import search
from rag.retrieval.types import RetrievalMode

RESULTS_DIR = Path("evals/results")


@dataclass
class Configuration:
    """One row of the experiment matrix."""

    name: str
    mode: RetrievalMode
    rerank: bool = False

    @property
    def slug(self) -> str:
        return self.name.lower().replace(" ", "-").replace("(", "").replace(")", "")


def default_matrix() -> list[Configuration]:
    """The matrix from plan.md §6, in the order the results should be read.

    The hybrid rows are the point; the single-arm rows are the baselines that make the
    hybrid number mean something, and they stay in the table whether they win or lose.
    """
    return [
        Configuration("vector", RetrievalMode.VECTOR),
        Configuration("keyword (ts_rank_cd)", RetrievalMode.KEYWORD),
        Configuration("bm25 (rank_bm25)", RetrievalMode.BM25),
        Configuration("hybrid RRF(vector + ts_rank_cd)", RetrievalMode.HYBRID),
        Configuration("hybrid RRF(vector + bm25)", RetrievalMode.HYBRID_BM25),
    ]


def resolve_policy_ids(session: Session, tenant_id: uuid.UUID, eval_set: EvalSet) -> dict:
    """Map each question's ``policy`` label to a real policy id in this tenant.

    Matching is on the product name, case-insensitively and ignoring punctuation, so the
    YAML can stay readable instead of carrying UUIDs that change on every re-ingest.
    """

    def normalise(value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    policies = list(session.scalars(scoped(select(Policy), Policy, tenant_id)))
    by_name = {normalise(p.product_name): p.id for p in policies}

    mapping: dict[str, uuid.UUID | None] = {}
    for question in eval_set.questions:
        key = normalise(question.policy)
        match = by_name.get(key)
        if match is None:
            # Fall back to a prefix match so "acme-health-guard" finds "Acme Health Guard".
            match = next((pid for name, pid in by_name.items() if key in name or name in key), None)
        mapping[question.policy] = match
    return mapping


def run_question(
    session: Session,
    tenant_id: uuid.UUID,
    question: EvalQuestion,
    config: Configuration,
    k: int,
    policy_id: uuid.UUID | None,
    query_vector: list[float] | None = None,
    generate: bool = False,
) -> QuestionResult:
    started = time.time()
    chunks = search(
        session,
        tenant_id,
        question.question,
        policy_ids=[policy_id] if policy_id else None,
        top_k=max(k, get_settings().retrieval_top_k),
        mode=config.mode,
        rerank=config.rerank,
        query_vector=query_vector,
    )
    latency = int((time.time() - started) * 1000)

    result = QuestionResult(
        question_id=question.id,
        answerable=question.answerable,
        recall=recall_at_k(question, chunks, k),
        rr=reciprocal_rank(question, chunks),
        ndcg=ndcg_at_k(question, chunks, k),
        section=section_hit(question, chunks, k),
        retrieved=len(chunks),
        latency_ms=latency,
    )

    if generate:
        from rag.generation.answer import generate_answer

        answer = generate_answer(question.question, chunks)
        result.abstained = not answer.answerable
        result.tokens = int(answer.usage.get("total_tokens", 0) or 0)
        result.latency_ms = latency + answer.latency_ms

    return result


def run_configuration(
    session: Session,
    tenant_id: uuid.UUID,
    config: Configuration,
    *,
    k: int = 5,
    eval_set: EvalSet | None = None,
    generate: bool = False,
    vector_cache: dict[str, list[float]] | None = None,
) -> RunMetrics:
    eval_set = eval_set or load_eval_set()
    policy_ids = resolve_policy_ids(session, tenant_id, eval_set)
    vector_cache = vector_cache if vector_cache is not None else {}

    results: list[QuestionResult] = []
    for question in eval_set.questions:
        # Embed for *every* mode, not just the vector-using ones. Keyword-only retrieval
        # still needs a cosine score for the confidence signal, so the embedding is part
        # of every request's cost. Leaving it out of the cache here made the keyword rows
        # pay a ~300ms network round trip inside the timed section and report it as
        # retrieval latency, which flattered vector search by 25x.
        if question.question not in vector_cache:
            vector_cache[question.question] = embed_query(question.question)
        query_vector = vector_cache[question.question]

        result = run_question(
            session,
            tenant_id,
            question,
            config,
            k,
            policy_ids.get(question.policy),
            query_vector=query_vector,
            generate=generate,
        )
        if policy_ids.get(question.policy) is None:
            result.notes = f"policy '{question.policy}' not ingested"
        results.append(result)

    return aggregate(config.name, k, results)


def save_run(metrics: RunMetrics, config: Configuration) -> Path:
    """Persist the raw per-question output so a table row can be traced back."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{config.slug}-k{metrics.k}.json"
    payload = asdict(metrics)
    payload["config"] = {"mode": str(config.mode), "rerank": config.rerank}
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
