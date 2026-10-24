"""Retrieval and abstention metrics.

Evidence is matched at **page level**, not by chunk id. That is a deliberate choice: the
chunker changes, and chunk-id labels would be invalidated by every chunking improvement --
turning the benchmark into something that silently measures whether chunk boundaries moved.
Page numbers are a property of the document, so a label written once stays valid.

Both abstention error directions are reported. Driving either to zero alone is trivial and
worthless: a system that abstains on everything is perfectly grounded and useless.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field

from rag.evaluation.dataset import EvalQuestion
from rag.retrieval.types import RetrievedChunk


def hits(question: EvalQuestion, chunks: list[RetrievedChunk]) -> list[bool]:
    """Per-rank relevance: does this chunk's page range cover any expected page?"""
    expected = set(question.expected_pages)
    return [any(chunk.covers_page(page) for page in expected) for chunk in chunks]


def section_hit(question: EvalQuestion, chunks: list[RetrievedChunk], k: int) -> bool:
    """Stricter secondary signal: did the right *section* surface, not just the page?

    A page can hold several clauses. This catches the case where retrieval lands on the
    correct page for the wrong reason.
    """
    if not question.expected_section:
        return False
    wanted = question.expected_section.strip().lower()
    for chunk in chunks[:k]:
        path = (chunk.section_path or "").strip().lower()
        if path and (path in wanted or wanted in path):
            return True
    return False


def recall_at_k(question: EvalQuestion, chunks: list[RetrievedChunk], k: int) -> float:
    """Did the expected evidence appear anywhere in the top k?

    Binary per question: these labels mark the page(s) that answer it, not an exhaustive
    set of every relevant passage, so a proportion-of-all-relevant recall would be
    measuring the completeness of the labelling rather than the retrieval.
    """
    return 1.0 if any(hits(question, chunks)[:k]) else 0.0


def reciprocal_rank(question: EvalQuestion, chunks: list[RetrievedChunk]) -> float:
    for index, hit in enumerate(hits(question, chunks), start=1):
        if hit:
            return 1.0 / index
    return 0.0


def ndcg_at_k(question: EvalQuestion, chunks: list[RetrievedChunk], k: int) -> float:
    """Binary-gain nDCG. With one relevant item the ideal DCG is 1, so this reduces to
    1/log2(rank+1) -- a gentler rank penalty than MRR, reported alongside it."""
    gains = hits(question, chunks)[:k]
    dcg = sum(1.0 / math.log2(i + 1) for i, hit in enumerate(gains, start=1) if hit)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, max(1, sum(gains))) + 1))
    return dcg / ideal if ideal else 0.0


@dataclass
class QuestionResult:
    question_id: str
    answerable: bool
    recall: float = 0.0
    rr: float = 0.0
    ndcg: float = 0.0
    section: bool = False
    retrieved: int = 0
    latency_ms: int = 0
    # Only set when a generation pass ran.
    abstained: bool | None = None
    tokens: int = 0
    notes: str = ""


@dataclass
class RunMetrics:
    name: str
    k: int
    questions: int
    answerable: int
    unanswerable: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    section_accuracy: float
    p50_latency_ms: float
    p95_latency_ms: float
    # None when the run was retrieval-only.
    false_answer_rate: float | None = None
    false_abstention_rate: float | None = None
    tokens_per_100q: int | None = None
    per_question: list[QuestionResult] = field(default_factory=list)

    def as_row(self) -> dict:
        def pct(value: float | None) -> str:
            return "—" if value is None else f"{value * 100:.0f}%"

        return {
            "experiment": self.name,
            "recall@k": f"{self.recall_at_k:.3f}",
            "mrr": f"{self.mrr:.3f}",
            "ndcg@k": f"{self.ndcg_at_k:.3f}",
            "section": f"{self.section_accuracy:.3f}",
            "false answer": pct(self.false_answer_rate),
            "false abstain": pct(self.false_abstention_rate),
            "p50 ms": f"{self.p50_latency_ms:.0f}",
            "tokens/100q": "—" if self.tokens_per_100q is None else f"{self.tokens_per_100q:,}",
        }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def aggregate(name: str, k: int, results: list[QuestionResult]) -> RunMetrics:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    latencies = [float(r.latency_ms) for r in results] or [0.0]

    generated = [r for r in results if r.abstained is not None]
    false_answer = false_abstention = tokens_per_100q = None
    if generated:
        bad_answers = [r for r in generated if not r.answerable and r.abstained is False]
        bad_abstentions = [r for r in generated if r.answerable and r.abstained is True]
        if unanswerable:
            false_answer = len(bad_answers) / len(unanswerable)
        if answerable:
            false_abstention = len(bad_abstentions) / len(answerable)
        total_tokens = sum(r.tokens for r in generated)
        tokens_per_100q = round(total_tokens / len(generated) * 100)

    return RunMetrics(
        name=name,
        k=k,
        questions=len(results),
        answerable=len(answerable),
        unanswerable=len(unanswerable),
        # Retrieval quality is only meaningful where there is evidence to find.
        recall_at_k=_mean([r.recall for r in answerable]),
        mrr=_mean([r.rr for r in answerable]),
        ndcg_at_k=_mean([r.ndcg for r in answerable]),
        section_accuracy=_mean([1.0 if r.section else 0.0 for r in answerable]),
        p50_latency_ms=statistics.median(latencies),
        p95_latency_ms=sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        false_answer_rate=false_answer,
        false_abstention_rate=false_abstention,
        tokens_per_100q=tokens_per_100q,
        per_question=results,
    )
