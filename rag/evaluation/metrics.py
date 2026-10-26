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


def conditions_covered(question: EvalQuestion, chunks: list[RetrievedChunk], k: int) -> bool:
    """Whether *every* labelled page reached the model, not merely one of them.

    A conditional question is only answered well if all the competing conditions are in
    front of the model. Surfacing one of three waiting periods and answering confidently
    is the failure this metric exists to catch -- and ordinary Recall@k would score it a
    full hit.
    """
    if not question.expected_pages:
        return False
    top = chunks[:k]
    return all(any(chunk.covers_page(page) for chunk in top) for page in question.expected_pages)


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
    # Conditional questions leave the binary abstention rates and are scored on whether
    # retrieval surfaced *every* labelled condition instead (see Answerability).
    conditional: bool = False
    conditions_covered: bool = False
    recall: float = 0.0
    rr: float = 0.0
    ndcg: float = 0.0
    section: bool = False
    retrieved: int = 0
    latency_ms: int = 0
    # Only set when a generation pass ran.
    abstained: bool | None = None
    # One entry per repeat when --repeats > 1. Generation is not deterministic even at
    # temperature 0: a benchmark question was measured answering 5 times and abstaining 5
    # out of 10, which is why a single run cannot tell an improvement from a coin landing
    # differently (docs/experiments.md, finding 7).
    abstained_runs: list[bool] = field(default_factory=list)
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
    # Half the range across repeats: the "± 6" in "12% ± 6". None for a single run, which
    # is the honest rendering -- one run carries no information about its own spread.
    false_answer_spread: float | None = None
    false_abstention_spread: float | None = None
    repeats: int = 1
    tokens_per_100q: int | None = None
    # None when the set contains no conditional questions.
    conditional_coverage: float | None = None
    per_question: list[QuestionResult] = field(default_factory=list)

    def as_row(self) -> dict:
        def pct(value: float | None, spread: float | None = None) -> str:
            if value is None:
                return "—"
            if spread is None:
                return f"{value * 100:.0f}%"
            return f"{value * 100:.0f}% ± {spread * 100:.0f}"

        return {
            "experiment": self.name,
            "recall@k": f"{self.recall_at_k:.3f}",
            "mrr": f"{self.mrr:.3f}",
            "ndcg@k": f"{self.ndcg_at_k:.3f}",
            "section": f"{self.section_accuracy:.3f}",
            "false answer": pct(self.false_answer_rate, self.false_answer_spread),
            "false abstain": pct(self.false_abstention_rate, self.false_abstention_spread),
            "cond. coverage": pct(self.conditional_coverage),
            "p50 ms": f"{self.p50_latency_ms:.0f}",
            "tokens/100q": "—" if self.tokens_per_100q is None else f"{self.tokens_per_100q:,}",
        }


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _outcome(result: QuestionResult, pass_index: int) -> bool | None:
    """This question's abstention outcome on one pass.

    ``abstained_runs`` is empty for a single-pass run, where ``abstained`` is the whole
    answer; it holds one entry per repeat otherwise.
    """
    if result.abstained_runs:
        if pass_index < len(result.abstained_runs):
            return result.abstained_runs[pass_index]
        return None
    return result.abstained if pass_index == 0 else None


def _rate(
    candidates: list[QuestionResult],
    denominator: list[QuestionResult],
    pass_index: int,
    *,
    abstained_is: bool,
) -> float | None:
    """How many of ``candidates`` went the wrong way on this pass, over ``denominator``."""
    if not denominator:
        return None
    wrong = [r for r in candidates if _outcome(r, pass_index) is abstained_is]
    return len(wrong) / len(denominator)


def _half_range(rates: list[float | None]) -> float | None:
    """Half the spread across passes: the "± 6" in "12% ± 6".

    Half-range rather than a standard deviation because the sample is small -- three to
    ten passes -- and "the rate landed anywhere in this band" is the claim the number can
    actually support.
    """
    present = [r for r in rates if r is not None]
    if len(present) < 2:
        return None
    return (max(present) - min(present)) / 2.0


def aggregate(name: str, k: int, results: list[QuestionResult]) -> RunMetrics:
    answerable = [r for r in results if r.answerable]
    unanswerable = [r for r in results if not r.answerable]
    latencies = [float(r.latency_ms) for r in results] or [0.0]

    conditional = [r for r in results if r.conditional]
    # Conditional questions are excluded from both binary rates: neither answering nor
    # abstaining is wrong when the honest response is "it depends".
    binary_answerable = [r for r in answerable if not r.conditional]

    generated = [r for r in results if r.abstained is not None]
    false_answer = false_abstention = tokens_per_100q = None
    false_answer_spread = false_abstention_spread = None
    repeats = max((len(r.abstained_runs) for r in generated), default=1) or 1
    if generated:
        # With repeats, each rate is computed once per pass and then averaged, so the
        # spread describes the variation in *the rate* -- which is the number that gets
        # published and compared between phases. Averaging the per-question outcomes
        # first would hide exactly the instability this exists to expose.
        answer_rates = [
            _rate(
                [r for r in generated if not r.answerable],
                unanswerable,
                pass_index,
                abstained_is=False,
            )
            for pass_index in range(repeats)
        ]
        abstain_rates = [
            _rate(
                [r for r in generated if r.answerable and not r.conditional],
                binary_answerable,
                pass_index,
                abstained_is=True,
            )
            for pass_index in range(repeats)
        ]
        false_answer = _mean([r for r in answer_rates if r is not None]) if unanswerable else None
        false_abstention = (
            _mean([r for r in abstain_rates if r is not None]) if binary_answerable else None
        )
        if repeats > 1:
            false_answer_spread = _half_range(answer_rates)
            false_abstention_spread = _half_range(abstain_rates)
        total_tokens = sum(r.tokens for r in generated)
        tokens_per_100q = round(total_tokens / len(generated) * 100)

    conditional_coverage = (
        _mean([1.0 if r.conditions_covered else 0.0 for r in conditional]) if conditional else None
    )

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
        false_answer_spread=false_answer_spread,
        false_abstention_spread=false_abstention_spread,
        repeats=repeats,
        tokens_per_100q=tokens_per_100q,
        conditional_coverage=conditional_coverage,
        per_question=results,
    )
