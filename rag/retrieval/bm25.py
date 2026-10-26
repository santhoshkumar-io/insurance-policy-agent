"""BM25 over a cached in-process index.

PostgreSQL cannot do BM25 (see rag/retrieval/keyword.py), and on the real corpus BM25
beat `ts_rank_cd` by 0.187 MRR -- large enough that the lexical arm was worth making
deployable rather than leaving as a benchmark curiosity.

The first version rebuilt a `BM25Okapi` index from a full table scan on *every query*.
That is fine for scoring a benchmark and unusable for serving, which is why the retrieval
default stayed on vector search despite hybrid measuring better. This version builds the
index once per scope and reuses it.

WHY NOT pg_search
    ParadeDB's `pg_search` extension is the right long-term answer and is not available on
    managed PostgreSQL (Neon, Supabase), which is the deployment target. Keeping BM25
    in-process keeps the cheapest hosting option open. When the corpus outgrows this, the
    `search()` interface means pg_search arrives as another arm rather than a rewrite.

WHAT THIS COSTS, STATED PLAINLY
    * Each worker process holds its own index, so memory and cold-start cost multiply by
      worker count.
    * The first query after an ingest pays a rebuild.
    * Sound at this corpus size; revisit well before ~10^5 chunks.
    `policy bm25-stats` reports the measured build cost rather than leaving it to faith.
"""

from __future__ import annotations

import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from db.models import Chunk
from db.session import scoped
from rag.config import get_settings
from rag.retrieval.types import RetrievedBy, RetrievedChunk
from rag.retrieval.vector import base_statement, build_chunk

_TOKEN = re.compile(r"[a-z0-9][a-z0-9./-]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# (tenant_id, policy scope) -> index. The tenant is part of the key, not an afterthought:
# a cache keyed on the policy set alone would serve one tenant's index to another, which
# is the single failure in this design that would be a security bug rather than a
# performance one. tests/test_bm25_cache.py asserts it.
ScopeKey = tuple[uuid.UUID, frozenset | None]


@dataclass
class _Index:
    """A built BM25 index plus what is needed to detect that it has gone stale."""

    bm25: object
    chunk_ids: list[uuid.UUID]
    fingerprint: tuple
    build_ms: int = 0
    hits: int = 0
    tokens: int = field(default=0)


class _Cache:
    """A small LRU of built indexes, with hit/miss counters for reporting."""

    def __init__(self) -> None:
        self._entries: OrderedDict[ScopeKey, _Index] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.rebuilds = 0

    @property
    def max_size(self) -> int:
        return max(1, get_settings().bm25_cache_size)

    def get(self, key: ScopeKey, fingerprint: tuple) -> _Index | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry.fingerprint != fingerprint:
            # The corpus changed under us: a stale index is worse than a slow one.
            self.rebuilds += 1
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        entry.hits += 1
        return entry

    def put(self, key: ScopeKey, entry: _Index) -> None:
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> dict:
        return {
            "scopes_cached": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "rebuilds_from_staleness": self.rebuilds,
            "entries": [
                {
                    "chunks": len(e.chunk_ids),
                    "tokens": e.tokens,
                    "build_ms": e.build_ms,
                    "hits": e.hits,
                }
                for e in self._entries.values()
            ],
        }


CACHE = _Cache()


def _scope_key(tenant_id: uuid.UUID, policy_ids: list[uuid.UUID] | None) -> ScopeKey:
    return (tenant_id, frozenset(policy_ids) if policy_ids else None)


def _fingerprint(
    session: Session, tenant_id: uuid.UUID, policy_ids: list[uuid.UUID] | None
) -> tuple:
    """A cheap summary of the scoped corpus, used to detect change.

    Deliberately a query rather than an ``invalidate()`` hook called from ingestion.
    Ingestion runs in a background task and, once deployed, possibly in another process
    entirely -- a hook would leave every other worker serving a stale index with no way
    to notice. Two indexed aggregates per query is a small price for not being silently
    wrong, and the alternative failure is invisible.
    """
    stmt = select(
        func.count(Chunk.id),
        func.max(Chunk.created_at),
        func.max(Chunk.ordinal),
    ).where(Chunk.embedding.isnot(None))
    stmt = scoped(stmt, Chunk, tenant_id)
    if policy_ids:
        stmt = stmt.where(Chunk.policy_id.in_(policy_ids))
    count, newest, max_ordinal = session.execute(stmt).one()
    return (int(count or 0), str(newest), int(max_ordinal or 0))


def _build(
    session: Session, tenant_id: uuid.UUID, policy_ids: list[uuid.UUID] | None, fingerprint: tuple
) -> _Index | None:
    from rank_bm25 import BM25Okapi

    started = time.perf_counter()

    # Only id and text: the index never holds chunk bodies, and the top-k rows are
    # fetched from the database after scoring.
    stmt = select(Chunk.id, Chunk.text).where(Chunk.embedding.isnot(None)).order_by(Chunk.id)
    stmt = scoped(stmt, Chunk, tenant_id)
    if policy_ids:
        stmt = stmt.where(Chunk.policy_id.in_(policy_ids))

    rows = session.execute(stmt).all()
    if not rows:
        return None

    corpus = [tokenize(text) for _, text in rows]
    index = _Index(
        bm25=BM25Okapi(corpus),
        chunk_ids=[chunk_id for chunk_id, _ in rows],
        fingerprint=fingerprint,
        build_ms=int((time.perf_counter() - started) * 1000),
        tokens=sum(len(doc) for doc in corpus),
    )
    return index


def get_index(
    session: Session, tenant_id: uuid.UUID, policy_ids: list[uuid.UUID] | None
) -> _Index | None:
    key = _scope_key(tenant_id, policy_ids)
    fingerprint = _fingerprint(session, tenant_id, policy_ids)

    entry = CACHE.get(key, fingerprint)
    if entry is not None:
        return entry

    entry = _build(session, tenant_id, policy_ids, fingerprint)
    if entry is not None:
        CACHE.put(key, entry)
    return entry


def retrieve(
    session: Session,
    tenant_id: uuid.UUID,
    question: str,
    policy_ids: list[uuid.UUID] | None = None,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or get_settings().retrieval_top_k

    index = get_index(session, tenant_id, policy_ids)
    if index is None:
        return []

    scores = index.bm25.get_scores(tokenize(question))
    ranked = sorted(
        zip(index.chunk_ids, scores, strict=True), key=lambda pair: float(pair[1]), reverse=True
    )
    # A zero score means no term overlap at all; padding the list would only add noise.
    wanted = [(cid, float(sc)) for cid, sc in ranked[:top_k] if float(sc) > 0.0]
    if not wanted:
        return []

    score_by_id = dict(wanted)
    rows = session.execute(
        base_statement(tenant_id, policy_ids).where(Chunk.id.in_(list(score_by_id)))
    ).all()
    by_id = {chunk.id: (chunk, filename, product) for chunk, filename, product in rows}

    results: list[RetrievedChunk] = []
    for position, (chunk_id, score) in enumerate(wanted, start=1):
        found = by_id.get(chunk_id)
        if found is None:
            # Scoped out between scoring and fetching -- the database is authoritative.
            continue
        chunk, filename, product_name = found
        results.append(
            build_chunk(
                chunk,
                filename,
                product_name,
                score=score,
                semantic_score=None,
                rank=position,
                retrieved_by=RetrievedBy.KEYWORD,
            )
        )
    return results
