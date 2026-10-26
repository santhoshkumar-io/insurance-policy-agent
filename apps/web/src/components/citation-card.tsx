"use client";

import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import type { CitationOut } from "@/lib/api";

function pageLabel(citation: CitationOut): string {
  return citation.page_from === citation.page_to
    ? `p. ${citation.page_from}`
    : `pp. ${citation.page_from}–${citation.page_to}`;
}

/**
 * One retrieved chunk, with its provenance and both of its scores.
 *
 * The two scores are shown side by side on purpose. `score` is whatever the active
 * strategy ranks by — a cosine under vector search, a reciprocal-rank sum under RRF, a
 * cross-encoder logit under reranking — so its scale moves when the mode does.
 * `semantic_score` is always a cosine, which is why it is the one the abstention
 * threshold reads. Treating those two as interchangeable was a real bug in this codebase
 * (docs/decisions.md), and a panel showing a single unlabelled "score" would hide exactly
 * the thing that made it hard to see.
 *
 * Chunks are clipped by default unless they were cited. Policy chunks run to several
 * hundred words, and eight of them unclipped buries the provenance this panel exists to
 * show — but truncating with no way to expand would hide the evidence itself, so the full
 * text is always one click away.
 */
export function CitationCard({
  citation,
  cited,
}: {
  citation: CitationOut;
  cited: boolean;
}) {
  const [expanded, setExpanded] = useState(cited);

  return (
    <div className="rounded-lg border p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={cited ? "default" : "outline"} className="font-mono">
          {citation.ref}
        </Badge>
        {!cited && (
          <Badge variant="ghost" className="text-muted-foreground">
            retrieved, not cited
          </Badge>
        )}
        <Badge variant="secondary">{citation.retrieved_by}</Badge>
        {citation.chunk_type !== "prose" && (
          <Badge variant="outline">{citation.chunk_type}</Badge>
        )}
        <span className="ml-auto text-xs text-muted-foreground">
          {citation.policy} · {citation.document}
        </span>
      </div>

      <div className="mt-3 flex flex-wrap items-baseline gap-x-3 gap-y-1 text-sm">
        <span className="font-medium tabular-nums">{pageLabel(citation)}</span>
        <span className="text-muted-foreground">{citation.section_path || "—"}</span>
      </div>

      <div className="mt-3 flex gap-6 text-xs">
        <div>
          <div className="text-muted-foreground">ranking score</div>
          <div className="font-mono tabular-nums">{citation.score.toFixed(4)}</div>
        </div>
        <div>
          <div className="text-muted-foreground">cosine</div>
          <div className="font-mono tabular-nums">
            {citation.semantic_score === null ? "—" : citation.semantic_score.toFixed(4)}
          </div>
        </div>
      </div>

      <Separator className="my-3" />
      <p
        className={`text-sm whitespace-pre-wrap ${expanded ? "" : "line-clamp-4"}`}
        data-testid="chunk-text"
      >
        {citation.text}
      </p>
      <Button
        variant="link"
        size="sm"
        className="mt-1 h-auto p-0"
        onClick={() => setExpanded((open) => !open)}
      >
        {expanded ? "Show less" : "Show the full chunk"}
      </Button>
    </div>
  );
}
