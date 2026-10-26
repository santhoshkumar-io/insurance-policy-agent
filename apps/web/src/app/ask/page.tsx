"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useMemo, useState } from "react";

import { CitationCard } from "@/components/citation-card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { usePolicies } from "@/hooks/use-policies";
import { ApiError, ask, type ChatOut } from "@/lib/api";

const EXAMPLES = [
  "What is the grace period for paying premiums?",
  "What is the UIN of this plan?",
  "What is the premium for a 35 year old non-smoker?",
];

function AskScreen() {
  const preselected = useSearchParams().get("policy");
  const { policies } = usePolicies();
  const [selected, setSelected] = useState<string[]>(
    preselected ? [preselected] : [],
  );
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<ChatOut | null>(null);
  const [error, setError] = useState<{ title: string; detail: string } | null>(null);
  const [pending, setPending] = useState(false);

  const ready = policies.filter((policy) =>
    policy.documents.some((document) => document.status === "ready"),
  );

  const citedRefs = useMemo(
    () => new Set((answer?.citations ?? []).map((citation) => citation.ref)),
    [answer],
  );
  // Everything retrieved, cited first. The uncited rows are the interesting ones: they
  // show what the model was shown and chose not to lean on.
  const panel = useMemo(() => {
    if (!answer) return [];
    const byRef = new Map(answer.retrieved.map((item) => [item.ref, item]));
    for (const citation of answer.citations) byRef.set(citation.ref, citation);
    return [...byRef.values()].sort(
      (a, b) => Number(citedRefs.has(b.ref)) - Number(citedRefs.has(a.ref)),
    );
  }, [answer, citedRefs]);

  async function submit(text: string) {
    if (text.trim().length < 3) return;
    setPending(true);
    setError(null);
    setAnswer(null);
    try {
      setAnswer(
        await ask({
          question: text.trim(),
          policy_ids: selected.length > 0 ? selected : null,
        }),
      );
    } catch (caught) {
      // A 429 is the demo's cost ceiling doing its job, not a failure. Saying "the
      // request failed" would send the reader looking for a bug that is not there.
      setError(
        caught instanceof ApiError
          ? {
              title: caught.status === 429 ? "Slow down" : "The request failed",
              detail: caught.message,
            }
          : { title: "The request failed", detail: String(caught) },
      );
    } finally {
      setPending(false);
    }
  }

  function toggle(id: string) {
    setSelected((current) =>
      current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
    );
  }

  return (
    <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="space-y-6">
        <div>
          <h1 className="font-heading text-xl font-semibold">Ask</h1>
          <p className="text-sm text-muted-foreground">
            Scope the question to one or more policies, or leave it unscoped to search
            everything in this tenant.
          </p>
        </div>

        {ready.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {ready.map((policy) => {
              const active = selected.includes(policy.id);
              return (
                <Button
                  key={policy.id}
                  size="sm"
                  variant={active ? "default" : "outline"}
                  onClick={() => toggle(policy.id)}
                >
                  {policy.product_name}
                </Button>
              );
            })}
          </div>
        )}

        <form
          onSubmit={(event) => {
            event.preventDefault();
            void submit(question);
          }}
          className="space-y-3"
        >
          <Textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="What does this policy say about…"
            rows={3}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                event.preventDefault();
                void submit(question);
              }
            }}
          />
          <div className="flex items-center gap-3">
            <Button type="submit" disabled={pending || question.trim().length < 3}>
              {pending ? "Retrieving…" : "Ask"}
            </Button>
            <span className="text-xs text-muted-foreground">⌘↵ to send</span>
          </div>
        </form>

        <div className="space-y-2">
          <p className="text-xs text-muted-foreground">
            Try one — the third has no answer in a policy wording, and the system should
            say so rather than guess:
          </p>
          <div className="flex flex-col items-start gap-1">
            {EXAMPLES.map((example) => (
              <Button
                key={example}
                variant="link"
                size="sm"
                className="h-auto p-0 text-left"
                onClick={() => {
                  setQuestion(example);
                  void submit(example);
                }}
              >
                {example}
              </Button>
            ))}
          </div>
        </div>

        {error && (
          <Alert variant={error.title === "Slow down" ? "default" : "destructive"}>
            <AlertTitle>{error.title}</AlertTitle>
            <AlertDescription>{error.detail}</AlertDescription>
          </Alert>
        )}

        {answer && (
          <Card>
            <CardHeader>
              <CardTitle>
                {answer.answerable ? "Answer" : "No answer — the system abstained"}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {answer.answerable ? (
                // The answer *is* its claims, so rendering both prints it twice. The
                // claim list is the better of the two: it carries the citation each
                // sentence was drawn from, which is the whole point.
                answer.claims.length > 0 ? (
                  <div className="space-y-3">
                    {answer.claims.map((claim, index) => (
                      <p key={index} className="text-sm">
                        <span className="whitespace-pre-wrap">{claim.text}</span>{" "}
                        {claim.citation_refs.map((ref) => (
                          <Badge key={ref} variant="outline" className="ml-1 font-mono">
                            {ref}
                          </Badge>
                        ))}
                      </p>
                    ))}
                    <p className="text-xs text-muted-foreground">
                      Every reference above is an id the model was served. One it invented
                      would have been rejected before this point.
                    </p>
                  </div>
                ) : (
                  <p className="text-sm whitespace-pre-wrap">{answer.answer}</p>
                )
              ) : (
                // An abstention is a result, not an empty state. Rendering it as a blank
                // answer box would undo the feature the whole system is built around.
                <Alert>
                  <AlertTitle>Declined to answer</AlertTitle>
                  <AlertDescription>
                    {answer.abstained_reason ??
                      "The retrieved text does not support an answer."}
                  </AlertDescription>
                </Alert>
              )}
              <p className="text-xs text-muted-foreground">
                {answer.model} · prompt {answer.prompt_version} · {answer.latency_ms} ms
              </p>
            </CardContent>
          </Card>
        )}
      </div>

      <div className="space-y-4">
        <div>
          <h2 className="font-heading text-base font-semibold">Evidence</h2>
          <p className="text-sm text-muted-foreground">
            Page and section are read from the database row, never from the model&apos;s
            output — so a plausible invented page number has no path to this panel. Two
            scores are shown because they are not the same number: the ranking score&apos;s
            scale depends on the retrieval mode, while the cosine is comparable across
            modes and is what the abstention threshold reads.
          </p>
        </div>
        {panel.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Nothing retrieved yet. Ask something to see what the model was given.
          </p>
        ) : (
          <div className="space-y-3">
            {panel.map((citation) => (
              <CitationCard
                key={citation.ref}
                citation={citation}
                cited={citedRefs.has(citation.ref)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function AskPage() {
  // useSearchParams needs a Suspense boundary to prerender.
  return (
    <Suspense fallback={<p className="text-sm text-muted-foreground">Loading…</p>}>
      <AskScreen />
    </Suspense>
  );
}
