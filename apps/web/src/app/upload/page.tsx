"use client";

import Link from "next/link";
import { useState } from "react";

import { StatusBadge } from "@/components/status-badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { usePolicies } from "@/hooks/use-policies";
import { ApiError, uploadPolicy, type UploadOut } from "@/lib/api";

export default function UploadPage() {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<UploadOut | null>(null);

  // The upload returns `pending`: ingestion runs in a background task. The live status
  // therefore has to come from the policy list, not from the upload response.
  const { policies, busy } = usePolicies(result !== null);
  const uploaded = result
    ? policies
        .flatMap((policy) => policy.documents)
        .find((candidate) => candidate.id === result.document_id)
    : undefined;

  async function onSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      setResult(await uploadPolicy(new FormData(event.currentTarget)));
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : String(caught));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h1 className="font-heading text-xl font-semibold">Upload a policy</h1>
        <p className="text-sm text-muted-foreground">
          PDF wordings only. The file is validated by its content, not its extension, and
          an identical file is recognised as a duplicate rather than ingested twice.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Document</CardTitle>
        </CardHeader>
        <CardContent>
          <form onSubmit={onSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="file">PDF</Label>
              <Input id="file" name="file" type="file" accept="application/pdf" required />
            </div>
            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="product_name">Product</Label>
                <Input
                  id="product_name"
                  name="product_name"
                  placeholder="Digi Term"
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="insurer">Insurer</Label>
                <Input id="insurer" name="insurer" placeholder="LIC" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="policy_version">Version</Label>
                <Input id="policy_version" name="policy_version" placeholder="2024-07" />
              </div>
            </div>
            <Button type="submit" disabled={submitting}>
              {submitting ? "Uploading…" : "Upload and ingest"}
            </Button>
          </form>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Upload rejected</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {result && (
        <Card>
          <CardHeader>
            <CardTitle>Ingestion</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm">
            {result.duplicate_of && (
              <p className="text-muted-foreground">
                This file is byte-identical to a document already ingested, so it was not
                processed again.
              </p>
            )}
            <div className="flex items-center gap-3">
              <StatusBadge status={uploaded?.status ?? result.status} />
              <span className="text-muted-foreground">
                {uploaded
                  ? `${uploaded.page_count} pages, ${uploaded.chunk_count} chunks`
                  : "waiting for the background task to start"}
              </span>
            </div>
            {uploaded?.error && <p className="text-destructive">{uploaded.error}</p>}
            {uploaded?.status === "ready" && (
              <Button variant="outline" nativeButton={false} render={<Link href="/ask" />}>
                Ask it something
              </Button>
            )}
            {busy && (
              <p className="text-xs text-muted-foreground">
                Extraction, chunking and embedding run in the background; this updates on
                its own.
              </p>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
