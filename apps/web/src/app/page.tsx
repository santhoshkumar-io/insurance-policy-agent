"use client";

import Link from "next/link";

import { StatusBadge } from "@/components/status-badge";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { usePolicies } from "@/hooks/use-policies";

export default function LibraryPage() {
  const { policies, error, loading, busy, refresh } = usePolicies(true);

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between gap-4">
        <div>
          <h1 className="font-heading text-xl font-semibold">Library</h1>
          <p className="text-sm text-muted-foreground">
            Ingested policies and their documents. Chunk counts come from the database, so
            a document with zero chunks really has nothing to retrieve.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => void refresh()}>
            Refresh
          </Button>
          <Button nativeButton={false} render={<Link href="/upload" />}>
            Upload a policy
          </Button>
        </div>
      </div>

      {error && (
        <Alert variant="destructive">
          <AlertTitle>Cannot load the library</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {busy && (
        <p className="text-sm text-muted-foreground">
          A document is still ingesting — this list refreshes itself.
        </p>
      )}

      {loading ? (
        <div className="space-y-2">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      ) : policies.length === 0 ? (
        <Alert>
          <AlertTitle>No policies yet</AlertTitle>
          <AlertDescription>
            Upload a policy wording to ingest it, or run{" "}
            <code className="font-mono text-xs">policy ingest</code> from the CLI.
          </AlertDescription>
        </Alert>
      ) : (
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Product</TableHead>
              <TableHead>Insurer</TableHead>
              <TableHead>Version</TableHead>
              <TableHead>Document</TableHead>
              <TableHead>Status</TableHead>
              <TableHead className="text-right">Pages</TableHead>
              <TableHead className="text-right">Chunks</TableHead>
              <TableHead />
            </TableRow>
          </TableHeader>
          <TableBody>
            {policies.flatMap((policy) =>
              (policy.documents.length > 0
                ? policy.documents
                : [null]
              ).map((document, index) => (
                <TableRow key={`${policy.id}-${document?.id ?? index}`}>
                  <TableCell className="font-medium">{policy.product_name}</TableCell>
                  <TableCell>{policy.insurer || "—"}</TableCell>
                  <TableCell>{policy.policy_version || "—"}</TableCell>
                  <TableCell className="max-w-56 truncate" title={document?.filename}>
                    {document?.filename ?? "—"}
                  </TableCell>
                  <TableCell>
                    {document ? <StatusBadge status={document.status} /> : "—"}
                    {document?.error && (
                      <p className="mt-1 text-xs text-destructive">{document.error}</p>
                    )}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {document?.page_count ?? 0}
                  </TableCell>
                  <TableCell className="text-right tabular-nums">
                    {document?.chunk_count ?? 0}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="sm"
                      nativeButton={false}
                      render={<Link href={`/ask?policy=${policy.id}`} />}
                    >
                      Ask
                    </Button>
                  </TableCell>
                </TableRow>
              )),
            )}
          </TableBody>
        </Table>
      )}
    </div>
  );
}
