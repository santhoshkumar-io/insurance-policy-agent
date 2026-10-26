import { Badge } from "@/components/ui/badge";
import type { DocumentStatus } from "@/lib/api";

const VARIANT: Record<DocumentStatus, "default" | "secondary" | "outline" | "destructive"> = {
  ready: "default",
  processing: "secondary",
  pending: "outline",
  failed: "destructive",
};

export function StatusBadge({ status }: { status: DocumentStatus }) {
  return <Badge variant={VARIANT[status] ?? "outline"}>{status}</Badge>;
}
