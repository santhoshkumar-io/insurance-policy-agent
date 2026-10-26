/**
 * The API client.
 *
 * Types mirror `services/api/schemas.py` by hand. That is a duplication worth being
 * honest about: generating them from the OpenAPI schema would be better and is a build
 * step this project does not have yet, so the risk is that the two drift. The pairing
 * that matters most — `score` and `semanticScore` — is asserted in a test on the Python
 * side, because the whole point of the citation panel is that those two numbers differ.
 */

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export type DocumentStatus = "pending" | "processing" | "ready" | "failed";

export interface DocumentOut {
  id: string;
  filename: string;
  status: DocumentStatus;
  page_count: number;
  chunk_count: number;
  error: string | null;
}

export interface PolicyOut {
  id: string;
  insurer: string;
  product_name: string;
  policy_version: string;
  documents: DocumentOut[];
}

export interface UploadOut {
  policy_id: string;
  document_id: string;
  status: DocumentStatus;
  duplicate_of: string | null;
}

export interface CitationOut {
  chunk_id: string;
  ref: string;
  document: string;
  policy: string;
  page_from: number;
  page_to: number;
  section_path: string;
  chunk_type: string;
  /** The active strategy's ranking score. Its scale depends on the mode. */
  score: number;
  /** Cosine similarity to the question. Comparable across every mode. */
  semantic_score: number | null;
  retrieved_by: string;
  text: string;
}

export interface ClaimOut {
  text: string;
  citation_refs: string[];
}

export interface ChatOut {
  answerable: boolean;
  answer: string;
  abstained_reason: string | null;
  claims: ClaimOut[];
  citations: CitationOut[];
  retrieved: CitationOut[];
  model: string;
  prompt_version: string;
  latency_ms: number;
  message_id: string | null;
}

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch {
    // A failed fetch to localhost is almost always "the API is not running", and saying
    // so is more useful than surfacing the browser's opaque "Failed to fetch".
    throw new ApiError(0, `Cannot reach the API at ${BASE}. Is uvicorn running?`);
  }
  if (!response.ok) {
    const body = await response.text();
    let detail = body;
    try {
      detail = (JSON.parse(body) as { detail?: string }).detail ?? body;
    } catch {
      // Not JSON; the raw body is the best message available.
    }
    throw new ApiError(response.status, detail || response.statusText);
  }
  return (await response.json()) as T;
}

export function listPolicies(): Promise<PolicyOut[]> {
  return request<PolicyOut[]>("/policies", { cache: "no-store" });
}

export function uploadPolicy(form: FormData): Promise<UploadOut> {
  return request<UploadOut>("/policies/upload", { method: "POST", body: form });
}

export function ask(payload: {
  question: string;
  policy_ids?: string[] | null;
  top_k?: number;
  mode?: string;
}): Promise<ChatOut> {
  return request<ChatOut>("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export const apiBaseUrl = BASE;
