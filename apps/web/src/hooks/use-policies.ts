"use client";

import { useCallback, useEffect, useState } from "react";

import { ApiError, listPolicies, type PolicyOut } from "@/lib/api";

const BUSY: ReadonlySet<string> = new Set(["pending", "processing"]);

export function isBusy(policies: PolicyOut[]): boolean {
  return policies.some((policy) => policy.documents.some((d) => BUSY.has(d.status)));
}

interface State {
  policies: PolicyOut[];
  error: string | null;
  loading: boolean;
}

/**
 * The policy list, polled only while something is still ingesting.
 *
 * Ingestion runs in a FastAPI background task, so the upload response says `pending` and
 * the real outcome arrives later. Polling on a fixed timer regardless of state would
 * hammer the API forever on an idle page; polling never would leave the uploader staring
 * at `pending`. So the loop runs exactly while a document is in flight, and it schedules
 * itself from inside the async fetch rather than from an effect that depends on the data
 * it just set — which re-subscribed on every response.
 */
export function usePolicies(poll = false) {
  const [state, setState] = useState<State>({
    policies: [],
    error: null,
    loading: true,
  });
  const [reloads, setReloads] = useState(0);

  const refresh = useCallback(() => setReloads((count) => count + 1), []);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const load = async () => {
      try {
        const policies = await listPolicies();
        if (cancelled) return;
        setState({ policies, error: null, loading: false });
        if (poll && isBusy(policies)) timer = setTimeout(() => void load(), 1500);
      } catch (caught) {
        if (cancelled) return;
        const error = caught instanceof ApiError ? caught.message : String(caught);
        setState((current) => ({ ...current, error, loading: false }));
      }
    };

    void load();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [poll, reloads]);

  return { ...state, refresh, busy: isBusy(state.policies) };
}
