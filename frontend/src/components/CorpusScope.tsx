import { useEffect, useState } from "react";
import { getHealth, HealthResponse } from "../api/client";
import { strings } from "../strings";

/** States plainly what is being searched: a small, fixed Wikipedia snapshot. */
export function CorpusScope({ className = "" }: { className?: string }) {
  const [corpus, setCorpus] = useState<HealthResponse["corpus"] | null>(null);
  useEffect(() => {
    getHealth().then((h) => setCorpus(h.corpus)).catch(() => undefined);
  }, []);
  const n = corpus?.documents;
  const date = corpus?.snapshot_at ? new Date(corpus.snapshot_at).toISOString().slice(0, 10) : null;
  return (
    <p className={"text-xs text-content-muted " + className}>
      {strings.scope(n ?? null, date, corpus?.dynamic_documents ?? 0)}
    </p>
  );
}
