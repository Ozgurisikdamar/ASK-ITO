// Backend API client. Requests go to /api and are proxied to the backend
// (see vite.config.ts) in dev.

export interface ComponentStatus {
  status: string;
  [key: string]: unknown;
}

export interface HealthResponse {
  status: string;
  components: Record<string, ComponentStatus>;
  corpus: {
    documents: number;
    chunks: number;
    snapshot_at: string | null;
  };
}

export async function getHealth(): Promise<HealthResponse> {
  const resp = await fetch("/api/health");
  if (!resp.ok) throw new Error(`health request failed: ${resp.status}`);
  return resp.json();
}

export interface TraceStep {
  step: string;
  input?: unknown;
  output?: unknown;
  duration_ms: number;
  candidate_ids?: number[];
  note?: string | null;
}

export interface Source {
  n?: number;
  chunk_id: number;
  doc_id?: number;
  doc_title: string;
  section: string | null;
  url: string;
  domain: string;
  score: number | null;
  snippet: string;
}

/** An organic (Google-style) result: one passage of a Wikipedia article. */
export interface ResultItem {
  chunk_id: number;
  doc_id: number;
  doc_title: string;
  section: string | null;
  url: string;
  domain: string;
  score: number;
  snippet: string;
}

export interface ResultsEvent {
  results: ResultItem[];
  total: number;
  stage?: string;
  timings: Record<string, number>;
  cached?: boolean;
}

export interface DoneEvent {
  answer: string;
  citations: Citation[];
  timings: Record<string, number>;
  cached: boolean;
  mode: string;
  model?: string;
  rounds?: number;
  llm_calls?: number;
}

export interface StreamHandlers {
  onResults?: (d: ResultsEvent) => void;
  onReranked?: (d: ResultsEvent) => void;
  onSources?: (sources: Source[]) => void;
  onStep?: (step: TraceStep) => void;
  onToken?: (text: string) => void;
  onDone?: (d: DoneEvent) => void;
  onError?: (d: { error: string; detail: string }) => void;
}

/**
 * Open a streaming search (Server-Sent Events). Returns a function that closes
 * the stream; closing also stops answer generation on the server.
 */
export function openSearchStream(
  p: { query: string; mode: string; domain: string },
  h: StreamHandlers
): () => void {
  const qs = new URLSearchParams({ q: p.query, mode: p.mode, domain: p.domain });
  const es = new EventSource(`/api/search/stream?${qs.toString()}`);
  let finished = false;
  const finish = () => {
    finished = true;
    es.close(); // stop EventSource from auto-reconnecting (which would re-run the search)
  };
  const on = <T,>(name: string, fn: (d: T) => void) =>
    es.addEventListener(name, (e) => fn(JSON.parse((e as MessageEvent).data)));

  on<ResultsEvent>("results", (d) => h.onResults?.(d));
  on<ResultsEvent>("reranked", (d) => h.onReranked?.(d));
  on<{ sources: Source[] }>("sources", (d) => h.onSources?.(d.sources));
  on<TraceStep>("step", (d) => h.onStep?.(d));
  on<{ t: string }>("token", (d) => h.onToken?.(d.t));
  on<DoneEvent>("done", (d) => {
    finish();
    h.onDone?.(d);
  });
  es.addEventListener("error", (e) => {
    const data = (e as MessageEvent).data;
    if (data) {
      finish();
      h.onError?.(JSON.parse(data));
    } else if (!finished) {
      finish();
      h.onError?.({ error: "connection", detail: "The connection to the server was lost." });
    }
  });
  return finish;
}

export async function getFeatured(): Promise<string[]> {
  const r = await fetch("/api/suggest/featured");
  if (!r.ok) throw new Error(`featured failed: ${r.status}`);
  return (await r.json()).queries as string[];
}

export interface Citation {
  n: number;
  chunk_id: number;
}

export interface SearchResponse {
  mode: string;
  answer: string;
  citations: Citation[];
  sources: Source[];
  trace: TraceStep[];
  rewritten_query?: string | null;
  timings: Record<string, number>;
  rounds?: number;
  llm_calls?: number;
}

export interface SearchRequest {
  query: string;
  mode: string;
  domain: string;
  final_k?: number;
}

export interface GoldChunk {
  chunk_id: number;
  doc_title: string;
  section: string | null;
  snippet: string;
}

export interface ReviewItem {
  id: string;
  question: string;
  reference: string | null;
  qtype: string;
  domain: string;
  gold_doc_ids: number[];
  gold_chunk_ids: number[];
  status: "pending" | "accepted" | "rejected";
  gold_chunks: GoldChunk[];
}

export interface ReviewResponse {
  items: ReviewItem[];
  summary: { total: number; accepted: number; rejected: number; pending: number };
}

export async function getReview(): Promise<ReviewResponse> {
  const resp = await fetch("/api/eval/review");
  if (!resp.ok) throw new Error(`review request failed: ${resp.status}`);
  return resp.json();
}

export async function postReview(action: {
  id: string;
  action: "accept" | "reject";
  question?: string;
  reference?: string | null;
}): Promise<{ id: string; status: string; accepted_total: number }> {
  const resp = await fetch("/api/eval/review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(action),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data?.detail || data?.error || "review update failed");
  return data;
}

export interface EvalRun {
  id: number;
  config_name: string;
  status: string;
  started_at: string | null;
  finished_at: string | null;
  n_questions: number | null;
  retrieval: Record<string, number | null>;
  ragas: Record<string, number | string | null>;
  operational: { latency_ms?: { p50: number | null; p95: number | null }; avg_llm_calls?: number };
  behaviour: Record<string, number | null>;
}

export async function getRuns(): Promise<{ runs: EvalRun[] }> {
  const r = await fetch("/api/eval/runs");
  if (!r.ok) throw new Error(`runs failed: ${r.status}`);
  return r.json();
}

export interface RunResult {
  question_id: string;
  question: string;
  qtype: string;
  domain: string;
  reference: string | null;
  gold_chunk_ids: number[];
  gold_doc_ids: number[];
  answer: string;
  retrieved_chunk_ids: number[];
  metrics: Record<string, number | null>;
  latency_ms: number;
  llm_calls: number;
  trace: unknown[];
}

export interface RunDetail {
  id: number;
  config_name: string;
  status: string;
  config_json: Record<string, unknown>;
  summary: Record<string, unknown>;
  results: RunResult[];
}

export async function getRunDetail(id: number): Promise<RunDetail> {
  const r = await fetch(`/api/eval/runs/${id}`);
  if (!r.ok) throw new Error(`run detail failed: ${r.status}`);
  return r.json();
}

export interface CompareRow {
  question_id: string;
  qtype: string;
  a: number;
  b: number;
  delta: number;
}

export async function getCompare(a: number, b: number): Promise<{
  metric: string;
  improved: CompareRow[];
  regressed: CompareRow[];
}> {
  const r = await fetch(`/api/eval/compare?a=${a}&b=${b}`);
  if (!r.ok) throw new Error(`compare failed: ${r.status}`);
  return r.json();
}

export async function startRun(configs: string[]): Promise<{ status: string; configs: string[] }> {
  const r = await fetch("/api/eval/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ configs }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data?.detail || data?.error || "start run failed");
  return data;
}

export async function search(
  req: SearchRequest,
  signal?: AbortSignal
): Promise<SearchResponse> {
  const resp = await fetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
    signal,
  });
  const data = await resp.json();
  if (!resp.ok) {
    throw new Error(data?.detail || data?.error || `search failed: ${resp.status}`);
  }
  return data as SearchResponse;
}
