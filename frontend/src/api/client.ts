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
  chunk_id: number;
  doc_title: string;
  section: string | null;
  url: string;
  domain: string;
  score: number | null;
  snippet: string;
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
