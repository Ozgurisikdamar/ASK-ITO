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
