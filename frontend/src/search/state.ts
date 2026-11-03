import type {
  Citation, DoneEvent, ResultItem, ResultsEvent, Source, StatusEvent, StreamError, TraceStep,
} from "../api/client";

export type Phase = "searching" | "answering" | "done" | "error";

export interface SearchState {
  phase: Phase;
  results: ResultItem[];
  resultsStage: string | null;
  sources: Source[];
  streamText: string; // raw tokens as they arrive
  answer: string | null; // validated final answer
  citations: Citation[];
  steps: TraceStep[];
  timings: Record<string, number>;
  cached: boolean; // answer served from the answer cache
  resultsCached: boolean; // ranked results served from the retrieval cache
  model: string | null;
  reranker: string | null;
  abstained: boolean;
  status: StatusEvent | null; // queued / loading the model
  error: string | null; // message shown to the user
  errorCode: string | null;
}

export const initialState: SearchState = {
  phase: "searching",
  results: [],
  resultsStage: null,
  sources: [],
  streamText: "",
  answer: null,
  citations: [],
  steps: [],
  timings: {},
  cached: false,
  resultsCached: false,
  model: null,
  reranker: null,
  abstained: false,
  status: null,
  error: null,
  errorCode: null,
};

export type Action =
  | { type: "reset" }
  | { type: "results"; data: ResultsEvent }
  | { type: "reranked"; data: ResultsEvent }
  | { type: "sources"; sources: Source[] }
  | { type: "step"; step: TraceStep }
  | { type: "token"; text: string }
  | { type: "done"; data: DoneEvent }
  | { type: "status"; data: StatusEvent }
  | { type: "error"; data: StreamError };

export function reducer(state: SearchState, action: Action): SearchState {
  switch (action.type) {
    case "reset":
      return initialState;
    case "results":
      return {
        ...state,
        results: action.data.results,
        resultsStage: action.data.stage ?? null,
        timings: { ...state.timings, ...action.data.timings },
        resultsCached: !!action.data.cached,
      };
    case "reranked":
      return {
        ...state,
        results: action.data.results,
        resultsStage: "reranked",
        timings: { ...state.timings, ...action.data.timings },
      };
    case "sources":
      return { ...state, sources: action.sources, phase: "answering" };
    case "step":
      return { ...state, steps: [...state.steps, action.step] };
    case "token":
      return { ...state, streamText: state.streamText + action.text, phase: "answering", status: null };
    case "done":
      return {
        ...state,
        phase: "done",
        answer: action.data.answer,
        citations: action.data.citations,
        timings: { ...state.timings, ...action.data.timings },
        cached: action.data.cached,
        model: action.data.model ?? null,
        reranker: action.data.reranker ?? null,
        abstained: !!action.data.abstained,
      };
    case "status":
      return { ...state, status: action.data };
    case "error":
      // keep results, sources and any partial answer: only the generation failed
      return {
        ...state,
        phase: "error",
        status: null,
        error: action.data.message || action.data.detail || action.data.error,
        errorCode: action.data.error,
      };
  }
}
