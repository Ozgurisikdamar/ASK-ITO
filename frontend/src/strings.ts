// All user-facing UI text in one place (corpus is English, so UI is English).
export const strings = {
  appName: "Ask ITO",
  tagline: "Search and ask about Nobel Prizes, quantum computing and AI — every answer cites its sources.",
  backendUnavailable: "backend unavailable",
  scope: (n: number | null, date: string | null, added = 0) =>
    `Searches ${n ? n.toLocaleString() : "a few hundred"} local English Wikipedia articles` +
    ` (Nobel Prizes, quantum computing, AI${date ? `; snapshot ${date}` : ""}` +
    `${added ? `; ${added} added by AI Mode` : ""}) — not all of Wikipedia.` +
    ` When the local evidence is weak, AI Mode looks up Wikipedia and adds a few articles.`,
  searchPlaceholder: "Ask anything about recent Nobel Prizes or quantum computing…",
  searchAria: "Search query",
  submit: "Search",
  ask: "Ask",
  aiModeHint:
    "AI Mode: a cited answer from a local model. Simple questions get one answer; comparisons and multi-part questions run a step-by-step search first.",
  modes: {
    search: "Search",
    answer: "AI answer",
    sources: "Sources",
    aiMode: "AI Mode",
    agentic: "Agentic",
    fast: "Fast",
    advanced: "Advanced",
  },
  domains: {
    all: "All",
    nobel: "Nobel",
    quantum: "Quantum",
    ai: "AI",
  },
  exampleChips: [
    "How is the 2025 Physics prize related to quantum computers?",
    "What is the attention mechanism in Transformers?",
    "What did Hopfield and Hinton contribute to machine learning?",
    "How do AI agents use tools to complete tasks?",
    "How do neural networks learn with backpropagation?",
    "Why do quantum computers need error correction?",
  ],
  evaluationLink: "Evaluation",
  ragasOffline:
    "RAGAS and retrieval metrics run offline as a developer evaluation; they are never computed during a search.",
  corpusLoading: "Loading corpus info…",
  corpusEmpty: "Corpus not indexed yet.",
  themeToggleAria: "Toggle dark mode",
  results: {
    overview: "AI Overview",
    generating: "Generating an AI overview from the sources…",
    cached: "cached answer",
    cachedResults: "cached results",
    webResults: "Results",
    stats: (n: number, ms: number) => `${n} results (${(ms / 1000).toFixed(2)} seconds)`,
    sourcesUsed: "Sources used in the answer",
    topPassages: "Top passages",
    sourcesOnly: "Sources only: ranked passages from the corpus, no AI answer generated.",
    citationDisclaimer:
      "Citation numbers point to the retrieved passages the answer was written from. They do not by themselves verify each claim — check the sources.",
    abstained:
      "No passage in the corpus was relevant enough to answer from, so no answer was generated. Try a narrower query or browse the results below.",
    latency: "Latency",
    stages: {
      embedding: "Embedding",
      bm25: "BM25",
      vector: "pgvector",
      rrf: "RRF fusion",
      rerank: "Rerank",
      context: "Context",
      first_token: "First token",
      answer: "Generation",
      total: "Total",
    } as Record<string, string>,
    reranker: "Reranker",
    rerankerOff: "off",
    llmModel: "LLM",
    sources: "Sources",
    howItSearched: "How it searched",
    copy: "Copy",
    copied: "Copied",
    searching: "Searching…",
    notFoundPhrase: "could not find",
    noAnswer: "No answer was produced.",
    answerFailed: "The AI answer could not be generated.",
    answerInterrupted: "The answer was interrupted.",
    retry: "Try again",
    wikiSearch: "Evidence in the local corpus is weak — searching Wikipedia…",
    wikiAdding: (title: string) => `Adding from Wikipedia: ${title}…`,
    wikiAdded: "Added to the corpus from Wikipedia for this question:",
    wikiUnavailable: "Wikipedia could not be reached, so the answer uses only the local corpus.",
    errorReason: "Model server",
    queued: "Waiting for the previous answer to stop…",
    modelLoading: (model: string, s: number) =>
      `Loading the local model${model ? ` (${model})` : ""} into memory… ${Math.floor(s / 60)}:${String(
        Math.round(s % 60),
      ).padStart(2, "0")} elapsed. The first load after a restart can take several minutes.`,
    agentStep: (step: string, input: unknown) => {
      const q = typeof input === "string" && input ? `: ${input.slice(0, 80)}` : "";
      const label: Record<string, string> = {
        analyze: "Analysing the question",
        rewrite: "Writing a search query",
        search: "Searching the corpus",
        bm25: "Searching the corpus",
        vector: "Searching the corpus",
        rrf: "Searching the corpus",
        rerank: "Ranking passages",
        sufficiency: "Checking whether the evidence is enough",
        final_rerank: "Choosing the best passages",
        llm_error: "A step took too long; continuing with what was found",
      };
      return `${label[step] ?? "Working"}${step === "search" || step === "rewrite" ? q : ""}…`;
    },
    errorTitle: "Something went wrong",
    sourceAria: (n: number) => `Jump to source ${n}`,
    openWiki: "Wikipedia",
  },
};
