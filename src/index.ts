/**
 * planet-capture SDK - Library mode for programmatic access.
 *
 * Usage:
 *   import { createStore } from 'planet-capture'
 *
 *   const store = await createStore({ dbPath: './my-index.sqlite' })
 *   const results = await store.search({ query: "how does auth work?" })
 *   await store.close()
 */

import {
  createStore as createStoreInternal,
  hybridQuery,
  structuredSearch,
  extractSnippet,
  addLineNumbers,
  DEFAULT_EMBED_MODEL,
  DEFAULT_MULTI_GET_MAX_BYTES,
  generateEmbeddings,
  type Store as InternalStore,
  type PageResult,
  type PageNotFound,
  type SearchResult,
  type HybridQueryResult,
  type HybridQueryOptions,
  type HybridQueryExplain,
  type ExpandedQuery,
  type StructuredSearchOptions,
  type IndexStatus,
  type IndexHealthInfo,
  type SearchHooks,
  type EmbedProgress,
  type EmbedResult,
  type ChunkStrategy,
  type IndexerRunState,
  type BrowserInfo,
} from "./store.js";
import { runIndex, type IndexOptions, type DiscoverResult, type FetchResult } from "./indexer.js";
import { LlamaCpp } from "./llm.js";

// Re-export types for SDK consumers
export type {
  PageResult,
  PageNotFound,
  SearchResult,
  HybridQueryResult,
  HybridQueryOptions,
  HybridQueryExplain,
  ExpandedQuery,
  StructuredSearchOptions,
  IndexStatus,
  IndexHealthInfo,
  SearchHooks,
  EmbedProgress,
  EmbedResult,
  IndexerRunState,
  BrowserInfo,
  IndexOptions,
  DiscoverResult,
  FetchResult,
};

export type { InternalStore };
export { extractSnippet, addLineNumbers, DEFAULT_MULTI_GET_MAX_BYTES };
export type { ChunkStrategy } from "./store.js";
export { getDefaultDbPath } from "./store.js";
export { Maintenance } from "./maintenance.js";

export interface SearchOptions {
  query?: string;
  queries?: ExpandedQuery[];
  intent?: string;
  rerank?: boolean;
  browser?: string;
  limit?: number;
  minScore?: number;
  explain?: boolean;
  chunkStrategy?: ChunkStrategy;
}

export interface LexSearchOptions {
  limit?: number;
  browser?: string;
}

export interface VectorSearchOptions {
  limit?: number;
  browser?: string;
}

export interface ExpandQueryOptions {
  intent?: string;
}

export interface StoreOptions {
  dbPath: string;
}

/**
 * The planet-capture SDK store.
 */
export interface PlanetCaptureStore {
  readonly internal: InternalStore;
  readonly dbPath: string;

  // Search
  search(options: SearchOptions): Promise<HybridQueryResult[]>;
  searchLex(query: string, options?: LexSearchOptions): Promise<SearchResult[]>;
  searchVector(query: string, options?: VectorSearchOptions): Promise<SearchResult[]>;
  expandQuery(query: string, options?: ExpandQueryOptions): Promise<ExpandedQuery[]>;

  // Page retrieval
  get(urlOrDocid: string, options?: { includeBody?: boolean }): Promise<PageResult | PageNotFound>;
  getPageBody(urlOrDocid: string, opts?: { fromLine?: number; maxLines?: number }): Promise<string | null>;

  // Browser management
  listBrowsers(): Promise<BrowserInfo[]>;

  // URL filter management
  addUrlFilter(pattern: string, type?: string): Promise<void>;
  removeUrlFilter(pattern: string): Promise<boolean>;
  listUrlFilters(): Promise<{ id: number; pattern: string; filter_type: string }[]>;

  // Indexing
  index(options?: IndexOptions): Promise<{ discover: DiscoverResult; fetch?: FetchResult }>;

  // Embeddings
  embed(options?: {
    force?: boolean;
    model?: string;
    maxDocsPerBatch?: number;
    maxBatchBytes?: number;
    chunkStrategy?: ChunkStrategy;
    onProgress?: (info: EmbedProgress) => void;
  }): Promise<EmbedResult>;

  // Index health
  getStatus(): Promise<IndexStatus>;
  getIndexHealth(): Promise<IndexHealthInfo>;

  // Lifecycle
  close(): Promise<void>;
}

/**
 * Create a planet-capture store.
 */
export async function createStore(options: StoreOptions): Promise<PlanetCaptureStore> {
  if (!options.dbPath) throw new Error("dbPath is required");

  const internal = createStoreInternal(options.dbPath);

  const llm = new LlamaCpp({
    inactivityTimeoutMs: 5 * 60 * 1000,
    disposeModelsOnInactivity: true,
  });
  internal.llm = llm;

  const store: PlanetCaptureStore = {
    internal,
    dbPath: internal.dbPath,

    search: async (opts) => {
      if (!opts.query && !opts.queries) {
        throw new Error("search() requires either 'query' or 'queries'");
      }
      const skipRerank = opts.rerank === false;

      if (opts.queries) {
        return structuredSearch(internal, opts.queries, {
          limit: opts.limit,
          minScore: opts.minScore,
          explain: opts.explain,
          intent: opts.intent,
          skipRerank,
          chunkStrategy: opts.chunkStrategy,
        });
      }

      return hybridQuery(internal, opts.query!, {
        collection: opts.browser,
        limit: opts.limit,
        minScore: opts.minScore,
        explain: opts.explain,
        intent: opts.intent,
        skipRerank,
        chunkStrategy: opts.chunkStrategy,
      });
    },

    searchLex: async (q, opts) => internal.searchFTS(q, opts?.limit, opts?.browser),
    searchVector: async (q, opts) => internal.searchVec(q, DEFAULT_EMBED_MODEL, opts?.limit, opts?.browser),
    expandQuery: async (q, opts) => internal.expandQuery(q, undefined, opts?.intent),

    get: async (urlOrDocid, opts) => internal.findPage(urlOrDocid, opts),
    getPageBody: async (urlOrDocid, opts) => {
      const result = internal.findPage(urlOrDocid, { includeBody: false });
      if ("error" in result) return null;
      return internal.getPageBody(result, opts?.fromLine, opts?.maxLines);
    },

    listBrowsers: async () => {
      const status = internal.getStatus();
      return status.browsers;
    },

    addUrlFilter: async (pattern, type) => internal.addUrlFilter(pattern, type),
    removeUrlFilter: async (pattern) => internal.removeUrlFilter(pattern),
    listUrlFilters: async () => internal.listUrlFilters(),

    index: async (indexOpts) => runIndex(internal, indexOpts),

    embed: async (embedOpts) => generateEmbeddings(internal, embedOpts),

    getStatus: async () => internal.getStatus(),
    getIndexHealth: async () => internal.getIndexHealth(),

    close: async () => {
      await llm.dispose();
      internal.close();
    },
  };

  return store;
}
