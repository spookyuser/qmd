/**
 * indexer.ts - Orchestrates the browser history indexing pipeline
 *
 * Three-phase pipeline:
 *   1. Discover - read history from browsers, upsert into pages + page_sources
 *   2. Fetch - download and extract content for pending pages
 *   3. Embed - generate vector embeddings (reuses store.ts generateEmbeddings)
 */

import {
  detectBrowsers,
  getDetectedBrowsers,
  readChromiumHistory,
  readChromiumBookmarks,
  readSafariHistory,
  shouldSkipUrl,
  matchesExcludeFilter,
  type BrowserInfo,
  type HistoryEntry,
  type BookmarkEntry,
} from "./browsers.js";
import {
  fetchPage,
  createRateLimiter,
  isFetchableUrl,
  type RateLimiter,
} from "./fetcher.js";
import { hashContent, type Store } from "./store.js";

// =============================================================================
// Types
// =============================================================================

export interface DiscoverProgress {
  browser: string;
  phase: "history" | "bookmarks";
  entriesFound: number;
}

export interface DiscoverResult {
  browsersScanned: number;
  urlsDiscovered: number;
  urlsUpdated: number;
  urlsSkipped: number;
}

export interface FetchProgress {
  url: string;
  current: number;
  total: number;
  status: "fetching" | "fetched" | "failed" | "skipped";
  error?: string;
}

export interface FetchResult {
  fetched: number;
  failed: number;
  skipped: number;
  unchanged: number;
  durationMs: number;
}

export interface IndexOptions {
  /** Only index specific browser */
  browser?: string;
  /** Only history after this date */
  since?: Date;
  /** Requests per second (default: 2) */
  rateLimit?: number;
  /** Max pages to fetch in this run */
  maxPages?: number;
  /** Just discover URLs, don't fetch content */
  discoverOnly?: boolean;
  /** Show what would be fetched without doing it */
  dryRun?: boolean;
  /** Progress callback for discover phase */
  onDiscoverProgress?: (info: DiscoverProgress) => void;
  /** Progress callback for fetch phase */
  onFetchProgress?: (info: FetchProgress) => void;
  /** Abort signal */
  signal?: AbortSignal;
}

// =============================================================================
// Phase 1: Discover - Read browser history and upsert into DB
// =============================================================================

/**
 * Discover URLs from browser history and bookmarks.
 * Upserts into pages + page_sources tables.
 */
export function discover(store: Store, options?: {
  browser?: string;
  since?: Date;
  onProgress?: (info: DiscoverProgress) => void;
}): DiscoverResult {
  const browsers = getDetectedBrowsers();
  const filteredBrowsers = options?.browser
    ? browsers.filter(b => b.name === options.browser)
    : browsers;

  if (filteredBrowsers.length === 0) {
    return { browsersScanned: 0, urlsDiscovered: 0, urlsUpdated: 0, urlsSkipped: 0 };
  }

  // Get user-defined exclude filters from DB
  const excludeFilters = store.getExcludeFilters();

  let totalDiscovered = 0;
  let totalUpdated = 0;
  let totalSkipped = 0;

  for (const browser of filteredBrowsers) {
    // Register/update browser in DB
    store.upsertBrowser(
      browser.name,
      browser.historyPath,
      browser.bookmarksPath ?? null
    );

    // Read history
    const historyEntries: HistoryEntry[] = browser.type === "chromium"
      ? readChromiumHistory(browser, { since: options?.since })
      : readSafariHistory(browser, { since: options?.since });

    options?.onProgress?.({
      browser: browser.name,
      phase: "history",
      entriesFound: historyEntries.length,
    });

    for (const entry of historyEntries) {
      if (matchesExcludeFilter(entry.url, excludeFilters)) {
        totalSkipped++;
        continue;
      }

      const pageId = store.upsertPage(
        entry.url,
        entry.title,
        null, // no content hash yet
        "pending"
      );

      store.upsertPageSource(
        pageId,
        entry.browser,
        "history",
        entry.visitCount,
        entry.lastVisitTime.toISOString()
      );

      totalDiscovered++;
    }

    // Read bookmarks (Chromium only)
    if (browser.type === "chromium") {
      const bookmarkEntries: BookmarkEntry[] = readChromiumBookmarks(browser);

      options?.onProgress?.({
        browser: browser.name,
        phase: "bookmarks",
        entriesFound: bookmarkEntries.length,
      });

      for (const entry of bookmarkEntries) {
        if (shouldSkipUrl(entry.url) || matchesExcludeFilter(entry.url, excludeFilters)) {
          totalSkipped++;
          continue;
        }

        const pageId = store.upsertPage(
          entry.url,
          entry.title,
          null,
          "pending"
        );

        store.upsertPageSource(
          pageId,
          entry.browser,
          "bookmark",
          0,
          null,
          entry.folder
        );

        totalDiscovered++;
      }
    }

    // Update browser sync time
    store.updateBrowserSyncTime(browser.name);
  }

  return {
    browsersScanned: filteredBrowsers.length,
    urlsDiscovered: totalDiscovered,
    urlsUpdated: totalUpdated,
    urlsSkipped: totalSkipped,
  };
}

// =============================================================================
// Phase 2: Fetch - Download and extract content for pending pages
// =============================================================================

/**
 * Fetch content for pages that haven't been fetched yet.
 */
export async function fetchPages(store: Store, options?: {
  rateLimit?: number;
  maxPages?: number;
  dryRun?: boolean;
  onProgress?: (info: FetchProgress) => void;
  signal?: AbortSignal;
}): Promise<FetchResult> {
  const startTime = Date.now();
  const limit = options?.maxPages ?? 1000;
  const rateLimiter = createRateLimiter(options?.rateLimit ?? 2);

  const pendingPages = store.getPendingPages(limit);

  if (options?.dryRun) {
    for (const page of pendingPages) {
      options?.onProgress?.({
        url: page.url,
        current: 0,
        total: pendingPages.length,
        status: "skipped",
      });
    }
    return {
      fetched: 0,
      failed: 0,
      skipped: pendingPages.length,
      unchanged: 0,
      durationMs: Date.now() - startTime,
    };
  }

  // Create indexer run for tracking
  const runId = store.createIndexerRun();
  store.updateIndexerRun(runId, { urls_discovered: pendingPages.length });

  let fetched = 0;
  let failed = 0;
  let skipped = 0;
  let unchanged = 0;

  for (let i = 0; i < pendingPages.length; i++) {
    if (options?.signal?.aborted) {
      store.updateIndexerRun(runId, {
        status: "interrupted",
        completed_at: new Date().toISOString(),
        urls_fetched: fetched,
        urls_failed: failed,
        urls_skipped: skipped,
        last_processed_url: pendingPages[i]!.url,
      });
      break;
    }

    const page = pendingPages[i]!;

    // Check if URL is fetchable
    if (!isFetchableUrl(page.url)) {
      store.updatePageFetchResult(page.url, null, page.title, "skipped", "Not fetchable");
      skipped++;
      options?.onProgress?.({
        url: page.url,
        current: i + 1,
        total: pendingPages.length,
        status: "skipped",
      });
      continue;
    }

    // Rate limit
    await rateLimiter.acquire();

    options?.onProgress?.({
      url: page.url,
      current: i + 1,
      total: pendingPages.length,
      status: "fetching",
    });

    const outcome = await fetchPage(page.url);

    if (outcome.ok) {
      const content = outcome.result.content;
      const contentHash = hashContent(content);

      // Check if content is the same as before
      const existing = store.getPageByUrl(page.url);
      if (existing?.hash === contentHash) {
        unchanged++;
        store.updatePageFetchResult(page.url, contentHash, outcome.result.title || page.title, "fetched");
        options?.onProgress?.({
          url: page.url,
          current: i + 1,
          total: pendingPages.length,
          status: "fetched",
        });
        continue;
      }

      // Store content and update page
      store.insertContent(contentHash, content, new Date().toISOString());
      store.updatePageFetchResult(
        page.url,
        contentHash,
        outcome.result.title || page.title,
        "fetched"
      );

      fetched++;
      options?.onProgress?.({
        url: page.url,
        current: i + 1,
        total: pendingPages.length,
        status: "fetched",
      });
    } else {
      store.updatePageFetchResult(page.url, null, page.title, "failed", outcome.error.error);
      failed++;
      options?.onProgress?.({
        url: page.url,
        current: i + 1,
        total: pendingPages.length,
        status: "failed",
        error: outcome.error.error,
      });
    }

    // Update indexer run periodically
    if (i % 50 === 0) {
      store.updateIndexerRun(runId, {
        urls_fetched: fetched,
        urls_failed: failed,
        urls_skipped: skipped,
        last_processed_url: page.url,
      });
    }
  }

  // Finalize indexer run
  store.updateIndexerRun(runId, {
    status: "completed",
    completed_at: new Date().toISOString(),
    urls_fetched: fetched,
    urls_failed: failed,
    urls_skipped: skipped,
  });

  return {
    fetched,
    failed,
    skipped,
    unchanged,
    durationMs: Date.now() - startTime,
  };
}

// =============================================================================
// Combined Index Pipeline
// =============================================================================

/**
 * Run the full indexing pipeline: discover + fetch.
 */
export async function runIndex(store: Store, options?: IndexOptions): Promise<{
  discover: DiscoverResult;
  fetch?: FetchResult;
}> {
  // Phase 1: Discover
  const discoverResult = discover(store, {
    browser: options?.browser,
    since: options?.since,
    onProgress: options?.onDiscoverProgress,
  });

  if (options?.discoverOnly) {
    return { discover: discoverResult };
  }

  // Phase 2: Fetch
  const fetchResult = await fetchPages(store, {
    rateLimit: options?.rateLimit,
    maxPages: options?.maxPages,
    dryRun: options?.dryRun,
    onProgress: options?.onFetchProgress,
    signal: options?.signal,
  });

  return { discover: discoverResult, fetch: fetchResult };
}
