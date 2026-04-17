/**
 * planet-capture CLI
 *
 * Commands:
 *   index          - Discover URLs from browsers, fetch pages, index content
 *   discover       - Only read browser history (no fetch)
 *   fetch          - Only fetch pending pages (no discover)
 *   search <q>     - BM25 full-text search
 *   vsearch <q>    - Vector similarity search
 *   query <q>      - Hybrid search (BM25 + vector + rerank)
 *   get <url|id>   - Show a single page
 *   status         - Show index counts and last run
 *   browsers       - List detected browsers
 *   filters        - Manage URL filters (list/add/remove)
 *   embed          - Generate vector embeddings
 *   mcp            - Start MCP server (stdio or HTTP)
 */

import { createStore as createInternalStore, enableProductionMode } from "../store.js";
import type { Store } from "../store.js";
import { runIndex, discover, fetchPages } from "../indexer.js";
import { detectBrowsers } from "../browsers.js";
import { generateEmbeddings } from "../store.js";
import {
  formatSearchResults,
  formatPage,
  type OutputFormat,
} from "./formatter.js";
import { hybridQuery } from "../store.js";

// =============================================================================
// Argument parsing
// =============================================================================

type ParsedArgs = {
  positional: string[];
  flags: Record<string, string | boolean>;
};

function parseArgs(argv: string[]): ParsedArgs {
  const positional: string[] = [];
  const flags: Record<string, string | boolean> = {};

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i]!;
    if (arg.startsWith("--")) {
      const eq = arg.indexOf("=");
      if (eq >= 0) {
        flags[arg.slice(2, eq)] = arg.slice(eq + 1);
      } else {
        const name = arg.slice(2);
        const next = argv[i + 1];
        if (next && !next.startsWith("-")) {
          flags[name] = next;
          i++;
        } else {
          flags[name] = true;
        }
      }
    } else if (arg.startsWith("-") && arg.length > 1) {
      const name = arg.slice(1);
      const next = argv[i + 1];
      if (next && !next.startsWith("-")) {
        flags[name] = next;
        i++;
      } else {
        flags[name] = true;
      }
    } else {
      positional.push(arg);
    }
  }

  return { positional, flags };
}

function parseOutputFormat(flags: Record<string, string | boolean>, fallback: OutputFormat = "cli"): OutputFormat {
  if (flags.json) return "json";
  if (flags.csv) return "csv";
  if (flags.md) return "md";
  if (flags.xml) return "xml";
  if (flags.files) return "files";
  return fallback;
}

function asInt(val: string | boolean | undefined, fallback: number): number {
  if (typeof val === "string") {
    const n = parseInt(val, 10);
    return Number.isFinite(n) ? n : fallback;
  }
  return fallback;
}

// =============================================================================
// Commands
// =============================================================================

async function cmdIndex(store: Store, args: ParsedArgs): Promise<void> {
  const browser = typeof args.flags.browser === "string" ? args.flags.browser : undefined;
  const rateLimit = asInt(args.flags["rate-limit"], 2);
  const maxPages = args.flags["max-pages"] ? asInt(args.flags["max-pages"], 1000) : undefined;
  const discoverOnly = !!args.flags["discover-only"];
  const dryRun = !!args.flags["dry-run"];
  const since = typeof args.flags.since === "string" ? new Date(args.flags.since) : undefined;

  // Ensure detected browsers are registered before discover runs
  const detected = detectBrowsers();
  for (const b of detected.filter(x => x.detected)) {
    store.upsertBrowser(b.name, b.historyPath, b.bookmarksPath ?? null);
  }

  console.log("Discovering URLs from browser history...");
  const result = await runIndex(store, {
    browser,
    since,
    rateLimit,
    maxPages,
    discoverOnly,
    dryRun,
    onDiscoverProgress: (info) => {
      console.log(`  ${info.browser} ${info.phase}: ${info.entriesFound} entries`);
    },
    onFetchProgress: (info) => {
      const pct = info.total > 0 ? Math.round((info.current / info.total) * 100) : 0;
      const statusEmoji = info.status === "fetched" ? "✓" : info.status === "failed" ? "✗" : "·";
      process.stdout.write(`\r  [${pct}%] ${statusEmoji} ${info.current}/${info.total} ${info.url.slice(0, 60)}`.padEnd(100) + "\n");
    },
  });

  console.log(`\nDiscover: ${result.discover.urlsDiscovered} URLs, ${result.discover.urlsSkipped} skipped`);
  if (result.fetch) {
    console.log(`Fetch: ${result.fetch.fetched} fetched, ${result.fetch.failed} failed, ${result.fetch.skipped} skipped, ${result.fetch.unchanged} unchanged (${result.fetch.durationMs}ms)`);
  }
}

async function cmdDiscover(store: Store, args: ParsedArgs): Promise<void> {
  const browser = typeof args.flags.browser === "string" ? args.flags.browser : undefined;
  const since = typeof args.flags.since === "string" ? new Date(args.flags.since) : undefined;

  const detected = detectBrowsers();
  for (const b of detected.filter(x => x.detected)) {
    store.upsertBrowser(b.name, b.historyPath, b.bookmarksPath ?? null);
  }

  const result = discover(store, {
    browser,
    since,
    onProgress: (info) => console.log(`  ${info.browser} ${info.phase}: ${info.entriesFound} entries`),
  });

  console.log(`\nDiscovered ${result.urlsDiscovered} URLs (${result.urlsSkipped} skipped) across ${result.browsersScanned} browsers`);
}

async function cmdFetch(store: Store, args: ParsedArgs): Promise<void> {
  const rateLimit = asInt(args.flags["rate-limit"], 2);
  const maxPages = args.flags["max-pages"] ? asInt(args.flags["max-pages"], 1000) : undefined;
  const dryRun = !!args.flags["dry-run"];

  const result = await fetchPages(store, {
    rateLimit,
    maxPages,
    dryRun,
    onProgress: (info) => {
      const statusEmoji = info.status === "fetched" ? "✓" : info.status === "failed" ? "✗" : "·";
      console.log(`  ${statusEmoji} ${info.current}/${info.total} ${info.url}`);
    },
  });

  console.log(`\nFetched ${result.fetched}, failed ${result.failed}, skipped ${result.skipped}, unchanged ${result.unchanged} (${result.durationMs}ms)`);
}

async function cmdSearch(store: Store, args: ParsedArgs): Promise<void> {
  const query = args.positional.join(" ");
  if (!query) {
    console.error("Usage: planet-capture search <query>");
    process.exit(1);
  }
  const limit = asInt(args.flags.n, 10);
  const browser = typeof args.flags.browser === "string" ? args.flags.browser : undefined;
  const format = parseOutputFormat(args.flags);

  const results = store.searchFTS(query, limit, browser);
  console.log(formatSearchResults(results, format, {
    query,
    full: !!args.flags.full,
    lineNumbers: !!args.flags["line-numbers"],
  }));
}

async function cmdVsearch(store: Store, args: ParsedArgs): Promise<void> {
  const query = args.positional.join(" ");
  if (!query) {
    console.error("Usage: planet-capture vsearch <query>");
    process.exit(1);
  }
  const limit = asInt(args.flags.n, 10);
  const browser = typeof args.flags.browser === "string" ? args.flags.browser : undefined;
  const format = parseOutputFormat(args.flags);

  const { DEFAULT_EMBED_MODEL } = await import("../store.js");
  const results = await store.searchVec(query, DEFAULT_EMBED_MODEL, limit, browser);
  console.log(formatSearchResults(results, format, {
    query,
    full: !!args.flags.full,
    lineNumbers: !!args.flags["line-numbers"],
  }));
}

async function cmdQuery(store: Store, args: ParsedArgs): Promise<void> {
  const query = args.positional.join(" ");
  if (!query) {
    console.error("Usage: planet-capture query <query>");
    process.exit(1);
  }
  const limit = asInt(args.flags.n, 10);
  const browser = typeof args.flags.browser === "string" ? args.flags.browser : undefined;
  const format = parseOutputFormat(args.flags);
  const intent = typeof args.flags.intent === "string" ? args.flags.intent : undefined;

  const results = await hybridQuery(store, query, {
    limit,
    collection: browser,
    intent,
    minScore: args.flags["min-score"] ? parseFloat(String(args.flags["min-score"])) : undefined,
  });

  // Project hybrid results into SearchResult-shaped records for the formatter
  const projected = results.map(r => ({
    url: r.file,
    title: r.title,
    hash: "",
    docid: r.docid,
    browsers: [],
    visitCount: 0,
    fetchStatus: "fetched",
    fetchedAt: null,
    modifiedAt: "",
    bodyLength: r.body.length,
    body: r.body,
    score: r.score,
    source: "fts" as const,
    chunkPos: r.bestChunkPos,
  }));

  console.log(formatSearchResults(projected, format, {
    query,
    full: !!args.flags.full,
    lineNumbers: !!args.flags["line-numbers"],
    intent,
  }));
}

async function cmdGet(store: Store, args: ParsedArgs): Promise<void> {
  const target = args.positional[0];
  if (!target) {
    console.error("Usage: planet-capture get <url-or-docid>");
    process.exit(1);
  }
  const format = parseOutputFormat(args.flags, "md");
  const result = store.findPage(target, { includeBody: true });

  if ("error" in result) {
    console.error(`Not found: ${target}`);
    if (result.similarUrls.length > 0) {
      console.error(`Did you mean: ${result.similarUrls.slice(0, 3).join(", ")}?`);
    }
    process.exit(1);
  }

  console.log(formatPage(result, format));
}

async function cmdStatus(store: Store, _args: ParsedArgs): Promise<void> {
  const status = store.getStatus();
  console.log(`Pages:`);
  console.log(`  total:        ${status.totalPages}`);
  console.log(`  fetched:      ${status.fetchedPages}`);
  console.log(`  pending:      ${status.pendingPages}`);
  console.log(`  failed:       ${status.failedPages}`);
  console.log(`  needs embed:  ${status.needsEmbedding}`);
  console.log(`  vector index: ${status.hasVectorIndex ? "yes" : "no"}`);

  if (status.browsers.length > 0) {
    console.log(`\nBrowsers:`);
    for (const b of status.browsers) {
      console.log(`  ${b.name.padEnd(10)} pages=${b.pages}  synced=${b.lastSyncedAt || "never"}`);
    }
  }

  if (status.lastRun) {
    console.log(`\nLast run:`);
    console.log(`  status:      ${status.lastRun.status}`);
    console.log(`  started:     ${status.lastRun.started_at}`);
    console.log(`  completed:   ${status.lastRun.completed_at || "—"}`);
    console.log(`  discovered:  ${status.lastRun.urls_discovered ?? 0}`);
    console.log(`  fetched:     ${status.lastRun.urls_fetched ?? 0}`);
    console.log(`  failed:      ${status.lastRun.urls_failed ?? 0}`);
  }
}

async function cmdBrowsers(_store: Store, _args: ParsedArgs): Promise<void> {
  const detected = detectBrowsers();
  for (const b of detected) {
    const mark = b.detected ? "✓" : " ";
    console.log(`  ${mark} ${b.name.padEnd(10)} ${b.type.padEnd(9)} ${b.historyPath}`);
  }
}

async function cmdFilters(store: Store, args: ParsedArgs): Promise<void> {
  const sub = args.positional[0];

  if (sub === "add") {
    const pattern = args.positional[1];
    if (!pattern) {
      console.error("Usage: planet-capture filters add <pattern>");
      process.exit(1);
    }
    store.addUrlFilter(pattern, "exclude");
    console.log(`Added exclude filter: ${pattern}`);
    return;
  }

  if (sub === "remove" || sub === "rm") {
    const pattern = args.positional[1];
    if (!pattern) {
      console.error("Usage: planet-capture filters remove <pattern>");
      process.exit(1);
    }
    const removed = store.removeUrlFilter(pattern);
    console.log(removed ? `Removed: ${pattern}` : `Not found: ${pattern}`);
    return;
  }

  // Default: list
  const filters = store.listUrlFilters();
  if (filters.length === 0) {
    console.log("(no filters)");
    return;
  }
  for (const f of filters) {
    console.log(`  [${f.filter_type}] ${f.pattern}`);
  }
}

async function cmdEmbed(store: Store, args: ParsedArgs): Promise<void> {
  const force = !!args.flags.force;
  console.log(force ? "Re-embedding all pages..." : "Generating embeddings for new pages...");

  const result = await generateEmbeddings(store, {
    force,
    onProgress: (info) => {
      const pct = info.totalBytes > 0 ? Math.round((info.bytesProcessed / info.totalBytes) * 100) : 0;
      process.stdout.write(`\r  [${pct}%] chunks=${info.chunksEmbedded}/${info.totalChunks} errors=${info.errors}`);
    },
  });
  console.log(`\nEmbedded ${result.chunksEmbedded} chunks across ${result.docsProcessed} pages (${result.errors} errors, ${result.durationMs}ms)`);
}

async function cmdMcp(_store: Store, args: ParsedArgs): Promise<void> {
  const { startStdio, startHttp } = await import("../mcp/server.js");
  if (args.flags.http) {
    const port = args.flags.port ? Number(args.flags.port) : 8181;
    await startHttp(port);
  } else {
    await startStdio();
  }
}

// =============================================================================
// Main
// =============================================================================

function printHelp(): void {
  console.log(`planet-capture — index and search your browser history

Usage: planet-capture <command> [options]

Commands:
  index              Discover URLs from browsers + fetch pages
  discover           Only read browser history (no fetch)
  fetch              Only fetch pending pages (no discover)
  search <query>     BM25 full-text search
  vsearch <query>    Vector similarity search
  query <query>      Hybrid search (BM25 + vector + rerank)
  get <url|#docid>   Show a single page
  status             Show index counts and last run
  browsers           List detected browsers
  filters [list|add|remove] [pattern]
                     Manage URL filters
  embed              Generate vector embeddings
  mcp                Start MCP server (stdio, default)
  mcp --http         Start MCP server (HTTP)
  mcp --http --port=N  HTTP on custom port (default 8181)

Options (by command):
  index/discover/fetch:
    --browser=NAME       Only scan this browser
    --since=DATE         Only include history after this date
    --rate-limit=N       Fetches per second (default 2)
    --max-pages=N        Max pages to fetch this run
    --discover-only      Skip fetching
    --dry-run            Show what would happen

  search/vsearch/query:
    -n N                 Number of results (default 10)
    --browser=NAME       Restrict to one browser
    --full               Show full body instead of snippet
    --line-numbers       Annotate with line numbers
    --intent=TEXT        Disambiguation intent (query only)
    --min-score=N        Minimum relevance score
    --json --csv --md --xml --files
                         Output format

  get:
    --full, --line-numbers, --json, --md

  embed:
    --force              Re-embed everything
`);
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  if (argv.length === 0 || argv[0] === "help" || argv[0] === "--help" || argv[0] === "-h") {
    printHelp();
    return;
  }

  const command = argv[0]!;
  const args = parseArgs(argv.slice(1));
  enableProductionMode();
  const store = createInternalStore();

  try {
    switch (command) {
      case "index":      await cmdIndex(store, args); break;
      case "discover":   await cmdDiscover(store, args); break;
      case "fetch":      await cmdFetch(store, args); break;
      case "search":     await cmdSearch(store, args); break;
      case "vsearch":    await cmdVsearch(store, args); break;
      case "query":      await cmdQuery(store, args); break;
      case "get":        await cmdGet(store, args); break;
      case "status":     await cmdStatus(store, args); break;
      case "browsers":   await cmdBrowsers(store, args); break;
      case "filters":    await cmdFilters(store, args); break;
      case "embed":      await cmdEmbed(store, args); break;
      case "mcp":        await cmdMcp(store, args); break;
      default:
        console.error(`Unknown command: ${command}`);
        console.error(`Run 'planet-capture help' for usage.`);
        process.exit(1);
    }
  } finally {
    store.close();
  }
}

main().catch(err => {
  console.error(err?.stack || err?.message || err);
  process.exit(1);
});
