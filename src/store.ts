/**
 * planet-capture Store - Core data access and retrieval functions
 *
 * This module provides all database operations, search functions, and page
 * retrieval for planet-capture. It returns raw data structures that can be
 * formatted by CLI or MCP consumers.
 *
 * Usage:
 *   const store = createStore("/path/to/db.sqlite");
 *   // or use default path:
 *   const store = createStore();
 */

import { openDatabase, loadSqliteVec } from "./db.js";
import type { Database } from "./db.js";
import { createHash } from "crypto";
import { mkdirSync } from "node:fs";
import {
  LlamaCpp,
  getDefaultLlamaCpp,
  formatQueryForEmbedding,
  formatDocForEmbedding,
  withLLMSessionForLlm,
  type RerankDocument,
  type ILLMSession,
} from "./llm.js";

// =============================================================================
// Configuration
// =============================================================================

const HOME = process.env.HOME || process.env.USERPROFILE || "/tmp";
export const DEFAULT_EMBED_MODEL = "embeddinggemma";
export const DEFAULT_RERANK_MODEL = "ExpedientFalcon/qwen3-reranker:0.6b-q8_0";
export const DEFAULT_QUERY_MODEL = "Qwen/Qwen3-1.7B";
export const DEFAULT_MULTI_GET_MAX_BYTES = 10 * 1024; // 10KB
export const DEFAULT_EMBED_MAX_DOCS_PER_BATCH = 64;
export const DEFAULT_EMBED_MAX_BATCH_BYTES = 64 * 1024 * 1024; // 64MB

// Chunking: 900 tokens per chunk with 15% overlap
// Increased from 800 to accommodate smart chunking finding natural break points
export const CHUNK_SIZE_TOKENS = 900;
export const CHUNK_OVERLAP_TOKENS = Math.floor(CHUNK_SIZE_TOKENS * 0.15);  // 135 tokens (15% overlap)
// Fallback char-based approximation for sync chunking (~4 chars per token)
export const CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * 4;  // 3600 chars
export const CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * 4;  // 540 chars
// Search window for finding optimal break points (in tokens, ~200 tokens)
export const CHUNK_WINDOW_TOKENS = 200;
export const CHUNK_WINDOW_CHARS = CHUNK_WINDOW_TOKENS * 4;  // 800 chars

/**
 * Get the LlamaCpp instance for a store — prefers the store's own instance,
 * falls back to the global singleton.
 */
function getLlm(store: Store): LlamaCpp {
  return store.llm ?? getDefaultLlamaCpp();
}

// =============================================================================
// Smart Chunking - Break Point Detection
// =============================================================================

/**
 * A potential break point in the document with a base score indicating quality.
 */
export interface BreakPoint {
  pos: number;    // character position
  score: number;  // base score (higher = better break point)
  type: string;   // for debugging: 'h1', 'h2', 'blank', etc.
}

/**
 * A region where a code fence exists (between ``` markers).
 * We should never split inside a code fence.
 */
export interface CodeFenceRegion {
  start: number;  // position of opening ```
  end: number;    // position of closing ``` (or document end if unclosed)
}

/**
 * Patterns for detecting break points in markdown documents.
 * Higher scores indicate better places to split.
 * Scores are spread wide so headings decisively beat lower-quality breaks.
 * Order matters for scoring - more specific patterns first.
 */
export const BREAK_PATTERNS: [RegExp, number, string][] = [
  [/\n#{1}(?!#)/g, 100, 'h1'],     // # but not ##
  [/\n#{2}(?!#)/g, 90, 'h2'],      // ## but not ###
  [/\n#{3}(?!#)/g, 80, 'h3'],      // ### but not ####
  [/\n#{4}(?!#)/g, 70, 'h4'],      // #### but not #####
  [/\n#{5}(?!#)/g, 60, 'h5'],      // ##### but not ######
  [/\n#{6}(?!#)/g, 50, 'h6'],      // ######
  [/\n```/g, 80, 'codeblock'],     // code block boundary (same as h3)
  [/\n(?:---|\*\*\*|___)\s*\n/g, 60, 'hr'],  // horizontal rule
  [/\n\n+/g, 20, 'blank'],         // paragraph boundary
  [/\n[-*]\s/g, 5, 'list'],        // unordered list item
  [/\n\d+\.\s/g, 5, 'numlist'],    // ordered list item
  [/\n/g, 1, 'newline'],           // minimal break
];

/**
 * Scan text for all potential break points.
 * Returns sorted array of break points with higher-scoring patterns taking precedence
 * when multiple patterns match the same position.
 */
export function scanBreakPoints(text: string): BreakPoint[] {
  const points: BreakPoint[] = [];
  const seen = new Map<number, BreakPoint>();  // pos -> best break point at that pos

  for (const [pattern, score, type] of BREAK_PATTERNS) {
    for (const match of text.matchAll(pattern)) {
      const pos = match.index!;
      const existing = seen.get(pos);
      // Keep higher score if position already seen
      if (!existing || score > existing.score) {
        const bp = { pos, score, type };
        seen.set(pos, bp);
      }
    }
  }

  // Convert to array and sort by position
  for (const bp of seen.values()) {
    points.push(bp);
  }
  return points.sort((a, b) => a.pos - b.pos);
}

/**
 * Find all code fence regions in the text.
 * Code fences are delimited by ``` and we should never split inside them.
 */
export function findCodeFences(text: string): CodeFenceRegion[] {
  const regions: CodeFenceRegion[] = [];
  const fencePattern = /\n```/g;
  let inFence = false;
  let fenceStart = 0;

  for (const match of text.matchAll(fencePattern)) {
    if (!inFence) {
      fenceStart = match.index!;
      inFence = true;
    } else {
      regions.push({ start: fenceStart, end: match.index! + match[0].length });
      inFence = false;
    }
  }

  // Handle unclosed fence - extends to end of document
  if (inFence) {
    regions.push({ start: fenceStart, end: text.length });
  }

  return regions;
}

/**
 * Check if a position is inside a code fence region.
 */
export function isInsideCodeFence(pos: number, fences: CodeFenceRegion[]): boolean {
  return fences.some(f => pos > f.start && pos < f.end);
}

/**
 * Find the best cut position using scored break points with distance decay.
 *
 * Uses squared distance for gentler early decay - headings far back still win
 * over low-quality breaks near the target.
 *
 * @param breakPoints - Pre-scanned break points from scanBreakPoints()
 * @param targetCharPos - The ideal cut position (e.g., maxChars boundary)
 * @param windowChars - How far back to search for break points (default ~200 tokens)
 * @param decayFactor - How much to penalize distance (0.7 = 30% score at window edge)
 * @param codeFences - Code fence regions to avoid splitting inside
 * @returns The best position to cut at
 */
export function findBestCutoff(
  breakPoints: BreakPoint[],
  targetCharPos: number,
  windowChars: number = CHUNK_WINDOW_CHARS,
  decayFactor: number = 0.7,
  codeFences: CodeFenceRegion[] = []
): number {
  const windowStart = targetCharPos - windowChars;
  let bestScore = -1;
  let bestPos = targetCharPos;

  for (const bp of breakPoints) {
    if (bp.pos < windowStart) continue;
    if (bp.pos > targetCharPos) break;  // sorted, so we can stop

    // Skip break points inside code fences
    if (isInsideCodeFence(bp.pos, codeFences)) continue;

    const distance = targetCharPos - bp.pos;
    // Squared distance decay: gentle early, steep late
    // At target: multiplier = 1.0
    // At 25% back: multiplier = 0.956
    // At 50% back: multiplier = 0.825
    // At 75% back: multiplier = 0.606
    // At window edge: multiplier = 0.3
    const normalizedDist = distance / windowChars;
    const multiplier = 1.0 - (normalizedDist * normalizedDist) * decayFactor;
    const finalScore = bp.score * multiplier;

    if (finalScore > bestScore) {
      bestScore = finalScore;
      bestPos = bp.pos;
    }
  }

  return bestPos;
}

// =============================================================================
// Chunk Strategy
// =============================================================================

export type ChunkStrategy = "auto" | "regex";

/**
 * Merge two sets of break points (e.g. regex + AST), keeping the highest
 * score at each position. Result is sorted by position.
 */
export function mergeBreakPoints(a: BreakPoint[], b: BreakPoint[]): BreakPoint[] {
  const seen = new Map<number, BreakPoint>();
  for (const bp of a) {
    const existing = seen.get(bp.pos);
    if (!existing || bp.score > existing.score) {
      seen.set(bp.pos, bp);
    }
  }
  for (const bp of b) {
    const existing = seen.get(bp.pos);
    if (!existing || bp.score > existing.score) {
      seen.set(bp.pos, bp);
    }
  }
  return Array.from(seen.values()).sort((a, b) => a.pos - b.pos);
}

/**
 * Core chunk algorithm that operates on precomputed break points and code fences.
 * This is the shared implementation used by both regex-only and AST-aware chunking.
 */
export function chunkDocumentWithBreakPoints(
  content: string,
  breakPoints: BreakPoint[],
  codeFences: CodeFenceRegion[],
  maxChars: number = CHUNK_SIZE_CHARS,
  overlapChars: number = CHUNK_OVERLAP_CHARS,
  windowChars: number = CHUNK_WINDOW_CHARS
): { text: string; pos: number }[] {
  if (content.length <= maxChars) {
    return [{ text: content, pos: 0 }];
  }

  const chunks: { text: string; pos: number }[] = [];
  let charPos = 0;

  while (charPos < content.length) {
    const targetEndPos = Math.min(charPos + maxChars, content.length);
    let endPos = targetEndPos;

    if (endPos < content.length) {
      const bestCutoff = findBestCutoff(
        breakPoints,
        targetEndPos,
        windowChars,
        0.7,
        codeFences
      );

      if (bestCutoff > charPos && bestCutoff <= targetEndPos) {
        endPos = bestCutoff;
      }
    }

    if (endPos <= charPos) {
      endPos = Math.min(charPos + maxChars, content.length);
    }

    chunks.push({ text: content.slice(charPos, endPos), pos: charPos });

    if (endPos >= content.length) {
      break;
    }
    charPos = endPos - overlapChars;
    const lastChunkPos = chunks.at(-1)!.pos;
    if (charPos <= lastChunkPos) {
      charPos = endPos;
    }
  }

  return chunks;
}

// Hybrid query: strong BM25 signal detection thresholds
// Skip expensive LLM expansion when top result is strong AND clearly separated from runner-up
export const STRONG_SIGNAL_MIN_SCORE = 0.85;
export const STRONG_SIGNAL_MIN_GAP = 0.15;
// Max candidates to pass to reranker — balances quality vs latency.
// 40 keeps rank 31-40 visible to the reranker (matters for recall on broad queries).
export const RERANK_CANDIDATE_LIMIT = 40;

/**
 * A typed query expansion result. Decoupled from llm.ts internal Queryable —
 * same shape, but store.ts owns its own public API type.
 *
 * - lex: keyword variant → routes to FTS only
 * - vec: semantic variant → routes to vector only
 * - hyde: hypothetical document → routes to vector only
 */
export type ExpandedQuery = {
  type: 'lex' | 'vec' | 'hyde';
  query: string;
  /** Optional line number for error reporting (CLI parser) */
  line?: number;
};

// =============================================================================
// Path utilities
// =============================================================================

export function homedir(): string {
  return HOME;
}

/**
 * Check if a path is absolute.
 * Supports:
 * - Unix paths: /path/to/file
 * - Windows native: C:\path or C:/path
 * - Git Bash: /c/path or /C/path (C-Z drives, excluding A/B floppy drives)
 * 
 * Note: /c without trailing slash is treated as Unix path (directory named "c"),
 * while /c/ or /c/path are treated as Git Bash paths (C: drive).
 */
export function isAbsolutePath(path: string): boolean {
  if (!path) return false;
  
  // Unix absolute path
  if (path.startsWith('/')) {
    // Check if it's a Git Bash style path like /c/ or /c/Users (C-Z only, not A or B)
    // Requires path[2] === '/' to distinguish from Unix paths like /c or /cache
    // Skipped on WSL where /c/ is a valid drvfs mount point, not a drive letter
    if (!isWSL() && path.length >= 3 && path[2] === '/') {
      const driveLetter = path[1];
      if (driveLetter && /[c-zC-Z]/.test(driveLetter)) {
        return true;
      }
    }
    // Any other path starting with / is Unix absolute
    return true;
  }
  
  // Windows native path: C:\ or C:/ (any letter A-Z)
  if (path.length >= 2 && /[a-zA-Z]/.test(path[0]!) && path[1] === ':') {
    return true;
  }
  
  return false;
}

/**
 * Normalize path separators to forward slashes.
 * Converts Windows backslashes to forward slashes.
 */
export function normalizePathSeparators(path: string): string {
  return path.replace(/\\/g, '/');
}

/**
 * Detect if running inside WSL (Windows Subsystem for Linux).
 * On WSL, paths like /c/work/... are valid drvfs mount points, not Git Bash paths.
 */
function isWSL(): boolean {
  return !!(process.env.WSL_DISTRO_NAME || process.env.WSL_INTEROP);
}

/**
 * Get the relative path from a prefix.
 * Returns null if path is not under prefix.
 * Returns empty string if path equals prefix.
 */
export function getRelativePathFromPrefix(path: string, prefix: string): string | null {
  // Empty prefix is invalid
  if (!prefix) {
    return null;
  }
  
  const normalizedPath = normalizePathSeparators(path);
  const normalizedPrefix = normalizePathSeparators(prefix);
  
  // Ensure prefix ends with / for proper matching
  const prefixWithSlash = !normalizedPrefix.endsWith('/') 
    ? normalizedPrefix + '/' 
    : normalizedPrefix;
  
  // Exact match
  if (normalizedPath === normalizedPrefix) {
    return '';
  }
  
  // Check if path starts with prefix
  if (normalizedPath.startsWith(prefixWithSlash)) {
    return normalizedPath.slice(prefixWithSlash.length);
  }
  
  return null;
}

export function resolve(...paths: string[]): string {
  if (paths.length === 0) {
    throw new Error("resolve: at least one path segment is required");
  }
  
  // Normalize all paths to use forward slashes
  const normalizedPaths = paths.map(normalizePathSeparators);
  
  let result = '';
  let windowsDrive = '';
  
  // Check if first path is absolute
  const firstPath = normalizedPaths[0]!;
  if (isAbsolutePath(firstPath)) {
    result = firstPath;
    
    // Extract Windows drive letter if present
    if (firstPath.length >= 2 && /[a-zA-Z]/.test(firstPath[0]!) && firstPath[1] === ':') {
      windowsDrive = firstPath.slice(0, 2);
      result = firstPath.slice(2);
    } else if (!isWSL() && firstPath.startsWith('/') && firstPath.length >= 3 && firstPath[2] === '/') {
      // Git Bash style: /c/ -> C: (C-Z drives only, not A or B)
      // Skipped on WSL where /c/ is a valid drvfs mount point, not a drive letter
      const driveLetter = firstPath[1];
      if (driveLetter && /[c-zC-Z]/.test(driveLetter)) {
        windowsDrive = driveLetter.toUpperCase() + ':';
        result = firstPath.slice(2);
      }
    }
  } else {
    // Start with PWD or cwd, then append the first relative path
    const pwd = normalizePathSeparators(process.env.PWD || process.cwd());
    
    // Extract Windows drive from PWD if present
    if (pwd.length >= 2 && /[a-zA-Z]/.test(pwd[0]!) && pwd[1] === ':') {
      windowsDrive = pwd.slice(0, 2);
      result = pwd.slice(2) + '/' + firstPath;
    } else {
      result = pwd + '/' + firstPath;
    }
  }
  
  // Process remaining paths
  for (let i = 1; i < normalizedPaths.length; i++) {
    const p = normalizedPaths[i]!;
    if (isAbsolutePath(p)) {
      // Absolute path replaces everything
      result = p;
      
      // Update Windows drive if present
      if (p.length >= 2 && /[a-zA-Z]/.test(p[0]!) && p[1] === ':') {
        windowsDrive = p.slice(0, 2);
        result = p.slice(2);
      } else if (!isWSL() && p.startsWith('/') && p.length >= 3 && p[2] === '/') {
        // Git Bash style (C-Z drives only, not A or B)
        // Skipped on WSL where /c/ is a valid drvfs mount point, not a drive letter
        const driveLetter = p[1];
        if (driveLetter && /[c-zC-Z]/.test(driveLetter)) {
          windowsDrive = driveLetter.toUpperCase() + ':';
          result = p.slice(2);
        } else {
          windowsDrive = '';
        }
      } else {
        windowsDrive = '';
      }
    } else {
      // Relative path - append
      result = result + '/' + p;
    }
  }
  
  // Normalize . and .. components
  const parts = result.split('/').filter(Boolean);
  const normalized: string[] = [];
  for (const part of parts) {
    if (part === '..') {
      normalized.pop();
    } else if (part !== '.') {
      normalized.push(part);
    }
  }
  
  // Build final path
  const finalPath = '/' + normalized.join('/');
  
  // Prepend Windows drive if present
  if (windowsDrive) {
    return windowsDrive + finalPath;
  }
  
  return finalPath;
}

// Flag to indicate production mode (set by qmd.ts at startup)
let _productionMode = false;

export function enableProductionMode(): void {
  _productionMode = true;
}

/** Reset production mode flag — only for testing. */
export function _resetProductionModeForTesting(): void {
  _productionMode = false;
}

export function getDefaultDbPath(indexName: string = "index"): string {
  // Always allow override via INDEX_PATH (for testing)
  if (process.env.INDEX_PATH) {
    return process.env.INDEX_PATH;
  }

  // In non-production mode (tests), require explicit path
  if (!_productionMode) {
    throw new Error(
      "Database path not set. Tests must set INDEX_PATH env var or use createStore() with explicit path. " +
      "This prevents tests from accidentally writing to the global index."
    );
  }

  const dataDir = process.env.PLANET_CAPTURE_DIR || resolve(homedir(), ".planet-capture");
  try { mkdirSync(dataDir, { recursive: true }); } catch { }
  return resolve(dataDir, `${indexName}.db`);
}

// =============================================================================
// Database initialization
// =============================================================================


function createSqliteVecUnavailableError(reason: string): Error {
  return new Error(
    "sqlite-vec extension is unavailable. " +
    `${reason}. ` +
    "Install Homebrew SQLite so the sqlite-vec extension can be loaded, " +
    "and set BREW_PREFIX if Homebrew is installed in a non-standard location."
  );
}

let _sqliteVecUnavailableReason: string | null = null;

function getErrorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function verifySqliteVecLoaded(db: Database): void {
  try {
    const row = db.prepare(`SELECT vec_version() AS version`).get() as { version?: string } | null;
    if (!row?.version || typeof row.version !== "string") {
      throw new Error("vec_version() returned no version");
    }
  } catch (err) {
    const message = getErrorMessage(err);
    throw createSqliteVecUnavailableError(`sqlite-vec probe failed (${message})`);
  }
}

let _sqliteVecAvailable: boolean | null = null;

function initializeDatabase(db: Database): void {
  try {
    loadSqliteVec(db);
    verifySqliteVecLoaded(db);
    _sqliteVecAvailable = true;
    _sqliteVecUnavailableReason = null;
  } catch (err) {
    // sqlite-vec is optional — vector search won't work but FTS is fine
    _sqliteVecAvailable = false;
    _sqliteVecUnavailableReason = getErrorMessage(err);
    console.warn(_sqliteVecUnavailableReason);
  }
  db.exec("PRAGMA journal_mode = WAL");
  db.exec("PRAGMA foreign_keys = ON");

  // Content-addressable storage - the source of truth for page content
  db.exec(`
    CREATE TABLE IF NOT EXISTS content (
      hash TEXT PRIMARY KEY,
      doc TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  // Pages table - maps URLs to fetched content
  db.exec(`
    CREATE TABLE IF NOT EXISTS pages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      url TEXT NOT NULL UNIQUE,
      title TEXT NOT NULL DEFAULT '',
      hash TEXT,
      fetch_status TEXT DEFAULT 'pending',
      fetch_error TEXT,
      fetched_at TEXT,
      active INTEGER NOT NULL DEFAULT 1,
      created_at TEXT NOT NULL,
      modified_at TEXT NOT NULL,
      FOREIGN KEY (hash) REFERENCES content(hash) ON DELETE SET NULL
    )
  `);

  db.exec(`CREATE INDEX IF NOT EXISTS idx_pages_hash ON pages(hash)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_pages_url ON pages(url)`);
  db.exec(`CREATE INDEX IF NOT EXISTS idx_pages_status ON pages(fetch_status, active)`);

  // Track which browsers contributed each URL
  db.exec(`
    CREATE TABLE IF NOT EXISTS page_sources (
      page_id INTEGER NOT NULL REFERENCES pages(id) ON DELETE CASCADE,
      browser TEXT NOT NULL,
      source_type TEXT NOT NULL,
      visit_count INTEGER DEFAULT 0,
      last_visit_time TEXT,
      first_visit_time TEXT,
      bookmark_folder TEXT,
      PRIMARY KEY (page_id, browser, source_type)
    )
  `);

  // Browser registry
  db.exec(`
    CREATE TABLE IF NOT EXISTS browsers (
      name TEXT PRIMARY KEY,
      history_path TEXT,
      bookmarks_path TEXT,
      detected_at TEXT NOT NULL,
      last_synced_at TEXT,
      enabled INTEGER DEFAULT 1
    )
  `);

  // Indexer run tracking for resumability
  db.exec(`
    CREATE TABLE IF NOT EXISTS indexer_state (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      started_at TEXT NOT NULL,
      completed_at TEXT,
      status TEXT DEFAULT 'running',
      urls_discovered INTEGER DEFAULT 0,
      urls_fetched INTEGER DEFAULT 0,
      urls_skipped INTEGER DEFAULT 0,
      urls_failed INTEGER DEFAULT 0,
      last_processed_url TEXT
    )
  `);

  // URL exclude/include filters (user-defined patterns)
  db.exec(`
    CREATE TABLE IF NOT EXISTS url_filters (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      pattern TEXT NOT NULL UNIQUE,
      filter_type TEXT DEFAULT 'exclude'
    )
  `);

  // Cache table for LLM API calls
  db.exec(`
    CREATE TABLE IF NOT EXISTS llm_cache (
      hash TEXT PRIMARY KEY,
      result TEXT NOT NULL,
      created_at TEXT NOT NULL
    )
  `);

  // Content vectors
  const cvInfo = db.prepare(`PRAGMA table_info(content_vectors)`).all() as { name: string }[];
  const hasSeqColumn = cvInfo.some(col => col.name === 'seq');
  if (cvInfo.length > 0 && !hasSeqColumn) {
    db.exec(`DROP TABLE IF EXISTS content_vectors`);
    db.exec(`DROP TABLE IF EXISTS vectors_vec`);
  }
  db.exec(`
    CREATE TABLE IF NOT EXISTS content_vectors (
      hash TEXT NOT NULL,
      seq INTEGER NOT NULL DEFAULT 0,
      pos INTEGER NOT NULL DEFAULT 0,
      model TEXT NOT NULL,
      embedded_at TEXT NOT NULL,
      PRIMARY KEY (hash, seq)
    )
  `);

  // Store config — key-value metadata
  db.exec(`
    CREATE TABLE IF NOT EXISTS store_config (
      key TEXT PRIMARY KEY,
      value TEXT
    )
  `);

  // FTS - filepath stores URL, body stores extracted text content
  db.exec(`
    CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
      filepath, title, body,
      tokenize='porter unicode61'
    )
  `);

  // Triggers to keep FTS in sync with pages
  db.exec(`
    CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages
    WHEN new.active = 1 AND new.hash IS NOT NULL
    BEGIN
      INSERT INTO documents_fts(rowid, filepath, title, body)
      SELECT new.id, new.url, new.title,
        (SELECT doc FROM content WHERE hash = new.hash)
      WHERE new.active = 1 AND new.hash IS NOT NULL;
    END
  `);

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
      DELETE FROM documents_fts WHERE rowid = old.id;
    END
  `);

  db.exec(`
    CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages
    BEGIN
      DELETE FROM documents_fts WHERE rowid = old.id;
      INSERT INTO documents_fts(rowid, filepath, title, body)
      SELECT new.id, new.url, new.title,
        (SELECT doc FROM content WHERE hash = new.hash)
      WHERE new.active = 1 AND new.hash IS NOT NULL;
    END
  `);
}

// =============================================================================
// Page CRUD operations
// =============================================================================

/**
 * Upsert a page by URL. Returns the page id (existing or new).
 */
export function upsertPage(
  db: Database,
  url: string,
  title: string,
  hash: string | null,
  fetchStatus: string
): number {
  const now = new Date().toISOString();
  const existing = db.prepare(`SELECT id, fetch_status FROM pages WHERE url = ?`).get(url) as { id: number; fetch_status: string } | null;

  if (existing) {
    // Update title if provided and not blank
    if (title) {
      db.prepare(`UPDATE pages SET title = ?, modified_at = ? WHERE id = ?`)
        .run(title, now, existing.id);
    }
    // Only reset to pending if currently not fetched
    if (fetchStatus === "pending" && existing.fetch_status !== "fetched") {
      db.prepare(`UPDATE pages SET fetch_status = 'pending', modified_at = ? WHERE id = ?`)
        .run(now, existing.id);
    }
    return existing.id;
  }

  const result = db.prepare(`
    INSERT INTO pages (url, title, hash, fetch_status, active, created_at, modified_at)
    VALUES (?, ?, ?, ?, 1, ?, ?)
  `).run(url, title, hash, fetchStatus, now, now);
  return Number(result.lastInsertRowid);
}

/**
 * Upsert a page_source entry (which browser/type contributed this URL).
 */
export function upsertPageSource(
  db: Database,
  pageId: number,
  browser: string,
  sourceType: string,
  visitCount: number,
  lastVisitTime: string | null,
  bookmarkFolder?: string
): void {
  db.prepare(`
    INSERT INTO page_sources (page_id, browser, source_type, visit_count, last_visit_time, bookmark_folder)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(page_id, browser, source_type) DO UPDATE SET
      visit_count = MAX(excluded.visit_count, visit_count),
      last_visit_time = CASE
        WHEN excluded.last_visit_time > last_visit_time THEN excluded.last_visit_time
        ELSE last_visit_time
      END,
      bookmark_folder = COALESCE(excluded.bookmark_folder, bookmark_folder)
  `).run(pageId, browser, sourceType, visitCount, lastVisitTime, bookmarkFolder ?? null);
}

/**
 * Update a page after fetching (content hash, title, status).
 */
export function updatePageFetchResult(
  db: Database,
  url: string,
  hash: string | null,
  title: string,
  fetchStatus: string,
  fetchError?: string
): void {
  const now = new Date().toISOString();
  db.prepare(`
    UPDATE pages SET
      hash = ?,
      title = CASE WHEN ? != '' THEN ? ELSE title END,
      fetch_status = ?,
      fetch_error = ?,
      fetched_at = ?,
      modified_at = ?
    WHERE url = ?
  `).run(hash, title, title, fetchStatus, fetchError ?? null, fetchStatus === "fetched" ? now : null, now, url);
}

/**
 * Get pending pages (not yet fetched or previously failed).
 * Ordered by most recently visited first.
 */
export function getPendingPages(
  db: Database,
  limit: number = 1000
): { id: number; url: string; title: string }[] {
  return db.prepare(`
    SELECT p.id, p.url, p.title
    FROM pages p
    LEFT JOIN (
      SELECT page_id, MAX(last_visit_time) as last_visit
      FROM page_sources
      GROUP BY page_id
    ) ps ON ps.page_id = p.id
    WHERE p.active = 1 AND p.fetch_status IN ('pending', 'failed')
    ORDER BY ps.last_visit DESC NULLS LAST
    LIMIT ?
  `).all(limit) as { id: number; url: string; title: string }[];
}

/**
 * Get a page by URL.
 */
export function getPageByUrl(
  db: Database,
  url: string
): { id: number; hash: string | null; title: string; fetch_status: string } | null {
  return db.prepare(`
    SELECT id, hash, title, fetch_status FROM pages WHERE url = ?
  `).get(url) as { id: number; hash: string | null; title: string; fetch_status: string } | null;
}

// =============================================================================
// Browser registry operations
// =============================================================================

/**
 * Register or update a browser in the registry.
 */
export function upsertBrowser(
  db: Database,
  name: string,
  historyPath: string | null,
  bookmarksPath: string | null
): void {
  const now = new Date().toISOString();
  db.prepare(`
    INSERT INTO browsers (name, history_path, bookmarks_path, detected_at)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(name) DO UPDATE SET
      history_path = excluded.history_path,
      bookmarks_path = excluded.bookmarks_path
  `).run(name, historyPath, bookmarksPath, now);
}

/**
 * Get all registered browsers.
 */
export function getBrowsers(db: Database): {
  name: string; history_path: string | null; bookmarks_path: string | null;
  detected_at: string; last_synced_at: string | null; enabled: number;
}[] {
  return db.prepare(`SELECT * FROM browsers`).all() as any[];
}

/**
 * Update the last_synced_at timestamp for a browser.
 */
export function updateBrowserSyncTime(db: Database, name: string): void {
  db.prepare(`UPDATE browsers SET last_synced_at = ? WHERE name = ?`).run(new Date().toISOString(), name);
}

// =============================================================================
// URL filter operations
// =============================================================================

/**
 * Get all exclude filter patterns.
 */
export function getExcludeFilters(db: Database): string[] {
  const rows = db.prepare(`SELECT pattern FROM url_filters WHERE filter_type = 'exclude'`).all() as { pattern: string }[];
  return rows.map(r => r.pattern);
}

/**
 * Add a URL filter pattern.
 */
export function addUrlFilter(db: Database, pattern: string, filterType: string = "exclude"): void {
  db.prepare(`INSERT OR IGNORE INTO url_filters (pattern, filter_type) VALUES (?, ?)`).run(pattern, filterType);
}

/**
 * Remove a URL filter pattern. Returns true if removed.
 */
export function removeUrlFilter(db: Database, pattern: string): boolean {
  const result = db.prepare(`DELETE FROM url_filters WHERE pattern = ?`).run(pattern);
  return result.changes > 0;
}

/**
 * List all URL filters.
 */
export function listUrlFilters(db: Database): { id: number; pattern: string; filter_type: string }[] {
  return db.prepare(`SELECT id, pattern, filter_type FROM url_filters ORDER BY id`).all() as any[];
}

// =============================================================================
// Indexer state operations
// =============================================================================

export type IndexerRunState = {
  id?: number;
  started_at?: string;
  completed_at?: string | null;
  status?: string;
  urls_discovered?: number;
  urls_fetched?: number;
  urls_skipped?: number;
  urls_failed?: number;
  last_processed_url?: string | null;
};

/**
 * Create a new indexer run and return its id.
 */
export function createIndexerRun(db: Database): number {
  const result = db.prepare(`INSERT INTO indexer_state (started_at, status) VALUES (?, 'running')`).run(new Date().toISOString());
  return Number(result.lastInsertRowid);
}

/**
 * Update an existing indexer run's state.
 */
export function updateIndexerRun(db: Database, id: number, updates: Partial<IndexerRunState>): void {
  const fields = Object.entries(updates)
    .filter(([k]) => k !== "id")
    .map(([k]) => `${k} = ?`).join(", ");
  const values = Object.entries(updates)
    .filter(([k]) => k !== "id")
    .map(([, v]) => v);
  if (fields) {
    db.prepare(`UPDATE indexer_state SET ${fields} WHERE id = ?`).run(...values, id);
  }
}

/**
 * Get the last indexer run.
 */
export function getLastIndexerRun(db: Database): IndexerRunState | null {
  return db.prepare(`SELECT * FROM indexer_state ORDER BY id DESC LIMIT 1`).get() as IndexerRunState | null;
}

export function isSqliteVecAvailable(): boolean {
  return _sqliteVecAvailable === true;
}

function ensureVecTableInternal(db: Database, dimensions: number): void {
  if (!_sqliteVecAvailable) {
    throw createSqliteVecUnavailableError(
      _sqliteVecUnavailableReason ?? "vector operations require a SQLite build with extension loading support"
    );
  }
  const tableInfo = db.prepare(`SELECT sql FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get() as { sql: string } | null;
  if (tableInfo) {
    const match = tableInfo.sql.match(/float\[(\d+)\]/);
    const hasHashSeq = tableInfo.sql.includes('hash_seq');
    const hasCosine = tableInfo.sql.includes('distance_metric=cosine');
    const existingDims = match?.[1] ? parseInt(match[1], 10) : null;
    if (existingDims === dimensions && hasHashSeq && hasCosine) return;
    if (existingDims !== null && existingDims !== dimensions) {
      throw new Error(
        `Embedding dimension mismatch: existing vectors are ${existingDims}d but the current model produces ${dimensions}d. ` +
        `Run 'qmd embed -f' to re-embed with the new model.`
      );
    }
    db.exec("DROP TABLE IF EXISTS vectors_vec");
  }
  db.exec(`CREATE VIRTUAL TABLE vectors_vec USING vec0(hash_seq TEXT PRIMARY KEY, embedding float[${dimensions}] distance_metric=cosine)`);
}

// =============================================================================
// Store Factory
// =============================================================================

export type Store = {
  db: Database;
  dbPath: string;
  /** Optional LlamaCpp instance for this store (overrides the global singleton) */
  llm?: LlamaCpp;
  close: () => void;
  ensureVecTable: (dimensions: number) => void;

  // Index health
  getHashesNeedingEmbedding: () => number;
  getIndexHealth: () => IndexHealthInfo;
  getStatus: () => IndexStatus;

  // Caching
  getCacheKey: typeof getCacheKey;
  getCachedResult: (cacheKey: string) => string | null;
  setCachedResult: (cacheKey: string, result: string) => void;
  clearCache: () => void;

  // Cleanup and maintenance
  deleteLLMCache: () => number;
  deleteInactivePages: () => number;
  cleanupOrphanedContent: () => number;
  cleanupOrphanedVectors: () => number;
  vacuumDatabase: () => void;

  // No-op context stub (pages have no context system)
  getContextForFile: (url: string) => string | null;

  // Search
  searchFTS: (query: string, limit?: number, browser?: string) => SearchResult[];
  searchVec: (query: string, model: string, limit?: number, browser?: string, session?: ILLMSession, precomputedEmbedding?: number[]) => Promise<SearchResult[]>;

  // Query expansion & reranking
  expandQuery: (query: string, model?: string, intent?: string) => Promise<ExpandedQuery[]>;
  rerank: (query: string, documents: { file: string; text: string }[], model?: string, intent?: string) => Promise<{ file: string; score: number }[]>;

  // Page retrieval
  findPage: (urlOrDocid: string, options?: { includeBody?: boolean }) => PageResult | PageNotFound;
  getPageBody: (page: PageResult | { url: string }, fromLine?: number, maxLines?: number) => string | null;

  // Page operations
  upsertPage: (url: string, title: string, hash: string | null, fetchStatus: string) => number;
  upsertPageSource: (pageId: number, browser: string, sourceType: string, visitCount: number, lastVisitTime: string | null, bookmarkFolder?: string) => void;
  updatePageFetchResult: (url: string, hash: string | null, title: string, fetchStatus: string, fetchError?: string) => void;
  getPendingPages: (limit?: number) => { id: number; url: string; title: string }[];
  getPageByUrl: (url: string) => { id: number; hash: string | null; title: string; fetch_status: string } | null;

  // Browser operations
  upsertBrowser: (name: string, historyPath: string | null, bookmarksPath: string | null) => void;
  getBrowsers: () => { name: string; history_path: string | null; bookmarks_path: string | null; detected_at: string; last_synced_at: string | null; enabled: number }[];
  updateBrowserSyncTime: (name: string) => void;

  // URL filter operations
  getExcludeFilters: () => string[];
  addUrlFilter: (pattern: string, filterType?: string) => void;
  removeUrlFilter: (pattern: string) => boolean;
  listUrlFilters: () => { id: number; pattern: string; filter_type: string }[];

  // Indexer state
  createIndexerRun: () => number;
  updateIndexerRun: (id: number, updates: Partial<IndexerRunState>) => void;
  getLastIndexerRun: () => IndexerRunState | null;

  // Docid lookup
  findPageByDocid: (docid: string) => { url: string; hash: string } | null;

  // Content operations
  insertContent: (hash: string, content: string, createdAt: string) => void;

  // Vector/embedding operations
  getHashesForEmbedding: () => { hash: string; body: string; path: string }[];
  clearAllEmbeddings: () => void;
  insertEmbedding: (hash: string, seq: number, pos: number, embedding: Float32Array, model: string, embeddedAt: string) => void;
};

// =============================================================================
// Embed — pure-logic functions for SDK and CLI
// =============================================================================

export type EmbedProgress = {
  chunksEmbedded: number;
  totalChunks: number;
  bytesProcessed: number;
  totalBytes: number;
  errors: number;
};

export type EmbedResult = {
  docsProcessed: number;
  chunksEmbedded: number;
  errors: number;
  durationMs: number;
};

export type EmbedOptions = {
  force?: boolean;
  model?: string;
  maxDocsPerBatch?: number;
  maxBatchBytes?: number;
  chunkStrategy?: ChunkStrategy;
  onProgress?: (info: EmbedProgress) => void;
};

type PendingEmbeddingDoc = {
  hash: string;
  path: string;
  bytes: number;
};

type EmbeddingDoc = PendingEmbeddingDoc & {
  body: string;
};

type ChunkItem = {
  hash: string;
  title: string;
  text: string;
  seq: number;
  pos: number;
  tokens: number;
  bytes: number;
};

function validatePositiveIntegerOption(name: string, value: number | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  if (!Number.isInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

function resolveEmbedOptions(options?: EmbedOptions): Required<Pick<EmbedOptions, "maxDocsPerBatch" | "maxBatchBytes">> {
  return {
    maxDocsPerBatch: validatePositiveIntegerOption("maxDocsPerBatch", options?.maxDocsPerBatch, DEFAULT_EMBED_MAX_DOCS_PER_BATCH),
    maxBatchBytes: validatePositiveIntegerOption("maxBatchBytes", options?.maxBatchBytes, DEFAULT_EMBED_MAX_BATCH_BYTES),
  };
}

function getPendingEmbeddingDocs(db: Database): PendingEmbeddingDoc[] {
  return db.prepare(`
    SELECT p.hash, MIN(p.url) as path, length(CAST(c.doc AS BLOB)) as bytes
    FROM pages p
    JOIN content c ON p.hash = c.hash
    LEFT JOIN content_vectors v ON p.hash = v.hash AND v.seq = 0
    WHERE p.active = 1 AND p.hash IS NOT NULL AND p.fetch_status = 'fetched' AND v.hash IS NULL
    GROUP BY p.hash
    ORDER BY MIN(p.url)
  `).all() as PendingEmbeddingDoc[];
}

function buildEmbeddingBatches(
  docs: PendingEmbeddingDoc[],
  maxDocsPerBatch: number,
  maxBatchBytes: number,
): PendingEmbeddingDoc[][] {
  const batches: PendingEmbeddingDoc[][] = [];
  let currentBatch: PendingEmbeddingDoc[] = [];
  let currentBytes = 0;

  for (const doc of docs) {
    const docBytes = Math.max(0, doc.bytes);
    const wouldExceedDocs = currentBatch.length >= maxDocsPerBatch;
    const wouldExceedBytes = currentBatch.length > 0 && (currentBytes + docBytes) > maxBatchBytes;

    if (wouldExceedDocs || wouldExceedBytes) {
      batches.push(currentBatch);
      currentBatch = [];
      currentBytes = 0;
    }

    currentBatch.push(doc);
    currentBytes += docBytes;
  }

  if (currentBatch.length > 0) {
    batches.push(currentBatch);
  }

  return batches;
}

function getEmbeddingDocsForBatch(db: Database, batch: PendingEmbeddingDoc[]): EmbeddingDoc[] {
  if (batch.length === 0) return [];

  const placeholders = batch.map(() => "?").join(",");
  const rows = db.prepare(`
    SELECT hash, doc as body
    FROM content
    WHERE hash IN (${placeholders})
  `).all(...batch.map(doc => doc.hash)) as { hash: string; body: string }[];
  const bodyByHash = new Map(rows.map(row => [row.hash, row.body]));

  return batch.map((doc) => ({
    ...doc,
    body: bodyByHash.get(doc.hash) ?? "",
  }));
}

/**
 * Generate vector embeddings for documents that need them.
 * Pure function — no console output, no db lifecycle management.
 * Uses the store's LlamaCpp instance if set, otherwise the global singleton.
 */
export async function generateEmbeddings(
  store: Store,
  options?: EmbedOptions
): Promise<EmbedResult> {
  const db = store.db;
  const model = options?.model ?? DEFAULT_EMBED_MODEL;
  const now = new Date().toISOString();
  const { maxDocsPerBatch, maxBatchBytes } = resolveEmbedOptions(options);
  const encoder = new TextEncoder();

  if (options?.force) {
    clearAllEmbeddings(db);
  }

  const docsToEmbed = getPendingEmbeddingDocs(db);

  if (docsToEmbed.length === 0) {
    return { docsProcessed: 0, chunksEmbedded: 0, errors: 0, durationMs: 0 };
  }
  const totalBytes = docsToEmbed.reduce((sum, doc) => sum + Math.max(0, doc.bytes), 0);
  const totalDocs = docsToEmbed.length;
  const startTime = Date.now();

  // Use store's LlamaCpp or global singleton, wrapped in a session
  const llm = getLlm(store);
  const embedModelUri = llm.embedModelName;

  // Create a session manager for this llm instance
  const result = await withLLMSessionForLlm(llm, async (session) => {
    let chunksEmbedded = 0;
    let errors = 0;
    let bytesProcessed = 0;
    let totalChunks = 0;
    let vectorTableInitialized = false;
    const BATCH_SIZE = 32;
    const batches = buildEmbeddingBatches(docsToEmbed, maxDocsPerBatch, maxBatchBytes);

    for (const batchMeta of batches) {
      // Abort early if session has been invalidated
      if (!session.isValid) {
        console.warn(`⚠ Session expired — skipping remaining document batches`);
        break;
      }

      const batchDocs = getEmbeddingDocsForBatch(db, batchMeta);
      const batchChunks: ChunkItem[] = [];
      const batchBytes = batchMeta.reduce((sum, doc) => sum + Math.max(0, doc.bytes), 0);

      for (const doc of batchDocs) {
        if (!doc.body.trim()) continue;

        // For pages, use the title from the pages table (derived from <title>),
        // falling back to the URL if not set.
        const titleRow = db.prepare(`SELECT title FROM pages WHERE hash = ? AND active = 1 LIMIT 1`).get(doc.hash) as { title: string } | null;
        const title = titleRow?.title || doc.path;
        const chunks = await chunkDocumentByTokens(
          doc.body,
          undefined, undefined, undefined,
          doc.path,
          options?.chunkStrategy,
          session.signal,
        );

        for (let seq = 0; seq < chunks.length; seq++) {
          batchChunks.push({
            hash: doc.hash,
            title,
            text: chunks[seq]!.text,
            seq,
            pos: chunks[seq]!.pos,
            tokens: chunks[seq]!.tokens,
            bytes: encoder.encode(chunks[seq]!.text).length,
          });
        }
      }

      totalChunks += batchChunks.length;

      if (batchChunks.length === 0) {
        bytesProcessed += batchBytes;
        options?.onProgress?.({ chunksEmbedded, totalChunks, bytesProcessed, totalBytes, errors });
        continue;
      }

      if (!vectorTableInitialized) {
        const firstChunk = batchChunks[0]!;
        const firstText = formatDocForEmbedding(firstChunk.text, firstChunk.title, embedModelUri);
        const firstResult = await session.embed(firstText, { model });
        if (!firstResult) {
          throw new Error("Failed to get embedding dimensions from first chunk");
        }
        store.ensureVecTable(firstResult.embedding.length);
        vectorTableInitialized = true;
      }

      const totalBatchChunkBytes = batchChunks.reduce((sum, chunk) => sum + chunk.bytes, 0);
      let batchChunkBytesProcessed = 0;

      for (let batchStart = 0; batchStart < batchChunks.length; batchStart += BATCH_SIZE) {
        // Abort early if session has been invalidated (e.g. max duration exceeded)
        if (!session.isValid) {
          const remaining = batchChunks.length - batchStart;
          errors += remaining;
          console.warn(`⚠ Session expired — skipping ${remaining} remaining chunks`);
          break;
        }

        // Abort early if error rate is too high (>80% of processed chunks failed)
        const processed = chunksEmbedded + errors;
        if (processed >= BATCH_SIZE && errors > processed * 0.8) {
          const remaining = batchChunks.length - batchStart;
          errors += remaining;
          console.warn(`⚠ Error rate too high (${errors}/${processed}) — aborting embedding`);
          break;
        }

        const batchEnd = Math.min(batchStart + BATCH_SIZE, batchChunks.length);
        const chunkBatch = batchChunks.slice(batchStart, batchEnd);
        const texts = chunkBatch.map(chunk => formatDocForEmbedding(chunk.text, chunk.title, embedModelUri));

        try {
          const embeddings = await session.embedBatch(texts, { model });
          for (let i = 0; i < chunkBatch.length; i++) {
            const chunk = chunkBatch[i]!;
            const embedding = embeddings[i];
            if (embedding) {
              insertEmbedding(db, chunk.hash, chunk.seq, chunk.pos, new Float32Array(embedding.embedding), model, now);
              chunksEmbedded++;
            } else {
              errors++;
            }
            batchChunkBytesProcessed += chunk.bytes;
          }
        } catch {
          // Batch failed — try individual embeddings as fallback
          // But skip if session is already invalid (avoids N doomed retries)
          if (!session.isValid) {
            errors += chunkBatch.length;
            batchChunkBytesProcessed += chunkBatch.reduce((sum, c) => sum + c.bytes, 0);
          } else {
            for (const chunk of chunkBatch) {
              try {
                const text = formatDocForEmbedding(chunk.text, chunk.title, embedModelUri);
                const result = await session.embed(text, { model });
                if (result) {
                  insertEmbedding(db, chunk.hash, chunk.seq, chunk.pos, new Float32Array(result.embedding), model, now);
                  chunksEmbedded++;
                } else {
                  errors++;
                }
              } catch {
                errors++;
              }
              batchChunkBytesProcessed += chunk.bytes;
            }
          }
        }

        const proportionalBytes = totalBatchChunkBytes === 0
          ? batchBytes
          : Math.min(batchBytes, Math.round((batchChunkBytesProcessed / totalBatchChunkBytes) * batchBytes));
        options?.onProgress?.({
          chunksEmbedded,
          totalChunks,
          bytesProcessed: bytesProcessed + proportionalBytes,
          totalBytes,
          errors,
        });
      }

      bytesProcessed += batchBytes;
      options?.onProgress?.({ chunksEmbedded, totalChunks, bytesProcessed, totalBytes, errors });
    }

    return { chunksEmbedded, errors };
  }, { maxDuration: 30 * 60 * 1000, name: 'generateEmbeddings' });

  return {
    docsProcessed: totalDocs,
    chunksEmbedded: result.chunksEmbedded,
    errors: result.errors,
    durationMs: Date.now() - startTime,
  };
}

/**
 * Create a new store instance with the given database path.
 * If no path is provided, uses the default path (~/.cache/qmd/index.sqlite).
 *
 * @param dbPath - Path to the SQLite database file
 * @returns Store instance with all methods bound to the database
 */
export function createStore(dbPath?: string): Store {
  const resolvedPath = dbPath || getDefaultDbPath();
  const db = openDatabase(resolvedPath);
  initializeDatabase(db);

  const store: Store = {
    db,
    dbPath: resolvedPath,
    close: () => db.close(),
    ensureVecTable: (dimensions: number) => ensureVecTableInternal(db, dimensions),

    // Index health
    getHashesNeedingEmbedding: () => getHashesNeedingEmbedding(db),
    getIndexHealth: () => getIndexHealth(db),
    getStatus: () => getStatus(db),

    // Caching
    getCacheKey,
    getCachedResult: (cacheKey: string) => getCachedResult(db, cacheKey),
    setCachedResult: (cacheKey: string, result: string) => setCachedResult(db, cacheKey, result),
    clearCache: () => clearCache(db),

    // Cleanup and maintenance
    deleteLLMCache: () => deleteLLMCache(db),
    deleteInactivePages: () => deleteInactivePages(db),
    cleanupOrphanedContent: () => cleanupOrphanedContent(db),
    cleanupOrphanedVectors: () => cleanupOrphanedVectors(db),
    vacuumDatabase: () => vacuumDatabase(db),

    // No-op context stub (pages have no context system)
    getContextForFile: (_url: string) => null,

    // Search
    searchFTS: (query: string, limit?: number, browser?: string) => searchFTS(db, query, limit, browser),
    searchVec: (query: string, model: string, limit?: number, browser?: string, session?: ILLMSession, precomputedEmbedding?: number[]) => searchVec(db, query, model, limit, browser, session, precomputedEmbedding),

    // Query expansion & reranking
    expandQuery: (query: string, model?: string, intent?: string) => expandQuery(query, model, db, intent, store.llm),
    rerank: (query: string, documents: { file: string; text: string }[], model?: string, intent?: string) => rerank(query, documents, model, db, intent, store.llm),

    // Page retrieval
    findPage: (urlOrDocid: string, options?: { includeBody?: boolean }) => findPage(db, urlOrDocid, options),
    getPageBody: (page: PageResult | { url: string }, fromLine?: number, maxLines?: number) => getPageBody(db, page, fromLine, maxLines),

    // Page operations
    upsertPage: (url: string, title: string, hash: string | null, fetchStatus: string) => upsertPage(db, url, title, hash, fetchStatus),
    upsertPageSource: (pageId: number, browser: string, sourceType: string, visitCount: number, lastVisitTime: string | null, bookmarkFolder?: string) => upsertPageSource(db, pageId, browser, sourceType, visitCount, lastVisitTime, bookmarkFolder),
    updatePageFetchResult: (url: string, hash: string | null, title: string, fetchStatus: string, fetchError?: string) => updatePageFetchResult(db, url, hash, title, fetchStatus, fetchError),
    getPendingPages: (limit?: number) => getPendingPages(db, limit),
    getPageByUrl: (url: string) => getPageByUrl(db, url),

    // Browser operations
    upsertBrowser: (name: string, historyPath: string | null, bookmarksPath: string | null) => upsertBrowser(db, name, historyPath, bookmarksPath),
    getBrowsers: () => getBrowsers(db),
    updateBrowserSyncTime: (name: string) => updateBrowserSyncTime(db, name),

    // URL filter operations
    getExcludeFilters: () => getExcludeFilters(db),
    addUrlFilter: (pattern: string, filterType?: string) => addUrlFilter(db, pattern, filterType),
    removeUrlFilter: (pattern: string) => removeUrlFilter(db, pattern),
    listUrlFilters: () => listUrlFilters(db),

    // Indexer state
    createIndexerRun: () => createIndexerRun(db),
    updateIndexerRun: (id: number, updates: Partial<IndexerRunState>) => updateIndexerRun(db, id, updates),
    getLastIndexerRun: () => getLastIndexerRun(db),

    // Docid lookup
    findPageByDocid: (docid: string) => findPageByDocid(db, docid),

    // Content operations
    insertContent: (hash: string, content: string, createdAt: string) => insertContent(db, hash, content, createdAt),

    // Vector/embedding operations
    getHashesForEmbedding: () => getHashesForEmbedding(db),
    clearAllEmbeddings: () => clearAllEmbeddings(db),
    insertEmbedding: (hash: string, seq: number, pos: number, embedding: Float32Array, model: string, embeddedAt: string) => insertEmbedding(db, hash, seq, pos, embedding, model, embeddedAt),
  };

  return store;
}

// =============================================================================
// Core Page Type
// =============================================================================

/**
 * Unified page result type with all metadata.
 * Body is optional - use getPageBody() to load it separately if needed.
 */
export type PageResult = {
  url: string;                // Canonical URL
  title: string;              // Page title (from <title> or og:title)
  hash: string;               // Content hash for caching/change detection
  docid: string;              // Short docid (first 6 chars of hash) for quick reference
  browsers: string[];         // Browsers this URL came from (e.g., ["chrome", "arc"])
  visitCount: number;         // Total visit count across browsers
  fetchStatus: string;        // "pending" | "fetched" | "failed" | "skipped"
  fetchedAt: string | null;   // When the page was last fetched
  modifiedAt: string;         // Last modification timestamp
  bodyLength: number;         // Body length in bytes (useful before loading)
  body?: string;              // Extracted text (optional, load with getPageBody)
};

/**
 * Extract short docid from a full hash (first 6 characters).
 */
export function getDocid(hash: string): string {
  return hash.slice(0, 6);
}

/**
 * Search result extends PageResult with score and source info
 */
export type SearchResult = PageResult & {
  score: number;              // Relevance score (0-1)
  source: "fts" | "vec";      // Search source (full-text or vector)
  chunkPos?: number;          // Character position of matching chunk (for vector search)
};

/**
 * Ranked result for RRF fusion (simplified, used internally)
 */
export type RankedResult = {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  score: number;
};

export type RRFContributionTrace = {
  listIndex: number;
  source: "fts" | "vec";
  queryType: "original" | "lex" | "vec" | "hyde";
  query: string;
  rank: number;            // 1-indexed rank within list
  weight: number;
  backendScore: number;    // Backend-normalized score before fusion
  rrfContribution: number; // weight / (k + rank)
};

export type RRFScoreTrace = {
  contributions: RRFContributionTrace[];
  baseScore: number;       // Sum of reciprocal-rank contributions
  topRank: number;         // Best (lowest) rank seen across lists
  topRankBonus: number;    // +0.05 for rank 1, +0.02 for rank 2-3
  totalScore: number;      // baseScore + topRankBonus
};

export type HybridQueryExplain = {
  ftsScores: number[];
  vectorScores: number[];
  rrf: {
    rank: number;          // Rank after RRF fusion (1-indexed)
    positionScore: number; // 1 / rank used in position-aware blending
    weight: number;        // Position-aware RRF weight (0.75 / 0.60 / 0.40)
    baseScore: number;
    topRankBonus: number;
    totalScore: number;
    contributions: RRFContributionTrace[];
  };
  rerankScore: number;
  blendedScore: number;
};

/**
 * Error result when page is not found
 */
export type PageNotFound = {
  error: "not_found";
  query: string;
  similarUrls: string[];
};

/**
 * Result from multi-get operations
 */
export type MultiGetResult = {
  page: PageResult;
  skipped: false;
} | {
  page: Pick<PageResult, "url">;
  skipped: true;
  skipReason: string;
};

export type BrowserInfo = {
  name: string;
  historyPath: string | null;
  bookmarksPath: string | null;
  detectedAt: string;
  lastSyncedAt: string | null;
  pages: number;
};

export type IndexStatus = {
  totalPages: number;
  fetchedPages: number;
  pendingPages: number;
  failedPages: number;
  needsEmbedding: number;
  hasVectorIndex: boolean;
  browsers: BrowserInfo[];
  lastRun: IndexerRunState | null;
};

// =============================================================================
// Index health
// =============================================================================

export function getHashesNeedingEmbedding(db: Database): number {
  const result = db.prepare(`
    SELECT COUNT(DISTINCT p.hash) as count
    FROM pages p
    LEFT JOIN content_vectors v ON p.hash = v.hash AND v.seq = 0
    WHERE p.active = 1 AND p.hash IS NOT NULL AND p.fetch_status = 'fetched' AND v.hash IS NULL
  `).get() as { count: number };
  return result.count;
}

export type IndexHealthInfo = {
  needsEmbedding: number;
  totalDocs: number;
  daysStale: number | null;
};

export function getIndexHealth(db: Database): IndexHealthInfo {
  const needsEmbedding = getHashesNeedingEmbedding(db);
  const totalDocs = (db.prepare(`SELECT COUNT(*) as count FROM pages WHERE active = 1 AND fetch_status = 'fetched'`).get() as { count: number }).count;

  const mostRecent = db.prepare(`SELECT MAX(modified_at) as latest FROM pages WHERE active = 1`).get() as { latest: string | null };
  let daysStale: number | null = null;
  if (mostRecent?.latest) {
    const lastUpdate = new Date(mostRecent.latest);
    daysStale = Math.floor((Date.now() - lastUpdate.getTime()) / (24 * 60 * 60 * 1000));
  }

  return { needsEmbedding, totalDocs, daysStale };
}

// =============================================================================
// Caching
// =============================================================================

export function getCacheKey(url: string, body: object): string {
  const hash = createHash("sha256");
  hash.update(url);
  hash.update(JSON.stringify(body));
  return hash.digest("hex");
}

export function getCachedResult(db: Database, cacheKey: string): string | null {
  const row = db.prepare(`SELECT result FROM llm_cache WHERE hash = ?`).get(cacheKey) as { result: string } | null;
  return row?.result || null;
}

export function setCachedResult(db: Database, cacheKey: string, result: string): void {
  const now = new Date().toISOString();
  db.prepare(`INSERT OR REPLACE INTO llm_cache (hash, result, created_at) VALUES (?, ?, ?)`).run(cacheKey, result, now);
  if (Math.random() < 0.01) {
    db.exec(`DELETE FROM llm_cache WHERE hash NOT IN (SELECT hash FROM llm_cache ORDER BY created_at DESC LIMIT 1000)`);
  }
}

export function clearCache(db: Database): void {
  db.exec(`DELETE FROM llm_cache`);
}

// =============================================================================
// Cleanup and maintenance operations
// =============================================================================

/**
 * Delete cached LLM API responses.
 * Returns the number of cached responses deleted.
 */
export function deleteLLMCache(db: Database): number {
  const result = db.prepare(`DELETE FROM llm_cache`).run();
  return result.changes;
}

/**
 * Remove inactive page records (active = 0).
 * Returns the number of inactive pages deleted.
 */
export function deleteInactivePages(db: Database): number {
  const result = db.prepare(`DELETE FROM pages WHERE active = 0`).run();
  return result.changes;
}

/**
 * Remove orphaned content hashes that are not referenced by any active page.
 * Returns the number of orphaned content hashes deleted.
 */
export function cleanupOrphanedContent(db: Database): number {
  const result = db.prepare(`
    DELETE FROM content
    WHERE hash NOT IN (SELECT DISTINCT hash FROM pages WHERE active = 1 AND hash IS NOT NULL)
  `).run();
  return result.changes;
}

/**
 * Remove orphaned vector embeddings that are not referenced by any active page.
 * Returns the number of orphaned embedding chunks deleted.
 */
export function cleanupOrphanedVectors(db: Database): number {
  // sqlite-vec may not be loaded (e.g. Bun's bun:sqlite lacks loadExtension).
  // The vectors_vec virtual table can appear in sqlite_master from a prior
  // session, but querying it without the vec0 module loaded will crash (#380).
  if (!isSqliteVecAvailable()) {
    return 0;
  }

  // The schema entry can exist even when sqlite-vec itself is unavailable
  // (for example when reopening a DB without vec0 loaded). In that case,
  // touching the virtual table throws "no such module: vec0" and cleanup
  // should degrade gracefully like the rest of the vector features.
  try {
    db.prepare(`SELECT 1 FROM vectors_vec LIMIT 0`).get();
  } catch {
    return 0;
  }

  // Count orphaned vectors first
  const countResult = db.prepare(`
    SELECT COUNT(*) as c FROM content_vectors cv
    WHERE NOT EXISTS (
      SELECT 1 FROM pages p WHERE p.hash = cv.hash AND p.active = 1
    )
  `).get() as { c: number };

  if (countResult.c === 0) {
    return 0;
  }

  // Delete from vectors_vec first
  db.exec(`
    DELETE FROM vectors_vec WHERE hash_seq IN (
      SELECT cv.hash || '_' || cv.seq FROM content_vectors cv
      WHERE NOT EXISTS (
        SELECT 1 FROM pages p WHERE p.hash = cv.hash AND p.active = 1
      )
    )
  `);

  // Delete from content_vectors
  db.exec(`
    DELETE FROM content_vectors WHERE hash NOT IN (
      SELECT hash FROM pages WHERE active = 1 AND hash IS NOT NULL
    )
  `);

  return countResult.c;
}

/**
 * Run VACUUM to reclaim unused space in the database.
 * This operation rebuilds the database file to eliminate fragmentation.
 */
export function vacuumDatabase(db: Database): void {
  db.exec(`VACUUM`);
}

// =============================================================================
// Document helpers
// =============================================================================

export function hashContent(content: string): string {
  const hash = createHash("sha256");
  hash.update(content);
  return hash.digest("hex");
}

const titleExtractors: Record<string, (content: string) => string | null> = {
  '.md': (content) => {
    const match = content.match(/^##?\s+(.+)$/m);
    if (match) {
      const title = (match[1] ?? "").trim();
      if (title === "📝 Notes" || title === "Notes") {
        const nextMatch = content.match(/^##\s+(.+)$/m);
        if (nextMatch?.[1]) return nextMatch[1].trim();
      }
      return title;
    }
    return null;
  },
  '.org': (content) => {
    const titleProp = content.match(/^#\+TITLE:\s*(.+)$/im);
    if (titleProp?.[1]) return titleProp[1].trim();
    const heading = content.match(/^\*+\s+(.+)$/m);
    if (heading?.[1]) return heading[1].trim();
    return null;
  },
};

export function extractTitle(content: string, filename: string): string {
  const ext = filename.slice(filename.lastIndexOf('.')).toLowerCase();
  const extractor = titleExtractors[ext];
  if (extractor) {
    const title = extractor(content);
    if (title) return title;
  }
  return filename.replace(/\.[^.]+$/, "").split("/").pop() || filename;
}

// =============================================================================
// Document indexing operations
// =============================================================================

/**
 * Insert content into the content table (content-addressable storage).
 * Uses INSERT OR IGNORE so duplicate hashes are skipped.
 */
export function insertContent(db: Database, hash: string, content: string, createdAt: string): void {
  db.prepare(`INSERT OR IGNORE INTO content (hash, doc, created_at) VALUES (?, ?, ?)`)
    .run(hash, content, createdAt);
}

export { formatQueryForEmbedding, formatDocForEmbedding };

/**
 * Chunk a document using regex-only break point detection.
 * This is the sync, backward-compatible API used by tests and legacy callers.
 */
export function chunkDocument(
  content: string,
  maxChars: number = CHUNK_SIZE_CHARS,
  overlapChars: number = CHUNK_OVERLAP_CHARS,
  windowChars: number = CHUNK_WINDOW_CHARS
): { text: string; pos: number }[] {
  const breakPoints = scanBreakPoints(content);
  const codeFences = findCodeFences(content);
  return chunkDocumentWithBreakPoints(content, breakPoints, codeFences, maxChars, overlapChars, windowChars);
}

/**
 * Async AST-aware chunking. Detects language from filepath, computes AST
 * break points for supported code files, merges with regex break points,
 * and delegates to the shared chunk algorithm.
 *
 * Falls back to regex-only when strategy is "regex", filepath is absent,
 * or language is unsupported.
 */
export async function chunkDocumentAsync(
  content: string,
  maxChars: number = CHUNK_SIZE_CHARS,
  overlapChars: number = CHUNK_OVERLAP_CHARS,
  windowChars: number = CHUNK_WINDOW_CHARS,
  filepath?: string,
  chunkStrategy: ChunkStrategy = "regex",
): Promise<{ text: string; pos: number }[]> {
  const regexPoints = scanBreakPoints(content);
  const codeFences = findCodeFences(content);

  // AST-aware chunking removed in planet-capture (code files aren't indexed).
  const breakPoints = regexPoints;
  void chunkStrategy;
  void filepath;

  return chunkDocumentWithBreakPoints(content, breakPoints, codeFences, maxChars, overlapChars, windowChars);
}

/**
 * Chunk a document by actual token count using the LLM tokenizer.
 * More accurate than character-based chunking but requires async.
 *
 * When filepath and chunkStrategy are provided, uses AST-aware break points
 * for supported code files.
 */
export async function chunkDocumentByTokens(
  content: string,
  maxTokens: number = CHUNK_SIZE_TOKENS,
  overlapTokens: number = CHUNK_OVERLAP_TOKENS,
  windowTokens: number = CHUNK_WINDOW_TOKENS,
  filepath?: string,
  chunkStrategy: ChunkStrategy = "regex",
  signal?: AbortSignal
): Promise<{ text: string; pos: number; tokens: number }[]> {
  const llm = getDefaultLlamaCpp();

  // Use moderate chars/token estimate (prose ~4, code ~2, mixed ~3)
  // If chunks exceed limit, they'll be re-split with actual ratio
  const avgCharsPerToken = 3;
  const maxChars = maxTokens * avgCharsPerToken;
  const overlapChars = overlapTokens * avgCharsPerToken;
  const windowChars = windowTokens * avgCharsPerToken;

  // Chunk in character space with conservative estimate
  // Use AST-aware chunking for the first pass when filepath/strategy provided
  let charChunks = await chunkDocumentAsync(content, maxChars, overlapChars, windowChars, filepath, chunkStrategy);

  // Tokenize and split any chunks that still exceed limit
  const results: { text: string; pos: number; tokens: number }[] = [];
  const clampOverlapChars = (value: number, maxChars: number): number => {
    if (maxChars <= 1) return 0;
    return Math.max(0, Math.min(maxChars - 1, Math.floor(value)));
  };

  const pushChunkWithinTokenLimit = async (text: string, pos: number): Promise<void> => {
    if (signal?.aborted) return;

    const tokens = await llm.tokenize(text);
    if (tokens.length <= maxTokens || text.length <= 1) {
      results.push({ text, pos, tokens: tokens.length });
      return;
    }

    const actualCharsPerToken = text.length / tokens.length;
    let safeMaxChars = Math.floor(maxTokens * actualCharsPerToken * 0.95);
    if (!Number.isFinite(safeMaxChars) || safeMaxChars < 1) {
      safeMaxChars = Math.floor(text.length / 2);
    }
    safeMaxChars = Math.max(1, Math.min(text.length - 1, safeMaxChars));

    let nextOverlapChars = clampOverlapChars(
      overlapChars * actualCharsPerToken / 2,
      safeMaxChars,
    );
    let nextWindowChars = Math.max(0, Math.floor(windowChars * actualCharsPerToken / 2));
    let subChunks = chunkDocument(text, safeMaxChars, nextOverlapChars, nextWindowChars);

    // Pathological single-line blobs can produce no meaningful breakpoint progress.
    // Fall back to a simple half split so every recursion step strictly shrinks.
    if (
      subChunks.length <= 1
      || subChunks[0]?.text.length === text.length
    ) {
      safeMaxChars = Math.max(1, Math.floor(text.length / 2));
      nextOverlapChars = 0;
      nextWindowChars = 0;
      subChunks = chunkDocument(text, safeMaxChars, nextOverlapChars, nextWindowChars);
    }

    if (
      subChunks.length <= 1
      || subChunks[0]?.text.length === text.length
    ) {
      const fallbackTokens = tokens.slice(0, Math.max(1, maxTokens));
      const truncatedText = await llm.detokenize(fallbackTokens);
      results.push({
        text: truncatedText,
        pos,
        tokens: fallbackTokens.length,
      });
      return;
    }

    for (const subChunk of subChunks) {
      await pushChunkWithinTokenLimit(text.slice(subChunk.pos, subChunk.pos + subChunk.text.length), pos + subChunk.pos);
    }
  };

  for (const chunk of charChunks) {
    await pushChunkWithinTokenLimit(chunk.text, chunk.pos);
  }

  return results;
}

// =============================================================================
// Fuzzy matching
// =============================================================================

function levenshtein(a: string, b: string): number {
  const m = a.length, n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  const dp: number[][] = Array.from({ length: m + 1 }, () => Array(n + 1).fill(0));
  for (let i = 0; i <= m; i++) dp[i]![0] = i;
  for (let j = 0; j <= n; j++) dp[0]![j] = j;
  for (let i = 1; i <= m; i++) {
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      dp[i]![j] = Math.min(
        dp[i - 1]![j]! + 1,
        dp[i]![j - 1]! + 1,
        dp[i - 1]![j - 1]! + cost
      );
    }
  }
  return dp[m]![n]!;
}

/**
 * Normalize a docid input by stripping surrounding quotes and leading #.
 * Handles: "#abc123", 'abc123', "abc123", #abc123, abc123
 * Returns the bare hex string.
 */
export function normalizeDocid(docid: string): string {
  let normalized = docid.trim();

  // Strip surrounding quotes (single or double)
  if ((normalized.startsWith('"') && normalized.endsWith('"')) ||
      (normalized.startsWith("'") && normalized.endsWith("'"))) {
    normalized = normalized.slice(1, -1);
  }

  // Strip leading # if present
  if (normalized.startsWith('#')) {
    normalized = normalized.slice(1);
  }

  return normalized;
}

/**
 * Check if a string looks like a docid reference.
 * Accepts: #abc123, abc123, "#abc123", "abc123", '#abc123', 'abc123'
 * Returns true if the normalized form is a valid hex string of 6+ chars.
 */
export function isDocid(input: string): boolean {
  const normalized = normalizeDocid(input);
  // Must be at least 6 hex characters
  return normalized.length >= 6 && /^[a-f0-9]+$/i.test(normalized);
}


// =============================================================================
// FTS Search
// =============================================================================

export function sanitizeFTS5Term(term: string): string {
  return term.replace(/[^\p{L}\p{N}'_]/gu, '').toLowerCase();
}

/**
 * Check if a token is a hyphenated compound word (e.g., multi-agent, DEC-0054, gpt-4).
 * Returns true if the token contains internal hyphens between word/digit characters.
 */
function isHyphenatedToken(token: string): boolean {
  return /^[\p{L}\p{N}][\p{L}\p{N}'-]*-[\p{L}\p{N}][\p{L}\p{N}'-]*$/u.test(token);
}

/**
 * Sanitize a hyphenated term into an FTS5 phrase by splitting on hyphens
 * and sanitizing each part. Returns the parts joined by spaces for use
 * inside FTS5 quotes: "multi agent" matches "multi-agent" in porter tokenizer.
 */
function sanitizeHyphenatedTerm(term: string): string {
  return term.split('-').map(t => sanitizeFTS5Term(t)).filter(t => t).join(' ');
}

/**
 * Parse lex query syntax into FTS5 query.
 *
 * Supports:
 * - Quoted phrases: "exact phrase" → "exact phrase" (exact match)
 * - Negation: -term or -"phrase" → uses FTS5 NOT operator
 * - Hyphenated tokens: multi-agent, DEC-0054, gpt-4 → treated as phrases
 * - Plain terms: term → "term"* (prefix match)
 *
 * FTS5 NOT is a binary operator: `term1 NOT term2` means "match term1 but not term2".
 * So `-term` only works when there are also positive terms.
 *
 * Hyphen disambiguation: `-sports` at a word boundary is negation, but `multi-agent`
 * (where `-` is between word characters) is treated as a hyphenated phrase.
 * When a leading `-` is followed by what looks like a hyphenated compound word
 * (e.g., `-multi-agent`), the entire token is treated as a negated phrase.
 *
 * Examples:
 *   performance -sports     → "performance"* NOT "sports"*
 *   "machine learning"      → "machine learning"
 *   multi-agent memory      → "multi agent" AND "memory"*
 *   DEC-0054               → "dec 0054"
 *   -multi-agent            → NOT "multi agent"
 */
function buildFTS5Query(query: string): string | null {
  const positive: string[] = [];
  const negative: string[] = [];

  let i = 0;
  const s = query.trim();

  while (i < s.length) {
    // Skip whitespace
    while (i < s.length && /\s/.test(s[i]!)) i++;
    if (i >= s.length) break;

    // Check for negation prefix
    const negated = s[i] === '-';
    if (negated) i++;

    // Check for quoted phrase
    if (s[i] === '"') {
      const start = i + 1;
      i++;
      while (i < s.length && s[i] !== '"') i++;
      const phrase = s.slice(start, i).trim();
      i++; // skip closing quote
      if (phrase.length > 0) {
        const sanitized = phrase.split(/\s+/).map(t => sanitizeFTS5Term(t)).filter(t => t).join(' ');
        if (sanitized) {
          const ftsPhrase = `"${sanitized}"`;  // Exact phrase, no prefix match
          if (negated) {
            negative.push(ftsPhrase);
          } else {
            positive.push(ftsPhrase);
          }
        }
      }
    } else {
      // Plain term (until whitespace or quote)
      const start = i;
      while (i < s.length && !/[\s"]/.test(s[i]!)) i++;
      const term = s.slice(start, i);

      // Handle hyphenated tokens: multi-agent, DEC-0054, gpt-4
      // These get split into phrase queries so FTS5 porter tokenizer matches them.
      if (isHyphenatedToken(term)) {
        const sanitized = sanitizeHyphenatedTerm(term);
        if (sanitized) {
          const ftsPhrase = `"${sanitized}"`;  // Phrase match (no prefix)
          if (negated) {
            negative.push(ftsPhrase);
          } else {
            positive.push(ftsPhrase);
          }
        }
      } else {
        const sanitized = sanitizeFTS5Term(term);
        if (sanitized) {
          const ftsTerm = `"${sanitized}"*`;  // Prefix match
          if (negated) {
            negative.push(ftsTerm);
          } else {
            positive.push(ftsTerm);
          }
        }
      }
    }
  }

  if (positive.length === 0 && negative.length === 0) return null;

  // If only negative terms, we can't search (FTS5 NOT is binary)
  if (positive.length === 0) return null;

  // Join positive terms with AND
  let result = positive.join(' AND ');

  // Add NOT clause for negative terms
  for (const neg of negative) {
    result = `${result} NOT ${neg}`;
  }

  return result;
}

/**
 * Validate that a vec/hyde query doesn't use lex-only syntax.
 * Returns error message if invalid, null if valid.
 */
export function validateSemanticQuery(query: string): string | null {
  // Check for negation syntax
  if (/-\w/.test(query) || /-"/.test(query)) {
    return 'Negation (-term) is not supported in vec/hyde queries. Use lex for exclusions.';
  }
  return null;
}

export function validateLexQuery(query: string): string | null {
  if (/[\r\n]/.test(query)) {
    return 'Lex queries must be a single line. Remove newline characters or split into separate lex: lines.';
  }
  const quoteCount = (query.match(/"/g) ?? []).length;
  if (quoteCount % 2 === 1) {
    return 'Lex query has an unmatched double quote ("). Add the closing quote or remove it.';
  }
  return null;
}

export function searchFTS(db: Database, query: string, limit: number = 20, browser?: string): SearchResult[] {
  const ftsQuery = buildFTS5Query(query);
  if (!ftsQuery) return [];

  const params: (string | number)[] = [ftsQuery];

  // When filtering by browser, fetch extra candidates from the FTS index
  // since some will be filtered out.
  const ftsLimit = browser ? limit * 10 : limit;

  let sql = `
    WITH fts_matches AS (
      SELECT rowid, bm25(documents_fts, 1.5, 4.0, 1.0) as bm25_score
      FROM documents_fts
      WHERE documents_fts MATCH ?
      ORDER BY bm25_score ASC
      LIMIT ${ftsLimit}
    )
    SELECT
      p.url,
      p.title,
      p.hash,
      p.fetch_status,
      p.fetched_at,
      p.modified_at,
      content.doc as body,
      fm.bm25_score,
      (SELECT GROUP_CONCAT(DISTINCT ps.browser) FROM page_sources ps WHERE ps.page_id = p.id) as browsers_csv,
      (SELECT COALESCE(SUM(ps.visit_count), 0) FROM page_sources ps WHERE ps.page_id = p.id) as visit_count
    FROM fts_matches fm
    JOIN pages p ON p.id = fm.rowid
    JOIN content ON content.hash = p.hash
    WHERE p.active = 1
  `;

  if (browser) {
    sql += ` AND EXISTS (SELECT 1 FROM page_sources ps WHERE ps.page_id = p.id AND ps.browser = ?)`;
    params.push(String(browser));
  }

  sql += ` ORDER BY fm.bm25_score ASC LIMIT ?`;
  params.push(limit);

  const rows = db.prepare(sql).all(...params) as {
    url: string; title: string; hash: string;
    fetch_status: string; fetched_at: string | null; modified_at: string;
    body: string; bm25_score: number;
    browsers_csv: string | null; visit_count: number;
  }[];
  return rows.map(row => {
    // |x| / (1 + |x|) maps: strong(-10)→0.91, medium(-2)→0.67, weak(-0.5)→0.33, none(0)→0.
    const score = Math.abs(row.bm25_score) / (1 + Math.abs(row.bm25_score));
    return {
      url: row.url,
      title: row.title,
      hash: row.hash,
      docid: getDocid(row.hash),
      browsers: row.browsers_csv ? row.browsers_csv.split(",") : [],
      visitCount: row.visit_count || 0,
      fetchStatus: row.fetch_status,
      fetchedAt: row.fetched_at,
      modifiedAt: row.modified_at,
      bodyLength: row.body.length,
      body: row.body,
      score,
      source: "fts" as const,
    };
  });
}

// =============================================================================
// Vector Search
// =============================================================================

export async function searchVec(db: Database, query: string, model: string, limit: number = 20, browser?: string, session?: ILLMSession, precomputedEmbedding?: number[]): Promise<SearchResult[]> {
  const tableExists = db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
  if (!tableExists) return [];

  const embedding = precomputedEmbedding ?? await getEmbedding(query, model, true, session);
  if (!embedding) return [];

  // IMPORTANT: We use a two-step query approach here because sqlite-vec virtual tables
  // hang indefinitely when combined with JOINs in the same query.

  // Step 1: Get vector matches from sqlite-vec (no JOINs allowed)
  const vecResults = db.prepare(`
    SELECT hash_seq, distance
    FROM vectors_vec
    WHERE embedding MATCH ? AND k = ?
  `).all(new Float32Array(embedding), limit * 3) as { hash_seq: string; distance: number }[];

  if (vecResults.length === 0) return [];

  // Step 2: Get chunk info and page data
  const hashSeqs = vecResults.map(r => r.hash_seq);
  const distanceMap = new Map(vecResults.map(r => [r.hash_seq, r.distance]));

  const placeholders = hashSeqs.map(() => '?').join(',');
  let docSql = `
    SELECT
      cv.hash || '_' || cv.seq as hash_seq,
      cv.hash,
      cv.pos,
      p.id as page_id,
      p.url,
      p.title,
      p.fetch_status,
      p.fetched_at,
      p.modified_at,
      content.doc as body,
      (SELECT GROUP_CONCAT(DISTINCT ps.browser) FROM page_sources ps WHERE ps.page_id = p.id) as browsers_csv,
      (SELECT COALESCE(SUM(ps.visit_count), 0) FROM page_sources ps WHERE ps.page_id = p.id) as visit_count
    FROM content_vectors cv
    JOIN pages p ON p.hash = cv.hash AND p.active = 1
    JOIN content ON content.hash = p.hash
    WHERE cv.hash || '_' || cv.seq IN (${placeholders})
  `;
  const params: string[] = [...hashSeqs];

  if (browser) {
    docSql += ` AND EXISTS (SELECT 1 FROM page_sources ps WHERE ps.page_id = p.id AND ps.browser = ?)`;
    params.push(browser);
  }

  const docRows = db.prepare(docSql).all(...params) as {
    hash_seq: string; hash: string; pos: number;
    page_id: number; url: string; title: string;
    fetch_status: string; fetched_at: string | null; modified_at: string;
    body: string;
    browsers_csv: string | null; visit_count: number;
  }[];

  // Combine with distances and dedupe by url
  const seen = new Map<string, { row: typeof docRows[0]; bestDist: number }>();
  for (const row of docRows) {
    const distance = distanceMap.get(row.hash_seq) ?? 1;
    const existing = seen.get(row.url);
    if (!existing || distance < existing.bestDist) {
      seen.set(row.url, { row, bestDist: distance });
    }
  }

  return Array.from(seen.values())
    .sort((a, b) => a.bestDist - b.bestDist)
    .slice(0, limit)
    .map(({ row, bestDist }) => {
      return {
        url: row.url,
        title: row.title,
        hash: row.hash,
        docid: getDocid(row.hash),
        browsers: row.browsers_csv ? row.browsers_csv.split(",") : [],
        visitCount: row.visit_count || 0,
        fetchStatus: row.fetch_status,
        fetchedAt: row.fetched_at,
        modifiedAt: row.modified_at,
        bodyLength: row.body.length,
        body: row.body,
        score: 1 - bestDist,
        source: "vec" as const,
        chunkPos: row.pos,
      };
    });
}

// =============================================================================
// Embeddings
// =============================================================================

async function getEmbedding(text: string, model: string, isQuery: boolean, session?: ILLMSession, llmOverride?: LlamaCpp): Promise<number[] | null> {
  // Format text using the appropriate prompt template
  const formattedText = isQuery ? formatQueryForEmbedding(text, model) : formatDocForEmbedding(text, undefined, model);
  const result = session
    ? await session.embed(formattedText, { model, isQuery })
    : await (llmOverride ?? getDefaultLlamaCpp()).embed(formattedText, { model, isQuery });
  return result?.embedding || null;
}

/**
 * Get all unique content hashes that need embeddings (from active fetched pages).
 * Returns hash, page body, and the URL for display purposes.
 */
export function getHashesForEmbedding(db: Database): { hash: string; body: string; path: string }[] {
  return db.prepare(`
    SELECT p.hash, c.doc as body, MIN(p.url) as path
    FROM pages p
    JOIN content c ON p.hash = c.hash
    LEFT JOIN content_vectors v ON p.hash = v.hash AND v.seq = 0
    WHERE p.active = 1 AND p.hash IS NOT NULL AND p.fetch_status = 'fetched' AND v.hash IS NULL
    GROUP BY p.hash
  `).all() as { hash: string; body: string; path: string }[];
}

/**
 * Clear all embeddings from the database (force re-index).
 * Deletes all rows from content_vectors and drops the vectors_vec table.
 */
export function clearAllEmbeddings(db: Database): void {
  db.exec(`DELETE FROM content_vectors`);
  db.exec(`DROP TABLE IF EXISTS vectors_vec`);
}

/**
 * Insert a single embedding into both content_vectors and vectors_vec tables.
 * The hash_seq key is formatted as "hash_seq" for the vectors_vec table.
 *
 * content_vectors is inserted first so that getHashesForEmbedding (which checks
 * only content_vectors) won't re-select the hash on a crash between the two inserts.
 *
 * vectors_vec uses DELETE + INSERT instead of INSERT OR REPLACE because sqlite-vec's
 * vec0 virtual tables silently ignore the OR REPLACE conflict clause.
 */
export function insertEmbedding(
  db: Database,
  hash: string,
  seq: number,
  pos: number,
  embedding: Float32Array,
  model: string,
  embeddedAt: string
): void {
  const hashSeq = `${hash}_${seq}`;

  // Insert content_vectors first — crash-safe ordering (see getHashesForEmbedding)
  const insertContentVectorStmt = db.prepare(`INSERT OR REPLACE INTO content_vectors (hash, seq, pos, model, embedded_at) VALUES (?, ?, ?, ?, ?)`);
  insertContentVectorStmt.run(hash, seq, pos, model, embeddedAt);

  // vec0 virtual tables don't support OR REPLACE — use DELETE + INSERT
  const deleteVecStmt = db.prepare(`DELETE FROM vectors_vec WHERE hash_seq = ?`);
  const insertVecStmt = db.prepare(`INSERT INTO vectors_vec (hash_seq, embedding) VALUES (?, ?)`);
  deleteVecStmt.run(hashSeq);
  insertVecStmt.run(hashSeq, embedding);
}

// =============================================================================
// Query expansion
// =============================================================================

export async function expandQuery(query: string, model: string = DEFAULT_QUERY_MODEL, db: Database, intent?: string, llmOverride?: LlamaCpp): Promise<ExpandedQuery[]> {
  // Check cache first — stored as JSON preserving types
  const cacheKey = getCacheKey("expandQuery", { query, model, ...(intent && { intent }) });
  const cached = getCachedResult(db, cacheKey);
  if (cached) {
    try {
      const parsed = JSON.parse(cached) as any[];
      // Migrate old cache format: { type, text } → { type, query }
      if (parsed.length > 0 && parsed[0].query) {
        return parsed as ExpandedQuery[];
      } else if (parsed.length > 0 && parsed[0].text) {
        return parsed.map((r: any) => ({ type: r.type, query: r.text }));
      }
    } catch {
      // Old cache format (pre-typed, newline-separated text) — re-expand
    }
  }

  const llm = llmOverride ?? getDefaultLlamaCpp();
  // Note: LlamaCpp uses hardcoded model, model parameter is ignored
  const results = await llm.expandQuery(query, { intent });

  // Map Queryable[] → ExpandedQuery[] (same shape, decoupled from llm.ts internals).
  // Filter out entries that duplicate the original query text.
  const expanded: ExpandedQuery[] = results
    .filter(r => r.text !== query)
    .map(r => ({ type: r.type, query: r.text }));

  if (expanded.length > 0) {
    setCachedResult(db, cacheKey, JSON.stringify(expanded));
  }

  return expanded;
}

// =============================================================================
// Reranking
// =============================================================================

export async function rerank(query: string, documents: { file: string; text: string }[], model: string = DEFAULT_RERANK_MODEL, db: Database, intent?: string, llmOverride?: LlamaCpp): Promise<{ file: string; score: number }[]> {
  // Prepend intent to rerank query so the reranker scores with domain context
  const rerankQuery = intent ? `${intent}\n\n${query}` : query;

  const cachedResults: Map<string, number> = new Map();
  const uncachedDocsByChunk: Map<string, RerankDocument> = new Map();

  // Check cache for each document
  // Cache key includes chunk text — different queries can select different chunks
  // from the same file, and the reranker score depends on which chunk was sent.
  // File path is excluded from the new cache key because the reranker score
  // depends on the chunk content, not where it came from.
  for (const doc of documents) {
    const cacheKey = getCacheKey("rerank", { query: rerankQuery, model, chunk: doc.text });
    const legacyCacheKey = getCacheKey("rerank", { query, file: doc.file, model, chunk: doc.text });
    const cached = getCachedResult(db, cacheKey) ?? getCachedResult(db, legacyCacheKey);
    if (cached !== null) {
      cachedResults.set(doc.text, parseFloat(cached));
    } else {
      uncachedDocsByChunk.set(doc.text, { file: doc.file, text: doc.text });
    }
  }

  // Rerank uncached documents using LlamaCpp
  if (uncachedDocsByChunk.size > 0) {
    const llm = llmOverride ?? getDefaultLlamaCpp();
    const uncachedDocs = [...uncachedDocsByChunk.values()];
    const rerankResult = await llm.rerank(rerankQuery, uncachedDocs, { model });

    // Cache results by chunk text so identical chunks across files are scored once.
    const textByFile = new Map(uncachedDocs.map(d => [d.file, d.text]));
    for (const result of rerankResult.results) {
      const chunk = textByFile.get(result.file) || "";
      const cacheKey = getCacheKey("rerank", { query: rerankQuery, model, chunk });
      setCachedResult(db, cacheKey, result.score.toString());
      cachedResults.set(chunk, result.score);
    }
  }

  // Return all results sorted by score
  return documents
    .map(doc => ({ file: doc.file, score: cachedResults.get(doc.text) || 0 }))
    .sort((a, b) => b.score - a.score);
}

// =============================================================================
// Reciprocal Rank Fusion
// =============================================================================

export function reciprocalRankFusion(
  resultLists: RankedResult[][],
  weights: number[] = [],
  k: number = 60
): RankedResult[] {
  const scores = new Map<string, { result: RankedResult; rrfScore: number; topRank: number }>();

  for (let listIdx = 0; listIdx < resultLists.length; listIdx++) {
    const list = resultLists[listIdx];
    if (!list) continue;
    const weight = weights[listIdx] ?? 1.0;

    for (let rank = 0; rank < list.length; rank++) {
      const result = list[rank];
      if (!result) continue;
      const rrfContribution = weight / (k + rank + 1);
      const existing = scores.get(result.file);

      if (existing) {
        existing.rrfScore += rrfContribution;
        existing.topRank = Math.min(existing.topRank, rank);
      } else {
        scores.set(result.file, {
          result,
          rrfScore: rrfContribution,
          topRank: rank,
        });
      }
    }
  }

  // Top-rank bonus
  for (const entry of scores.values()) {
    if (entry.topRank === 0) {
      entry.rrfScore += 0.05;
    } else if (entry.topRank <= 2) {
      entry.rrfScore += 0.02;
    }
  }

  return Array.from(scores.values())
    .sort((a, b) => b.rrfScore - a.rrfScore)
    .map(e => ({ ...e.result, score: e.rrfScore }));
}

/**
 * Build per-document RRF contribution traces for explain/debug output.
 */
export function buildRrfTrace(
  resultLists: RankedResult[][],
  weights: number[] = [],
  listMeta: RankedListMeta[] = [],
  k: number = 60
): Map<string, RRFScoreTrace> {
  const traces = new Map<string, RRFScoreTrace>();

  for (let listIdx = 0; listIdx < resultLists.length; listIdx++) {
    const list = resultLists[listIdx];
    if (!list) continue;
    const weight = weights[listIdx] ?? 1.0;
    const meta = listMeta[listIdx] ?? {
      source: "fts",
      queryType: "original",
      query: "",
    } as const;

    for (let rank0 = 0; rank0 < list.length; rank0++) {
      const result = list[rank0];
      if (!result) continue;
      const rank = rank0 + 1; // 1-indexed rank for explain output
      const contribution = weight / (k + rank);
      const existing = traces.get(result.file);

      const detail: RRFContributionTrace = {
        listIndex: listIdx,
        source: meta.source,
        queryType: meta.queryType,
        query: meta.query,
        rank,
        weight,
        backendScore: result.score,
        rrfContribution: contribution,
      };

      if (existing) {
        existing.baseScore += contribution;
        existing.topRank = Math.min(existing.topRank, rank);
        existing.contributions.push(detail);
      } else {
        traces.set(result.file, {
          contributions: [detail],
          baseScore: contribution,
          topRank: rank,
          topRankBonus: 0,
          totalScore: 0,
        });
      }
    }
  }

  for (const trace of traces.values()) {
    let bonus = 0;
    if (trace.topRank === 1) bonus = 0.05;
    else if (trace.topRank <= 3) bonus = 0.02;
    trace.topRankBonus = bonus;
    trace.totalScore = trace.baseScore + bonus;
  }

  return traces;
}

// =============================================================================
// Page retrieval
// =============================================================================

type DbPageRow = {
  url: string;
  title: string;
  hash: string | null;
  fetch_status: string;
  fetched_at: string | null;
  modified_at: string;
  body_length: number;
  body?: string;
  browsers_csv: string | null;
  visit_count: number;
};

/**
 * Find similar URLs using Levenshtein distance against the URL field.
 */
export function findSimilarUrls(db: Database, query: string, maxDistance: number = 5, limit: number = 5): string[] {
  const rows = db.prepare(`SELECT url FROM pages WHERE active = 1`).all() as { url: string }[];
  const q = query.toLowerCase();
  const scored = rows
    .map(r => ({ url: r.url, dist: levenshtein(r.url.toLowerCase(), q) }))
    .filter(r => r.dist <= maxDistance)
    .sort((a, b) => a.dist - b.dist)
    .slice(0, limit);
  return scored.map(r => r.url);
}

/**
 * Find a page by its short docid (first 6 characters of hash).
 * Returns the matching URL + full hash, or null.
 *
 * Accepts lenient input: #abc123, abc123, "#abc123", "abc123"
 */
export function findPageByDocid(db: Database, docid: string): { url: string; hash: string } | null {
  const shortHash = normalizeDocid(docid);
  if (shortHash.length < 1) return null;

  const row = db.prepare(`
    SELECT p.url, p.hash
    FROM pages p
    WHERE p.hash LIKE ? AND p.active = 1 AND p.hash IS NOT NULL
    LIMIT 1
  `).get(`${shortHash}%`) as { url: string; hash: string } | null;

  return row;
}

/**
 * Find a page by URL or docid.
 * Returns page metadata without body by default.
 *
 * Supports:
 * - Exact URL match
 * - Suffix URL match (fuzzy)
 * - Short docid: #abc123 (first 6 chars of hash)
 */
export function findPage(db: Database, urlOrDocid: string, options: { includeBody?: boolean } = {}): PageResult | PageNotFound {
  let lookup = urlOrDocid.trim();

  // Docid lookup
  if (isDocid(lookup)) {
    const hit = findPageByDocid(db, lookup);
    if (hit) {
      lookup = hit.url;
    } else {
      return { error: "not_found", query: urlOrDocid, similarUrls: [] };
    }
  }

  const bodyCol = options.includeBody ? `, content.doc as body` : ``;
  const selectCols = `
    p.id as page_id,
    p.url,
    p.title,
    p.hash,
    p.fetch_status,
    p.fetched_at,
    p.modified_at,
    COALESCE(LENGTH(content.doc), 0) as body_length,
    (SELECT GROUP_CONCAT(DISTINCT ps.browser) FROM page_sources ps WHERE ps.page_id = p.id) as browsers_csv,
    (SELECT COALESCE(SUM(ps.visit_count), 0) FROM page_sources ps WHERE ps.page_id = p.id) as visit_count
    ${bodyCol}
  `;

  // Exact URL match
  let row = db.prepare(`
    SELECT ${selectCols}
    FROM pages p
    LEFT JOIN content ON content.hash = p.hash
    WHERE p.url = ? AND p.active = 1
  `).get(lookup) as (DbPageRow & { page_id: number }) | null;

  // Suffix match
  if (!row) {
    row = db.prepare(`
      SELECT ${selectCols}
      FROM pages p
      LEFT JOIN content ON content.hash = p.hash
      WHERE p.url LIKE ? AND p.active = 1
      LIMIT 1
    `).get(`%${lookup}`) as (DbPageRow & { page_id: number }) | null;
  }

  if (!row) {
    const similar = findSimilarUrls(db, lookup, 5, 5);
    return { error: "not_found", query: urlOrDocid, similarUrls: similar };
  }

  return {
    url: row.url,
    title: row.title,
    hash: row.hash || "",
    docid: row.hash ? getDocid(row.hash) : "",
    browsers: row.browsers_csv ? row.browsers_csv.split(",") : [],
    visitCount: row.visit_count || 0,
    fetchStatus: row.fetch_status,
    fetchedAt: row.fetched_at,
    modifiedAt: row.modified_at,
    bodyLength: row.body_length,
    ...(options.includeBody && row.body !== undefined && { body: row.body }),
  };
}

/**
 * Get the body content for a page.
 * Optionally slice by line range.
 */
export function getPageBody(db: Database, page: PageResult | { url: string }, fromLine?: number, maxLines?: number): string | null {
  const row = db.prepare(`
    SELECT content.doc as body
    FROM pages p
    JOIN content ON content.hash = p.hash
    WHERE p.url = ? AND p.active = 1
  `).get(page.url) as { body: string } | null;

  if (!row) return null;

  let body = row.body;
  if (fromLine !== undefined || maxLines !== undefined) {
    const lines = body.split('\n');
    const start = (fromLine || 1) - 1;
    const end = maxLines !== undefined ? start + maxLines : lines.length;
    body = lines.slice(start, end).join('\n');
  }
  return body;
}


// =============================================================================
// Status
// =============================================================================

export function getStatus(db: Database): IndexStatus {
  const pageCounts = db.prepare(`
    SELECT
      COUNT(*) as total,
      SUM(CASE WHEN fetch_status = 'fetched' THEN 1 ELSE 0 END) as fetched,
      SUM(CASE WHEN fetch_status = 'pending' THEN 1 ELSE 0 END) as pending,
      SUM(CASE WHEN fetch_status = 'failed' THEN 1 ELSE 0 END) as failed
    FROM pages
    WHERE active = 1
  `).get() as { total: number; fetched: number; pending: number; failed: number };

  // Per-browser counts: distinct pages reachable from each browser via page_sources
  const browserRows = db.prepare(`
    SELECT
      b.name,
      b.history_path,
      b.bookmarks_path,
      b.detected_at,
      b.last_synced_at,
      (SELECT COUNT(DISTINCT ps.page_id)
         FROM page_sources ps
         JOIN pages p ON p.id = ps.page_id
         WHERE ps.browser = b.name AND p.active = 1) as pages
    FROM browsers b
    ORDER BY b.name
  `).all() as {
    name: string; history_path: string | null; bookmarks_path: string | null;
    detected_at: string; last_synced_at: string | null; pages: number;
  }[];

  const browsers: BrowserInfo[] = browserRows.map(b => ({
    name: b.name,
    historyPath: b.history_path,
    bookmarksPath: b.bookmarks_path,
    detectedAt: b.detected_at,
    lastSyncedAt: b.last_synced_at,
    pages: b.pages || 0,
  }));

  const needsEmbedding = getHashesNeedingEmbedding(db);
  const hasVectors = !!db.prepare(`SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`).get();
  const lastRun = getLastIndexerRun(db);

  return {
    totalPages: pageCounts?.total || 0,
    fetchedPages: pageCounts?.fetched || 0,
    pendingPages: pageCounts?.pending || 0,
    failedPages: pageCounts?.failed || 0,
    needsEmbedding,
    hasVectorIndex: hasVectors,
    browsers,
    lastRun,
  };
}

// =============================================================================
// Snippet extraction
// =============================================================================

export type SnippetResult = {
  line: number;           // 1-indexed line number of best match
  snippet: string;        // The snippet text with diff-style header
  linesBefore: number;    // Lines in document before snippet
  linesAfter: number;     // Lines in document after snippet
  snippetLines: number;   // Number of lines in snippet
};

/** Weight for intent terms relative to query terms (1.0) in snippet scoring */
export const INTENT_WEIGHT_SNIPPET = 0.3;

/** Weight for intent terms relative to query terms (1.0) in chunk selection */
export const INTENT_WEIGHT_CHUNK = 0.5;

// Common stop words filtered from intent strings before tokenization.
// Seeded from finetune/reward.py KEY_TERM_STOPWORDS, extended with common
// 2-3 char function words so the length threshold can drop to >1 and let
// short domain terms (API, SQL, LLM, CPU, CDN, …) survive.
const INTENT_STOP_WORDS = new Set([
  // 2-char function words
  "am", "an", "as", "at", "be", "by", "do", "he", "if",
  "in", "is", "it", "me", "my", "no", "of", "on", "or", "so",
  "to", "up", "us", "we",
  // 3-char function words
  "all", "and", "any", "are", "but", "can", "did", "for", "get",
  "has", "her", "him", "his", "how", "its", "let", "may", "not",
  "our", "out", "the", "too", "was", "who", "why", "you",
  // 4+ char common words
  "also", "does", "find", "from", "have", "into", "more", "need",
  "show", "some", "tell", "that", "them", "this", "want", "what",
  "when", "will", "with", "your",
  // Search-context noise
  "about", "looking", "notes", "search", "where", "which",
]);

/**
 * Extract meaningful terms from an intent string, filtering stop words and punctuation.
 * Uses Unicode-aware punctuation stripping so domain terms like "API" survive.
 * Returns lowercase terms suitable for text matching.
 */
export function extractIntentTerms(intent: string): string[] {
  return intent.toLowerCase().split(/\s+/)
    .map(t => t.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ""))
    .filter(t => t.length > 1 && !INTENT_STOP_WORDS.has(t));
}

export function extractSnippet(body: string, query: string, maxLen = 500, chunkPos?: number, chunkLen?: number, intent?: string): SnippetResult {
  const totalLines = body.split('\n').length;
  let searchBody = body;
  let lineOffset = 0;

  if (chunkPos && chunkPos > 0) {
    // Search within the chunk region, with some padding for context
    // Use provided chunkLen or fall back to max chunk size (covers variable-length chunks)
    const searchLen = chunkLen || CHUNK_SIZE_CHARS;
    const contextStart = Math.max(0, chunkPos - 100);
    const contextEnd = Math.min(body.length, chunkPos + searchLen + 100);
    searchBody = body.slice(contextStart, contextEnd);
    if (contextStart > 0) {
      lineOffset = body.slice(0, contextStart).split('\n').length - 1;
    }
  }

  const lines = searchBody.split('\n');
  const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 0);
  const intentTerms = intent ? extractIntentTerms(intent) : [];
  let bestLine = 0, bestScore = -1;

  for (let i = 0; i < lines.length; i++) {
    const lineLower = (lines[i] ?? "").toLowerCase();
    let score = 0;
    for (const term of queryTerms) {
      if (lineLower.includes(term)) score += 1.0;
    }
    for (const term of intentTerms) {
      if (lineLower.includes(term)) score += INTENT_WEIGHT_SNIPPET;
    }
    if (score > bestScore) {
      bestScore = score;
      bestLine = i;
    }
  }

  const start = Math.max(0, bestLine - 1);
  const end = Math.min(lines.length, bestLine + 3);
  const snippetLines = lines.slice(start, end);
  let snippetText = snippetLines.join('\n');

  // If we focused on a chunk window and it produced an empty/whitespace-only snippet,
  // fall back to a full-document snippet so we always show something useful.
  if (chunkPos && chunkPos > 0 && snippetText.trim().length === 0) {
    return extractSnippet(body, query, maxLen, undefined, undefined, intent);
  }

  if (snippetText.length > maxLen) snippetText = snippetText.substring(0, maxLen - 3) + "...";

  const absoluteStart = lineOffset + start + 1; // 1-indexed
  const snippetLineCount = snippetLines.length;
  const linesBefore = absoluteStart - 1;
  const linesAfter = totalLines - (absoluteStart + snippetLineCount - 1);

  // Format with diff-style header: @@ -start,count @@ (linesBefore before, linesAfter after)
  const header = `@@ -${absoluteStart},${snippetLineCount} @@ (${linesBefore} before, ${linesAfter} after)`;
  const snippet = `${header}\n${snippetText}`;

  return {
    line: lineOffset + bestLine + 1,
    snippet,
    linesBefore,
    linesAfter,
    snippetLines: snippetLineCount,
  };
}

// =============================================================================
// Shared helpers (used by both CLI and MCP)
// =============================================================================

/**
 * Add line numbers to text content.
 * Each line becomes: "{lineNum}: {content}"
 */
export function addLineNumbers(text: string, startLine: number = 1): string {
  const lines = text.split('\n');
  return lines.map((line, i) => `${startLine + i}: ${line}`).join('\n');
}

// =============================================================================
// Shared search orchestration
//
// hybridQuery() and vectorSearchQuery() are standalone functions (not Store
// methods) because they are orchestration over primitives — same rationale as
// reciprocalRankFusion(). They take a Store as first argument so both CLI
// and MCP can share the identical pipeline.
// =============================================================================

/**
 * Optional progress hooks for search orchestration.
 * CLI wires these to stderr for user feedback; MCP leaves them unset.
 */
export interface SearchHooks {
  /** BM25 probe found strong signal — expansion will be skipped */
  onStrongSignal?: (topScore: number) => void;
  /** Query expansion starting */
  onExpandStart?: () => void;
  /** Query expansion complete. Empty array = strong signal skip. elapsedMs = time taken. */
  onExpand?: (original: string, expanded: ExpandedQuery[], elapsedMs: number) => void;
  /** Embedding starting (vec/hyde queries) */
  onEmbedStart?: (count: number) => void;
  /** Embedding complete */
  onEmbedDone?: (elapsedMs: number) => void;
  /** Reranking is about to start */
  onRerankStart?: (chunkCount: number) => void;
  /** Reranking finished */
  onRerankDone?: (elapsedMs: number) => void;
}

export interface HybridQueryOptions {
  collection?: string;
  limit?: number;           // default 10
  minScore?: number;        // default 0
  candidateLimit?: number;  // default RERANK_CANDIDATE_LIMIT
  explain?: boolean;        // include backend/RRF/rerank score traces
  intent?: string;          // domain intent hint for disambiguation
  skipRerank?: boolean;     // skip LLM reranking, use only RRF scores
  chunkStrategy?: ChunkStrategy;
  hooks?: SearchHooks;
}

export interface HybridQueryResult {
  file: string;             // internal filepath (qmd://collection/path)
  displayPath: string;
  title: string;
  body: string;             // full document body (for snippet extraction)
  bestChunk: string;        // best chunk text
  bestChunkPos: number;     // char offset of best chunk in body
  score: number;            // blended score (full precision)
  context: string | null;   // user-set context
  docid: string;            // content hash prefix (6 chars)
  explain?: HybridQueryExplain;
}

export type RankedListMeta = {
  source: "fts" | "vec";
  queryType: "original" | "lex" | "vec" | "hyde";
  query: string;
};

/**
 * Hybrid search: BM25 + vector + query expansion + RRF + chunked reranking.
 *
 * Pipeline:
 * 1. BM25 probe → skip expansion if strong signal
 * 2. expandQuery() → typed query variants (lex/vec/hyde)
 * 3. Type-routed search: original→vector, lex→FTS, vec/hyde→vector
 * 4. RRF fusion → slice to candidateLimit
 * 5. chunkDocument() + keyword-best-chunk selection
 * 6. rerank on chunks (NOT full bodies — O(tokens) trap)
 * 7. Position-aware score blending (RRF rank × reranker score)
 * 8. Dedup by file, filter by minScore, slice to limit
 */
export async function hybridQuery(
  store: Store,
  query: string,
  options?: HybridQueryOptions
): Promise<HybridQueryResult[]> {
  const limit = options?.limit ?? 10;
  const minScore = options?.minScore ?? 0;
  const candidateLimit = options?.candidateLimit ?? RERANK_CANDIDATE_LIMIT;
  const collection = options?.collection;
  const explain = options?.explain ?? false;
  const intent = options?.intent;
  const skipRerank = options?.skipRerank ?? false;
  const hooks = options?.hooks;

  const rankedLists: RankedResult[][] = [];
  const rankedListMeta: RankedListMeta[] = [];
  const docidMap = new Map<string, string>(); // filepath -> docid
  const hasVectors = !!store.db.prepare(
    `SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`
  ).get();

  // Step 1: BM25 probe — strong signal skips expensive LLM expansion
  // When intent is provided, disable strong-signal bypass — the obvious BM25
  // match may not be what the caller wants (e.g. "performance" with intent
  // "web page load times" should NOT shortcut to a sports-performance doc).
  // Pass collection directly into FTS query (filter at SQL level, not post-hoc)
  const initialFts = store.searchFTS(query, 20, collection);
  const topScore = initialFts[0]?.score ?? 0;
  const secondScore = initialFts[1]?.score ?? 0;
  const hasStrongSignal = !intent && initialFts.length > 0
    && topScore >= STRONG_SIGNAL_MIN_SCORE
    && (topScore - secondScore) >= STRONG_SIGNAL_MIN_GAP;

  if (hasStrongSignal) hooks?.onStrongSignal?.(topScore);

  // Step 2: Expand query (or skip if strong signal)
  hooks?.onExpandStart?.();
  const expandStart = Date.now();
  const expanded = hasStrongSignal
    ? []
    : await store.expandQuery(query, undefined, intent);

  hooks?.onExpand?.(query, expanded, Date.now() - expandStart);

  // Seed with initial FTS results (avoid re-running original query FTS)
  if (initialFts.length > 0) {
    for (const r of initialFts) docidMap.set(r.url, r.docid);
    rankedLists.push(initialFts.map(r => ({
      file: r.url, displayPath: r.url,
      title: r.title, body: r.body || "", score: r.score,
    })));
    rankedListMeta.push({ source: "fts", queryType: "original", query });
  }

  // Step 3: Route searches by query type
  //
  // Strategy: run all FTS queries immediately (they're sync/instant), then
  // batch-embed all vector queries in one embedBatch() call, then run
  // sqlite-vec lookups with pre-computed embeddings.

  // 3a: Run FTS for all lex expansions right away (no LLM needed)
  for (const q of expanded) {
    if (q.type === 'lex') {
      const ftsResults = store.searchFTS(q.query, 20, collection);
      if (ftsResults.length > 0) {
        for (const r of ftsResults) docidMap.set(r.url, r.docid);
        rankedLists.push(ftsResults.map(r => ({
          file: r.url, displayPath: r.url,
          title: r.title, body: r.body || "", score: r.score,
        })));
        rankedListMeta.push({ source: "fts", queryType: "lex", query: q.query });
      }
    }
  }

  // 3b: Collect all texts that need vector search (original query + vec/hyde expansions)
  if (hasVectors) {
    const vecQueries: { text: string; queryType: "original" | "vec" | "hyde" }[] = [
      { text: query, queryType: "original" },
    ];
    for (const q of expanded) {
      if (q.type === 'vec' || q.type === 'hyde') {
        vecQueries.push({ text: q.query, queryType: q.type });
      }
    }

    // Batch embed all vector queries in a single call
    const llm = getLlm(store);
    const textsToEmbed = vecQueries.map(q => formatQueryForEmbedding(q.text, llm.embedModelName));
    hooks?.onEmbedStart?.(textsToEmbed.length);
    const embedStart = Date.now();
    const embeddings = await llm.embedBatch(textsToEmbed);
    hooks?.onEmbedDone?.(Date.now() - embedStart);

    // Run sqlite-vec lookups with pre-computed embeddings
    for (let i = 0; i < vecQueries.length; i++) {
      const embedding = embeddings[i]?.embedding;
      if (!embedding) continue;

      const vecResults = await store.searchVec(
        vecQueries[i]!.text, DEFAULT_EMBED_MODEL, 20, collection,
        undefined, embedding
      );
      if (vecResults.length > 0) {
        for (const r of vecResults) docidMap.set(r.url, r.docid);
        rankedLists.push(vecResults.map(r => ({
          file: r.url, displayPath: r.url,
          title: r.title, body: r.body || "", score: r.score,
        })));
        rankedListMeta.push({
          source: "vec",
          queryType: vecQueries[i]!.queryType,
          query: vecQueries[i]!.text,
        });
      }
    }
  }

  // Step 4: RRF fusion — first 2 lists (original FTS + first vec) get 2x weight
  const weights = rankedLists.map((_, i) => i < 2 ? 2.0 : 1.0);
  const fused = reciprocalRankFusion(rankedLists, weights);
  const rrfTraceByFile = explain ? buildRrfTrace(rankedLists, weights, rankedListMeta) : null;
  const candidates = fused.slice(0, candidateLimit);

  if (candidates.length === 0) return [];

  // Step 5: Chunk documents, pick best chunk per doc for reranking.
  // Reranking full bodies is O(tokens) — the critical perf lesson that motivated this refactor.
  const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);
  const intentTerms = intent ? extractIntentTerms(intent) : [];
  const docChunkMap = new Map<string, { chunks: { text: string; pos: number }[]; bestIdx: number }>();

  const chunkStrategy = options?.chunkStrategy;
  for (const cand of candidates) {
    const chunks = await chunkDocumentAsync(cand.body, undefined, undefined, undefined, cand.file, chunkStrategy);
    if (chunks.length === 0) continue;

    // Pick chunk with most keyword overlap (fallback: first chunk)
    // Intent terms contribute at INTENT_WEIGHT_CHUNK (0.5) relative to query terms (1.0)
    let bestIdx = 0;
    let bestScore = -1;
    for (let i = 0; i < chunks.length; i++) {
      const chunkLower = chunks[i]!.text.toLowerCase();
      let score = queryTerms.reduce((acc, term) => acc + (chunkLower.includes(term) ? 1 : 0), 0);
      for (const term of intentTerms) {
        if (chunkLower.includes(term)) score += INTENT_WEIGHT_CHUNK;
      }
      if (score > bestScore) { bestScore = score; bestIdx = i; }
    }

    docChunkMap.set(cand.file, { chunks, bestIdx });
  }

  if (skipRerank) {
    // Skip LLM reranking — return candidates scored by RRF only
    const seenFiles = new Set<string>();
    return candidates
      .map((cand, i) => {
        const chunkInfo = docChunkMap.get(cand.file);
        const bestIdx = chunkInfo?.bestIdx ?? 0;
        const bestChunk = chunkInfo?.chunks[bestIdx]?.text || cand.body || "";
        const bestChunkPos = chunkInfo?.chunks[bestIdx]?.pos || 0;
        const rrfRank = i + 1;
        const rrfScore = 1 / rrfRank;
        const trace = rrfTraceByFile?.get(cand.file);
        const explainData: HybridQueryExplain | undefined = explain ? {
          ftsScores: trace?.contributions.filter(c => c.source === "fts").map(c => c.backendScore) ?? [],
          vectorScores: trace?.contributions.filter(c => c.source === "vec").map(c => c.backendScore) ?? [],
          rrf: {
            rank: rrfRank,
            positionScore: rrfScore,
            weight: 1.0,
            baseScore: trace?.baseScore ?? 0,
            topRankBonus: trace?.topRankBonus ?? 0,
            totalScore: trace?.totalScore ?? 0,
            contributions: trace?.contributions ?? [],
          },
          rerankScore: 0,
          blendedScore: rrfScore,
        } : undefined;

        return {
          file: cand.file,
          displayPath: cand.displayPath,
          title: cand.title,
          body: cand.body,
          bestChunk,
          bestChunkPos,
          score: rrfScore,
          context: store.getContextForFile(cand.file),
          docid: docidMap.get(cand.file) || "",
          ...(explainData ? { explain: explainData } : {}),
        };
      })
      .filter(r => {
        if (seenFiles.has(r.file)) return false;
        seenFiles.add(r.file);
        return true;
      })
      .filter(r => r.score >= minScore)
      .slice(0, limit);
  }

  // Step 6: Rerank chunks (NOT full bodies)
  const chunksToRerank: { file: string; text: string }[] = [];
  for (const cand of candidates) {
    const chunkInfo = docChunkMap.get(cand.file);
    if (chunkInfo) {
      chunksToRerank.push({ file: cand.file, text: chunkInfo.chunks[chunkInfo.bestIdx]!.text });
    }
  }

  hooks?.onRerankStart?.(chunksToRerank.length);
  const rerankStart = Date.now();
  const reranked = await store.rerank(query, chunksToRerank, undefined, intent);
  hooks?.onRerankDone?.(Date.now() - rerankStart);

  // Step 7: Blend RRF position score with reranker score
  // Position-aware weights: top retrieval results get more protection from reranker disagreement
  const candidateMap = new Map(candidates.map(c => [c.file, {
    displayPath: c.displayPath, title: c.title, body: c.body,
  }]));
  const rrfRankMap = new Map(candidates.map((c, i) => [c.file, i + 1]));

  const blended = reranked.map(r => {
    const rrfRank = rrfRankMap.get(r.file) || candidateLimit;
    let rrfWeight: number;
    if (rrfRank <= 3) rrfWeight = 0.75;
    else if (rrfRank <= 10) rrfWeight = 0.60;
    else rrfWeight = 0.40;
    const rrfScore = 1 / rrfRank;
    const blendedScore = rrfWeight * rrfScore + (1 - rrfWeight) * r.score;

    const candidate = candidateMap.get(r.file);
    const chunkInfo = docChunkMap.get(r.file);
    const bestIdx = chunkInfo?.bestIdx ?? 0;
    const bestChunk = chunkInfo?.chunks[bestIdx]?.text || candidate?.body || "";
    const bestChunkPos = chunkInfo?.chunks[bestIdx]?.pos || 0;
    const trace = rrfTraceByFile?.get(r.file);
    const explainData: HybridQueryExplain | undefined = explain ? {
      ftsScores: trace?.contributions.filter(c => c.source === "fts").map(c => c.backendScore) ?? [],
      vectorScores: trace?.contributions.filter(c => c.source === "vec").map(c => c.backendScore) ?? [],
      rrf: {
        rank: rrfRank,
        positionScore: rrfScore,
        weight: rrfWeight,
        baseScore: trace?.baseScore ?? 0,
        topRankBonus: trace?.topRankBonus ?? 0,
        totalScore: trace?.totalScore ?? 0,
        contributions: trace?.contributions ?? [],
      },
      rerankScore: r.score,
      blendedScore,
    } : undefined;

    return {
      file: r.file,
      displayPath: candidate?.displayPath || "",
      title: candidate?.title || "",
      body: candidate?.body || "",
      bestChunk,
      bestChunkPos,
      score: blendedScore,
      context: store.getContextForFile(r.file),
      docid: docidMap.get(r.file) || "",
      ...(explainData ? { explain: explainData } : {}),
    };
  }).sort((a, b) => b.score - a.score);

  // Step 8: Dedup by file (safety net — prevents duplicate output)
  const seenFiles = new Set<string>();
  return blended
    .filter(r => {
      if (seenFiles.has(r.file)) return false;
      seenFiles.add(r.file);
      return true;
    })
    .filter(r => r.score >= minScore)
    .slice(0, limit);
}

export interface VectorSearchOptions {
  collection?: string;
  limit?: number;           // default 10
  minScore?: number;        // default 0.3
  intent?: string;          // domain intent hint for disambiguation
  hooks?: Pick<SearchHooks, 'onExpand'>;
}

export interface VectorSearchResult {
  file: string;
  displayPath: string;
  title: string;
  body: string;
  score: number;
  context: string | null;
  docid: string;
}

/**
 * Vector-only semantic search with query expansion.
 *
 * Pipeline:
 * 1. expandQuery() → typed variants, filter to vec/hyde only (lex irrelevant here)
 * 2. searchVec() for original + vec/hyde variants (sequential — node-llama-cpp embed limitation)
 * 3. Dedup by filepath (keep max score)
 * 4. Sort by score descending, filter by minScore, slice to limit
 */
export async function vectorSearchQuery(
  store: Store,
  query: string,
  options?: VectorSearchOptions
): Promise<VectorSearchResult[]> {
  const limit = options?.limit ?? 10;
  const minScore = options?.minScore ?? 0.3;
  const collection = options?.collection;
  const intent = options?.intent;

  const hasVectors = !!store.db.prepare(
    `SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`
  ).get();
  if (!hasVectors) return [];

  // Expand query — filter to vec/hyde only (lex queries target FTS, not vector)
  const expandStart = Date.now();
  const allExpanded = await store.expandQuery(query, undefined, intent);
  const vecExpanded = allExpanded.filter(q => q.type !== 'lex');
  options?.hooks?.onExpand?.(query, vecExpanded, Date.now() - expandStart);

  // Run original + vec/hyde expanded through vector, sequentially — concurrent embed() hangs
  const queryTexts = [query, ...vecExpanded.map(q => q.query)];
  const allResults = new Map<string, VectorSearchResult>();
  for (const q of queryTexts) {
    const vecResults = await store.searchVec(q, DEFAULT_EMBED_MODEL, limit, collection);
    for (const r of vecResults) {
      const existing = allResults.get(r.url);
      if (!existing || r.score > existing.score) {
        allResults.set(r.url, {
          file: r.url,
          displayPath: r.url,
          title: r.title,
          body: r.body || "",
          score: r.score,
          context: store.getContextForFile(r.url),
          docid: r.docid,
        });
      }
    }
  }

  return Array.from(allResults.values())
    .sort((a, b) => b.score - a.score)
    .filter(r => r.score >= minScore)
    .slice(0, limit);
}

// =============================================================================
// Structured search — pre-expanded queries from LLM
// =============================================================================

/**
 * A single sub-search in a structured search request.
 * Matches the format used in QMD training data.
 */
export interface StructuredSearchOptions {
  collections?: string[];   // Filter to specific collections (OR match)
  limit?: number;           // default 10
  minScore?: number;        // default 0
  candidateLimit?: number;  // default RERANK_CANDIDATE_LIMIT
  explain?: boolean;        // include backend/RRF/rerank score traces
  /** Domain intent hint for disambiguation — steers reranking and chunk selection */
  intent?: string;
  /** Skip LLM reranking, use only RRF scores */
  skipRerank?: boolean;
  chunkStrategy?: ChunkStrategy;
  hooks?: SearchHooks;
}

/**
 * Structured search: execute pre-expanded queries without LLM query expansion.
 *
 * Designed for LLM callers (MCP/HTTP) that generate their own query expansions.
 * Skips the internal expandQuery() step — goes directly to:
 *
 * Pipeline:
 * 1. Route searches: lex→FTS, vec/hyde→vector (batch embed)
 * 2. RRF fusion across all result lists
 * 3. Chunk documents + keyword-best-chunk selection
 * 4. Rerank on chunks
 * 5. Position-aware score blending
 * 6. Dedup, filter, slice
 *
 * This is the recommended endpoint for capable LLMs — they can generate
 * better query variations than our small local model, especially for
 * domain-specific or nuanced queries.
 */
export async function structuredSearch(
  store: Store,
  searches: ExpandedQuery[],
  options?: StructuredSearchOptions
): Promise<HybridQueryResult[]> {
  const limit = options?.limit ?? 10;
  const minScore = options?.minScore ?? 0;
  const candidateLimit = options?.candidateLimit ?? RERANK_CANDIDATE_LIMIT;
  const explain = options?.explain ?? false;
  const intent = options?.intent;
  const skipRerank = options?.skipRerank ?? false;
  const hooks = options?.hooks;

  const collections = options?.collections;

  if (searches.length === 0) return [];

  // Validate queries before executing
  for (const search of searches) {
    const location = search.line ? `Line ${search.line}` : 'Structured search';
    if (/[\r\n]/.test(search.query)) {
      throw new Error(`${location} (${search.type}): queries must be single-line. Remove newline characters.`);
    }
    if (search.type === 'lex') {
      const error = validateLexQuery(search.query);
      if (error) {
        throw new Error(`${location} (lex): ${error}`);
      }
    } else if (search.type === 'vec' || search.type === 'hyde') {
      const error = validateSemanticQuery(search.query);
      if (error) {
        throw new Error(`${location} (${search.type}): ${error}`);
      }
    }
  }

  const rankedLists: RankedResult[][] = [];
  const rankedListMeta: RankedListMeta[] = [];
  const docidMap = new Map<string, string>(); // filepath -> docid
  const hasVectors = !!store.db.prepare(
    `SELECT name FROM sqlite_master WHERE type='table' AND name='vectors_vec'`
  ).get();

  // Helper to run search across collections (or all if undefined)
  const collectionList = collections ?? [undefined]; // undefined = all collections

  // Step 1: Run FTS for all lex searches (sync, instant)
  for (const search of searches) {
    if (search.type === 'lex') {
      for (const coll of collectionList) {
        const ftsResults = store.searchFTS(search.query, 20, coll);
        if (ftsResults.length > 0) {
          for (const r of ftsResults) docidMap.set(r.url, r.docid);
          rankedLists.push(ftsResults.map(r => ({
            file: r.url, displayPath: r.url,
            title: r.title, body: r.body || "", score: r.score,
          })));
          rankedListMeta.push({
            source: "fts",
            queryType: "lex",
            query: search.query,
          });
        }
      }
    }
  }

  // Step 2: Batch embed and run vector searches for vec/hyde
  if (hasVectors) {
    const vecSearches = searches.filter(
      (s): s is ExpandedQuery & { type: 'vec' | 'hyde' } =>
        s.type === 'vec' || s.type === 'hyde'
    );
    if (vecSearches.length > 0) {
      const llm = getLlm(store);
      const textsToEmbed = vecSearches.map(s => formatQueryForEmbedding(s.query, llm.embedModelName));
      hooks?.onEmbedStart?.(textsToEmbed.length);
      const embedStart = Date.now();
      const embeddings = await llm.embedBatch(textsToEmbed);
      hooks?.onEmbedDone?.(Date.now() - embedStart);

      for (let i = 0; i < vecSearches.length; i++) {
        const embedding = embeddings[i]?.embedding;
        if (!embedding) continue;

        for (const coll of collectionList) {
          const vecResults = await store.searchVec(
            vecSearches[i]!.query, DEFAULT_EMBED_MODEL, 20, coll,
            undefined, embedding
          );
          if (vecResults.length > 0) {
            for (const r of vecResults) docidMap.set(r.url, r.docid);
            rankedLists.push(vecResults.map(r => ({
              file: r.url, displayPath: r.url,
              title: r.title, body: r.body || "", score: r.score,
            })));
            rankedListMeta.push({
              source: "vec",
              queryType: vecSearches[i]!.type,
              query: vecSearches[i]!.query,
            });
          }
        }
      }
    }
  }

  if (rankedLists.length === 0) return [];

  // Step 3: RRF fusion — first list gets 2x weight (assume caller ordered by importance)
  const weights = rankedLists.map((_, i) => i === 0 ? 2.0 : 1.0);
  const fused = reciprocalRankFusion(rankedLists, weights);
  const rrfTraceByFile = explain ? buildRrfTrace(rankedLists, weights, rankedListMeta) : null;
  const candidates = fused.slice(0, candidateLimit);

  if (candidates.length === 0) return [];

  hooks?.onExpand?.("", [], 0); // Signal no expansion (pre-expanded)

  // Step 4: Chunk documents, pick best chunk per doc for reranking
  // Use first lex query as the "query" for keyword matching, or first vec if no lex
  const primaryQuery = searches.find(s => s.type === 'lex')?.query
    || searches.find(s => s.type === 'vec')?.query
    || searches[0]?.query || "";
  const queryTerms = primaryQuery.toLowerCase().split(/\s+/).filter(t => t.length > 2);
  const intentTerms = intent ? extractIntentTerms(intent) : [];
  const docChunkMap = new Map<string, { chunks: { text: string; pos: number }[]; bestIdx: number }>();
  const ssChunkStrategy = options?.chunkStrategy;

  for (const cand of candidates) {
    const chunks = await chunkDocumentAsync(cand.body, undefined, undefined, undefined, cand.file, ssChunkStrategy);
    if (chunks.length === 0) continue;

    // Pick chunk with most keyword overlap
    // Intent terms contribute at INTENT_WEIGHT_CHUNK (0.5) relative to query terms (1.0)
    let bestIdx = 0;
    let bestScore = -1;
    for (let i = 0; i < chunks.length; i++) {
      const chunkLower = chunks[i]!.text.toLowerCase();
      let score = queryTerms.reduce((acc, term) => acc + (chunkLower.includes(term) ? 1 : 0), 0);
      for (const term of intentTerms) {
        if (chunkLower.includes(term)) score += INTENT_WEIGHT_CHUNK;
      }
      if (score > bestScore) { bestScore = score; bestIdx = i; }
    }

    docChunkMap.set(cand.file, { chunks, bestIdx });
  }

  if (skipRerank) {
    // Skip LLM reranking — return candidates scored by RRF only
    const seenFiles = new Set<string>();
    return candidates
      .map((cand, i) => {
        const chunkInfo = docChunkMap.get(cand.file);
        const bestIdx = chunkInfo?.bestIdx ?? 0;
        const bestChunk = chunkInfo?.chunks[bestIdx]?.text || cand.body || "";
        const bestChunkPos = chunkInfo?.chunks[bestIdx]?.pos || 0;
        const rrfRank = i + 1;
        const rrfScore = 1 / rrfRank;
        const trace = rrfTraceByFile?.get(cand.file);
        const explainData: HybridQueryExplain | undefined = explain ? {
          ftsScores: trace?.contributions.filter(c => c.source === "fts").map(c => c.backendScore) ?? [],
          vectorScores: trace?.contributions.filter(c => c.source === "vec").map(c => c.backendScore) ?? [],
          rrf: {
            rank: rrfRank,
            positionScore: rrfScore,
            weight: 1.0,
            baseScore: trace?.baseScore ?? 0,
            topRankBonus: trace?.topRankBonus ?? 0,
            totalScore: trace?.totalScore ?? 0,
            contributions: trace?.contributions ?? [],
          },
          rerankScore: 0,
          blendedScore: rrfScore,
        } : undefined;

        return {
          file: cand.file,
          displayPath: cand.displayPath,
          title: cand.title,
          body: cand.body,
          bestChunk,
          bestChunkPos,
          score: rrfScore,
          context: store.getContextForFile(cand.file),
          docid: docidMap.get(cand.file) || "",
          ...(explainData ? { explain: explainData } : {}),
        };
      })
      .filter(r => {
        if (seenFiles.has(r.file)) return false;
        seenFiles.add(r.file);
        return true;
      })
      .filter(r => r.score >= minScore)
      .slice(0, limit);
  }

  // Step 5: Rerank chunks
  const chunksToRerank: { file: string; text: string }[] = [];
  for (const cand of candidates) {
    const chunkInfo = docChunkMap.get(cand.file);
    if (chunkInfo) {
      chunksToRerank.push({ file: cand.file, text: chunkInfo.chunks[chunkInfo.bestIdx]!.text });
    }
  }

  hooks?.onRerankStart?.(chunksToRerank.length);
  const rerankStart2 = Date.now();
  const reranked = await store.rerank(primaryQuery, chunksToRerank, undefined, intent);
  hooks?.onRerankDone?.(Date.now() - rerankStart2);

  // Step 6: Blend RRF position score with reranker score
  const candidateMap = new Map(candidates.map(c => [c.file, {
    displayPath: c.displayPath, title: c.title, body: c.body,
  }]));
  const rrfRankMap = new Map(candidates.map((c, i) => [c.file, i + 1]));

  const blended = reranked.map(r => {
    const rrfRank = rrfRankMap.get(r.file) || candidateLimit;
    let rrfWeight: number;
    if (rrfRank <= 3) rrfWeight = 0.75;
    else if (rrfRank <= 10) rrfWeight = 0.60;
    else rrfWeight = 0.40;
    const rrfScore = 1 / rrfRank;
    const blendedScore = rrfWeight * rrfScore + (1 - rrfWeight) * r.score;

    const candidate = candidateMap.get(r.file);
    const chunkInfo = docChunkMap.get(r.file);
    const bestIdx = chunkInfo?.bestIdx ?? 0;
    const bestChunk = chunkInfo?.chunks[bestIdx]?.text || candidate?.body || "";
    const bestChunkPos = chunkInfo?.chunks[bestIdx]?.pos || 0;
    const trace = rrfTraceByFile?.get(r.file);
    const explainData: HybridQueryExplain | undefined = explain ? {
      ftsScores: trace?.contributions.filter(c => c.source === "fts").map(c => c.backendScore) ?? [],
      vectorScores: trace?.contributions.filter(c => c.source === "vec").map(c => c.backendScore) ?? [],
      rrf: {
        rank: rrfRank,
        positionScore: rrfScore,
        weight: rrfWeight,
        baseScore: trace?.baseScore ?? 0,
        topRankBonus: trace?.topRankBonus ?? 0,
        totalScore: trace?.totalScore ?? 0,
        contributions: trace?.contributions ?? [],
      },
      rerankScore: r.score,
      blendedScore,
    } : undefined;

    return {
      file: r.file,
      displayPath: candidate?.displayPath || "",
      title: candidate?.title || "",
      body: candidate?.body || "",
      bestChunk,
      bestChunkPos,
      score: blendedScore,
      context: store.getContextForFile(r.file),
      docid: docidMap.get(r.file) || "",
      ...(explainData ? { explain: explainData } : {}),
    };
  }).sort((a, b) => b.score - a.score);

  // Step 7: Dedup by file
  const seenFiles = new Set<string>();
  return blended
    .filter(r => {
      if (seenFiles.has(r.file)) return false;
      seenFiles.add(r.file);
      return true;
    })
    .filter(r => r.score >= minScore)
    .slice(0, limit);
}
