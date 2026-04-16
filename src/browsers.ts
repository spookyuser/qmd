/**
 * browsers.ts - Browser detection and history/bookmark reading
 *
 * Supports Chrome, Arc, Brave, and Safari on macOS.
 * Copies browser databases to a temp directory before reading (browsers lock their DBs).
 */

import { existsSync, copyFileSync, mkdirSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { tmpdir } from "node:os";
import { openDatabase } from "./db.js";
import type { Database } from "./db.js";

// =============================================================================
// Types
// =============================================================================

export interface BrowserInfo {
  name: string;
  type: "chromium" | "safari";
  historyPath: string;
  bookmarksPath?: string; // Only for Chromium browsers (JSON file)
  detected: boolean;
}

export interface HistoryEntry {
  url: string;
  title: string;
  visitCount: number;
  lastVisitTime: Date;
  browser: string;
  sourceType: "history";
}

export interface BookmarkEntry {
  url: string;
  title: string;
  folder: string;
  browser: string;
  sourceType: "bookmark";
}

export type BrowserEntry = HistoryEntry | BookmarkEntry;

// =============================================================================
// Browser Paths (macOS)
// =============================================================================

const HOME = homedir();

const BROWSER_CONFIGS: Omit<BrowserInfo, "detected">[] = [
  {
    name: "chrome",
    type: "chromium",
    historyPath: join(HOME, "Library/Application Support/Google/Chrome/Default/History"),
    bookmarksPath: join(HOME, "Library/Application Support/Google/Chrome/Default/Bookmarks"),
  },
  {
    name: "arc",
    type: "chromium",
    historyPath: join(HOME, "Library/Application Support/Arc/User Data/Default/History"),
    bookmarksPath: join(HOME, "Library/Application Support/Arc/User Data/Default/Bookmarks"),
  },
  {
    name: "brave",
    type: "chromium",
    historyPath: join(HOME, "Library/Application Support/BraveSoftware/Brave-Browser/Default/History"),
    bookmarksPath: join(HOME, "Library/Application Support/BraveSoftware/Brave-Browser/Default/Bookmarks"),
  },
  {
    name: "safari",
    type: "safari",
    historyPath: join(HOME, "Library/Safari/History.db"),
  },
];

// =============================================================================
// URL Filtering
// =============================================================================

/** Built-in URL prefixes to skip */
const SKIP_PREFIXES = [
  "chrome://",
  "chrome-extension://",
  "about:",
  "data:",
  "blob:",
  "file://",
  "javascript:",
  "mailto:",
  "tel:",
  "arc://",
  "brave://",
  "edge://",
  "opera://",
  "vivaldi://",
  "chrome-devtools://",
  "devtools://",
  "view-source:",
];

/** Built-in URL patterns to skip (regex) */
const SKIP_PATTERNS = [
  /^https?:\/\/localhost/,
  /^https?:\/\/127\.\d+\.\d+\.\d+/,
  /^https?:\/\/\[::1\]/,
  /^https?:\/\/0\.0\.0\.0/,
  /^https?:\/\/192\.168\./,
  /^https?:\/\/10\./,
];

/**
 * Check if a URL should be skipped based on built-in filters.
 */
export function shouldSkipUrl(url: string): boolean {
  if (!url) return true;
  for (const prefix of SKIP_PREFIXES) {
    if (url.startsWith(prefix)) return true;
  }
  for (const pattern of SKIP_PATTERNS) {
    if (pattern.test(url)) return true;
  }
  return false;
}

/**
 * Check if a URL should be skipped based on user-defined exclude patterns.
 */
export function matchesExcludeFilter(url: string, patterns: string[]): boolean {
  for (const pattern of patterns) {
    try {
      if (new RegExp(pattern).test(url)) return true;
    } catch {
      // Invalid regex — treat as literal substring match
      if (url.includes(pattern)) return true;
    }
  }
  return false;
}

// =============================================================================
// Browser Detection
// =============================================================================

/**
 * Detect which browsers are installed by checking for their history databases.
 */
export function detectBrowsers(): BrowserInfo[] {
  return BROWSER_CONFIGS.map((config) => ({
    ...config,
    detected: existsSync(config.historyPath),
  }));
}

/**
 * Get only the detected (installed) browsers.
 */
export function getDetectedBrowsers(): BrowserInfo[] {
  return detectBrowsers().filter((b) => b.detected);
}

// =============================================================================
// Database Copy (browsers lock their DBs)
// =============================================================================

/**
 * Copy a browser database to a temp directory for safe reading.
 * Browsers hold a WAL lock on their databases while running.
 * Returns the path to the temporary copy.
 */
function copyBrowserDb(sourcePath: string): string {
  const tempDir = join(tmpdir(), "planet-capture");
  mkdirSync(tempDir, { recursive: true });

  const tempPath = join(tempDir, `${Date.now()}-${Math.random().toString(36).slice(2)}.db`);
  copyFileSync(sourcePath, tempPath);

  // Also copy WAL and SHM files if they exist (for consistent reads)
  const walPath = sourcePath + "-wal";
  const shmPath = sourcePath + "-shm";
  if (existsSync(walPath)) copyFileSync(walPath, tempPath + "-wal");
  if (existsSync(shmPath)) copyFileSync(shmPath, tempPath + "-shm");

  return tempPath;
}

/**
 * Clean up a temporary database copy.
 */
function cleanupTempDb(tempPath: string): void {
  try {
    rmSync(tempPath, { force: true });
    rmSync(tempPath + "-wal", { force: true });
    rmSync(tempPath + "-shm", { force: true });
  } catch {
    // Best effort cleanup
  }
}

// =============================================================================
// Chromium History Reading
// =============================================================================

/**
 * Convert Chromium timestamp (microseconds since 1601-01-01) to JS Date.
 * Chromium uses Windows FILETIME epoch.
 */
export function chromiumTimestampToDate(microseconds: number): Date {
  // Chromium epoch offset: microseconds between 1601-01-01 and 1970-01-01
  const CHROMIUM_EPOCH_OFFSET = 11644473600000000n;
  const unixMicroseconds = BigInt(microseconds) - CHROMIUM_EPOCH_OFFSET;
  return new Date(Number(unixMicroseconds / 1000n));
}

/**
 * Read browsing history from a Chromium-based browser.
 * Copies the database to a temp location first.
 */
export function readChromiumHistory(
  browser: BrowserInfo,
  options?: { since?: Date }
): HistoryEntry[] {
  if (!existsSync(browser.historyPath)) return [];

  const tempPath = copyBrowserDb(browser.historyPath);
  let db: Database | null = null;

  try {
    db = openDatabase(tempPath);

    let query = `
      SELECT url, title, visit_count, last_visit_time
      FROM urls
      WHERE url LIKE 'http%'
      ORDER BY last_visit_time DESC
    `;
    const params: any[] = [];

    if (options?.since) {
      // Convert JS Date to Chromium timestamp
      const CHROMIUM_EPOCH_OFFSET = 11644473600000000n;
      const chromiumTime = BigInt(options.since.getTime()) * 1000n + CHROMIUM_EPOCH_OFFSET;
      query = `
        SELECT url, title, visit_count, last_visit_time
        FROM urls
        WHERE url LIKE 'http%' AND last_visit_time >= ?
        ORDER BY last_visit_time DESC
      `;
      params.push(Number(chromiumTime));
    }

    const rows = db.prepare(query).all(...params);

    return rows
      .filter((row: any) => !shouldSkipUrl(row.url))
      .map((row: any) => ({
        url: row.url,
        title: row.title || "",
        visitCount: row.visit_count || 0,
        lastVisitTime: chromiumTimestampToDate(row.last_visit_time),
        browser: browser.name,
        sourceType: "history" as const,
      }));
  } finally {
    if (db) db.close();
    cleanupTempDb(tempPath);
  }
}

// =============================================================================
// Chromium Bookmarks Reading
// =============================================================================

interface ChromiumBookmarkNode {
  type: string;
  name: string;
  url?: string;
  children?: ChromiumBookmarkNode[];
}

/**
 * Recursively walk a Chromium bookmarks tree to extract URLs.
 */
function walkBookmarks(
  node: ChromiumBookmarkNode,
  folderPath: string,
  browser: string,
  results: BookmarkEntry[]
): void {
  if (node.type === "url" && node.url) {
    if (!shouldSkipUrl(node.url)) {
      results.push({
        url: node.url,
        title: node.name || "",
        folder: folderPath,
        browser,
        sourceType: "bookmark",
      });
    }
  }
  if (node.children) {
    const currentFolder = folderPath ? `${folderPath}/${node.name}` : node.name;
    for (const child of node.children) {
      walkBookmarks(child, currentFolder, browser, results);
    }
  }
}

/**
 * Read bookmarks from a Chromium-based browser.
 */
export function readChromiumBookmarks(browser: BrowserInfo): BookmarkEntry[] {
  if (!browser.bookmarksPath || !existsSync(browser.bookmarksPath)) return [];

  try {
    const raw = readFileSync(browser.bookmarksPath, "utf-8");
    const data = JSON.parse(raw);
    const results: BookmarkEntry[] = [];

    const roots = data.roots;
    if (roots) {
      for (const key of ["bookmark_bar", "other", "synced"]) {
        if (roots[key]) {
          walkBookmarks(roots[key], "", browser.name, results);
        }
      }
    }

    return results;
  } catch {
    return [];
  }
}

// =============================================================================
// Safari History Reading
// =============================================================================

/**
 * Convert Safari/Core Data timestamp (seconds since 2001-01-01) to JS Date.
 */
export function safariTimestampToDate(seconds: number): Date {
  // Core Data epoch offset: seconds between 2001-01-01 and 1970-01-01
  const CORE_DATA_EPOCH_OFFSET = 978307200;
  return new Date((seconds + CORE_DATA_EPOCH_OFFSET) * 1000);
}

/**
 * Read browsing history from Safari.
 * Safari uses a different schema than Chromium browsers.
 */
export function readSafariHistory(
  browser: BrowserInfo,
  options?: { since?: Date }
): HistoryEntry[] {
  if (!existsSync(browser.historyPath)) return [];

  const tempPath = copyBrowserDb(browser.historyPath);
  let db: Database | null = null;

  try {
    db = openDatabase(tempPath);

    let query = `
      SELECT
        hi.url,
        COALESCE(hi.title, '') as title,
        hi.visit_count,
        MAX(hv.visit_time) as last_visit_time
      FROM history_items hi
      LEFT JOIN history_visits hv ON hi.id = hv.history_item
      WHERE hi.url LIKE 'http%'
      GROUP BY hi.id
      ORDER BY last_visit_time DESC
    `;
    const params: any[] = [];

    if (options?.since) {
      const CORE_DATA_EPOCH_OFFSET = 978307200;
      const safariTime = options.since.getTime() / 1000 - CORE_DATA_EPOCH_OFFSET;
      query = `
        SELECT
          hi.url,
          COALESCE(hi.title, '') as title,
          hi.visit_count,
          MAX(hv.visit_time) as last_visit_time
        FROM history_items hi
        LEFT JOIN history_visits hv ON hi.id = hv.history_item
        WHERE hi.url LIKE 'http%' AND hv.visit_time >= ?
        GROUP BY hi.id
        ORDER BY last_visit_time DESC
      `;
      params.push(safariTime);
    }

    const rows = db.prepare(query).all(...params);

    return rows
      .filter((row: any) => !shouldSkipUrl(row.url))
      .map((row: any) => ({
        url: row.url,
        title: row.title || "",
        visitCount: row.visit_count || 0,
        lastVisitTime: row.last_visit_time
          ? safariTimestampToDate(row.last_visit_time)
          : new Date(),
        browser: browser.name,
        sourceType: "history" as const,
      }));
  } finally {
    if (db) db.close();
    cleanupTempDb(tempPath);
  }
}

// =============================================================================
// Unified Reading Interface
// =============================================================================

/**
 * Read all history and bookmarks from a single browser.
 */
export function readBrowser(
  browser: BrowserInfo,
  options?: { since?: Date }
): BrowserEntry[] {
  const entries: BrowserEntry[] = [];

  if (browser.type === "chromium") {
    entries.push(...readChromiumHistory(browser, options));
    entries.push(...readChromiumBookmarks(browser));
  } else if (browser.type === "safari") {
    entries.push(...readSafariHistory(browser, options));
  }

  return entries;
}

/**
 * Read history and bookmarks from all detected browsers.
 */
export function readAllBrowsers(options?: { since?: Date }): BrowserEntry[] {
  const browsers = getDetectedBrowsers();
  const entries: BrowserEntry[] = [];

  for (const browser of browsers) {
    entries.push(...readBrowser(browser, options));
  }

  return entries;
}
