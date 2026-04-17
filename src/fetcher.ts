/**
 * fetcher.ts - Page fetching and content extraction
 *
 * Downloads web pages and extracts readable text content using
 * Mozilla Readability + linkedom for DOM parsing.
 */

import { Readability } from "@mozilla/readability";
import { parseHTML } from "linkedom";

// =============================================================================
// Types
// =============================================================================

export interface FetchResult {
  url: string;
  title: string;
  content: string; // Extracted text content
  byteLength: number;
}

export interface FetchError {
  url: string;
  error: string;
}

export type FetchOutcome =
  | { ok: true; result: FetchResult }
  | { ok: false; error: FetchError };

// =============================================================================
// Rate Limiter
// =============================================================================

export interface RateLimiter {
  acquire(): Promise<void>;
}

/**
 * Create a simple rate limiter that enforces minimum delay between requests.
 */
export function createRateLimiter(requestsPerSecond: number = 2): RateLimiter {
  const minDelay = 1000 / requestsPerSecond;
  let lastFetchTime = 0;

  return {
    async acquire() {
      const now = Date.now();
      const elapsed = now - lastFetchTime;
      const waitTime = Math.max(0, minDelay - elapsed);
      if (waitTime > 0) {
        await new Promise((resolve) => setTimeout(resolve, waitTime));
      }
      lastFetchTime = Date.now();
    },
  };
}

// =============================================================================
// URL Filtering
// =============================================================================

/** File extensions that indicate binary/non-text content */
const BINARY_EXTENSIONS = new Set([
  ".pdf", ".zip", ".gz", ".tar", ".rar", ".7z",
  ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".ico", ".bmp", ".tiff",
  ".mp3", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".wav", ".flac", ".ogg",
  ".exe", ".dmg", ".app", ".msi", ".deb", ".rpm",
  ".woff", ".woff2", ".ttf", ".eot", ".otf",
  ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
  ".iso", ".img", ".bin",
]);

/**
 * Check if a URL points to a likely binary resource.
 */
export function isBinaryUrl(url: string): boolean {
  try {
    const pathname = new URL(url).pathname.toLowerCase();
    for (const ext of BINARY_EXTENSIONS) {
      if (pathname.endsWith(ext)) return true;
    }
  } catch {
    // Invalid URL
  }
  return false;
}

/**
 * Check if a URL is fetchable (http/https, not binary).
 */
export function isFetchableUrl(url: string): boolean {
  if (!url) return false;
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return false;
  } catch {
    return false;
  }
  if (isBinaryUrl(url)) return false;
  return true;
}

// =============================================================================
// Content Extraction
// =============================================================================

/**
 * Extract readable text content from HTML using Mozilla Readability.
 * Falls back to basic text extraction if Readability fails.
 */
export function extractContent(html: string, url: string): { title: string; content: string } {
  void url;
  try {
    const { document } = parseHTML(html);

    // Try Readability first
    const reader = new Readability(document as any);
    const article = reader.parse();

    if (article && article.textContent && article.textContent.trim().length > 50) {
      return {
        title: article.title || "",
        content: cleanText(article.textContent),
      };
    }

    // Fallback: extract text from body
    const body = document.querySelector("body");
    const text = body ? body.textContent || "" : "";
    const title = document.querySelector("title")?.textContent || "";

    return {
      title,
      content: cleanText(text),
    };
  } catch {
    return { title: "", content: "" };
  }
}

/**
 * Clean extracted text: normalize whitespace, remove excessive blank lines.
 */
function cleanText(text: string): string {
  return text
    .replace(/\t/g, " ")
    .replace(/[ ]+/g, " ") // collapse multiple spaces
    .replace(/\n[ ]+/g, "\n") // remove leading spaces on lines
    .replace(/[ ]+\n/g, "\n") // remove trailing spaces on lines
    .replace(/\n{3,}/g, "\n\n") // collapse 3+ newlines to 2
    .trim();
}

// =============================================================================
// Page Fetching
// =============================================================================

const DEFAULT_TIMEOUT = 15000; // 15 seconds
const DEFAULT_USER_AGENT =
  "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36";

/**
 * Fetch a page and extract its text content.
 */
export async function fetchPage(
  url: string,
  options?: { timeout?: number; userAgent?: string }
): Promise<FetchOutcome> {
  const timeout = options?.timeout ?? DEFAULT_TIMEOUT;
  const userAgent = options?.userAgent ?? DEFAULT_USER_AGENT;

  if (!isFetchableUrl(url)) {
    return {
      ok: false,
      error: { url, error: "URL not fetchable (non-http or binary)" },
    };
  }

  try {
    const response = await fetch(url, {
      signal: AbortSignal.timeout(timeout),
      headers: {
        "User-Agent": userAgent,
        Accept: "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
      },
      redirect: "follow",
    });

    if (!response.ok) {
      return {
        ok: false,
        error: { url, error: `HTTP ${response.status} ${response.statusText}` },
      };
    }

    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("text/html") && !contentType.includes("application/xhtml")) {
      return {
        ok: false,
        error: { url, error: `Non-HTML content type: ${contentType}` },
      };
    }

    const html = await response.text();
    const { title, content } = extractContent(html, url);

    if (!content || content.length < 50) {
      return {
        ok: false,
        error: { url, error: "Extracted content too short (< 50 chars)" },
      };
    }

    return {
      ok: true,
      result: {
        url,
        title,
        content,
        byteLength: new TextEncoder().encode(content).length,
      },
    };
  } catch (err: any) {
    const message = err?.name === "TimeoutError"
      ? `Timeout after ${timeout}ms`
      : err?.message || "Unknown error";

    return {
      ok: false,
      error: { url, error: message },
    };
  }
}
