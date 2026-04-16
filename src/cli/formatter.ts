/**
 * formatter.ts - Output formatting utilities for planet-capture
 *
 * Provides methods to format search results and pages into various output formats:
 * JSON, CSV, XML, Markdown, files list, and CLI (colored terminal output).
 */

import { extractSnippet } from "../store.js";
import type { SearchResult, PageResult } from "../store.js";

// =============================================================================
// Types
// =============================================================================

// Re-export store types for convenience
export type { SearchResult, PageResult };

export type OutputFormat = "cli" | "csv" | "md" | "xml" | "files" | "json";

export type FormatOptions = {
  full?: boolean;       // Show full page content instead of snippet
  query?: string;       // Query for snippet extraction and highlighting
  useColor?: boolean;   // Enable terminal colors (default: false for non-CLI)
  lineNumbers?: boolean;// Add line numbers to output
  intent?: string;      // Domain intent for snippet extraction disambiguation
};

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Add line numbers to text content.
 * Each line becomes: "{lineNum}: {content}"
 * @param text The text to add line numbers to
 * @param startLine Optional starting line number (default: 1)
 */
export function addLineNumbers(text: string, startLine: number = 1): string {
  const lines = text.split('\n');
  return lines.map((line, i) => `${startLine + i}: ${line}`).join('\n');
}

/**
 * Extract short docid from a full hash (first 6 characters).
 */
export function getDocid(hash: string): string {
  return hash.slice(0, 6);
}

// =============================================================================
// Escape Helpers
// =============================================================================

export function escapeCSV(value: string | null | number): string {
  if (value === null || value === undefined) return "";
  const str = String(value);
  if (str.includes(",") || str.includes('"') || str.includes("\n")) {
    return `"${str.replace(/"/g, '""')}"`;
  }
  return str;
}

export function escapeXml(str: string): string {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

// =============================================================================
// Search Results Formatters
// =============================================================================

/**
 * Format search results as JSON
 */
export function searchResultsToJson(
  results: SearchResult[],
  opts: FormatOptions = {}
): string {
  const query = opts.query || "";
  const output = results.map(row => {
    const bodyStr = row.body || "";
    const snippetInfo = bodyStr
      ? extractSnippet(bodyStr, query, 300, row.chunkPos, undefined, opts.intent)
      : undefined;
    let body = opts.full ? bodyStr : undefined;
    let snippet = !opts.full ? snippetInfo?.snippet : undefined;

    if (opts.lineNumbers) {
      if (body) body = addLineNumbers(body);
      if (snippet) snippet = addLineNumbers(snippet);
    }

    return {
      docid: `#${row.docid}`,
      score: Math.round(row.score * 100) / 100,
      url: row.url,
      ...(snippetInfo && { line: snippetInfo.line }),
      title: row.title,
      browsers: row.browsers,
      visitCount: row.visitCount,
      ...(body && { body }),
      ...(snippet && { snippet }),
    };
  });
  return JSON.stringify(output, null, 2);
}

/**
 * Format search results as CSV
 */
export function searchResultsToCsv(
  results: SearchResult[],
  opts: FormatOptions = {}
): string {
  const query = opts.query || "";
  const header = "docid,score,url,title,browsers,visits,line,snippet";
  const rows = results.map(row => {
    const bodyStr = row.body || "";
    const { line, snippet } = extractSnippet(bodyStr, query, 500, row.chunkPos, undefined, opts.intent);
    let content = opts.full ? bodyStr : snippet;
    if (opts.lineNumbers && content) {
      content = addLineNumbers(content);
    }
    return [
      `#${row.docid}`,
      row.score.toFixed(4),
      escapeCSV(row.url),
      escapeCSV(row.title),
      escapeCSV(row.browsers.join(";")),
      row.visitCount,
      line,
      escapeCSV(content),
    ].join(",");
  });
  return [header, ...rows].join("\n");
}

/**
 * Format search results as simple files list
 */
export function searchResultsToFiles(results: SearchResult[]): string {
  return results.map(row => {
    return `#${row.docid},${row.score.toFixed(2)},${row.url}`;
  }).join("\n");
}

/**
 * Format search results as Markdown
 */
export function searchResultsToMarkdown(
  results: SearchResult[],
  opts: FormatOptions = {}
): string {
  const query = opts.query || "";
  return results.map(row => {
    const heading = row.title || row.url;
    const bodyStr = row.body || "";
    let content: string;
    if (opts.full) {
      content = bodyStr;
    } else {
      content = extractSnippet(bodyStr, query, 500, row.chunkPos, undefined, opts.intent).snippet;
    }
    if (opts.lineNumbers) {
      content = addLineNumbers(content);
    }
    const browserLine = row.browsers.length > 0 ? `**browsers:** ${row.browsers.join(", ")}\n` : "";
    return `---\n# ${heading}\n\n**docid:** \`#${row.docid}\` **url:** ${row.url}\n${browserLine}\n${content}\n`;
  }).join("\n");
}

/**
 * Format search results as XML
 */
export function searchResultsToXml(
  results: SearchResult[],
  opts: FormatOptions = {}
): string {
  const query = opts.query || "";
  const items = results.map(row => {
    const titleAttr = row.title ? ` title="${escapeXml(row.title)}"` : "";
    const bodyStr = row.body || "";
    let content = opts.full ? bodyStr : extractSnippet(bodyStr, query, 500, row.chunkPos, undefined, opts.intent).snippet;
    if (opts.lineNumbers) {
      content = addLineNumbers(content);
    }
    return `<page docid="#${row.docid}" url="${escapeXml(row.url)}"${titleAttr}>\n${escapeXml(content)}\n</page>`;
  });
  return items.join("\n\n");
}

/**
 * Format search results for MCP (simpler CSV format with pre-extracted snippets)
 */
export function searchResultsToMcpCsv(
  results: { docid: string; url: string; title: string; score: number; snippet: string }[]
): string {
  const header = "docid,url,title,score,snippet";
  const rows = results.map(r =>
    [`#${r.docid}`, r.url, r.title, r.score, r.snippet].map(escapeCSV).join(",")
  );
  return [header, ...rows].join("\n");
}

// =============================================================================
// Page Formatters
// =============================================================================

/**
 * Format a single PageResult as JSON
 */
export function pageToJson(page: PageResult): string {
  return JSON.stringify({
    url: page.url,
    title: page.title,
    hash: page.hash,
    docid: page.docid,
    browsers: page.browsers,
    visitCount: page.visitCount,
    fetchStatus: page.fetchStatus,
    fetchedAt: page.fetchedAt,
    modifiedAt: page.modifiedAt,
    bodyLength: page.bodyLength,
    ...(page.body !== undefined && { body: page.body }),
  }, null, 2);
}

/**
 * Format a single PageResult as Markdown
 */
export function pageToMarkdown(page: PageResult): string {
  let md = `# ${page.title || page.url}\n\n`;
  md += `**URL:** ${page.url}\n`;
  md += `**Browsers:** ${page.browsers.join(", ")}\n`;
  md += `**Visits:** ${page.visitCount}\n`;
  md += `**Fetched:** ${page.fetchedAt || "not yet"}\n\n`;
  if (page.body !== undefined) {
    md += "---\n\n" + page.body + "\n";
  }
  return md;
}

/**
 * Format a single page to the specified format
 */
export function formatPage(page: PageResult, format: OutputFormat): string {
  switch (format) {
    case "json":
      return pageToJson(page);
    case "md":
      return pageToMarkdown(page);
    default:
      return pageToMarkdown(page);
  }
}

// =============================================================================
// Universal Format Function
// =============================================================================

/**
 * Format search results to the specified output format
 */
export function formatSearchResults(
  results: SearchResult[],
  format: OutputFormat,
  opts: FormatOptions = {}
): string {
  switch (format) {
    case "json":
      return searchResultsToJson(results, opts);
    case "csv":
      return searchResultsToCsv(results, opts);
    case "files":
      return searchResultsToFiles(results);
    case "md":
      return searchResultsToMarkdown(results, opts);
    case "xml":
      return searchResultsToXml(results, opts);
    case "cli":
      return searchResultsToMarkdown(results, opts);
    default:
      return searchResultsToJson(results, opts);
  }
}
