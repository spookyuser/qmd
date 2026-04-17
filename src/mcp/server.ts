import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { z } from "zod";
import { createServer } from "node:http";
import { randomUUID } from "node:crypto";
import {
  createStore,
  enableProductionMode,
  hybridQuery,
  extractSnippet,
  addLineNumbers,
  DEFAULT_MULTI_GET_MAX_BYTES,
  type Store,
} from "../store.js";
import { LlamaCpp } from "../llm.js";

enableProductionMode();

function createMcpServer(store: Store): McpServer {
  const server = new McpServer(
    { name: "planet-capture", version: "0.1.0" },
    { capabilities: { tools: {} } },
  );

  // ---------------------------------------------------------------------------
  // Tool: search
  // ---------------------------------------------------------------------------
  server.tool(
    "search",
    "Search your browser history using hybrid BM25 + vector + LLM reranking. Returns the most relevant pages matching the query.",
    {
      query: z.string().describe("Search query"),
      limit: z.number().int().min(1).max(50).optional().describe("Max results (default 10)"),
      browser: z.string().optional().describe("Restrict to a specific browser"),
      rerank: z.boolean().optional().describe("Use LLM reranking (default true)"),
    },
    async ({ query, limit, browser, rerank }) => {
      const results = await hybridQuery(store, query, {
        collection: browser,
        limit: limit ?? 10,
        skipRerank: rerank === false,
      });

      if (results.length === 0) {
        return { content: [{ type: "text", text: `No results found for "${query}".` }] };
      }

      const text = results
        .map((r, i) => {
          const snippet = extractSnippet(r.body, query, 300, r.bestChunkPos);
          return [
            `${i + 1}. ${r.title}`,
            `   URL: ${r.file}`,
            `   Score: ${r.score.toFixed(3)}  Docid: #${r.docid}`,
            `   ${snippet.snippet}`,
          ].join("\n");
        })
        .join("\n\n");

      return { content: [{ type: "text", text }] };
    },
  );

  // ---------------------------------------------------------------------------
  // Tool: get
  // ---------------------------------------------------------------------------
  server.tool(
    "get",
    "Retrieve a page by URL or docid (#abc123). Returns page metadata and extracted text content.",
    {
      page: z.string().describe("URL or docid (e.g. #abc123)"),
      fromLine: z.number().int().min(1).optional().describe("Start from this line number"),
      maxLines: z.number().int().min(1).optional().describe("Maximum lines to return"),
    },
    async ({ page, fromLine, maxLines }) => {
      const result = store.findPage(page, { includeBody: false });

      if ("error" in result) {
        const msg = result.similarUrls.length
          ? `Page not found: "${page}"\n\nSimilar URLs:\n${result.similarUrls.map(u => `  - ${u}`).join("\n")}`
          : `Page not found: "${page}"`;
        return { content: [{ type: "text", text: msg }], isError: true };
      }

      let body = store.getPageBody(result, fromLine, maxLines) ?? "(no content)";
      if (fromLine || maxLines) {
        body = addLineNumbers(body, fromLine ?? 1);
      }

      const header = [
        `Title: ${result.title}`,
        `URL: ${result.url}`,
        `Docid: #${result.docid}`,
        `Browsers: ${result.browsers.join(", ") || "unknown"}`,
        `Visits: ${result.visitCount}`,
        `Fetched: ${result.fetchedAt ?? "pending"}`,
        `Body length: ${result.bodyLength} bytes`,
        `---`,
      ].join("\n");

      return { content: [{ type: "text", text: `${header}\n${body}` }] };
    },
  );

  // ---------------------------------------------------------------------------
  // Tool: recent
  // ---------------------------------------------------------------------------
  server.tool(
    "recent",
    "List recently visited pages from your browser history.",
    {
      limit: z.number().int().min(1).max(100).optional().describe("Max results (default 20)"),
      browser: z.string().optional().describe("Restrict to a specific browser"),
    },
    async ({ limit, browser }) => {
      const n = limit ?? 20;
      let query = `
        SELECT
          p.url, p.title, p.hash, p.fetch_status, p.modified_at,
          (SELECT GROUP_CONCAT(DISTINCT ps.browser) FROM page_sources ps WHERE ps.page_id = p.id) as browsers_csv,
          (SELECT COALESCE(SUM(ps.visit_count), 0) FROM page_sources ps WHERE ps.page_id = p.id) as visit_count,
          (SELECT MAX(ps.last_visit) FROM page_sources ps WHERE ps.page_id = p.id) as last_visit
        FROM pages p
        WHERE p.active = 1 AND p.fetch_status = 'fetched'
      `;
      const params: (string | number)[] = [];

      if (browser) {
        query += ` AND EXISTS (SELECT 1 FROM page_sources ps WHERE ps.page_id = p.id AND ps.browser = ?)`;
        params.push(browser);
      }

      query += ` ORDER BY last_visit DESC NULLS LAST LIMIT ?`;
      params.push(n);

      const rows = store.db.prepare(query).all(...params) as {
        url: string; title: string; hash: string | null;
        fetch_status: string; modified_at: string;
        browsers_csv: string | null; visit_count: number;
        last_visit: string | null;
      }[];

      if (rows.length === 0) {
        return { content: [{ type: "text", text: "No recently visited pages found." }] };
      }

      const text = rows
        .map((r, i) => {
          const browsers = r.browsers_csv ? r.browsers_csv.split(",").join(", ") : "unknown";
          const docid = r.hash ? `#${r.hash.slice(0, 6)}` : "";
          return `${i + 1}. ${r.title}\n   ${r.url}\n   Browsers: ${browsers}  Visits: ${r.visit_count}  ${docid}`;
        })
        .join("\n\n");

      return { content: [{ type: "text", text }] };
    },
  );

  // ---------------------------------------------------------------------------
  // Tool: status
  // ---------------------------------------------------------------------------
  server.tool(
    "status",
    "Show the current state of the planet-capture index: page counts, browsers, embedding status.",
    {},
    async () => {
      const s = store.getStatus();
      const lines = [
        `Pages: ${s.totalPages} total, ${s.fetchedPages} fetched, ${s.pendingPages} pending, ${s.failedPages} failed`,
        `Embeddings: ${s.needsEmbedding} pages need embedding, vector index ${s.hasVectorIndex ? "available" : "not built"}`,
      ];

      if (s.browsers.length > 0) {
        lines.push("", "Browsers:");
        for (const b of s.browsers) {
          lines.push(`  ${b.name}: ${b.pages} pages (last synced: ${b.lastSyncedAt ?? "never"})`);
        }
      }

      if (s.lastRun) {
        const d = s.lastRun;
        lines.push("", `Last run: ${d.started_at} — discovered ${d.urls_discovered ?? 0}, fetched ${d.urls_fetched ?? 0}, errors ${d.urls_failed ?? 0}`);
      }

      return { content: [{ type: "text", text: lines.join("\n") }] };
    },
  );

  // ---------------------------------------------------------------------------
  // Tool: multi_get
  // ---------------------------------------------------------------------------
  server.tool(
    "multi_get",
    "Retrieve multiple pages by their docids. Pass a comma-separated list of docids.",
    {
      docids: z.string().describe("Comma-separated docids, e.g. '#abc123, #def456'"),
      maxLines: z.number().int().min(1).optional().describe("Max lines per page (default unlimited)"),
      maxBytes: z.number().int().min(1).optional().describe("Skip pages larger than this (default 10KB)"),
    },
    async ({ docids, maxLines, maxBytes }) => {
      const ids = docids.split(",").map(d => d.trim()).filter(Boolean);
      const limit = maxBytes ?? DEFAULT_MULTI_GET_MAX_BYTES;
      const parts: string[] = [];

      for (const id of ids) {
        const result = store.findPage(id, { includeBody: false });
        if ("error" in result) {
          parts.push(`--- ${id} ---\nNot found\n`);
          continue;
        }
        if (result.bodyLength > limit) {
          parts.push(`--- ${result.url} (#${result.docid}) ---\nSkipped: ${result.bodyLength} bytes exceeds limit of ${limit}\n`);
          continue;
        }
        let body = store.getPageBody(result, undefined, maxLines) ?? "(no content)";
        if (maxLines) {
          body = addLineNumbers(body);
        }
        parts.push(`--- ${result.url} (#${result.docid}) ---\n${result.title}\n\n${body}\n`);
      }

      return { content: [{ type: "text", text: parts.join("\n") }] };
    },
  );

  return server;
}

// =============================================================================
// Transport: stdio
// =============================================================================

export async function startStdio(dbPath?: string): Promise<void> {
  const store = createStore(dbPath);
  const llm = new LlamaCpp({
    inactivityTimeoutMs: 5 * 60 * 1000,
    disposeModelsOnInactivity: true,
  });
  store.llm = llm;

  const server = createMcpServer(store);
  const transport = new StdioServerTransport();

  process.on("SIGINT", async () => {
    await server.close();
    await llm.dispose();
    store.close();
    process.exit(0);
  });

  await server.connect(transport);
}

// =============================================================================
// Transport: HTTP (Streamable HTTP)
// =============================================================================

export async function startHttp(port: number = 8181, dbPath?: string): Promise<void> {
  const store = createStore(dbPath);
  const llm = new LlamaCpp({
    inactivityTimeoutMs: 5 * 60 * 1000,
    disposeModelsOnInactivity: true,
  });
  store.llm = llm;

  const server = createMcpServer(store);
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: () => randomUUID(),
  });

  await server.connect(transport);

  const httpServer = createServer(async (req, res) => {
    const url = new URL(req.url ?? "/", `http://localhost:${port}`);
    if (url.pathname === "/mcp") {
      await transport.handleRequest(req, res);
    } else {
      res.writeHead(404);
      res.end("Not Found");
    }
  });

  httpServer.listen(port, () => {
    console.error(`planet-capture MCP server listening on http://localhost:${port}/mcp`);
  });

  const shutdown = async () => {
    httpServer.close();
    await server.close();
    await llm.dispose();
    store.close();
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}
