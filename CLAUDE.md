# planet-capture

Index and search your browser history with hybrid BM25 + vector search + LLM reranking.

Use Bun instead of Node.js (`bun` not `node`, `bun install` not `npm install`).

## Commands

```sh
planet-capture index              # Discover URLs from browsers + fetch pages
planet-capture discover           # Only read browser history (no fetch)
planet-capture fetch              # Only fetch pending pages (no discover)
planet-capture search <query>     # BM25 full-text search
planet-capture vsearch <query>    # Vector similarity search
planet-capture query <query>      # Hybrid search (BM25 + vector + rerank)
planet-capture get <url|#docid>   # Show a single page
planet-capture status             # Show index counts and browsers
planet-capture browsers           # List detected browsers
planet-capture filters list       # List URL exclusion filters
planet-capture filters add <pat>  # Add URL filter pattern
planet-capture filters remove <p> # Remove URL filter
planet-capture embed              # Generate vector embeddings
planet-capture mcp                # Start MCP server (stdio)
planet-capture mcp --http         # Start MCP server (HTTP, default port 8181)
planet-capture mcp --http --port=N  # HTTP on custom port
```

## Document IDs (docid)

Each page has a unique short ID (docid) — the first 6 characters of its content hash.
Docids are shown in search results as `#abc123` and can be used with `get`:

```sh
planet-capture get "#abc123"
planet-capture get abc123    # Leading # is optional
```

## Options

```sh
# Search & retrieval
--browser=NAME       # Restrict search to a specific browser
-n <num>             # Number of results
--min-score <num>    # Minimum score threshold
--full               # Show full page content
--line-numbers       # Add line numbers to output
--intent=TEXT        # Disambiguation intent (query only)

# Index options
--since=DATE         # Only include history after this date
--rate-limit=N       # Fetches per second (default 2)
--max-pages=N        # Max pages to fetch this run
--discover-only      # Skip fetching
--dry-run            # Show what would happen

# Output formats
--json, --csv, --md, --xml, --files
```

## Development

```sh
bun src/cli/planet-capture.ts <command>   # Run from source
bun link                                  # Install globally
```

## Tests

All tests live in `test/`. Run everything:

```sh
npx vitest run --reporter=verbose test/
```

## Architecture

- SQLite FTS5 for full-text search (BM25)
- sqlite-vec for vector similarity search
- node-llama-cpp for embeddings (embeddinggemma), reranking (qwen3-reranker), and query expansion (Qwen3)
- Reciprocal Rank Fusion (RRF) for combining results
- Smart chunking: 900 tokens/chunk with 15% overlap, prefers markdown headings as boundaries
- Supports Chrome, Arc, Brave, and Safari on macOS
- Pages table with content-addressable storage (hash → content)
- page_sources tracks which browsers contributed each URL

## Important: Do NOT run automatically

- Never run `planet-capture index`, `planet-capture embed`, or `planet-capture discover` automatically
- Never modify the SQLite database directly
- Write out example commands for the user to run manually
- Index is stored at `~/.planet-capture/index.db`

## Do NOT compile

- Never run `bun build --compile` — it overwrites the shell wrapper and breaks sqlite-vec
- The `planet-capture` bin file is a shell script that runs compiled JS from `dist/`
- `npm run build` compiles TypeScript to `dist/` via `tsc -p tsconfig.build.json`
