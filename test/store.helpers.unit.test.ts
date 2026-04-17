/**
 * Store helper-level unit tests (pure logic, no model/runtime dependency).
 */

import { describe, test, expect } from "vitest";
import {
  homedir,
  resolve,
  getDefaultDbPath,
  _resetProductionModeForTesting,
  normalizeDocid,
  isDocid,
  cleanupOrphanedVectors,
  sanitizeFTS5Term,
} from "../src/store";

// =============================================================================
// Path Utilities
// =============================================================================

describe("Path Utilities", () => {
  test("homedir returns HOME environment variable", () => {
    expect(homedir()).toBe(process.env.HOME || "/tmp");
  });

  test("resolve handles absolute paths", () => {
    expect(resolve("/foo/bar")).toBe("/foo/bar");
    expect(resolve("/foo", "/bar")).toBe("/bar");
  });

  test("resolve handles relative paths", () => {
    const pwd = process.env.PWD || process.cwd();
    expect(resolve("foo")).toBe(`${pwd}/foo`);
    expect(resolve("foo", "bar")).toBe(`${pwd}/foo/bar`);
  });

  test("resolve normalizes . and ..", () => {
    expect(resolve("/foo/bar/./baz")).toBe("/foo/bar/baz");
    expect(resolve("/foo/bar/../baz")).toBe("/foo/baz");
    expect(resolve("/foo/bar/../../baz")).toBe("/baz");
  });

  test("getDefaultDbPath throws in test mode without INDEX_PATH", () => {
    const originalIndexPath = process.env.INDEX_PATH;
    delete process.env.INDEX_PATH;
    // Reset production mode in case another test file set it (bun runs all
    // files in a single process, so module state leaks between files).
    _resetProductionModeForTesting();

    expect(() => getDefaultDbPath()).toThrow("Database path not set");

    if (originalIndexPath) {
      process.env.INDEX_PATH = originalIndexPath;
    }
  });

  test("getDefaultDbPath uses INDEX_PATH when set", () => {
    const originalIndexPath = process.env.INDEX_PATH;
    process.env.INDEX_PATH = "/tmp/test-index.sqlite";

    expect(getDefaultDbPath()).toBe("/tmp/test-index.sqlite");
    expect(getDefaultDbPath("custom")).toBe("/tmp/test-index.sqlite");

    if (originalIndexPath) {
      process.env.INDEX_PATH = originalIndexPath;
    } else {
      delete process.env.INDEX_PATH;
    }
  });

});

// =============================================================================
// Handelize Tests
// =============================================================================

describe("cleanupOrphanedVectors", () => {
  test("returns 0 when vec table exists in schema but sqlite-vec is unavailable", () => {
    const prepare = (sql: string) => {
      if (sql.includes("sqlite_master") && sql.includes("vectors_vec")) {
        return { get: () => ({ name: "vectors_vec" }) };
      }
      if (sql.includes("SELECT 1 FROM vectors_vec LIMIT 0")) {
        return { get: () => { throw new Error("no such module: vec0"); } };
      }
      throw new Error(`Unexpected SQL in test: ${sql}`);
    };

    const db = {
      prepare,
      exec: () => {
        throw new Error("cleanup should not execute vector deletes when sqlite-vec is unavailable");
      },
    } as any;

    expect(cleanupOrphanedVectors(db)).toBe(0);
  });
});

// =============================================================================
// Docid Tests
// =============================================================================

describe("docid utilities", () => {
  test("normalizes docids", () => {
    expect(normalizeDocid("123456")).toBe("123456");
    expect(normalizeDocid("#123456")).toBe("123456");
  });

  test("checks docid validity", () => {
    expect(isDocid("123456")).toBe(true);
    expect(isDocid("#123456")).toBe(true);
    expect(isDocid("bad-id")).toBe(false);
    expect(isDocid("12345")).toBe(false);
  });
});

// =============================================================================
// sanitizeFTS5Term Tests
// =============================================================================

describe("sanitizeFTS5Term", () => {
  test("preserves underscores in snake_case identifiers", () => {
    expect(sanitizeFTS5Term("my_variable")).toBe("my_variable");
    expect(sanitizeFTS5Term("MAX_RETRIES")).toBe("max_retries");
    expect(sanitizeFTS5Term("__init__")).toBe("__init__");
  });

  test("preserves alphanumeric characters", () => {
    expect(sanitizeFTS5Term("hello123")).toBe("hello123");
    expect(sanitizeFTS5Term("test")).toBe("test");
  });

  test("preserves apostrophes for contractions", () => {
    expect(sanitizeFTS5Term("don't")).toBe("don't");
    expect(sanitizeFTS5Term("it's")).toBe("it's");
  });

  test("strips other punctuation", () => {
    expect(sanitizeFTS5Term("hello!")).toBe("hello");
    expect(sanitizeFTS5Term("test@value")).toBe("testvalue");
    expect(sanitizeFTS5Term("a.b")).toBe("ab");
  });

  test("lowercases output", () => {
    expect(sanitizeFTS5Term("Hello")).toBe("hello");
    expect(sanitizeFTS5Term("MY_VAR")).toBe("my_var");
  });

  test("handles unicode letters and numbers", () => {
    expect(sanitizeFTS5Term("café")).toBe("café");
    expect(sanitizeFTS5Term("日本語")).toBe("日本語");
  });
});
