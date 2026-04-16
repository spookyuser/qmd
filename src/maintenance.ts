/**
 * Maintenance - Database cleanup operations for QMD.
 *
 * Wraps low-level store operations that the CLI needs for housekeeping.
 * Takes an internal Store in the constructor — allowed to access DB directly.
 */

import type { Store } from "./store.js";
import {
  vacuumDatabase,
  cleanupOrphanedContent,
  cleanupOrphanedVectors,
  deleteLLMCache,
  deleteInactivePages,
  clearAllEmbeddings,
} from "./store.js";

export class Maintenance {
  private store: Store;

  constructor(store: Store) {
    this.store = store;
  }

  /** Run VACUUM on the SQLite database to reclaim space */
  vacuum(): void {
    vacuumDatabase(this.store.db);
  }

  /** Remove content rows that are no longer referenced by any document */
  cleanupOrphanedContent(): number {
    return cleanupOrphanedContent(this.store.db);
  }

  /** Remove vector embeddings for content that no longer exists */
  cleanupOrphanedVectors(): number {
    return cleanupOrphanedVectors(this.store.db);
  }

  /** Clear the LLM response cache (query expansion, reranking) */
  clearLLMCache(): number {
    return deleteLLMCache(this.store.db);
  }

  /** Delete pages marked as inactive */
  deleteInactivePages(): number {
    return deleteInactivePages(this.store.db);
  }

  /** Clear all vector embeddings (forces re-embedding) */
  clearEmbeddings(): void {
    clearAllEmbeddings(this.store.db);
  }
}
