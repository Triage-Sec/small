/**
 * Delta LTSC SDK - Lossless Token Sequence Compression
 *
 * A TypeScript SDK for compressing LLM token sequences to reduce
 * inference costs and context length requirements.
 *
 * @packageDocumentation
 */

// Main compression API
export { compress, discoverPatterns } from './compress.js';
export { decompress, extractDictionary, extractBody, isCompressed } from './decompress.js';

// Streaming API
export {
  createStreamingCompressor,
  compressStream,
  processInChunks,
  type StreamingCompressor,
} from './streaming.js';

// Worker API
export {
  createWorkerPool,
  compressInWorker,
  decompressInWorker,
  type WorkerPool,
} from './worker.js';

// Configuration
export {
  type CompressionConfig,
  type DecompressionConfig,
  type SelectionMode,
  DEFAULT_CONFIG,
  mergeConfig,
} from './config.js';

// Types
export {
  type Token,
  type TokenSeq,
  type TokenInput,
  type CompressionResult,
  type CompressionMetrics,
  type DiscoveredPattern,
  normalizeTokens,
  isTokenSeq,
} from './types.js';

// Static dictionaries
export {
  loadStaticDictionary,
  createStaticDictionary,
  listStaticDictionaries,
  isBuiltinDictionary,
  type StaticDictionary,
  type StaticDictionaryId,
  STATIC_DICTIONARIES,
} from './dictionaries/index.js';

// WASM initialization
export {
  initWasm,
  initWasmFromModule,
  initWasmFromBytes,
  isWasmInitialized,
  getWasmVersion,
} from './wasm/loader.js';

// Re-export version from package.json at runtime
export const VERSION = '0.1.0';
