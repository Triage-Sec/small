/**
 * Core type definitions for Delta LTSC SDK.
 */

/**
 * Token type - represented as unsigned 32-bit integers.
 */
export type Token = number;

/**
 * A sequence of tokens (array of u32).
 */
export type TokenSeq = readonly number[] | number[] | Uint32Array;

/**
 * Result of a compression operation.
 */
export interface CompressionResult {
  /**
   * Original tokens that were compressed.
   */
  readonly originalTokens: readonly number[];

  /**
   * Fully serialized output (dictionary + body).
   */
  readonly serializedTokens: readonly number[];

  /**
   * Dictionary tokens only.
   */
  readonly dictionaryTokens: readonly number[];

  /**
   * Body tokens only (with meta-token references).
   */
  readonly bodyTokens: readonly number[];

  /**
   * Original sequence length.
   */
  readonly originalLength: number;

  /**
   * Compressed sequence length.
   */
  readonly compressedLength: number;

  /**
   * Compression ratio (compressed/original).
   * Values < 1.0 indicate compression savings.
   */
  readonly compressionRatio: number;

  /**
   * Mapping from meta-token to its expansion.
   */
  readonly dictionaryMap: ReadonlyMap<number, readonly number[]>;

  /**
   * Static dictionary ID if one was used.
   */
  readonly staticDictionaryId?: string;

  /**
   * Performance metrics from the compression operation.
   */
  readonly metrics?: CompressionMetrics;
}

/**
 * Performance metrics from compression.
 */
export interface CompressionMetrics {
  /**
   * Time spent in pattern discovery (ms).
   */
  readonly discoveryTimeMs: number;

  /**
   * Time spent in pattern selection (ms).
   */
  readonly selectionTimeMs: number;

  /**
   * Time spent in serialization (ms).
   */
  readonly serializationTimeMs: number;

  /**
   * Total compression time (ms).
   */
  readonly totalTimeMs: number;

  /**
   * Peak memory usage (bytes).
   */
  readonly peakMemoryBytes: number;
}

/**
 * A discovered pattern with its positions.
 */
export interface DiscoveredPattern {
  /**
   * The token pattern.
   */
  readonly pattern: readonly number[];

  /**
   * Length of the pattern.
   */
  readonly length: number;

  /**
   * Non-overlapping positions where the pattern occurs.
   */
  readonly positions: readonly number[];

  /**
   * Number of occurrences.
   */
  readonly count: number;
}

/**
 * Input that can be used for compression/decompression.
 */
export type TokenInput = TokenSeq;

/**
 * Normalize any token input to Uint32Array.
 */
export function normalizeTokens(tokens: TokenInput): Uint32Array {
  if (tokens instanceof Uint32Array) {
    return tokens;
  }
  return new Uint32Array(tokens);
}

/**
 * Check if a value is a valid token sequence.
 */
export function isTokenSeq(value: unknown): value is TokenSeq {
  if (value instanceof Uint32Array) {
    return true;
  }
  if (Array.isArray(value)) {
    return value.every(
      (v) => typeof v === 'number' && Number.isInteger(v) && v >= 0
    );
  }
  return false;
}
