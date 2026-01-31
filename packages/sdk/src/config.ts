/**
 * TypeScript configuration types for Delta LTSC.
 */

import type { StaticDictionary } from './dictionaries/index.js';

/**
 * Selection mode for pattern selection.
 */
export type SelectionMode = 'greedy' | 'optimal' | 'beam';

/**
 * Configuration for compression operations.
 */
export interface CompressionConfig {
  /**
   * Minimum pattern length to consider.
   * @default 2
   */
  minSubsequenceLength?: number;

  /**
   * Maximum pattern length to consider.
   * @default 8
   */
  maxSubsequenceLength?: number;

  /**
   * Selection algorithm to use.
   * - "greedy": Fast, good results for most inputs
   * - "optimal": Uses DP for theoretically optimal selection (slower)
   * - "beam": Beam search compromise between speed and quality
   * @default "greedy"
   */
  selectionMode?: SelectionMode;

  /**
   * Beam width for beam search selection.
   * Only used when selectionMode is "beam".
   * @default 8
   */
  beamWidth?: number;

  /**
   * Enable hierarchical compression (meta-tokens can reference other meta-tokens).
   * @default true
   */
  hierarchicalEnabled?: boolean;

  /**
   * Maximum depth for hierarchical compression.
   * @default 3
   */
  hierarchicalMaxDepth?: number;

  /**
   * Static dictionary to use for pre-defined patterns.
   * Can be a built-in dictionary ID or a custom dictionary.
   */
  staticDictionary?: string | StaticDictionary;

  /**
   * Input size threshold above which streaming mode is automatically enabled.
   * @default 50000
   */
  streamingThreshold?: number;

  /**
   * Maximum memory usage for WASM in MB.
   * @default 256
   */
  maxMemoryMb?: number;

  /**
   * Enable round-trip verification after compression.
   * When enabled, decompresses the result and verifies it matches the original.
   * @default false
   */
  verify?: boolean;

  /**
   * Dictionary start delimiter token.
   * @default 0xFFFFFFF0
   */
  dictStartToken?: number;

  /**
   * Dictionary end delimiter token.
   * @default 0xFFFFFFF1
   */
  dictEndToken?: number;

  /**
   * Starting meta-token ID for new dictionary entries.
   * @default 0xFFFF0000
   */
  nextMetaToken?: number;
}

/**
 * Configuration for decompression operations.
 */
export interface DecompressionConfig {
  /**
   * Dictionary start delimiter token.
   * Must match the token used during compression.
   */
  dictStartToken?: number;

  /**
   * Dictionary end delimiter token.
   * Must match the token used during compression.
   */
  dictEndToken?: number;
}

/**
 * Default configuration values.
 */
export const DEFAULT_CONFIG: Required<
  Omit<CompressionConfig, 'staticDictionary'>
> = {
  minSubsequenceLength: 2,
  maxSubsequenceLength: 8,
  selectionMode: 'greedy',
  beamWidth: 8,
  hierarchicalEnabled: true,
  hierarchicalMaxDepth: 3,
  streamingThreshold: 50000,
  maxMemoryMb: 256,
  verify: false,
  dictStartToken: 0xfffffff0,
  dictEndToken: 0xfffffff1,
  nextMetaToken: 0xffff0000,
} as const;

/**
 * Merge user config with defaults.
 */
export function mergeConfig(
  userConfig?: CompressionConfig
): Required<Omit<CompressionConfig, 'staticDictionary'>> & {
  staticDictionary?: string | StaticDictionary;
} {
  return {
    ...DEFAULT_CONFIG,
    ...userConfig,
  };
}

/**
 * Convert SDK config to WASM config format.
 */
export function toWasmConfig(config: CompressionConfig): Record<string, unknown> {
  return {
    min_subsequence_length: config.minSubsequenceLength,
    max_subsequence_length: config.maxSubsequenceLength,
    selection_mode: config.selectionMode,
    beam_width: config.beamWidth,
    hierarchical_enabled: config.hierarchicalEnabled,
    hierarchical_max_depth: config.hierarchicalMaxDepth,
    verify: config.verify,
    dict_start_token: config.dictStartToken,
    dict_end_token: config.dictEndToken,
    next_meta_token: config.nextMetaToken,
  };
}
