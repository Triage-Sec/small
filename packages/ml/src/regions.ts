/**
 * Adaptive region detection for context-aware compression.
 *
 * Detects semantic regions in token sequences (system prompts, user input,
 * injected context) and applies different compression strategies.
 *
 * Port of `delta/adaptive.py` region detection to TypeScript.
 */

import type { TokenSeq, DiscoveredPattern } from '@delta-ltsc/sdk';

/**
 * Region types with different compression strategies.
 */
export enum RegionType {
  /** System instructions - minimal compression */
  SYSTEM = 'system',
  /** User input - moderate compression */
  USER = 'user',
  /** Injected context - aggressive compression */
  CONTEXT = 'context',
  /** Code blocks - moderate compression */
  CODE = 'code',
  /** Unknown region - default compression */
  UNKNOWN = 'unknown',
}

/**
 * A detected region in the token sequence.
 */
export interface Region {
  /** Region type */
  type: RegionType;
  /** Start position (inclusive) */
  start: number;
  /** End position (exclusive) */
  end: number;
  /** Compression retention target (0-1) */
  retention: number;
}

/**
 * Configuration for region detection.
 */
export interface RegionConfig {
  /** Token patterns that mark system region start */
  systemMarkers?: number[][];
  /** Token patterns that mark user region start */
  userMarkers?: number[][];
  /** Token patterns that mark context region start */
  contextMarkers?: number[][];
  /** Token patterns that mark code region start */
  codeMarkers?: number[][];
  /** Retention targets for each region type */
  retentionTargets?: Partial<Record<RegionType, number>>;
}

const DEFAULT_RETENTION_TARGETS: Record<RegionType, number> = {
  [RegionType.SYSTEM]: 0.98,
  [RegionType.USER]: 0.85,
  [RegionType.CONTEXT]: 0.6,
  [RegionType.CODE]: 0.8,
  [RegionType.UNKNOWN]: 0.75,
};

/**
 * Detect regions in a token sequence.
 *
 * Uses marker patterns to identify boundaries between different
 * semantic regions in the input.
 *
 * @param tokens - Token sequence to analyze
 * @param config - Region detection configuration
 * @returns Array of detected regions
 */
export function detectRegions(tokens: TokenSeq, config?: RegionConfig): Region[] {
  const tokenArray = Array.isArray(tokens) ? tokens : Array.from(tokens);
  const n = tokenArray.length;

  if (n === 0) {
    return [];
  }

  const retentionTargets = {
    ...DEFAULT_RETENTION_TARGETS,
    ...config?.retentionTargets,
  };

  // Find all marker positions
  const markers: { pos: number; type: RegionType }[] = [];

  const systemMarkers = config?.systemMarkers ?? DEFAULT_SYSTEM_MARKERS;
  const userMarkers = config?.userMarkers ?? DEFAULT_USER_MARKERS;
  const contextMarkers = config?.contextMarkers ?? DEFAULT_CONTEXT_MARKERS;
  const codeMarkers = config?.codeMarkers ?? DEFAULT_CODE_MARKERS;

  // Search for markers
  for (const pattern of systemMarkers) {
    for (const pos of findPattern(tokenArray, pattern)) {
      markers.push({ pos, type: RegionType.SYSTEM });
    }
  }

  for (const pattern of userMarkers) {
    for (const pos of findPattern(tokenArray, pattern)) {
      markers.push({ pos, type: RegionType.USER });
    }
  }

  for (const pattern of contextMarkers) {
    for (const pos of findPattern(tokenArray, pattern)) {
      markers.push({ pos, type: RegionType.CONTEXT });
    }
  }

  for (const pattern of codeMarkers) {
    for (const pos of findPattern(tokenArray, pattern)) {
      markers.push({ pos, type: RegionType.CODE });
    }
  }

  // Sort markers by position
  markers.sort((a, b) => a.pos - b.pos);

  // Build regions from markers
  const regions: Region[] = [];

  if (markers.length === 0) {
    // No markers found - treat entire sequence as unknown
    regions.push({
      type: RegionType.UNKNOWN,
      start: 0,
      end: n,
      retention: retentionTargets[RegionType.UNKNOWN],
    });
  } else {
    // Add initial region if first marker is not at start
    if (markers[0].pos > 0) {
      regions.push({
        type: RegionType.UNKNOWN,
        start: 0,
        end: markers[0].pos,
        retention: retentionTargets[RegionType.UNKNOWN],
      });
    }

    // Add regions from markers
    for (let i = 0; i < markers.length; i++) {
      const marker = markers[i];
      const nextPos = i < markers.length - 1 ? markers[i + 1].pos : n;

      regions.push({
        type: marker.type,
        start: marker.pos,
        end: nextPos,
        retention: retentionTargets[marker.type],
      });
    }
  }

  return regions;
}

/**
 * Heuristic-based region detection without explicit markers.
 *
 * Uses statistical features to guess region boundaries.
 */
export function detectRegionsHeuristic(tokens: TokenSeq): Region[] {
  const tokenArray = Array.isArray(tokens) ? tokens : Array.from(tokens);
  const n = tokenArray.length;

  if (n === 0) {
    return [];
  }

  // Simple heuristic: first 10% is likely system, rest is context
  const systemEnd = Math.floor(n * 0.1);

  return [
    {
      type: RegionType.SYSTEM,
      start: 0,
      end: Math.max(systemEnd, 1),
      retention: DEFAULT_RETENTION_TARGETS[RegionType.SYSTEM],
    },
    {
      type: RegionType.CONTEXT,
      start: Math.max(systemEnd, 1),
      end: n,
      retention: DEFAULT_RETENTION_TARGETS[RegionType.CONTEXT],
    },
  ];
}

/**
 * Filter patterns based on region retention targets.
 *
 * Removes patterns that would compress high-retention regions.
 *
 * @param patterns - Discovered patterns
 * @param regions - Detected regions
 * @param tokens - Original token sequence
 * @returns Filtered patterns respecting region constraints
 */
export function filterPatternsByRegion(
  patterns: DiscoveredPattern[],
  regions: Region[],
  _tokens: TokenSeq
): DiscoveredPattern[] {
  if (regions.length === 0) {
    return patterns;
  }

  return patterns.map((pattern) => {
    // Filter positions based on region retention
    const filteredPositions = pattern.positions.filter((pos) => {
      const region = findRegionAtPosition(regions, pos);
      if (!region) return true;

      // Keep position if region allows compression (low retention)
      // High retention regions should preserve patterns
      return region.retention < 0.9;
    });

    return {
      ...pattern,
      positions: filteredPositions,
      count: filteredPositions.length,
    };
  }).filter((pattern) => pattern.positions.length >= 2);
}

/**
 * Find the region containing a position.
 */
function findRegionAtPosition(regions: Region[], pos: number): Region | null {
  for (const region of regions) {
    if (pos >= region.start && pos < region.end) {
      return region;
    }
  }
  return null;
}

/**
 * Find all occurrences of a pattern in tokens.
 */
function findPattern(tokens: number[], pattern: number[]): number[] {
  const positions: number[] = [];
  const n = tokens.length;
  const m = pattern.length;

  for (let i = 0; i <= n - m; i++) {
    let match = true;
    for (let j = 0; j < m; j++) {
      if (tokens[i + j] !== pattern[j]) {
        match = false;
        break;
      }
    }
    if (match) {
      positions.push(i);
    }
  }

  return positions;
}

// Default marker patterns (tiktoken cl100k_base token IDs for common markers)
// These are approximate - actual token IDs depend on the tokenizer

/** Default system region markers */
const DEFAULT_SYSTEM_MARKERS: number[][] = [
  // [SYSTEM], <<SYS>>, etc.
  [58, 71905, 60],     // [SYSTEM]
  [27, 27, 71905, 2083, 2083], // <<SYS>>
];

/** Default user region markers */
const DEFAULT_USER_MARKERS: number[][] = [
  // [USER], [INST], etc.
  [58, 35295, 60],     // [USER]
  [58, 96746, 60],     // [INST]
];

/** Default context region markers */
const DEFAULT_CONTEXT_MARKERS: number[][] = [
  // [CONTEXT], [DOC], etc.
  [58, 94034, 60],     // [CONTEXT]
  [58, 44184, 60],     // [DOC]
];

/** Default code region markers */
const DEFAULT_CODE_MARKERS: number[][] = [
  // ```python, ```typescript, etc.
  [74694, 12958],      // ```python
  [74694, 92459],      // ```typescript
  [74694, 13210],      // ```javascript
];

/**
 * Get compression settings for a region type.
 */
export function getRegionCompressionSettings(regionType: RegionType): {
  maxSubsequenceLength: number;
  minOccurrences: number;
} {
  switch (regionType) {
    case RegionType.SYSTEM:
      return { maxSubsequenceLength: 4, minOccurrences: 5 };
    case RegionType.USER:
      return { maxSubsequenceLength: 6, minOccurrences: 3 };
    case RegionType.CONTEXT:
      return { maxSubsequenceLength: 10, minOccurrences: 2 };
    case RegionType.CODE:
      return { maxSubsequenceLength: 6, minOccurrences: 3 };
    default:
      return { maxSubsequenceLength: 8, minOccurrences: 3 };
  }
}
