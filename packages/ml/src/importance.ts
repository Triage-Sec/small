/**
 * Pattern importance scoring for ML-aware compression.
 *
 * Port of `delta/pattern_importance.py` to TypeScript.
 */

import type { DiscoveredPattern, TokenSeq } from '@delta-ltsc/sdk';

/**
 * Interface for embedding providers.
 */
export interface EmbeddingProvider {
  /**
   * Get embeddings for a sequence of tokens.
   *
   * @param tokens - Token sequence to embed
   * @returns Promise resolving to embedding vector (Float32Array)
   */
  embed(tokens: TokenSeq): Promise<Float32Array>;

  /**
   * Get embedding dimension.
   */
  dimension(): number;
}

/**
 * Configuration for importance scoring.
 */
export interface ImportanceConfig {
  /**
   * Position decay rate for positional scoring.
   * Higher values = more weight on early positions.
   * @default 2.0
   */
  decayRate?: number;

  /**
   * Context window size for embedding-based scoring.
   * @default 5
   */
  contextWindow?: number;

  /**
   * Weight for positional importance (vs. embedding-based).
   * @default 0.3
   */
  positionalWeight?: number;
}

const DEFAULT_IMPORTANCE_CONFIG: Required<ImportanceConfig> = {
  decayRate: 2.0,
  contextWindow: 5,
  positionalWeight: 0.3,
};

/**
 * Interface for pattern importance scorers.
 */
export interface ImportanceScorer {
  /**
   * Score patterns by importance.
   *
   * Higher scores indicate more important patterns that should be
   * preserved (less aggressively compressed).
   *
   * @param tokens - Original token sequence
   * @param patterns - Discovered patterns to score
   * @returns Promise resolving to importance scores (0-1 range)
   */
  scorePatterns(tokens: TokenSeq, patterns: DiscoveredPattern[]): Promise<number[]>;
}

/**
 * Positional importance scorer.
 *
 * Scores patterns based on their position in the sequence,
 * with earlier positions receiving higher importance (useful for
 * system prompts that typically appear at the start).
 */
export class PositionalImportanceScorer implements ImportanceScorer {
  private decayRate: number;

  constructor(config?: ImportanceConfig) {
    const cfg = { ...DEFAULT_IMPORTANCE_CONFIG, ...config };
    this.decayRate = cfg.decayRate;
  }

  async scorePatterns(tokens: TokenSeq, patterns: DiscoveredPattern[]): Promise<number[]> {
    const n = tokens.length;
    if (n === 0) {
      return patterns.map(() => 0);
    }

    return patterns.map((pattern) => {
      if (pattern.positions.length === 0) {
        return 0;
      }

      // Compute average positional importance across all occurrences
      let totalImportance = 0;
      for (const pos of pattern.positions) {
        // Exponential decay from start of sequence
        const normalizedPos = pos / n;
        const importance = Math.exp(-this.decayRate * normalizedPos);
        totalImportance += importance;
      }

      return totalImportance / pattern.positions.length;
    });
  }
}

/**
 * Embedding-based importance scorer.
 *
 * Uses an embedding model to determine if a pattern appears in
 * diverse semantic contexts (important, should preserve) vs.
 * similar contexts (redundant, safe to compress).
 */
export class EmbeddingImportanceScorer implements ImportanceScorer {
  private provider: EmbeddingProvider;
  private contextWindow: number;

  constructor(provider: EmbeddingProvider, config?: ImportanceConfig) {
    const cfg = { ...DEFAULT_IMPORTANCE_CONFIG, ...config };
    this.provider = provider;
    this.contextWindow = cfg.contextWindow;
  }

  async scorePatterns(tokens: TokenSeq, patterns: DiscoveredPattern[]): Promise<number[]> {
    const tokenArray = Array.isArray(tokens) ? tokens : Array.from(tokens);
    const n = tokenArray.length;

    if (n === 0) {
      return patterns.map(() => 0);
    }

    const scores: number[] = [];

    for (const pattern of patterns) {
      if (pattern.positions.length <= 1) {
        // Single occurrence - can't compute diversity
        scores.push(0.5);
        continue;
      }

      // Extract context windows around each occurrence
      const contextEmbeddings: Float32Array[] = [];

      for (const pos of pattern.positions) {
        const start = Math.max(0, pos - this.contextWindow);
        const end = Math.min(n, pos + pattern.length + this.contextWindow);
        const context = tokenArray.slice(start, end);

        const embedding = await this.provider.embed(context);
        contextEmbeddings.push(embedding);
      }

      // Compute pairwise cosine similarities
      const similarities: number[] = [];
      for (let i = 0; i < contextEmbeddings.length; i++) {
        for (let j = i + 1; j < contextEmbeddings.length; j++) {
          const sim = cosineSimilarity(contextEmbeddings[i], contextEmbeddings[j]);
          similarities.push(sim);
        }
      }

      // Low average similarity = diverse contexts = high importance
      const avgSimilarity =
        similarities.length > 0
          ? similarities.reduce((a, b) => a + b, 0) / similarities.length
          : 0;

      // Convert to importance (invert similarity)
      scores.push(1 - avgSimilarity);
    }

    return scores;
  }
}

/**
 * Combined importance scorer that uses both positional and embedding-based scoring.
 */
export class CombinedImportanceScorer implements ImportanceScorer {
  private positionalScorer: PositionalImportanceScorer;
  private embeddingScorer: EmbeddingImportanceScorer | null;
  private positionalWeight: number;

  constructor(provider?: EmbeddingProvider, config?: ImportanceConfig) {
    const cfg = { ...DEFAULT_IMPORTANCE_CONFIG, ...config };
    this.positionalScorer = new PositionalImportanceScorer(config);
    this.embeddingScorer = provider ? new EmbeddingImportanceScorer(provider, config) : null;
    this.positionalWeight = cfg.positionalWeight;
  }

  async scorePatterns(tokens: TokenSeq, patterns: DiscoveredPattern[]): Promise<number[]> {
    const positionalScores = await this.positionalScorer.scorePatterns(tokens, patterns);

    if (!this.embeddingScorer) {
      return positionalScores;
    }

    const embeddingScores = await this.embeddingScorer.scorePatterns(tokens, patterns);

    // Weighted combination
    return positionalScores.map((posScore, i) => {
      const embScore = embeddingScores[i];
      return this.positionalWeight * posScore + (1 - this.positionalWeight) * embScore;
    });
  }
}

/**
 * Compute cosine similarity between two vectors.
 */
function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Adjust pattern priorities based on importance scores.
 *
 * Patterns with high importance get lower priority (compressed later),
 * patterns with low importance get higher priority (compressed first).
 *
 * @param patterns - Patterns to adjust
 * @param scores - Importance scores from an ImportanceScorer
 * @param threshold - Patterns above this importance threshold get negative priority
 * @returns Patterns with adjusted priorities
 */
export function adjustPrioritiesByImportance(
  patterns: DiscoveredPattern[],
  scores: number[],
  threshold = 0.7
): DiscoveredPattern[] {
  return patterns.map((pattern, i) => {
    const importance = scores[i];

    // High importance = low priority (preserve)
    // Low importance = high priority (compress)
    let priorityAdjustment: number;

    if (importance > threshold) {
      // Very important - negative priority to preserve
      priorityAdjustment = -10 * importance;
    } else {
      // Less important - positive priority to compress
      priorityAdjustment = 10 * (1 - importance);
    }

    return {
      ...pattern,
      // Store adjusted priority (would need to extend type)
      _importanceScore: importance,
      _adjustedPriority: priorityAdjustment,
    } as DiscoveredPattern & { _importanceScore: number; _adjustedPriority: number };
  });
}

/**
 * Filter patterns to only those below an importance threshold.
 *
 * @param patterns - Patterns to filter
 * @param scores - Importance scores
 * @param threshold - Maximum importance score to include
 * @returns Filtered patterns that are safe to compress
 */
export function filterByImportance(
  patterns: DiscoveredPattern[],
  scores: number[],
  threshold = 0.8
): DiscoveredPattern[] {
  return patterns.filter((_, i) => scores[i] < threshold);
}
