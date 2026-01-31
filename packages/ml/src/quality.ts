/**
 * Quality prediction for compression validation.
 *
 * Predicts whether a compressed sequence will maintain
 * sufficient quality for transformer models to learn from.
 *
 * Port of `delta/quality_predictor.py` to TypeScript.
 */

import type { CompressionResult } from '@delta-ltsc/sdk';
import type { EmbeddingProvider } from './importance.js';

/**
 * Configuration for quality prediction.
 */
export interface QualityConfig {
  /**
   * Maximum acceptable compression ratio.
   * @default 0.5
   */
  maxCompressionRatio?: number;

  /**
   * Maximum acceptable dictionary overhead ratio.
   * @default 0.3
   */
  maxDictionaryOverhead?: number;

  /**
   * Minimum embedding similarity between original and compressed.
   * @default 0.7
   */
  minEmbeddingSimilarity?: number;

  /**
   * Maximum acceptable token diversity reduction.
   * @default 0.4
   */
  maxDiversityReduction?: number;
}

const DEFAULT_QUALITY_CONFIG: Required<QualityConfig> = {
  maxCompressionRatio: 0.5,
  maxDictionaryOverhead: 0.3,
  minEmbeddingSimilarity: 0.7,
  maxDiversityReduction: 0.4,
};

/**
 * Result of quality prediction.
 */
export interface QualityPrediction {
  /**
   * Overall quality score (0-1, higher is better).
   */
  score: number;

  /**
   * Whether the compression passes quality threshold.
   */
  acceptable: boolean;

  /**
   * Probability of quality degradation.
   */
  degradationProbability: number;

  /**
   * Detailed feature scores.
   */
  features: QualityFeatures;

  /**
   * Recommendation for how to proceed.
   */
  recommendation: 'accept' | 'retry_conservative' | 'skip_compression';
}

/**
 * Feature scores used in quality prediction.
 */
export interface QualityFeatures {
  /**
   * Compression ratio feature (lower is more aggressive).
   */
  compressionRatio: number;

  /**
   * Dictionary overhead ratio.
   */
  dictionaryOverhead: number;

  /**
   * Token diversity change (0-1, 0 = no change, 1 = complete loss).
   */
  diversityReduction: number;

  /**
   * Average pattern length feature.
   */
  averagePatternLength: number;

  /**
   * Pattern count feature.
   */
  patternCount: number;

  /**
   * Embedding similarity (if available).
   */
  embeddingSimilarity?: number;
}

/**
 * Quality predictor interface.
 */
export interface QualityPredictor {
  /**
   * Predict quality of compressed output.
   *
   * @param result - Compression result to evaluate
   * @returns Promise resolving to quality prediction
   */
  predict(result: CompressionResult): Promise<QualityPrediction>;
}

/**
 * Heuristic-based quality predictor.
 *
 * Uses a combination of handcrafted features and thresholds
 * to predict compression quality.
 */
export class HeuristicQualityPredictor implements QualityPredictor {
  private config: Required<QualityConfig>;

  constructor(config?: QualityConfig) {
    this.config = { ...DEFAULT_QUALITY_CONFIG, ...config };
  }

  async predict(result: CompressionResult): Promise<QualityPrediction> {
    const features = this.extractFeatures(result);
    const score = this.computeScore(features);
    const degradationProbability = this.computeDegradationProbability(features);
    const acceptable = score >= 0.6 && degradationProbability < 0.3;

    let recommendation: QualityPrediction['recommendation'];
    if (acceptable) {
      recommendation = 'accept';
    } else if (degradationProbability < 0.5) {
      recommendation = 'retry_conservative';
    } else {
      recommendation = 'skip_compression';
    }

    return {
      score,
      acceptable,
      degradationProbability,
      features,
      recommendation,
    };
  }

  /**
   * Extract quality features from compression result.
   */
  private extractFeatures(result: CompressionResult): QualityFeatures {
    // Compression ratio
    const compressionRatio = result.compressionRatio;

    // Dictionary overhead
    const dictionaryOverhead =
      result.originalLength > 0
        ? result.dictionaryTokens.length / result.originalLength
        : 0;

    // Token diversity
    const originalDiversity = new Set(result.originalTokens).size;
    const compressedDiversity = new Set(result.serializedTokens).size;
    const diversityReduction =
      originalDiversity > 0 ? 1 - compressedDiversity / originalDiversity : 0;

    // Pattern statistics
    const patternCount = result.dictionaryMap.size;
    let totalPatternLength = 0;
    for (const [, pattern] of result.dictionaryMap) {
      totalPatternLength += pattern.length;
    }
    const averagePatternLength = patternCount > 0 ? totalPatternLength / patternCount : 0;

    return {
      compressionRatio,
      dictionaryOverhead,
      diversityReduction,
      averagePatternLength,
      patternCount,
    };
  }

  /**
   * Compute overall quality score from features.
   */
  private computeScore(features: QualityFeatures): number {
    let score = 1.0;

    // Penalize extreme compression
    if (features.compressionRatio < this.config.maxCompressionRatio) {
      const penalty = (this.config.maxCompressionRatio - features.compressionRatio) * 0.5;
      score -= penalty;
    }

    // Penalize high dictionary overhead
    if (features.dictionaryOverhead > this.config.maxDictionaryOverhead) {
      const penalty = (features.dictionaryOverhead - this.config.maxDictionaryOverhead) * 0.3;
      score -= penalty;
    }

    // Penalize diversity loss
    if (features.diversityReduction > this.config.maxDiversityReduction) {
      const penalty = (features.diversityReduction - this.config.maxDiversityReduction) * 0.4;
      score -= penalty;
    }

    // Bonus for reasonable pattern lengths
    if (features.averagePatternLength >= 3 && features.averagePatternLength <= 6) {
      score += 0.1;
    }

    // Use embedding similarity if available
    if (features.embeddingSimilarity !== undefined) {
      if (features.embeddingSimilarity < this.config.minEmbeddingSimilarity) {
        score -= (this.config.minEmbeddingSimilarity - features.embeddingSimilarity) * 0.5;
      }
    }

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Compute probability of quality degradation.
   */
  private computeDegradationProbability(features: QualityFeatures): number {
    let prob = 0;

    // Very aggressive compression increases risk
    if (features.compressionRatio < 0.4) {
      prob += 0.3;
    } else if (features.compressionRatio < 0.5) {
      prob += 0.15;
    }

    // High dictionary overhead increases risk
    if (features.dictionaryOverhead > 0.3) {
      prob += 0.2;
    }

    // Large diversity reduction increases risk
    if (features.diversityReduction > 0.3) {
      prob += 0.25;
    } else if (features.diversityReduction > 0.2) {
      prob += 0.1;
    }

    // Very short patterns are risky
    if (features.averagePatternLength < 2.5) {
      prob += 0.15;
    }

    // Low embedding similarity is a strong signal
    if (features.embeddingSimilarity !== undefined && features.embeddingSimilarity < 0.7) {
      prob += 0.3 * (0.7 - features.embeddingSimilarity);
    }

    return Math.max(0, Math.min(1, prob));
  }
}

/**
 * Embedding-enhanced quality predictor.
 *
 * Adds embedding similarity comparison to the heuristic predictor
 * for more accurate quality assessment.
 */
export class EmbeddingQualityPredictor implements QualityPredictor {
  private provider: EmbeddingProvider;
  private heuristicPredictor: HeuristicQualityPredictor;
  private config: Required<QualityConfig>;

  constructor(provider: EmbeddingProvider, config?: QualityConfig) {
    this.provider = provider;
    this.heuristicPredictor = new HeuristicQualityPredictor(config);
    this.config = { ...DEFAULT_QUALITY_CONFIG, ...config };
  }

  async predict(result: CompressionResult): Promise<QualityPrediction> {
    // Get base prediction
    const basePrediction = await this.heuristicPredictor.predict(result);

    // Compute embedding similarity
    const originalEmbedding = await this.provider.embed(result.originalTokens);
    const compressedEmbedding = await this.provider.embed(result.serializedTokens);
    const similarity = this.cosineSimilarity(originalEmbedding, compressedEmbedding);

    // Update features with embedding similarity
    const features: QualityFeatures = {
      ...basePrediction.features,
      embeddingSimilarity: similarity,
    };

    // Recompute scores with embedding
    let score = basePrediction.score;
    let degradationProbability = basePrediction.degradationProbability;

    if (similarity < this.config.minEmbeddingSimilarity) {
      const penalty = (this.config.minEmbeddingSimilarity - similarity) * 0.3;
      score -= penalty;
      degradationProbability += 0.2;
    } else {
      // High similarity is a good signal
      score += (similarity - this.config.minEmbeddingSimilarity) * 0.1;
    }

    score = Math.max(0, Math.min(1, score));
    degradationProbability = Math.max(0, Math.min(1, degradationProbability));

    const acceptable = score >= 0.6 && degradationProbability < 0.3;

    let recommendation: QualityPrediction['recommendation'];
    if (acceptable) {
      recommendation = 'accept';
    } else if (degradationProbability < 0.5) {
      recommendation = 'retry_conservative';
    } else {
      recommendation = 'skip_compression';
    }

    return {
      score,
      acceptable,
      degradationProbability,
      features,
      recommendation,
    };
  }

  private cosineSimilarity(a: Float32Array, b: Float32Array): number {
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
}

/**
 * Create a quality predictor.
 *
 * @param provider - Optional embedding provider for enhanced prediction
 * @param config - Quality configuration
 * @returns Quality predictor instance
 */
export function createQualityPredictor(
  provider?: EmbeddingProvider,
  config?: QualityConfig
): QualityPredictor {
  if (provider) {
    return new EmbeddingQualityPredictor(provider, config);
  }
  return new HeuristicQualityPredictor(config);
}
