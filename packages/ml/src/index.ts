/**
 * Delta LTSC ML - Machine Learning Features
 *
 * Optional ML features for enhanced compression quality:
 * - Pattern importance scoring
 * - Quality prediction
 * - Adaptive region detection
 *
 * @packageDocumentation
 */

// Importance scoring
export {
  type EmbeddingProvider,
  type ImportanceConfig,
  type ImportanceScorer,
  PositionalImportanceScorer,
  EmbeddingImportanceScorer,
  CombinedImportanceScorer,
  adjustPrioritiesByImportance,
  filterByImportance,
} from './importance.js';

// Quality prediction
export {
  type QualityConfig,
  type QualityPrediction,
  type QualityFeatures,
  type QualityPredictor,
  HeuristicQualityPredictor,
  EmbeddingQualityPredictor,
  createQualityPredictor,
} from './quality.js';

// Region detection
export {
  RegionType,
  type Region,
  type RegionConfig,
  detectRegions,
  detectRegionsHeuristic,
  filterPatternsByRegion,
  getRegionCompressionSettings,
} from './regions.js';

// Version
export const VERSION = '0.1.0';
