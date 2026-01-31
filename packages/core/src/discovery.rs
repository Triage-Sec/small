//! Pattern discovery algorithms.
//!
//! Implements suffix array-based pattern discovery for finding repeated
//! subsequences in token sequences.
//!
//! Port of `delta/discovery_sa.py`.

use crate::suffix_array::{non_overlapping_positions, SuffixArray};
use crate::types::{is_compressible, min_count_for_compressibility, Candidate, Token};
use std::collections::HashMap;

/// Configuration for pattern discovery.
#[derive(Debug, Clone)]
pub struct DiscoveryConfig {
    /// Minimum pattern length to consider
    pub min_length: usize,
    /// Maximum pattern length to consider
    pub max_length: usize,
    /// Minimum occurrences for a pattern to be considered
    pub min_occurrences: usize,
    /// Extra cost per pattern (e.g., length token)
    pub extra_cost: usize,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            min_length: 2,
            max_length: 8,
            min_occurrences: 2,
            extra_cost: 1,
        }
    }
}

/// Discover candidate patterns using suffix array.
///
/// Uses LCP intervals to efficiently find all repeated subsequences.
pub fn discover_candidates(tokens: &[Token], config: &DiscoveryConfig) -> Vec<Candidate> {
    if tokens.len() < config.min_length * 2 {
        return Vec::new();
    }

    let sa = SuffixArray::build(tokens);
    let intervals = sa.lcp_intervals(config.min_length);

    let mut candidates: Vec<Candidate> = Vec::new();
    let mut seen: HashMap<Vec<Token>, usize> = HashMap::new(); // pattern -> candidate index

    for (start_idx, end_idx, lcp_len) in intervals {
        // Number of suffixes in this interval
        let count = end_idx - start_idx + 1;
        if count < config.min_occurrences {
            continue;
        }

        // Extract positions from suffix array
        let positions: Vec<usize> = sa.suffix_array[start_idx..=end_idx].to_vec();

        // Try different lengths up to lcp_len
        for length in config.min_length..=lcp_len.min(config.max_length) {
            // Check if this length could be compressible
            let min_count = min_count_for_compressibility(length, config.extra_cost);
            if count < min_count {
                continue;
            }

            // Extract the pattern
            if positions.is_empty() {
                continue;
            }
            let first_pos = positions[0];
            if first_pos + length > tokens.len() {
                continue;
            }
            let pattern: Vec<Token> = tokens[first_pos..first_pos + length].to_vec();

            // Skip if we've already seen this pattern
            if seen.contains_key(&pattern) {
                continue;
            }

            // Get non-overlapping positions
            let mut sorted_positions = positions.clone();
            sorted_positions.sort_unstable();
            let non_overlapping = non_overlapping_positions(&sorted_positions, length);

            // Check if still compressible after removing overlaps
            if non_overlapping.len() < min_count {
                continue;
            }

            // Verify compressibility
            if !is_compressible(length, non_overlapping.len(), config.extra_cost) {
                continue;
            }

            // Create candidate
            let candidate = Candidate::new(pattern.clone(), non_overlapping);
            seen.insert(pattern, candidates.len());
            candidates.push(candidate);
        }
    }

    // Sort candidates by potential savings (higher first)
    candidates.sort_by(|a, b| {
        let savings_a = compute_potential_savings(a, config.extra_cost);
        let savings_b = compute_potential_savings(b, config.extra_cost);
        savings_b.cmp(&savings_a)
    });

    candidates
}

/// Compute potential savings for a candidate.
fn compute_potential_savings(candidate: &Candidate, extra_cost: usize) -> i64 {
    let count = candidate.positions.len();
    let length = candidate.length;

    if count == 0 {
        return 0;
    }

    let original = (length * count) as i64;
    let compressed = (1 + length + count + extra_cost) as i64; // meta-token + definition + count replacements + extra
    (original - compressed).max(0)
}

/// Discover patterns optimized for hierarchical compression.
///
/// Discovers patterns at multiple granularities, starting with longer
/// patterns that may contain shorter patterns.
pub fn discover_hierarchical(
    tokens: &[Token],
    config: &DiscoveryConfig,
    depth: usize,
) -> Vec<Vec<Candidate>> {
    let mut all_candidates: Vec<Vec<Candidate>> = Vec::new();
    let current_tokens = tokens.to_vec();

    for _level in 0..depth {
        let candidates = discover_candidates(&current_tokens, config);
        if candidates.is_empty() {
            break;
        }

        all_candidates.push(candidates);

        // For hierarchical, we would apply compression and discover again
        // For now, just return the first level
        break;
    }

    all_candidates
}

/// Fast hash-based pattern discovery for specific lengths.
///
/// More efficient than suffix array for when you know the exact lengths
/// you're looking for.
pub fn discover_fixed_length(
    tokens: &[Token],
    length: usize,
    extra_cost: usize,
) -> Vec<Candidate> {
    if tokens.len() < length {
        return Vec::new();
    }

    let min_count = min_count_for_compressibility(length, extra_cost);

    // Count occurrences of each n-gram
    let mut pattern_positions: HashMap<Vec<Token>, Vec<usize>> = HashMap::new();

    for i in 0..=tokens.len() - length {
        let pattern: Vec<Token> = tokens[i..i + length].to_vec();
        pattern_positions.entry(pattern).or_default().push(i);
    }

    // Filter and create candidates
    let mut candidates = Vec::new();

    for (pattern, positions) in pattern_positions {
        if positions.len() < min_count {
            continue;
        }

        let non_overlapping = non_overlapping_positions(&positions, length);
        if non_overlapping.len() < min_count {
            continue;
        }

        if !is_compressible(length, non_overlapping.len(), extra_cost) {
            continue;
        }

        candidates.push(Candidate::new(pattern, non_overlapping));
    }

    candidates.sort_by(|a, b| {
        let savings_a = compute_potential_savings(a, extra_cost);
        let savings_b = compute_potential_savings(b, extra_cost);
        savings_b.cmp(&savings_a)
    });

    candidates
}

/// Deduplicate candidates that have the same subsequence.
pub fn deduplicate_candidates(candidates: Vec<Candidate>) -> Vec<Candidate> {
    let mut seen: HashMap<Vec<Token>, usize> = HashMap::new();
    let mut result: Vec<Candidate> = Vec::new();

    for candidate in candidates {
        if let Some(&idx) = seen.get(&candidate.subsequence) {
            // Merge positions
            let existing: &mut Candidate = &mut result[idx];
            let mut all_positions: Vec<usize> = existing.positions.clone();
            all_positions.extend(&candidate.positions);
            all_positions.sort_unstable();
            all_positions.dedup();
            existing.positions = all_positions;
        } else {
            seen.insert(candidate.subsequence.clone(), result.len());
            result.push(candidate);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discover_empty() {
        let config = DiscoveryConfig::default();
        let candidates = discover_candidates(&[], &config);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_discover_too_short() {
        let config = DiscoveryConfig::default();
        let candidates = discover_candidates(&[1, 2], &config);
        assert!(candidates.is_empty());
    }

    #[test]
    fn test_discover_repeated_pattern() {
        let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
        let config = DiscoveryConfig {
            min_length: 2,
            max_length: 4,
            min_occurrences: 2,
            extra_cost: 1,
        };

        let candidates = discover_candidates(&tokens, &config);

        // Should find at least the [1, 2, 3] pattern
        assert!(!candidates.is_empty());

        // Check that all candidates are compressible
        for cand in &candidates {
            assert!(is_compressible(
                cand.length,
                cand.positions.len(),
                config.extra_cost
            ));
        }
    }

    #[test]
    fn test_discover_fixed_length() {
        let tokens = vec![1, 2, 1, 2, 1, 2, 1, 2, 1, 2];
        let candidates = discover_fixed_length(&tokens, 2, 1);

        // Should find [1, 2] pattern
        assert!(!candidates.is_empty());
        let found = candidates.iter().any(|c| c.subsequence == vec![1, 2]);
        assert!(found);
    }

    #[test]
    fn test_deduplicate_candidates() {
        let c1 = Candidate::new(vec![1, 2], vec![0, 4, 8]);
        let c2 = Candidate::new(vec![1, 2], vec![2, 6, 10]);
        let c3 = Candidate::new(vec![3, 4], vec![1, 5]);

        let result = deduplicate_candidates(vec![c1, c2, c3]);

        assert_eq!(result.len(), 2);

        // First candidate should have merged positions
        let merged = result.iter().find(|c| c.subsequence == vec![1, 2]).unwrap();
        assert_eq!(merged.positions.len(), 6);
    }

    #[test]
    fn test_potential_savings() {
        // Length 3, count 5: original = 15, compressed = 1 + 3 + 5 + 1 = 10
        let candidate = Candidate::new(vec![1, 2, 3], vec![0, 4, 8, 12, 16]);
        let savings = compute_potential_savings(&candidate, 1);
        assert_eq!(savings, 5);
    }

    #[test]
    fn test_non_overlapping_filtering() {
        // Overlapping positions should be filtered
        let tokens = vec![1, 2, 1, 2, 1, 2]; // Positions 0, 2, 4 for [1, 2]
        let candidates = discover_fixed_length(&tokens, 2, 1);

        for cand in &candidates {
            // Verify positions don't overlap
            let mut prev_end = 0;
            for &pos in &cand.positions {
                assert!(pos >= prev_end, "Positions should not overlap");
                prev_end = pos + cand.length;
            }
        }
    }
}
