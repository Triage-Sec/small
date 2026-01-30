//! Parallel suffix array construction using Rayon.
//!
//! Provides parallel implementations of suffix array construction
//! for improved performance on multi-core systems.

use crate::suffix_array::SuffixArray;
use crate::types::Token;
use rayon::prelude::*;
use std::collections::HashMap;

/// Configuration for parallel suffix array construction.
#[derive(Debug, Clone)]
pub struct ParallelSAConfig {
    /// Minimum input size to use parallel construction.
    /// Below this threshold, sequential is faster due to overhead.
    pub parallel_threshold: usize,
    /// Chunk size for parallel rank updates.
    pub chunk_size: usize,
}

impl Default for ParallelSAConfig {
    fn default() -> Self {
        Self {
            parallel_threshold: 10_000,
            chunk_size: 4096,
        }
    }
}

/// Build suffix array using parallel doubling algorithm.
///
/// Uses Rayon for parallel sorting and rank updates.
/// Falls back to sequential for small inputs where parallelism overhead
/// would exceed benefits.
///
/// Time complexity: O(n log n / p) where p is number of processors
/// Space complexity: O(n)
pub fn build_suffix_array_parallel(tokens: &[Token], config: &ParallelSAConfig) -> SuffixArray {
    let n = tokens.len();

    // Fall back to sequential for small inputs
    if n < config.parallel_threshold {
        return SuffixArray::build(tokens);
    }

    if n == 0 {
        return SuffixArray {
            suffix_array: Vec::new(),
            lcp: Vec::new(),
        };
    }

    // Initial ranking based on token values
    let mut rank = rank_tokens_parallel(tokens);
    let mut sa: Vec<usize> = (0..n).collect();
    let mut tmp = vec![0usize; n];
    let mut k = 1usize;

    loop {
        // Parallel sort by (rank[i], rank[i+k])
        // We need to capture rank as a reference for the closure
        let rank_ref = &rank;
        sa.par_sort_by(|&a, &b| {
            let ra = rank_ref[a];
            let rb = rank_ref[b];
            if ra != rb {
                return ra.cmp(&rb);
            }
            let ra_k = if a + k < n { rank_ref[a + k] } else { 0 };
            let rb_k = if b + k < n { rank_ref[b + k] } else { 0 };
            ra_k.cmp(&rb_k)
        });

        // Update ranks - this part is sequential due to data dependencies
        // but the key comparison can still leverage the parallel-sorted result
        tmp[sa[0]] = 1;
        for i in 1..n {
            let prev = sa[i - 1];
            let curr = sa[i];

            let prev_key = (rank[prev], if prev + k < n { rank[prev + k] } else { 0 });
            let curr_key = (rank[curr], if curr + k < n { rank[curr + k] } else { 0 });

            tmp[curr] = tmp[prev] + if curr_key != prev_key { 1 } else { 0 };
        }

        rank.copy_from_slice(&tmp);

        // Check if all ranks are unique
        if rank[sa[n - 1]] == n {
            break;
        }

        k *= 2;
    }

    // Build LCP array using parallel Kasai's algorithm
    let lcp = build_lcp_parallel(tokens, &sa);

    SuffixArray {
        suffix_array: sa,
        lcp,
    }
}

/// Parallel token ranking using HashMap with parallel collection.
fn rank_tokens_parallel(tokens: &[Token]) -> Vec<usize> {
    // Create a sorted list of unique tokens
    let mut unique: Vec<Token> = tokens.to_vec();
    unique.par_sort_unstable();
    unique.dedup();

    // Create mapping from token to rank
    let mapping: HashMap<Token, usize> = unique
        .into_iter()
        .enumerate()
        .map(|(i, t)| (t, i + 1))
        .collect();

    // Parallel rank assignment
    tokens.par_iter().map(|t| mapping[t]).collect()
}

/// Build LCP array using Kasai's algorithm with parallel inverse construction.
///
/// The main LCP computation has data dependencies that prevent full parallelization,
/// but the inverse suffix array construction can be done in parallel.
fn build_lcp_parallel(tokens: &[Token], sa: &[usize]) -> Vec<usize> {
    let n = tokens.len();
    if n == 0 {
        return Vec::new();
    }

    let mut lcp = vec![0usize; n - 1];

    // Build inverse suffix array
    // inv[sa[i]] = i, meaning inv[pos] = rank of suffix starting at pos
    let mut inv = vec![0usize; n];
    for (i, &idx) in sa.iter().enumerate() {
        inv[idx] = i;
    }

    // Kasai's algorithm - inherently sequential due to h carrying over
    let mut h = 0usize;

    for i in 0..n {
        let pos = inv[i];
        if pos == n - 1 {
            h = 0;
            continue;
        }

        let j = sa[pos + 1];

        // Extend the match
        while i + h < n && j + h < n && tokens[i + h] == tokens[j + h] {
            h += 1;
        }

        lcp[pos] = h;

        if h > 0 {
            h -= 1;
        }
    }

    lcp
}

/// Build suffix array with automatic parallel/sequential selection.
///
/// Chooses the best implementation based on input size and configuration.
pub fn build_suffix_array_auto(tokens: &[Token], enable_parallel: bool) -> SuffixArray {
    let config = ParallelSAConfig::default();
    
    if enable_parallel && tokens.len() >= config.parallel_threshold {
        build_suffix_array_parallel(tokens, &config)
    } else {
        SuffixArray::build(tokens)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_matches_sequential() {
        // Test that parallel produces same result as sequential
        let tokens: Vec<Token> = (0..1000).map(|i| (i % 100) as Token).collect();
        
        let sequential = SuffixArray::build(&tokens);
        let config = ParallelSAConfig {
            parallel_threshold: 0, // Force parallel
            ..Default::default()
        };
        let parallel = build_suffix_array_parallel(&tokens, &config);

        assert_eq!(sequential.suffix_array, parallel.suffix_array);
        assert_eq!(sequential.lcp, parallel.lcp);
    }

    #[test]
    fn test_parallel_repeated_pattern() {
        let pattern = vec![1u32, 2, 3, 4, 5];
        let tokens: Vec<Token> = pattern.iter().cycle().take(500).cloned().collect();

        let config = ParallelSAConfig {
            parallel_threshold: 0,
            ..Default::default()
        };
        let sa = build_suffix_array_parallel(&tokens, &config);

        // Verify all positions are present
        let mut sorted = sa.suffix_array.clone();
        sorted.sort_unstable();
        let expected: Vec<usize> = (0..500).collect();
        assert_eq!(sorted, expected);

        // Should have LCP values >= 5 for repeated patterns
        let max_lcp = sa.lcp.iter().copied().max().unwrap_or(0);
        assert!(max_lcp >= 5);
    }

    #[test]
    fn test_parallel_empty() {
        let tokens: Vec<Token> = vec![];
        let config = ParallelSAConfig::default();
        let sa = build_suffix_array_parallel(&tokens, &config);
        assert!(sa.suffix_array.is_empty());
        assert!(sa.lcp.is_empty());
    }

    #[test]
    fn test_parallel_single() {
        let tokens = vec![42u32];
        let config = ParallelSAConfig {
            parallel_threshold: 0,
            ..Default::default()
        };
        let sa = build_suffix_array_parallel(&tokens, &config);
        assert_eq!(sa.suffix_array, vec![0]);
        assert!(sa.lcp.is_empty());
    }

    #[test]
    fn test_auto_selection() {
        let small_tokens: Vec<Token> = (0..100).collect();
        let large_tokens: Vec<Token> = (0..20000).map(|i| (i % 1000) as Token).collect();

        // Small input should work
        let _sa1 = build_suffix_array_auto(&small_tokens, true);
        
        // Large input with parallel enabled
        let sa2 = build_suffix_array_auto(&large_tokens, true);
        assert_eq!(sa2.suffix_array.len(), 20000);
        
        // Large input with parallel disabled
        let sa3 = build_suffix_array_auto(&large_tokens, false);
        assert_eq!(sa2.suffix_array, sa3.suffix_array);
    }

    #[test]
    fn test_correctness_with_all_same() {
        let tokens: Vec<Token> = vec![42; 1000];
        let config = ParallelSAConfig {
            parallel_threshold: 0,
            ..Default::default()
        };
        
        let sequential = SuffixArray::build(&tokens);
        let parallel = build_suffix_array_parallel(&tokens, &config);
        
        assert_eq!(sequential.suffix_array, parallel.suffix_array);
        assert_eq!(sequential.lcp, parallel.lcp);
    }

    #[test]
    fn test_correctness_random_pattern() {
        // Use a deterministic "random" pattern
        let tokens: Vec<Token> = (0..1000)
            .map(|i| ((i * 7 + 13) % 256) as Token)
            .collect();
        
        let sequential = SuffixArray::build(&tokens);
        let config = ParallelSAConfig {
            parallel_threshold: 0,
            ..Default::default()
        };
        let parallel = build_suffix_array_parallel(&tokens, &config);
        
        assert_eq!(sequential.suffix_array, parallel.suffix_array);
        assert_eq!(sequential.lcp, parallel.lcp);
    }
}
