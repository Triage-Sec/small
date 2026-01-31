//! Suffix array construction and LCP computation.
//!
//! Implements O(n log n) suffix array construction using the doubling algorithm
//! and O(n) LCP computation using Kasai's algorithm.
//!
//! Port of `delta/suffix_array.py` and `delta/suffix_array_fast.py`.

use crate::types::Token;
use std::collections::HashMap;

/// Suffix array with LCP (Longest Common Prefix) array.
#[derive(Debug, Clone)]
pub struct SuffixArray {
    /// The suffix array: suffix_array[i] is the starting position of the i-th
    /// lexicographically smallest suffix.
    pub suffix_array: Vec<usize>,
    /// LCP array: lcp[i] is the length of the longest common prefix between
    /// suffix_array[i] and suffix_array[i+1].
    pub lcp: Vec<usize>,
}

impl SuffixArray {
    /// Build a suffix array from a token sequence using the doubling algorithm.
    ///
    /// Time complexity: O(n log n)
    /// Space complexity: O(n)
    pub fn build(tokens: &[Token]) -> Self {
        let n = tokens.len();
        if n == 0 {
            return Self {
                suffix_array: Vec::new(),
                lcp: Vec::new(),
            };
        }

        // Initial ranking based on token values
        let mut rank = rank_tokens(tokens);
        let mut sa: Vec<usize> = (0..n).collect();
        let mut tmp = vec![0usize; n];
        let mut k = 1usize;

        loop {
            // Sort by (rank[i], rank[i+k])
            sa.sort_by(|&a, &b| {
                let ra = rank[a];
                let rb = rank[b];
                if ra != rb {
                    return ra.cmp(&rb);
                }
                let ra_k = if a + k < n { rank[a + k] } else { 0 };
                let rb_k = if b + k < n { rank[b + k] } else { 0 };
                ra_k.cmp(&rb_k)
            });

            // Update ranks
            tmp[sa[0]] = 1;
            for i in 1..n {
                let prev = sa[i - 1];
                let curr = sa[i];

                let prev_key = (
                    rank[prev],
                    if prev + k < n { rank[prev + k] } else { 0 },
                );
                let curr_key = (
                    rank[curr],
                    if curr + k < n { rank[curr + k] } else { 0 },
                );

                tmp[curr] = tmp[prev] + if curr_key != prev_key { 1 } else { 0 };
            }

            rank.copy_from_slice(&tmp);

            // Check if all ranks are unique
            if rank[sa[n - 1]] == n {
                break;
            }

            k *= 2;
        }

        // Build LCP array using Kasai's algorithm
        let lcp = build_lcp(tokens, &sa);

        Self {
            suffix_array: sa,
            lcp,
        }
    }

    /// Extract LCP intervals representing repeated substrings.
    ///
    /// Returns intervals as (start_idx, end_idx, lcp_value) where:
    /// - start_idx, end_idx are indices into the suffix array
    /// - lcp_value is the minimum LCP in that interval
    ///
    /// Only returns intervals with lcp_value >= min_len.
    pub fn lcp_intervals(&self, min_len: usize) -> Vec<(usize, usize, usize)> {
        if self.lcp.is_empty() {
            return Vec::new();
        }

        let mut intervals = Vec::new();
        let mut stack: Vec<(usize, usize)> = Vec::new(); // (start, lcp_value)

        for (i, &lcp_value) in self.lcp.iter().enumerate() {
            let mut start = i;

            while !stack.is_empty() && stack.last().unwrap().1 > lcp_value {
                let (prev_start, prev_lcp) = stack.pop().unwrap();
                if prev_lcp >= min_len {
                    intervals.push((prev_start, i, prev_lcp));
                }
                start = prev_start;
            }

            if stack.is_empty() || stack.last().unwrap().1 < lcp_value {
                stack.push((start, lcp_value));
            }
        }

        // Process remaining stack
        let n = self.lcp.len();
        while let Some((start, lcp_value)) = stack.pop() {
            if lcp_value >= min_len {
                intervals.push((start, n, lcp_value));
            }
        }

        intervals
    }
}

/// Rank tokens to integers for suffix array construction.
fn rank_tokens(tokens: &[Token]) -> Vec<usize> {
    // Create a sorted list of unique tokens
    let mut unique: Vec<Token> = tokens.iter().copied().collect();
    unique.sort_unstable();
    unique.dedup();

    // Create mapping from token to rank
    let mapping: HashMap<Token, usize> = unique
        .into_iter()
        .enumerate()
        .map(|(i, t)| (t, i + 1))
        .collect();

    tokens.iter().map(|t| mapping[t]).collect()
}

/// Build LCP array using Kasai's algorithm.
///
/// Time complexity: O(n)
fn build_lcp(tokens: &[Token], sa: &[usize]) -> Vec<usize> {
    let n = tokens.len();
    if n == 0 {
        return Vec::new();
    }

    let mut lcp = vec![0usize; n - 1];

    // Build inverse suffix array
    let mut inv = vec![0usize; n];
    for (i, &idx) in sa.iter().enumerate() {
        inv[idx] = i;
    }

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

/// Count non-overlapping occurrences of a pattern of given length.
#[inline]
pub fn count_non_overlapping(positions: &[usize], length: usize) -> usize {
    let mut count = 0;
    let mut next_free = 0;

    for &pos in positions {
        if pos >= next_free {
            count += 1;
            next_free = pos + length;
        }
    }

    count
}

/// Extract non-overlapping positions from a sorted position list.
pub fn non_overlapping_positions(positions: &[usize], length: usize) -> Vec<usize> {
    let mut result = Vec::with_capacity(positions.len());
    let mut next_free = 0;

    for &pos in positions {
        if pos >= next_free {
            result.push(pos);
            next_free = pos + length;
        }
    }

    result
}

/// Build suffix array with automatic parallel/sequential selection.
///
/// When compiled with the `parallel` feature and input size exceeds threshold,
/// uses parallel construction for improved performance.
#[cfg(feature = "parallel")]
pub fn build_suffix_array_auto(tokens: &[Token], enable_parallel: bool) -> SuffixArray {
    crate::suffix_array_parallel::build_suffix_array_auto(tokens, enable_parallel)
}

/// Build suffix array (sequential only when parallel feature not enabled).
#[cfg(not(feature = "parallel"))]
pub fn build_suffix_array_auto(tokens: &[Token], _enable_parallel: bool) -> SuffixArray {
    SuffixArray::build(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suffix_array_simple() {
        let tokens = vec![1, 2, 3, 1, 2, 3];
        let sa = SuffixArray::build(&tokens);

        // Verify suffix array is valid (all positions present)
        let mut sorted = sa.suffix_array.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_suffix_array_empty() {
        let tokens: Vec<Token> = vec![];
        let sa = SuffixArray::build(&tokens);
        assert!(sa.suffix_array.is_empty());
        assert!(sa.lcp.is_empty());
    }

    #[test]
    fn test_suffix_array_single() {
        let tokens = vec![42];
        let sa = SuffixArray::build(&tokens);
        assert_eq!(sa.suffix_array, vec![0]);
        assert!(sa.lcp.is_empty());
    }

    #[test]
    fn test_suffix_array_repeated() {
        // "abab" pattern should have LCP values > 0
        let tokens = vec![1, 2, 1, 2];
        let sa = SuffixArray::build(&tokens);

        // Find the maximum LCP - should be 2 (for "ab" repeated)
        let max_lcp = sa.lcp.iter().copied().max().unwrap_or(0);
        assert!(max_lcp >= 2);
    }

    #[test]
    fn test_lcp_intervals() {
        let tokens = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
        let sa = SuffixArray::build(&tokens);

        let intervals = sa.lcp_intervals(2);
        // Should find intervals for repeated patterns
        assert!(!intervals.is_empty());
    }

    #[test]
    fn test_count_non_overlapping() {
        let positions = vec![0, 2, 4, 6, 8];
        // Length 2: 0, 2, 4, 6, 8 are all non-overlapping
        assert_eq!(count_non_overlapping(&positions, 2), 5);
        // Length 3: 0, 4, 8 (skipping 2, 6)
        assert_eq!(count_non_overlapping(&positions, 3), 3);
    }

    #[test]
    fn test_non_overlapping_positions() {
        let positions = vec![0, 1, 2, 5, 6, 10];
        let result = non_overlapping_positions(&positions, 3);
        // 0 (takes 0-2), skip 1 and 2, 5 (takes 5-7), skip 6, 10 (takes 10-12)
        assert_eq!(result, vec![0, 5, 10]);
    }
}
