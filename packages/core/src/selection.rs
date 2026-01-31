//! Pattern selection algorithms.
//!
//! Implements greedy selection with iterative refinement for choosing
//! non-overlapping pattern occurrences that maximize compression savings.
//!
//! Port of `delta/selection.py`.

use crate::types::{is_compressible, min_count_for_compressibility, Candidate, Occurrence, Token};
use std::collections::{HashMap, HashSet};

/// Result of pattern selection.
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Selected occurrences, sorted by start position
    pub selected: Vec<Occurrence>,
}

/// Compute savings-density score for an occurrence.
///
/// Higher values indicate better compression value per position consumed.
#[inline]
fn savings_density(occ: &Occurrence) -> f64 {
    if occ.length <= 1 {
        return 0.0;
    }
    let pattern_savings = occ.length as f64 - 1.0;
    let density = pattern_savings / occ.length as f64;
    density + occ.priority as f64 * 0.1
}

/// Build occurrence list from candidates.
fn build_occurrences(candidates: &[Candidate]) -> Vec<Occurrence> {
    let mut occurrences = Vec::new();

    for cand in candidates {
        for &pos in &cand.positions {
            let patches = cand.patches.get(&pos).cloned().unwrap_or_default();
            occurrences.push(Occurrence {
                start: pos,
                length: cand.length,
                subsequence: cand.subsequence.clone(),
                priority: cand.priority,
                patches,
            });
        }
    }

    occurrences.sort_by_key(|occ| (occ.start + occ.length, occ.start));
    occurrences
}

/// Group occurrences by their subsequence.
#[allow(dead_code)]
fn group_by_subsequence(occurrences: &[Occurrence]) -> HashMap<Vec<Token>, Vec<&Occurrence>> {
    let mut grouped: HashMap<Vec<Token>, Vec<&Occurrence>> = HashMap::new();
    for occ in occurrences {
        grouped
            .entry(occ.subsequence.clone())
            .or_default()
            .push(occ);
    }
    grouped
}

/// Estimate non-overlapping count for a group of occurrences.
fn estimate_non_overlapping_count(occs: &[&Occurrence]) -> usize {
    if occs.is_empty() {
        return 0;
    }

    let mut sorted: Vec<_> = occs.iter().map(|o| o.start).collect();
    sorted.sort_unstable();

    let length = occs[0].length;
    let mut count = 0;
    let mut next_free = 0;

    for &pos in &sorted {
        if pos >= next_free {
            count += 1;
            next_free = pos + length;
        }
    }

    count
}

/// Greedy selection with iterative refinement.
///
/// Uses an iterative refinement approach:
/// 1. Pre-filter patterns that can't possibly achieve compressibility
/// 2. Greedily select non-overlapping occurrences
/// 3. Release positions from patterns that didn't achieve compressibility
/// 4. Repeat until stable (all selected patterns are compressible)
pub fn select_greedy(candidates: &[Candidate], extra_cost: usize) -> SelectionResult {
    if candidates.is_empty() {
        return SelectionResult {
            selected: Vec::new(),
        };
    }

    let occurrences = build_occurrences(candidates);
    if occurrences.is_empty() {
        return SelectionResult {
            selected: Vec::new(),
        };
    }

    // Pre-compute minimum counts for each pattern length
    let mut min_counts_cache: HashMap<usize, usize> = HashMap::new();
    let get_min_count = |length: usize, cache: &mut HashMap<usize, usize>| -> usize {
        *cache
            .entry(length)
            .or_insert_with(|| min_count_for_compressibility(length, extra_cost))
    };

    // Group occurrences by subsequence
    let mut subseq_to_occs: HashMap<Vec<Token>, Vec<usize>> = HashMap::new();
    for (i, occ) in occurrences.iter().enumerate() {
        subseq_to_occs
            .entry(occ.subsequence.clone())
            .or_default()
            .push(i);
    }

    // Filter out patterns that can never be compressible
    let mut viable_subseqs: HashSet<Vec<Token>> = HashSet::new();
    for (subseq, indices) in &subseq_to_occs {
        let min_count = get_min_count(subseq.len(), &mut min_counts_cache);
        if indices.len() >= min_count {
            viable_subseqs.insert(subseq.clone());
        }
    }

    // Filter occurrences to only viable patterns
    let mut viable_indices: Vec<usize> = (0..occurrences.len())
        .filter(|&i| viable_subseqs.contains(&occurrences[i].subsequence))
        .collect();

    if viable_indices.is_empty() {
        return SelectionResult {
            selected: Vec::new(),
        };
    }

    // Iterative refinement loop
    let max_iterations = 10;
    let mut selected_indices: Vec<usize> = Vec::new();
    let mut subseq_counts: HashMap<Vec<Token>, usize> = HashMap::new();

    for _iteration in 0..max_iterations {
        // Sort by savings-density (highest first)
        viable_indices.sort_by(|&a, &b| {
            let da = savings_density(&occurrences[a]);
            let db = savings_density(&occurrences[b]);
            db.partial_cmp(&da)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| occurrences[a].start.cmp(&occurrences[b].start))
        });

        selected_indices.clear();
        subseq_counts.clear();
        let mut occupied: HashSet<usize> = HashSet::new();

        for &idx in &viable_indices {
            let occ = &occurrences[idx];
            let positions: HashSet<usize> = (occ.start..occ.start + occ.length).collect();

            if !positions.is_disjoint(&occupied) {
                continue;
            }

            selected_indices.push(idx);
            occupied.extend(positions);
            *subseq_counts.entry(occ.subsequence.clone()).or_default() += 1;
        }

        // Find patterns that achieved compressibility
        let mut compressible_subseqs: HashSet<Vec<Token>> = HashSet::new();
        let mut non_compressible_subseqs: HashSet<Vec<Token>> = HashSet::new();

        for (subseq, &count) in &subseq_counts {
            if is_compressible(subseq.len(), count, extra_cost) {
                compressible_subseqs.insert(subseq.clone());
            } else {
                non_compressible_subseqs.insert(subseq.clone());
            }
        }

        // If all selected patterns are compressible, we're done
        if non_compressible_subseqs.is_empty() {
            break;
        }

        // Remove non-compressible patterns from viable set and retry
        for subseq in &non_compressible_subseqs {
            viable_subseqs.remove(subseq);
        }
        viable_indices.retain(|&i| viable_subseqs.contains(&occurrences[i].subsequence));

        if viable_indices.is_empty() {
            selected_indices.clear();
            break;
        }
    }

    // Final filter: only keep compressible patterns
    let mut final_selected: Vec<Occurrence> = Vec::new();
    for &idx in &selected_indices {
        let occ = &occurrences[idx];
        let count = subseq_counts.get(&occ.subsequence).copied().unwrap_or(0);
        if is_compressible(occ.length, count, extra_cost) {
            final_selected.push(occ.clone());
        }
    }

    final_selected.sort_by_key(|occ| occ.start);

    SelectionResult {
        selected: final_selected,
    }
}

/// Weighted interval scheduling with proper savings calculation.
///
/// Uses dynamic programming to find optimal non-overlapping selection,
/// with iterative refinement for compressibility constraints.
pub fn select_optimal(candidates: &[Candidate], extra_cost: usize) -> SelectionResult {
    if candidates.is_empty() {
        return SelectionResult {
            selected: Vec::new(),
        };
    }

    let occurrences = build_occurrences(candidates);
    if occurrences.is_empty() {
        return SelectionResult {
            selected: Vec::new(),
        };
    }

    // Pre-filter patterns that can never be compressible
    let mut subseq_to_occs: HashMap<Vec<Token>, Vec<usize>> = HashMap::new();
    for (i, occ) in occurrences.iter().enumerate() {
        subseq_to_occs
            .entry(occ.subsequence.clone())
            .or_default()
            .push(i);
    }

    let mut viable_subseqs: HashSet<Vec<Token>> = HashSet::new();
    for (subseq, indices) in &subseq_to_occs {
        let min_count = min_count_for_compressibility(subseq.len(), extra_cost);
        if indices.len() >= min_count {
            viable_subseqs.insert(subseq.clone());
        }
    }

    let mut viable_indices: Vec<usize> = (0..occurrences.len())
        .filter(|&i| viable_subseqs.contains(&occurrences[i].subsequence))
        .collect();

    if viable_indices.is_empty() {
        return SelectionResult {
            selected: Vec::new(),
        };
    }

    // Iterative refinement loop
    let max_iterations = 10;
    let mut selected_indices: Vec<usize>;
    let mut grouped: HashMap<Vec<Token>, Vec<usize>> = HashMap::new();

    for _iteration in 0..max_iterations {
        // Sort by end position
        viable_indices.sort_by_key(|&i| (occurrences[i].start + occurrences[i].length, occurrences[i].start));

        if viable_indices.is_empty() {
            return SelectionResult {
                selected: Vec::new(),
            };
        }

        let n = viable_indices.len();
        let ends: Vec<usize> = viable_indices
            .iter()
            .map(|&i| occurrences[i].start + occurrences[i].length)
            .collect();

        // p[i]: last index < i that doesn't overlap
        let mut p: Vec<i32> = vec![-1; n];
        for i in 0..n {
            let occ_start = occurrences[viable_indices[i]].start;
            let mut lo = 0i32;
            let mut hi = i as i32 - 1;
            let mut idx = -1i32;

            while lo <= hi {
                let mid = (lo + hi) / 2;
                if ends[mid as usize] <= occ_start {
                    idx = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            p[i] = idx;
        }

        // Estimate non-overlapping counts for weight calculation
        let mut subseq_expected: HashMap<Vec<Token>, usize> = HashMap::new();
        for &i in &viable_indices {
            let subseq = &occurrences[i].subsequence;
            let indices_ref: Vec<&Occurrence> = viable_indices
                .iter()
                .filter(|&&j| occurrences[j].subsequence == *subseq)
                .map(|&j| &occurrences[j])
                .collect();
            subseq_expected.insert(subseq.clone(), estimate_non_overlapping_count(&indices_ref));
        }

        // Compute weights
        let weights: Vec<f64> = viable_indices
            .iter()
            .map(|&i| {
                let occ = &occurrences[i];
                let expected = *subseq_expected.get(&occ.subsequence).unwrap_or(&1);
                let dict_cost = (1 + occ.length + extra_cost) as f64 / expected as f64;
                let savings = occ.length as f64 - 1.0 - dict_cost;
                savings.max(0.0) + occ.priority as f64 * 0.5
            })
            .collect();

        // DP
        let mut dp = vec![0.0; n];
        let mut choose = vec![false; n];

        for i in 0..n {
            let take = weights[i] + if p[i] >= 0 { dp[p[i] as usize] } else { 0.0 };
            let skip = if i > 0 { dp[i - 1] } else { 0.0 };

            if take > skip {
                dp[i] = take;
                choose[i] = true;
            } else {
                dp[i] = skip;
            }
        }

        // Reconstruct
        selected_indices = Vec::new();
        let mut i = n as i32 - 1;
        while i >= 0 {
            if choose[i as usize] {
                selected_indices.push(viable_indices[i as usize]);
                i = p[i as usize];
            } else {
                i -= 1;
            }
        }
        selected_indices.reverse();

        // Check compressibility
        grouped.clear();
        for &idx in &selected_indices {
            grouped
                .entry(occurrences[idx].subsequence.clone())
                .or_default()
                .push(idx);
        }

        let mut non_compressible: HashSet<Vec<Token>> = HashSet::new();
        for (subseq, indices) in &grouped {
            if !is_compressible(subseq.len(), indices.len(), extra_cost) {
                non_compressible.insert(subseq.clone());
            }
        }

        if non_compressible.is_empty() {
            break;
        }

        // Remove non-compressible and retry
        for subseq in &non_compressible {
            viable_subseqs.remove(subseq);
        }
        viable_indices.retain(|&i| viable_subseqs.contains(&occurrences[i].subsequence));
    }

    // Final selection
    let mut final_selected: Vec<Occurrence> = Vec::new();
    for (subseq, indices) in &grouped {
        if is_compressible(subseq.len(), indices.len(), extra_cost) {
            for &idx in indices {
                final_selected.push(occurrences[idx].clone());
            }
        }
    }

    final_selected.sort_by_key(|occ| occ.start);

    SelectionResult {
        selected: final_selected,
    }
}

/// Select occurrences using the specified mode.
pub fn select_occurrences(
    candidates: &[Candidate],
    mode: &str,
    extra_cost: usize,
) -> SelectionResult {
    match mode {
        "greedy" => select_greedy(candidates, extra_cost),
        "optimal" => select_optimal(candidates, extra_cost),
        _ => select_greedy(candidates, extra_cost), // Default to greedy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(subseq: Vec<Token>, positions: Vec<usize>) -> Candidate {
        Candidate::new(subseq, positions)
    }

    #[test]
    fn test_select_greedy_empty() {
        let result = select_greedy(&[], 1);
        assert!(result.selected.is_empty());
    }

    #[test]
    fn test_select_greedy_simple() {
        // Pattern [1, 2] at positions 0, 4, 8 (non-overlapping)
        let cand = make_candidate(vec![1, 2], vec![0, 4, 8]);
        let result = select_greedy(&[cand], 1);

        // Length 2, count 3: not compressible (2*3=6 > 1+2+3+1=7 is false)
        // So no selection expected
        assert!(result.selected.is_empty());
    }

    #[test]
    fn test_select_greedy_compressible() {
        // Pattern [1, 2] at 5 positions
        let cand = make_candidate(vec![1, 2], vec![0, 3, 6, 9, 12]);
        let result = select_greedy(&[cand], 1);

        // Length 2, count 5: 2*5=10 > 1+2+5+1=9 - compressible!
        assert_eq!(result.selected.len(), 5);
    }

    #[test]
    fn test_select_optimal_compressible() {
        let cand = make_candidate(vec![1, 2], vec![0, 3, 6, 9, 12]);
        let result = select_optimal(&[cand], 1);

        assert_eq!(result.selected.len(), 5);
    }

    #[test]
    fn test_select_non_overlapping() {
        // Two patterns that overlap
        let cand1 = make_candidate(vec![1, 2, 3], vec![0, 6, 12]);
        let cand2 = make_candidate(vec![2, 3, 4], vec![1, 7, 13]);

        let result = select_greedy(&[cand1, cand2], 1);

        // Check that selected occurrences don't overlap
        let mut occupied: HashSet<usize> = HashSet::new();
        for occ in &result.selected {
            for pos in occ.start..occ.start + occ.length {
                assert!(
                    occupied.insert(pos),
                    "Position {} is covered by multiple occurrences",
                    pos
                );
            }
        }
    }

    #[test]
    fn test_savings_density() {
        let occ = Occurrence {
            start: 0,
            length: 4,
            subsequence: vec![1, 2, 3, 4],
            priority: 0,
            patches: vec![],
        };

        let density = savings_density(&occ);
        // (4-1)/4 = 0.75
        assert!((density - 0.75).abs() < 0.001);
    }
}
