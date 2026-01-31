# Algorithm Details

This document describes the core algorithms used in Delta for pattern discovery, selection, and compression.

## Overview

Delta's compression pipeline consists of three main algorithmic phases:

1. **Discovery**: Find repeated subsequences in the input
2. **Selection**: Choose non-overlapping occurrences that maximize savings
3. **Replacement**: Build the dictionary and substitute patterns

## Pattern Discovery

### Suffix Array Discovery (Default)

The suffix array approach provides O(n log n) discovery for all repeated substrings.

**Algorithm**:
1. Build suffix array using the doubling algorithm
2. Compute LCP (Longest Common Prefix) array using Kasai's algorithm
3. Extract LCP intervals representing repeated substrings
4. Filter by compressibility constraints

```python
# Suffix array construction: O(n log n)
def build_suffix_array(tokens):
    n = len(tokens)
    rank = initial_ranking(tokens)
    sa = list(range(n))
    k = 1
    while max(rank) < n - 1:
        sa.sort(key=lambda i: (rank[i], rank[i+k] if i+k < n else -1))
        update_ranks(sa, rank, k)
        k *= 2
    return sa

# LCP computation: O(n) using Kasai's algorithm
def build_lcp(tokens, sa):
    inv = inverse(sa)
    lcp = [0] * (n - 1)
    h = 0
    for i in range(n):
        if inv[i] > 0:
            j = sa[inv[i] - 1]
            while tokens[i + h] == tokens[j + h]:
                h += 1
            lcp[inv[i] - 1] = h
            h = max(0, h - 1)
    return lcp
```

**Optimization**: Maximal repeat extraction processes LCP intervals from longest to shortest, breaking early when compressibility is achieved.

### BPE-Style Discovery

Byte-Pair Encoding style discovery iteratively finds the best pair to merge.

**Algorithm**:
1. Count all adjacent token pairs
2. Find pair with highest compression savings
3. Conceptually merge the pair
4. Repeat until no beneficial merges remain

```python
def discover_bpe(tokens, max_iterations):
    working = list(tokens)
    candidates = []
    
    for _ in range(max_iterations):
        pair_counts = count_adjacent_pairs(working)
        best_pair = max(pair_counts, key=lambda p: savings(p, pair_counts[p]))
        
        if savings(best_pair, pair_counts[best_pair]) <= 0:
            break
            
        candidates.append(make_candidate(best_pair, working))
        working = apply_merge(working, best_pair)
    
    return candidates
```

**Advantage**: Often finds better hierarchical patterns than one-shot discovery.

### Sliding Window Discovery

Simple O(n × L) discovery for each pattern length L.

```python
def discover_length(tokens, length):
    positions = defaultdict(list)
    for i in range(len(tokens) - length + 1):
        subseq = tuple(tokens[i:i+length])
        positions[subseq].append(i)
    return [make_candidate(s, p) for s, p in positions.items()]
```

## Pattern Selection

### Compressibility Constraint

A pattern with length L appearing C times is compressible if:

```
L × C > 1 + L + C + extra_cost
```

Where:
- `L × C`: Original tokens consumed
- `1`: Meta-token entry in dictionary
- `L`: Pattern definition
- `C`: References in body
- `extra_cost`: 1 if length tokens enabled, else 0

Solving for minimum count: `C > (1 + L + extra_cost) / (L - 1)`

### Greedy Selection

**Algorithm**:
1. Sort occurrences by savings-density: `(L - 1) / L + priority × 0.1`
2. Greedily select non-overlapping occurrences
3. Post-filter patterns that achieved compressibility

```python
def greedy_select(occurrences, config):
    sorted_occs = sorted(occurrences, key=lambda o: -savings_density(o))
    selected = []
    occupied = set()
    
    for occ in sorted_occs:
        positions = set(range(occ.start, occ.start + occ.length))
        if not positions & occupied:
            selected.append(occ)
            occupied |= positions
    
    return filter_compressible(selected, config)
```

**Complexity**: O(n log n) for sorting, O(n) for selection.

### Weighted Interval Scheduling

Dynamic programming for optimal non-overlapping selection.

**Algorithm**:
1. Sort occurrences by end position
2. Compute predecessor array using binary search
3. DP: `dp[i] = max(dp[i-1], weight[i] + dp[pred[i]])`
4. Backtrack to reconstruct solution

```python
def optimal_select(occurrences):
    occs = sorted(occurrences, key=lambda o: o.start + o.length)
    pred = compute_predecessors(occs)  # Binary search: O(n log n)
    weights = compute_weights(occs)     # Marginal savings
    
    dp = [0] * len(occs)
    choose = [False] * len(occs)
    
    for i in range(len(occs)):
        take = weights[i] + (dp[pred[i]] if pred[i] >= 0 else 0)
        skip = dp[i-1] if i > 0 else 0
        dp[i], choose[i] = (take, True) if take > skip else (skip, False)
    
    return backtrack(occs, choose, pred)
```

**Complexity**: O(n log n) for sorting and predecessor computation, O(n) for DP.

### Beam Search

Explores multiple selection paths with bounded memory.

**Algorithm**:
1. Maintain top-k states: (score, last_end, selected, pattern_counts)
2. For each occurrence, expand all states with skip/take options
3. Prune to top-k by score
4. Select best final state

```python
def beam_select(occurrences, width):
    states = [(0, -1, [], {})]  # (score, last_end, selected, counts)
    
    for occ in sorted(occurrences, key=lambda o: o.start):
        new_states = []
        for score, last_end, selected, counts in states:
            # Skip option
            new_states.append((score, last_end, selected, counts))
            # Take option (if non-overlapping)
            if occ.start >= last_end:
                new_score = score + marginal_savings(occ, counts)
                new_states.append((new_score, occ.end, selected + [occ], updated_counts))
        
        states = sorted(new_states, key=lambda s: -s[0])[:width]
    
    return best_state(states).selected
```

**Complexity**: O(n × width) time, O(width) space.

### Integer Linear Programming

Globally optimal selection using ILP solver.

**Formulation**:
- Variables: `x_i ∈ {0,1}` for each occurrence, `y_p ∈ {0,1}` for each pattern
- Objective: Maximize `Σ(L_i - 1)x_i - Σ(1 + L_p + extra)y_p`
- Constraints:
  - Non-overlapping: `Σ x_i ≤ 1` for each position
  - Pattern activation: `x_i ≤ y_p` for occurrence i of pattern p
  - Compressibility: `Σ x_i ≥ min_count × y_p` for pattern p

**Complexity**: NP-hard in general, but practical for small instances with modern solvers.

## Subsumption Analysis

Pattern "abcd" subsumes "ab", "bc", "cd". Subsumption analysis avoids redundant dictionary entries.

**Algorithm**:
1. Build subsumption graph: directed edges from longer to shorter patterns
2. For each subsumed pattern, compute covered positions
3. Keep pattern only if it has sufficient independent occurrences

```python
def prune_subsumed(candidates):
    graph = build_subsumption_graph(candidates)
    keep = []
    
    for i, cand in enumerate(candidates):
        if not graph.subsumed_by[i]:
            keep.append(cand)  # Maximal pattern
        else:
            # Check for independent occurrences
            covered = positions_covered_by_subsuming(i, graph, candidates)
            independent = [p for p in cand.positions if p not in covered]
            if len(independent) >= min_independent:
                keep.append(cand.with_positions(independent))
    
    return keep
```

## Hierarchical Compression

Multiple compression passes allow meta-tokens to reference other meta-tokens.

**Algorithm**:
1. Compress input sequence
2. If compression occurred and depth < max_depth:
   - Treat compressed body as new input
   - Repeat from step 1
3. Early stopping: halt if improvement < 2%

**Dictionary ordering**: Topologically sorted so dependencies appear before dependents.

## Complexity Summary

| Operation | Time | Space |
|-----------|------|-------|
| Suffix array construction | O(n log n) | O(n) |
| LCP computation | O(n) | O(n) |
| Greedy selection | O(n log n) | O(n) |
| Optimal selection (DP) | O(n log n) | O(n) |
| Beam selection | O(n × width) | O(width) |
| ILP selection | Exponential | O(n²) |
| Hierarchical (d passes) | O(d × n log n) | O(n) |

## References

1. Kärkkäinen, J., Sanders, P., & Burkhardt, S. (2006). Linear work suffix array construction.
2. Kasai, T., et al. (2001). Linear-time longest-common-prefix computation.
3. Kleinberg, J., & Tardos, É. (2006). Algorithm Design. Chapter on Weighted Interval Scheduling.
4. Sennrich, R., Haddow, B., & Birch, A. (2016). Neural machine translation of rare words with subword units.
