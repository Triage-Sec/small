"""Integer Linear Programming based pattern selection for optimal compression.

This module provides globally optimal pattern selection using ILP solvers.
For high-value compression scenarios where computational cost is acceptable,
ILP can find solutions that greedy/DP approaches miss.

Requires scipy >= 1.11.0 for milp solver. Falls back to beam search if unavailable.

Formulation:
- Binary variable x_i for each occurrence i: 1 if selected, 0 otherwise
- Binary variable y_p for each pattern p: 1 if pattern is used, 0 otherwise
- Objective: maximize total savings
- Constraints:
  - Non-overlapping: for each position, at most one occurrence can cover it
  - Pattern activation: if any x_i is selected, y_p must be 1
  - Compressibility: if y_p = 1, enough occurrences must be selected
"""

from __future__ import annotations

import math
from typing import Iterable

from .config import CompressionConfig
from .types import Candidate, Occurrence
from .utils import is_compressible


def _min_count_for_compressibility(length: int, extra_cost: int) -> int:
    """Compute minimum occurrence count for compressibility."""
    if length <= 1:
        return 1_000_000_000
    return math.ceil((2 + length + extra_cost) / (length - 1))


def select_occurrences_ilp(
    candidates: Iterable[Candidate],
    config: CompressionConfig,
    time_limit_seconds: float | None = None,
) -> list[Occurrence]:
    """Select occurrences using Integer Linear Programming for optimal compression.

    Args:
        candidates: Iterable of candidate patterns
        config: Compression configuration
        time_limit_seconds: Maximum time for solver (uses config.ilp_time_limit if None)

    Returns:
        List of selected Occurrence objects
    """
    if time_limit_seconds is None:
        time_limit_seconds = config.ilp_time_limit

    try:
        import numpy as np
        from scipy.optimize import milp, LinearConstraint, Bounds
    except ImportError:
        # Fall back to beam search if scipy not available
        from .selection import _beam_search_with_savings, _build_occurrences

        fallback_occurrences = _build_occurrences(candidates)
        return _beam_search_with_savings(
            fallback_occurrences, config.beam_width, config
        )

    # Build occurrence list and pattern mapping
    occurrences: list[Occurrence] = []
    pattern_to_occs: dict[tuple, list[int]] = {}
    candidates_list = list(candidates)

    for cand in candidates_list:
        for pos in cand.positions:
            occ_idx = len(occurrences)
            occurrences.append(
                Occurrence(
                    start=pos,
                    length=cand.length,
                    subsequence=cand.subsequence,
                    priority=cand.priority,
                    patches=cand.patches.get(pos, ()),
                )
            )
            pattern_to_occs.setdefault(cand.subsequence, []).append(occ_idx)

    if not occurrences:
        return []

    n_occs = len(occurrences)
    patterns = list(pattern_to_occs.keys())
    n_patterns = len(patterns)
    pattern_idx = {p: i for i, p in enumerate(patterns)}

    # For large problems, use configurable fallback
    fallback_threshold = config.ilp_fallback_threshold
    if n_occs > fallback_threshold:
        # Use LP relaxation if enabled, otherwise beam search
        if config.ilp_use_relaxation:
            return select_occurrences_ilp_relaxed(candidates_list, config)
        else:
            from .selection import _beam_search_with_savings

            return _beam_search_with_savings(occurrences, config.beam_width, config)

    extra_cost = 1 if config.dict_length_enabled else 0

    # Variables: [x_0, ..., x_{n_occs-1}, y_0, ..., y_{n_patterns-1}]
    n_vars = n_occs + n_patterns

    # Objective: maximize savings (minimize negative savings)
    # For each occurrence: save (length - 1) tokens
    # For each activated pattern: pay dictionary cost (1 + length + extra_cost)
    c = np.zeros(n_vars)

    for i, occ in enumerate(occurrences):
        # Savings from selecting this occurrence
        c[i] = -(occ.length - 1 + occ.priority * 0.1)  # Negative for minimization

    for p_idx, pattern in enumerate(patterns):
        # Dictionary cost when pattern is activated
        dict_cost = 1 + len(pattern) + extra_cost
        c[n_occs + p_idx] = dict_cost  # Positive cost

    # Build constraints
    A_ub_rows: list[np.ndarray] = []
    b_ub: list[float] = []

    # 1. Non-overlapping constraints
    # For each position in the sequence, at most one occurrence can cover it
    # Build position-to-occurrences mapping
    pos_to_occs: dict[int, list[int]] = {}
    for i, occ in enumerate(occurrences):
        for pos in range(occ.start, occ.start + occ.length):
            pos_to_occs.setdefault(pos, []).append(i)

    for pos, covering in pos_to_occs.items():
        if len(covering) > 1:
            A_row = np.zeros(n_vars)
            for i in covering:
                A_row[i] = 1
            A_ub_rows.append(A_row)
            b_ub.append(1.0)

    # 2. Pattern activation: x_i <= y_p for all i belonging to pattern p
    for pattern, occ_indices in pattern_to_occs.items():
        p_idx = pattern_idx[pattern]
        for occ_idx in occ_indices:
            # x_i - y_p <= 0
            A_row = np.zeros(n_vars)
            A_row[occ_idx] = 1
            A_row[n_occs + p_idx] = -1
            A_ub_rows.append(A_row)
            b_ub.append(0.0)

    # 3. Compressibility: sum(x_i for i in pattern p) >= min_count * y_p
    # Rewritten as: min_count * y_p - sum(x_i) <= 0
    for pattern, occ_indices in pattern_to_occs.items():
        p_idx = pattern_idx[pattern]
        length = len(pattern)
        min_count = _min_count_for_compressibility(length, extra_cost)

        if min_count > len(occ_indices):
            # Pattern can never be compressible, force y_p = 0
            A_row = np.zeros(n_vars)
            A_row[n_occs + p_idx] = 1
            A_ub_rows.append(A_row)
            b_ub.append(0.0)
        else:
            # min_count * y_p - sum(x_i) <= 0
            A_row = np.zeros(n_vars)
            A_row[n_occs + p_idx] = min_count
            for occ_idx in occ_indices:
                A_row[occ_idx] = -1
            A_ub_rows.append(A_row)
            b_ub.append(0.0)

    # Convert to arrays
    if A_ub_rows:
        A_ub = np.vstack(A_ub_rows)
        b_ub_arr = np.array(b_ub)
        constraints = [LinearConstraint(A_ub, -np.inf, b_ub_arr)]
    else:
        constraints = []

    # Bounds: all variables are binary [0, 1]
    bounds = Bounds(lb=np.zeros(n_vars), ub=np.ones(n_vars))

    # All variables are integers (binary)
    integrality = np.ones(n_vars, dtype=int)

    # Solve
    try:
        result = milp(
            c=c,
            constraints=constraints,
            bounds=bounds,
            integrality=integrality,
            options={"time_limit": time_limit_seconds},
        )

        if result.success or (hasattr(result, "x") and result.x is not None):
            selected: list[Occurrence] = []
            x = result.x

            # Extract selected occurrences
            for i, occ in enumerate(occurrences):
                if x[i] > 0.5:
                    selected.append(occ)

            # Verify compressibility of selected patterns
            grouped: dict[tuple, list[Occurrence]] = {}
            for occ in selected:
                grouped.setdefault(occ.subsequence, []).append(occ)

            final_selected: list[Occurrence] = []
            for subseq, group in grouped.items():
                if is_compressible(len(subseq), len(group), extra_cost):
                    final_selected.extend(group)

            final_selected.sort(key=lambda o: o.start)
            return final_selected
    except Exception:
        pass

    # Fall back to beam search on solver failure
    from .selection import _beam_search_with_savings

    return _beam_search_with_savings(occurrences, config.beam_width, config)


def select_occurrences_ilp_relaxed(
    candidates: Iterable[Candidate],
    config: CompressionConfig,
    time_limit_seconds: float = 0.5,
) -> list[Occurrence]:
    """Relaxed ILP selection for faster approximate solutions.

    Uses LP relaxation with rounding heuristics for larger problems.
    """
    try:
        import numpy as np
        from scipy.optimize import linprog
    except ImportError:
        from .selection import _beam_search_with_savings, _build_occurrences

        fallback_occurrences = _build_occurrences(candidates)
        return _beam_search_with_savings(
            fallback_occurrences, config.beam_width, config
        )

    # Build occurrence list
    occurrences: list[Occurrence] = []
    pattern_to_occs: dict[tuple, list[int]] = {}
    candidates_list = list(candidates)

    for cand in candidates_list:
        for pos in cand.positions:
            occ_idx = len(occurrences)
            occurrences.append(
                Occurrence(
                    start=pos,
                    length=cand.length,
                    subsequence=cand.subsequence,
                    priority=cand.priority,
                    patches=cand.patches.get(pos, ()),
                )
            )
            pattern_to_occs.setdefault(cand.subsequence, []).append(occ_idx)

    if not occurrences:
        return []

    n_occs = len(occurrences)
    extra_cost = 1 if config.dict_length_enabled else 0

    # Simplified LP: just maximize savings without pattern variables
    # Objective: maximize sum of (length - 1) for selected occurrences
    c = np.array([-(occ.length - 1 + occ.priority * 0.1) for occ in occurrences])

    # Non-overlapping constraints
    pos_to_occs: dict[int, list[int]] = {}
    for i, occ in enumerate(occurrences):
        for pos in range(occ.start, occ.start + occ.length):
            pos_to_occs.setdefault(pos, []).append(i)

    A_ub_rows: list[list[float]] = []
    b_ub: list[float] = []

    for pos, covering in pos_to_occs.items():
        if len(covering) > 1:
            row = [0.0] * n_occs
            for i in covering:
                row[i] = 1.0
            A_ub_rows.append(row)
            b_ub.append(1.0)

    if A_ub_rows:
        A_ub = np.array(A_ub_rows)
        b_ub_arr = np.array(b_ub)
    else:
        A_ub = None
        b_ub_arr = None

    bounds = [(0, 1) for _ in range(n_occs)]

    try:
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub_arr,
            bounds=bounds,
            method="highs",
        )

        if result.success:
            # Round LP solution with threshold
            threshold = 0.5
            selected_indices = [i for i, val in enumerate(result.x) if val > threshold]

            # Greedily resolve remaining conflicts
            selected_indices.sort(key=lambda i: -result.x[i])
            occupied: set[int] = set()
            final_indices: list[int] = []

            for i in selected_indices:
                occ = occurrences[i]
                positions = set(range(occ.start, occ.start + occ.length))
                if not positions & occupied:
                    final_indices.append(i)
                    occupied |= positions

            selected = [occurrences[i] for i in final_indices]

            # Filter for compressibility
            grouped: dict[tuple, list[Occurrence]] = {}
            for occ in selected:
                grouped.setdefault(occ.subsequence, []).append(occ)

            final_selected: list[Occurrence] = []
            for subseq, group in grouped.items():
                if is_compressible(len(subseq), len(group), extra_cost):
                    final_selected.extend(group)

            final_selected.sort(key=lambda o: o.start)
            return final_selected
    except Exception:
        pass

    # Fall back
    from .selection import _beam_search_with_savings

    return _beam_search_with_savings(occurrences, config.beam_width, config)
