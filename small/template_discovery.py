"""Template discovery for parameterized pattern compression.

This module implements alignment-based template mining to find repeated
structures with variable "slots". Unlike exact pattern matching, templates
can compress patterns like:

    logger.info("User: alice")
    logger.info("User: bob")
    logger.info("User: charlie")

Into a single template with one slot for the username.

Algorithm:
1. Find candidate groups using n-gram fingerprinting
2. Align sequences within each group to find conserved positions
3. Extract templates from alignments
4. Filter templates by compression benefit
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .config import CompressionConfig
from .template_types import (
    FrameElement,
    SlotMarker,
    Template,
    TemplateCandidate,
    TemplateInstance,
    compute_frame_ratio,
)
from .types import Token, TokenSeq


@dataclass
class _SequenceGroup:
    """A group of similar sequences for alignment."""
    sequences: list[tuple[int, tuple[Token, ...]]]  # (position, tokens)
    representative: tuple[Token, ...]


def discover_templates(
    tokens: TokenSeq,
    config: CompressionConfig,
) -> list[TemplateCandidate]:
    """Discover parameterized templates in a token sequence.
    
    Uses n-gram fingerprinting to find similar sequences, then aligns
    them to extract templates with fixed frames and variable slots.
    
    Args:
        tokens: Input token sequence
        config: Compression configuration
        
    Returns:
        List of template candidates sorted by estimated savings
    """
    if len(tokens) < config.min_subsequence_length * 2:
        return []
    
    min_len = config.min_subsequence_length
    max_len = config.max_subsequence_length
    min_instances = getattr(config, 'template_min_instances', 3)
    max_slots = getattr(config, 'template_max_slots', 3)
    max_slot_length = getattr(config, 'template_max_slot_length', 10)
    min_frame_ratio = getattr(config, 'template_min_frame_ratio', 0.6)
    
    candidates: list[TemplateCandidate] = []
    
    # Try different window sizes
    for window_size in range(max_len, min_len - 1, -1):
        if window_size > len(tokens):
            continue
            
        # Find similar sequence groups using fingerprinting
        groups = _find_similar_groups(
            tokens, 
            window_size, 
            min_instances,
            max_slot_length,
        )
        
        # Extract templates from each group
        for group in groups:
            template = _extract_template_from_group(
                group,
                max_slots,
                max_slot_length,
                min_frame_ratio,
                config,
            )
            if template is not None:
                candidates.append(template)
    
    # Deduplicate and filter
    candidates = _deduplicate_templates(candidates)
    
    # Sort by savings (descending)
    candidates.sort(key=lambda c: -c.savings)
    
    return candidates


def _find_similar_groups(
    tokens: TokenSeq,
    window_size: int,
    min_instances: int,
    max_slot_length: int,
) -> list[_SequenceGroup]:
    """Find groups of similar sequences using n-gram fingerprinting.
    
    Uses a sparse fingerprint (every 3rd token) to group sequences
    that share structure but may differ in specific positions.
    """
    # Build fingerprints for each window position
    # Fingerprint = tuple of tokens at fixed positions (e.g., 0, 3, 6, ...)
    fingerprint_to_positions: dict[tuple, list[int]] = defaultdict(list)
    
    # Use stride of 3 for fingerprint - captures structure while allowing variation
    stride = 3
    
    for i in range(len(tokens) - window_size + 1):
        window = tuple(tokens[i:i + window_size])
        # Create sparse fingerprint
        fingerprint = tuple(window[j] for j in range(0, window_size, stride))
        fingerprint_to_positions[fingerprint].append(i)
    
    # Group positions with same fingerprint
    groups: list[_SequenceGroup] = []
    seen_positions: set[int] = set()
    
    for fingerprint, positions in fingerprint_to_positions.items():
        # Filter to non-overlapping positions
        non_overlapping = _filter_non_overlapping(positions, window_size)
        
        if len(non_overlapping) < min_instances:
            continue
            
        # Skip if all positions already used
        new_positions = [p for p in non_overlapping if p not in seen_positions]
        if len(new_positions) < min_instances:
            continue
        
        # Build the group
        sequences = [
            (pos, tuple(tokens[pos:pos + window_size]))
            for pos in new_positions
        ]
        
        # Check if sequences are actually similar (not identical)
        # Templates are for parameterized patterns, not exact matches
        if _all_identical(sequences):
            continue
        
        # Check if differences are within slot length limits
        if not _differences_within_limits(sequences, max_slot_length):
            continue
        
        groups.append(_SequenceGroup(
            sequences=sequences,
            representative=sequences[0][1],
        ))
        
        # Mark positions as seen
        seen_positions.update(new_positions)
    
    return groups


def _filter_non_overlapping(positions: list[int], length: int) -> list[int]:
    """Filter positions to non-overlapping set."""
    if not positions:
        return []
    
    sorted_positions = sorted(positions)
    result = [sorted_positions[0]]
    next_free = sorted_positions[0] + length
    
    for pos in sorted_positions[1:]:
        if pos >= next_free:
            result.append(pos)
            next_free = pos + length
    
    return result


def _all_identical(sequences: list[tuple[int, tuple[Token, ...]]]) -> bool:
    """Check if all sequences are identical."""
    if len(sequences) < 2:
        return True
    first = sequences[0][1]
    return all(seq == first for _, seq in sequences[1:])


def _differences_within_limits(
    sequences: list[tuple[int, tuple[Token, ...]]],
    max_slot_length: int,
) -> bool:
    """Check if sequence differences are within acceptable limits for templates."""
    if len(sequences) < 2:
        return True
    
    # Find positions where sequences differ
    first = sequences[0][1]
    diff_positions: set[int] = set()
    
    for _, seq in sequences[1:]:
        for i, (a, b) in enumerate(zip(first, seq)):
            if a != b:
                diff_positions.add(i)
    
    # Check that differences don't span too large a region
    if not diff_positions:
        return False  # All identical - not a template case
    
    # Group consecutive differences
    sorted_diffs = sorted(diff_positions)
    max_gap = 0
    current_run = 1
    
    for i in range(1, len(sorted_diffs)):
        if sorted_diffs[i] == sorted_diffs[i-1] + 1:
            current_run += 1
        else:
            max_gap = max(max_gap, current_run)
            current_run = 1
    max_gap = max(max_gap, current_run)
    
    return max_gap <= max_slot_length


def _extract_template_from_group(
    group: _SequenceGroup,
    max_slots: int,
    max_slot_length: int,
    min_frame_ratio: float,
    config: CompressionConfig,
) -> TemplateCandidate | None:
    """Extract a template from a group of similar sequences.
    
    Aligns sequences to find conserved (fixed) and variable (slot) positions.
    """
    sequences = [seq for _, seq in group.sequences]
    positions = [pos for pos, _ in group.sequences]
    
    if len(sequences) < 2:
        return None
    
    # Align sequences to find conserved positions
    alignment = _align_sequences(sequences, max_slot_length)
    
    if alignment is None:
        return None
    
    frame, slot_regions = alignment
    
    # Check constraints
    if len(slot_regions) > max_slots:
        return None
    
    if compute_frame_ratio(frame) < min_frame_ratio:
        return None
    
    # Build instances
    instances: list[TemplateInstance] = []
    for i, (pos, seq) in enumerate(group.sequences):
        slot_values = _extract_slot_values(seq, slot_regions)
        if slot_values is None:
            continue
        
        instances.append(TemplateInstance(
            position=pos,
            slot_values=tuple(tuple(sv) for sv in slot_values),
            original_length=len(seq),
        ))
    
    if len(instances) < getattr(config, 'template_min_instances', 3):
        return None
    
    # Build candidate
    template = Template(
        frame=frame,
        slot_count=len(slot_regions),
        instances=tuple(instances),
    )
    
    extra_cost = 1 if config.dict_length_enabled else 0
    savings = template.estimated_savings(extra_cost)
    
    if savings <= 0:
        return None
    
    return TemplateCandidate(
        frame=frame,
        slot_positions=tuple(i for i, elem in enumerate(frame) if isinstance(elem, SlotMarker)),
        instances=tuple(instances),
        frame_length=template.frame_length(),
        savings=savings,
        priority=1,  # Templates get slight priority
    )


def _align_sequences(
    sequences: list[tuple[Token, ...]],
    max_slot_length: int,
) -> tuple[tuple[FrameElement, ...], list[tuple[int, int]]] | None:
    """Align sequences to extract template frame and slot regions.
    
    Returns:
        Tuple of (frame, slot_regions) where slot_regions is list of (start, end) tuples,
        or None if alignment fails.
    """
    if not sequences:
        return None
    
    # All sequences should be same length for simple alignment
    length = len(sequences[0])
    if not all(len(seq) == length for seq in sequences):
        return None
    
    # Find positions where all sequences agree (conserved) vs differ (variable)
    conserved_mask: list[bool] = []
    
    for i in range(length):
        tokens_at_pos = set(seq[i] for seq in sequences)
        conserved_mask.append(len(tokens_at_pos) == 1)
    
    # Build frame with slots
    frame: list[FrameElement] = []
    slot_regions: list[tuple[int, int]] = []
    slot_index = 0
    
    i = 0
    while i < length:
        if conserved_mask[i]:
            # Conserved position - add token to frame
            frame.append(sequences[0][i])
            i += 1
        else:
            # Start of variable region - find extent
            start = i
            while i < length and not conserved_mask[i]:
                i += 1
            end = i
            
            # Check slot length constraint
            if end - start > max_slot_length:
                return None  # Slot too long
            
            slot_regions.append((start, end))
            frame.append(SlotMarker(index=slot_index, min_length=1, max_length=end - start))
            slot_index += 1
    
    if not slot_regions:
        return None  # No slots means exact match, not template
    
    return tuple(frame), slot_regions


def _extract_slot_values(
    sequence: tuple[Token, ...],
    slot_regions: list[tuple[int, int]],
) -> list[list[Token]] | None:
    """Extract slot values from a sequence given slot regions."""
    slot_values: list[list[Token]] = []
    
    for start, end in slot_regions:
        if end > len(sequence):
            return None
        slot_values.append(list(sequence[start:end]))
    
    return slot_values


def _deduplicate_templates(
    candidates: list[TemplateCandidate],
) -> list[TemplateCandidate]:
    """Remove duplicate or subsumed templates."""
    if not candidates:
        return []
    
    # Sort by savings descending - keep better templates
    sorted_candidates = sorted(candidates, key=lambda c: -c.savings)
    
    result: list[TemplateCandidate] = []
    used_positions: set[int] = set()
    
    for candidate in sorted_candidates:
        # Check if this template's positions overlap with already-selected templates
        candidate_positions = {inst.position for inst in candidate.instances}
        
        # Allow some overlap, but not complete
        overlap = candidate_positions & used_positions
        if len(overlap) > len(candidate_positions) * 0.5:
            continue  # Too much overlap
        
        result.append(candidate)
        used_positions.update(candidate_positions)
    
    return result


def discover_templates_simple(
    tokens: TokenSeq,
    config: CompressionConfig,
) -> list[TemplateCandidate]:
    """Simplified template discovery for common patterns.
    
    This is a faster, simpler algorithm that looks for patterns
    with exactly one variable position (single-slot templates).
    """
    if len(tokens) < config.min_subsequence_length * 2:
        return []
    
    min_len = config.min_subsequence_length
    max_len = config.max_subsequence_length
    min_instances = getattr(config, 'template_min_instances', 3)
    
    candidates: list[TemplateCandidate] = []
    
    # For each possible template length
    for length in range(max_len, min_len - 1, -1):
        if length > len(tokens):
            continue
        
        # Find patterns that differ in exactly one position
        patterns = _find_single_slot_patterns(tokens, length, min_instances)
        
        for frame, instances in patterns:
            template = Template(
                frame=frame,
                slot_count=1,
                instances=tuple(instances),
            )
            
            extra_cost = 1 if config.dict_length_enabled else 0
            savings = template.estimated_savings(extra_cost)
            
            if savings <= 0:
                continue
            
            candidates.append(TemplateCandidate(
                frame=frame,
                slot_positions=tuple(i for i, e in enumerate(frame) if isinstance(e, SlotMarker)),
                instances=tuple(instances),
                frame_length=template.frame_length(),
                savings=savings,
                priority=1,
            ))
    
    # Deduplicate
    candidates = _deduplicate_templates(candidates)
    candidates.sort(key=lambda c: -c.savings)
    
    return candidates


def _find_single_slot_patterns(
    tokens: TokenSeq,
    length: int,
    min_instances: int,
) -> list[tuple[tuple[FrameElement, ...], list[TemplateInstance]]]:
    """Find patterns with exactly one variable position."""
    # Group windows by their "signature" (all positions except one)
    # signature_key -> (slot_position, [(window_start, slot_value)])
    
    results: list[tuple[tuple[FrameElement, ...], list[TemplateInstance]]] = []
    
    # Try each position as the slot
    for slot_pos in range(length):
        # Build signature excluding slot position
        sig_to_instances: dict[tuple, list[tuple[int, Token]]] = defaultdict(list)
        
        for i in range(len(tokens) - length + 1):
            window = tuple(tokens[i:i + length])
            # Signature is window with slot position masked
            sig = window[:slot_pos] + (None,) + window[slot_pos + 1:]
            slot_value = window[slot_pos]
            sig_to_instances[sig].append((i, slot_value))
        
        # Find signatures with enough diverse instances
        for sig, instances in sig_to_instances.items():
            # Filter non-overlapping
            non_overlapping = []
            next_free = -1
            for pos, val in sorted(instances):
                if pos >= next_free:
                    non_overlapping.append((pos, val))
                    next_free = pos + length
            
            if len(non_overlapping) < min_instances:
                continue
            
            # Check that slot values are actually different
            slot_values = set(val for _, val in non_overlapping)
            if len(slot_values) < 2:
                continue  # All same value - not a template
            
            # Build frame
            frame: list[FrameElement] = []
            for j, tok in enumerate(sig):
                if j == slot_pos:
                    frame.append(SlotMarker(index=0))
                else:
                    frame.append(tok)
            
            # Build instances
            template_instances = [
                TemplateInstance(
                    position=pos,
                    slot_values=((val,),),
                    original_length=length,
                )
                for pos, val in non_overlapping
            ]
            
            results.append((tuple(frame), template_instances))
    
    return results
