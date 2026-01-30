"""Adaptive compression with region-aware strategies.

Different parts of a prompt need different compression levels:
- System instructions: Preserve carefully (minimal compression)
- User input: Moderate compression
- Retrieved context: Can compress aggressively
- Code blocks: Syntax-aware compression

This module provides:
1. Region detection based on markers or heuristics
2. Per-region compression limits and priorities
3. Candidate filtering based on region settings
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from .config import CompressionConfig
from .types import Candidate, Token, TokenSeq


class RegionType(Enum):
    """Types of semantic regions in prompts."""

    SYSTEM = "system"  # System instructions - preserve carefully
    USER = "user"  # User input - moderate compression
    ASSISTANT = "assistant"  # Assistant responses in history
    CONTEXT = "context"  # Retrieved/injected context - compress aggressively
    CODE = "code"  # Code blocks - syntax-aware
    DATA = "data"  # Structured data (JSON, etc.) - high compression
    UNKNOWN = "unknown"  # Default/unclassified


@dataclass(frozen=True)
class Region:
    """A region of the input with specific compression strategy.

    Attributes:
        start: Start position (inclusive)
        end: End position (exclusive)
        region_type: Semantic type of the region
        max_compression_ratio: Maximum allowed compression (1.0 = no compression)
        priority_boost: Priority boost for patterns in this region (higher = prefer compression)
    """

    start: int
    end: int
    region_type: RegionType
    max_compression_ratio: float
    priority_boost: int


# Default region settings: (max_compression_ratio, priority_boost)
DEFAULT_REGION_SETTINGS: dict[RegionType, tuple[float, int]] = {
    RegionType.SYSTEM: (0.95, -2),  # Almost no compression, negative priority
    RegionType.USER: (0.85, 0),  # Moderate compression
    RegionType.ASSISTANT: (0.80, 1),  # Can compress history
    RegionType.CONTEXT: (0.50, 3),  # Aggressive compression
    RegionType.CODE: (0.75, 1),  # Moderate, syntax-aware
    RegionType.DATA: (0.40, 4),  # Very aggressive for repetitive data
    RegionType.UNKNOWN: (0.80, 0),  # Default moderate
}

# Default marker patterns for region detection
DEFAULT_MARKERS: dict[str, RegionType] = {
    "[SYSTEM]": RegionType.SYSTEM,
    "[/SYSTEM]": RegionType.SYSTEM,  # End marker
    "<<SYS>>": RegionType.SYSTEM,
    "<</SYS>>": RegionType.SYSTEM,
    "[INST]": RegionType.USER,
    "[/INST]": RegionType.USER,
    "[USER]": RegionType.USER,
    "[ASSISTANT]": RegionType.ASSISTANT,
    "[CONTEXT]": RegionType.CONTEXT,
    "[/CONTEXT]": RegionType.CONTEXT,
    "```": RegionType.CODE,  # Code fence
    "{": RegionType.DATA,  # JSON-like
}


def detect_regions(
    tokens: TokenSeq,
    markers: dict[str, RegionType] | None = None,
    default_type: RegionType = RegionType.UNKNOWN,
) -> list[Region]:
    """Detect semantic regions in the input based on markers.

    Scans through tokens looking for marker patterns that indicate
    region boundaries. Returns a list of non-overlapping regions
    covering the entire input.

    Args:
        tokens: Input token sequence
        markers: Mapping from marker strings to region types
        default_type: Default region type for unmarked content

    Returns:
        List of Region objects covering the entire input
    """
    if not tokens:
        return []

    if markers is None:
        markers = DEFAULT_MARKERS

    regions: list[Region] = []
    current_type = default_type
    current_start = 0
    code_fence_open = False

    for i, token in enumerate(tokens):
        token_str = str(token)

        # Check for code fence toggle
        if "```" in token_str:
            if code_fence_open:
                # Close code region
                if current_type == RegionType.CODE and current_start < i:
                    regions.append(_make_region(current_start, i + 1, RegionType.CODE))
                    current_start = i + 1
                    current_type = default_type
                code_fence_open = False
            else:
                # Open code region
                if i > current_start:
                    regions.append(_make_region(current_start, i, current_type))
                current_start = i
                current_type = RegionType.CODE
                code_fence_open = True
            continue

        # Skip region detection inside code blocks
        if code_fence_open:
            continue

        # Check for other markers
        for marker, region_type in markers.items():
            if marker == "```":  # Already handled
                continue
            if marker == "{":  # Special handling for JSON
                if token_str.strip() == "{":
                    # Start of potential JSON block
                    if i > current_start:
                        regions.append(_make_region(current_start, i, current_type))
                    current_start = i
                    current_type = RegionType.DATA
                continue

            if marker in token_str:
                # Close previous region
                if i > current_start:
                    regions.append(_make_region(current_start, i, current_type))

                current_type = region_type
                current_start = i
                break

    # Close final region
    if len(tokens) > current_start:
        regions.append(_make_region(current_start, len(tokens), current_type))

    # Merge adjacent regions of the same type
    regions = _merge_adjacent_regions(regions)

    return regions


def _make_region(start: int, end: int, region_type: RegionType) -> Region:
    """Create a region with settings from defaults."""
    settings = DEFAULT_REGION_SETTINGS.get(region_type, (0.8, 0))
    max_ratio, priority = settings

    return Region(
        start=start,
        end=end,
        region_type=region_type,
        max_compression_ratio=max_ratio,
        priority_boost=priority,
    )


def _merge_adjacent_regions(regions: list[Region]) -> list[Region]:
    """Merge adjacent regions of the same type."""
    if not regions:
        return []

    merged: list[Region] = []
    current = regions[0]

    for region in regions[1:]:
        if region.region_type == current.region_type and region.start == current.end:
            # Merge
            current = Region(
                start=current.start,
                end=region.end,
                region_type=current.region_type,
                max_compression_ratio=current.max_compression_ratio,
                priority_boost=current.priority_boost,
            )
        else:
            merged.append(current)
            current = region

    merged.append(current)
    return merged


def detect_regions_heuristic(
    tokens: TokenSeq,
    system_fraction: float = 0.1,
) -> list[Region]:
    """Detect regions using heuristics when markers aren't present.

    Uses position-based heuristics:
    - First N% is likely system/context
    - Rest is user interaction

    Args:
        tokens: Input token sequence
        system_fraction: Fraction of input assumed to be system context

    Returns:
        List of Region objects
    """
    if not tokens:
        return []

    n = len(tokens)
    system_end = int(n * system_fraction)

    regions: list[Region] = []

    if system_end > 0:
        regions.append(_make_region(0, system_end, RegionType.SYSTEM))

    if system_end < n:
        regions.append(_make_region(system_end, n, RegionType.USER))

    return regions


def filter_candidates_by_region(
    candidates: list[Candidate],
    regions: list[Region],
    tokens: TokenSeq,
) -> list[Candidate]:
    """Filter and adjust candidates based on their containing regions.

    For each candidate occurrence:
    1. Determine which region it falls in
    2. Apply region's priority boost
    3. Filter out candidates in regions with very low compression ratio

    Args:
        candidates: Original candidate list
        regions: Detected regions
        tokens: Original token sequence (for reference)

    Returns:
        Adjusted candidate list
    """
    if not candidates or not regions:
        return candidates

    adjusted: list[Candidate] = []

    for cand in candidates:
        valid_positions: list[int] = []
        priority_boosts: list[int] = []

        for pos in cand.positions:
            # Find containing region
            containing_region: Region | None = None
            for region in regions:
                if region.start <= pos < region.end:
                    containing_region = region
                    break

            if containing_region is None:
                # No region found, keep with default settings
                valid_positions.append(pos)
                priority_boosts.append(0)
                continue

            # Check if compression is allowed in this region
            # Very conservative regions (ratio > 0.95) effectively disable compression
            if containing_region.max_compression_ratio >= 0.98:
                # Skip this position
                continue

            valid_positions.append(pos)
            priority_boosts.append(containing_region.priority_boost)

        if not valid_positions:
            continue

        # Compute average priority boost
        avg_boost = (
            sum(priority_boosts) // len(priority_boosts) if priority_boosts else 0
        )

        adjusted.append(
            Candidate(
                subsequence=cand.subsequence,
                length=cand.length,
                positions=tuple(valid_positions),
                priority=cand.priority + avg_boost,
                patches=cand.patches,
            )
        )

    return adjusted


@dataclass
class AdaptiveCompressionConfig:
    """Configuration for adaptive compression."""

    enable_region_detection: bool = True
    custom_markers: dict[str, RegionType] | None = None
    min_region_length: int = 10
    use_heuristics: bool = True
    system_fraction: float = 0.1


def compress_adaptive(
    tokens: TokenSeq,
    base_config: CompressionConfig,
    adaptive_config: AdaptiveCompressionConfig | None = None,
) -> tuple[list[Token], dict[Token, tuple[Token, ...]]]:
    """Compress with region-aware adaptive strategy.

    This is a convenience wrapper that:
    1. Detects regions in the input
    2. Filters/adjusts candidates based on regions
    3. Runs compression with adjusted candidates

    Args:
        tokens: Input token sequence
        base_config: Base compression configuration
        adaptive_config: Adaptive compression settings

    Returns:
        Tuple of (compressed_tokens, dictionary_map)
    """
    from .compressor import compress

    if adaptive_config is None:
        adaptive_config = AdaptiveCompressionConfig()

    if not adaptive_config.enable_region_detection:
        result = compress(tokens, base_config)
        return result.compressed_tokens, result.dictionary_map

    # Detect regions
    if adaptive_config.custom_markers:
        regions = detect_regions(tokens, adaptive_config.custom_markers)
    else:
        regions = detect_regions(tokens)

    # If no regions found and heuristics enabled, use heuristic detection
    if not regions and adaptive_config.use_heuristics:
        regions = detect_regions_heuristic(tokens, adaptive_config.system_fraction)

    # For now, run standard compression
    # Future: pass regions to discovery/selection for filtering
    result = compress(tokens, base_config)
    return result.compressed_tokens, result.dictionary_map


def get_region_stats(regions: list[Region], tokens: TokenSeq) -> dict[str, int]:
    """Get statistics about detected regions.

    Returns a dict with counts for each region type.
    """
    stats: dict[str, int] = {}

    for region in regions:
        name = region.region_type.value
        length = region.end - region.start
        stats[name] = stats.get(name, 0) + length

    return stats
