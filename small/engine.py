"""Compression engine pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

from .config import CompressionConfig
from .dictionary import build_body_tokens
from .discovery import discover_candidates, discover_candidates_chunked
from .discovery_parallel import discover_candidates_parallel
from .discovery_sa import discover_candidates_sa
from .fuzzy import discover_fuzzy_candidates
from .template_discovery import discover_templates
from .adaptive import (
    detect_regions,
    detect_regions_heuristic,
    filter_candidates_by_region,
    RegionType,
)
from .pattern_importance import (
    create_default_scorer,
    adjust_candidate_priorities,
    filter_high_importance_candidates,
)
from .subsumption import prune_subsumed_candidates, deduplicate_candidates
from .swap import perform_swaps
from .types import Candidate, Token, TokenSeq
from .validation import validate_config


@dataclass(frozen=True)
class DiscoveryStage:
    name: str

    def discover(self, tokens: TokenSeq, config: CompressionConfig) -> list[Candidate]:
        raise NotImplementedError


@dataclass(frozen=True)
class ExactDiscoveryStage(DiscoveryStage):
    use_suffix_array: bool = True

    def discover(self, tokens: TokenSeq, config: CompressionConfig) -> list[Candidate]:
        if self.use_suffix_array and config.discovery_mode == "suffix-array":
            return discover_candidates_sa(tokens, config)
        if (
            config.parallel_discovery
            and len(tokens) >= config.parallel_length_threshold
        ):
            return discover_candidates_parallel(tokens, config)
        if config.chunk_size and len(tokens) >= config.chunk_size:
            return discover_candidates_chunked(tokens, config)
        return discover_candidates(tokens, config.max_subsequence_length, config)


@dataclass(frozen=True)
class FuzzyDiscoveryStage(DiscoveryStage):
    def discover(self, tokens: TokenSeq, config: CompressionConfig) -> list[Candidate]:
        return discover_fuzzy_candidates(tokens, config)


@dataclass(frozen=True)
class TemplateDiscoveryStage(DiscoveryStage):
    """Discovery stage for parameterized template patterns."""

    def discover(self, tokens: TokenSeq, config: CompressionConfig) -> list[Candidate]:
        """Discover template patterns.
        
        Note: This is a discovery-only stage for now. It identifies parameterized
        patterns but doesn't convert them to standard Candidates because:
        1. Template frames contain slots - the actual tokens at instance positions
           include slot values that differ between instances
        2. Full template compression requires specialized serialization
        
        The discovered templates are logged/tracked but actual compression
        relies on the exact pattern discovery finding the common subsequences.
        Future: Integrate full template serialization for slot-aware compression.
        """
        # Discover templates for analysis/logging purposes
        # Template discovery itself runs, but we don't convert to standard
        # candidates since the positions don't map to exact matches
        _template_candidates = discover_templates(tokens, config)
        
        # For now, return empty - exact discovery will find what it can
        # Full template integration would require serialization changes
        # to handle <SlotVal> markers in the body
        return []


@dataclass(frozen=True)
class CompressionEngine:
    discovery_stages: tuple[DiscoveryStage, ...]
    last_candidates_discovered: int = 0
    min_improvement_ratio: float = 0.02  # Stop if < 2% improvement
    min_efficiency_ratio: float = 1.5  # Stop if body_savings/dict_growth < this

    def compress_tokens(
        self, tokens: TokenSeq, config: CompressionConfig
    ) -> tuple[list[Token], dict[Token, tuple[Token, ...]]]:
        for warning in validate_config(config):
            warnings.warn(warning.message, RuntimeWarning)
        working_tokens = list(tokens)
        dictionary_map: dict[Token, tuple[Token, ...]] = {}
        depth_limit = (
            config.hierarchical_max_depth if config.hierarchical_enabled else 1
        )
        total_candidates = 0

        prev_length = len(working_tokens)

        for depth in range(depth_limit):
            candidates: list[Candidate] = []
            for stage in self.discovery_stages:
                candidates.extend(stage.discover(working_tokens, config))

            if not candidates:
                break

            # Deduplicate candidates from different discovery stages
            candidates = deduplicate_candidates(candidates)

            # Adaptive region-aware compression
            if config.enable_adaptive_regions and candidates:
                # Detect regions in the input
                if config.region_markers:
                    # Convert string region types to RegionType enum
                    marker_mapping = {
                        marker: RegionType(region_type)
                        for marker, region_type in config.region_markers.items()
                    }
                    regions = detect_regions(working_tokens, markers=marker_mapping)
                else:
                    regions = detect_regions(working_tokens)

                # Fall back to heuristic detection if no regions found
                if not regions:
                    regions = detect_regions_heuristic(
                        working_tokens,
                        system_fraction=config.adaptive_system_fraction,
                    )

                # Filter and adjust candidates based on regions
                if regions:
                    candidates = filter_candidates_by_region(
                        candidates, regions, working_tokens
                    )

            # Prune subsumed patterns to reduce dictionary redundancy
            if config.enable_subsumption_pruning:
                candidates = prune_subsumed_candidates(
                    candidates,
                    config,
                    min_independent_occurrences=config.subsumption_min_independent,
                )

            # ML integration: Score pattern importance and adjust priorities
            if config.use_importance_scoring and candidates:
                scorer = create_default_scorer()
                importance_scores = scorer.score_patterns(working_tokens, candidates)

                # Filter out high-importance patterns (semantically critical)
                if config.importance_filter_threshold < 1.0:
                    candidates = filter_high_importance_candidates(
                        candidates,
                        importance_scores,
                        threshold=config.importance_filter_threshold,
                    )
                    # Recompute scores for remaining candidates
                    if candidates:
                        importance_scores = scorer.score_patterns(
                            working_tokens, candidates
                        )

                # Adjust priorities based on importance (low importance = compress more)
                if candidates and importance_scores:
                    candidates = adjust_candidate_priorities(
                        candidates,
                        importance_scores,
                        importance_weight=config.importance_weight,
                    )

            total_candidates += len(candidates)
            if not candidates:
                break

            swap_result = perform_swaps(working_tokens, candidates, config)
            if not swap_result.dictionary_map:
                break
            dictionary_map.update(swap_result.dictionary_map)
            working_tokens = build_body_tokens(
                working_tokens, swap_result.replacements, config
            )

            # Early stopping: check for diminishing returns using multiple criteria
            new_length = len(working_tokens)
            new_dict_size = sum(
                1 + len(seq) + (1 if config.dict_length_enabled else 0)
                for seq in swap_result.dictionary_map.values()
            )

            if prev_length > 0 and depth > 0:
                # Criterion 1: Relative improvement in body size
                improvement = (prev_length - new_length) / prev_length

                # Criterion 2: Efficiency ratio (body savings vs dictionary growth)
                body_savings = prev_length - new_length
                dict_growth = new_dict_size
                efficiency = (
                    body_savings / dict_growth if dict_growth > 0 else float("inf")
                )

                # Stop if either:
                # 1. Body improvement is too small, OR
                # 2. The ratio of body compression to dictionary expansion is unfavorable
                if improvement < self.min_improvement_ratio:
                    break
                if efficiency < self.min_efficiency_ratio and dict_growth > 0:
                    # Not efficient enough - dictionary is growing faster than body is shrinking
                    break

            prev_length = new_length

            if not config.hierarchical_enabled:
                break

        object.__setattr__(self, "last_candidates_discovered", total_candidates)
        return working_tokens, dictionary_map


def default_engine(config: CompressionConfig) -> CompressionEngine:
    stages: list[DiscoveryStage] = [
        ExactDiscoveryStage(name="exact-sa", use_suffix_array=True)
    ]
    if config.fuzzy_enabled:
        stages.insert(0, FuzzyDiscoveryStage(name="fuzzy"))
    if config.enable_template_extraction:
        stages.append(TemplateDiscoveryStage(name="template"))
    return CompressionEngine(tuple(stages))
