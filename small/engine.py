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
        if config.parallel_discovery and len(tokens) >= config.parallel_length_threshold:
            return discover_candidates_parallel(tokens, config)
        if config.chunk_size and len(tokens) >= config.chunk_size:
            return discover_candidates_chunked(tokens, config)
        return discover_candidates(tokens, config.max_subsequence_length, config)


@dataclass(frozen=True)
class FuzzyDiscoveryStage(DiscoveryStage):
    def discover(self, tokens: TokenSeq, config: CompressionConfig) -> list[Candidate]:
        return discover_fuzzy_candidates(tokens, config)


@dataclass(frozen=True)
class CompressionEngine:
    discovery_stages: tuple[DiscoveryStage, ...]
    last_candidates_discovered: int = 0

    def compress_tokens(self, tokens: TokenSeq, config: CompressionConfig) -> tuple[list[Token], dict[Token, tuple[Token, ...]]]:
        for warning in validate_config(config):
            warnings.warn(warning.message, RuntimeWarning)
        working_tokens = list(tokens)
        dictionary_map: dict[Token, tuple[Token, ...]] = {}
        depth_limit = config.hierarchical_max_depth if config.hierarchical_enabled else 1
        total_candidates = 0

        for _ in range(depth_limit):
            candidates: list[Candidate] = []
            for stage in self.discovery_stages:
                candidates.extend(stage.discover(working_tokens, config))
            total_candidates += len(candidates)
            if not candidates:
                break
            swap_result = perform_swaps(working_tokens, candidates, config)
            if not swap_result.dictionary_map:
                break
            dictionary_map.update(swap_result.dictionary_map)
            working_tokens = build_body_tokens(working_tokens, swap_result.replacements, config)
            if not config.hierarchical_enabled:
                break

        object.__setattr__(self, "last_candidates_discovered", total_candidates)
        return working_tokens, dictionary_map


def default_engine(config: CompressionConfig) -> CompressionEngine:
    stages: list[DiscoveryStage] = [ExactDiscoveryStage(name="exact-sa", use_suffix_array=True)]
    if config.fuzzy_enabled:
        stages.insert(0, FuzzyDiscoveryStage(name="fuzzy"))
    return CompressionEngine(tuple(stages))
