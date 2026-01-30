"""Core compression and decompression APIs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .config import CompressionConfig
from .dictionary import build_body_tokens
from .dictionary_store import CompressionDictionary
from .ast_python import discover_ast_candidates
from .domain import detect_domain
from .engine import CompressionEngine, default_engine
from .metrics import compute_metrics, log_metrics
from .metrics_writer import write_combined_metrics_jsonl, write_metrics_jsonl
from .serialization import serialize
from .static_dicts import (
    DOMAIN_TO_STATIC_ID,
    get_static_dictionary,
    parse_static_dictionary_marker,
)
from .types import CompressionResult, Token, TokenSeq
from .quality_predictor import create_predictor
from .utils import (
    is_meta_token,
    parse_length_token,
    parse_patch_index_token,
    require_no_reserved_tokens,
)


def _apply_static_dictionary(
    tokens: list[Token],
    static_dict: Mapping[Token, tuple[Token, ...]],
    config: CompressionConfig,
) -> tuple[list[Token], dict[int, tuple[int, Token, tuple[Any, ...]]]]:
    if any(token in static_dict for token in tokens):
        raise ValueError("Input sequence contains static meta-tokens.")
    occupied = [False] * len(tokens)
    replacements: dict[int, tuple[int, Token, tuple[Any, ...]]] = {}
    entries = sorted(static_dict.items(), key=lambda item: len(item[1]), reverse=True)
    for meta, subseq in entries:
        if len(subseq) < config.static_dictionary_min_length:
            continue
        idx = 0
        while idx <= len(tokens) - len(subseq):
            if any(occupied[idx : idx + len(subseq)]):
                idx += 1
                continue
            if tuple(tokens[idx : idx + len(subseq)]) == subseq:
                replacements[idx] = (len(subseq), meta, ())
                for pos in range(idx, idx + len(subseq)):
                    occupied[pos] = True
                idx += len(subseq)
            else:
                idx += 1
    if not replacements:
        return tokens, replacements
    return build_body_tokens(tokens, replacements, config), replacements


def _select_static_dictionary(
    tokens: list[Token], config: CompressionConfig
) -> str | None:
    if config.static_dictionary_id:
        return config.static_dictionary_id
    if not config.static_dictionary_auto:
        return None
    detection = detect_domain(tokens, config)
    if (
        detection.domain is None
        or detection.confidence < config.static_dictionary_min_confidence
    ):
        return None
    return DOMAIN_TO_STATIC_ID.get(detection.domain)


def _compress_internal(
    tokens: TokenSeq,
    config: CompressionConfig,
    preferred_candidates: list | None = None,
) -> CompressionResult:
    cfg = config or CompressionConfig()
    require_no_reserved_tokens(tokens, cfg)

    working_tokens = list(tokens)
    static_id = _select_static_dictionary(working_tokens, cfg)
    static_dict: Mapping[Token, tuple[Token, ...]] | None = None
    static_replacements: dict[int, tuple[int, Token, tuple[Any, ...]]] = {}
    if static_id:
        static_entry = get_static_dictionary(static_id)
        if static_entry is None:
            raise ValueError("Unknown static dictionary id.")
        static_dict = static_entry.entries
        working_tokens, static_replacements = _apply_static_dictionary(
            working_tokens, static_dict, cfg
        )

    engine = default_engine(cfg)
    if preferred_candidates:
        # Prepend preferred candidates by injecting a temporary discovery stage.
        class _PreferredStage:
            name = "preferred"

            def discover(self, tokens: TokenSeq, config: CompressionConfig) -> list:
                return preferred_candidates or []

        engine = CompressionEngine((_PreferredStage(),) + engine.discovery_stages)  # type: ignore[arg-type]

    working_tokens, dictionary_map = engine.compress_tokens(working_tokens, cfg)

    body_tokens = working_tokens
    dictionary = CompressionDictionary(
        meta_to_seq=dict(dictionary_map),
        seq_to_meta={},
        max_entries=cfg.meta_token_pool_size,
    )
    for meta, subseq in dictionary_map.items():
        dictionary.seq_to_meta[subseq] = meta
    serialized = serialize(dictionary, body_tokens, cfg, static_dictionary_id=static_id)
    compressed_tokens = list(body_tokens)

    if len(serialized.tokens) > len(tokens):
        compressed_tokens = list(tokens)
        serialized_tokens = list(tokens)
        dictionary_tokens = []
        body_tokens = list(tokens)
        dictionary_map = {}
        dictionary = CompressionDictionary(
            meta_to_seq={}, seq_to_meta={}, max_entries=cfg.meta_token_pool_size
        )
        static_id = None
    else:
        serialized_tokens = serialized.tokens
        dictionary_tokens = serialized.dictionary_tokens

    # Build initial result for quality prediction
    initial_result = CompressionResult(
        original_tokens=tuple(tokens),
        compressed_tokens=list(body_tokens),
        serialized_tokens=serialized_tokens,
        dictionary_tokens=dictionary_tokens,
        body_tokens=body_tokens,
        dictionary_map=dictionary_map,
        meta_tokens_used=tuple(dictionary_map.keys()),
        original_length=len(tokens),
        compressed_length=len(serialized_tokens),
        static_dictionary_id=static_id,
        dictionary=dictionary,
    )

    # Quality prediction validation pass
    if cfg.enable_quality_prediction and dictionary_map:
        predictor = create_predictor(cfg.quality_task_type)
        if cfg.quality_conservative:
            predictor = create_predictor(cfg.quality_task_type)
            predictor = type(predictor)(
                model_path=predictor.model_path,
                task_type=predictor.task_type,
                conservative=True,
            )

        prediction = predictor.predict(tokens, initial_result)

        if prediction.predicted_degradation > cfg.max_predicted_degradation:
            # Quality risk too high - try with conservative settings or return original
            if prediction.recommendation == "partial" and len(dictionary_map) > 0:
                # Retry with more conservative settings (shorter max_subsequence_length)
                conservative_cfg = CompressionConfig(
                    **{
                        **cfg.__dict__,
                        "max_subsequence_length": min(4, cfg.max_subsequence_length),
                        "hierarchical_enabled": False,
                        "enable_quality_prediction": False,  # Don't recurse
                    }
                )
                return _compress_internal(
                    tokens, conservative_cfg, preferred_candidates
                )
            elif prediction.recommendation == "skip":
                # Return original tokens (no compression)
                return CompressionResult(
                    original_tokens=tuple(tokens),
                    compressed_tokens=list(tokens),
                    serialized_tokens=list(tokens),
                    dictionary_tokens=[],
                    body_tokens=list(tokens),
                    dictionary_map={},
                    meta_tokens_used=(),
                    original_length=len(tokens),
                    compressed_length=len(tokens),
                    static_dictionary_id=None,
                    dictionary=CompressionDictionary(
                        meta_to_seq={},
                        seq_to_meta={},
                        max_entries=cfg.meta_token_pool_size,
                    ),
                )

    result = CompressionResult(
        original_tokens=tuple(tokens),
        compressed_tokens=compressed_tokens,
        serialized_tokens=serialized_tokens,
        dictionary_tokens=dictionary_tokens,
        body_tokens=body_tokens,
        dictionary_map=dictionary_map,
        meta_tokens_used=tuple(dictionary_map.keys()),
        original_length=len(tokens),
        compressed_length=len(serialized_tokens),
        static_dictionary_id=static_id,
        dictionary=dictionary,
    )

    if cfg.metrics_enabled:
        metrics = compute_metrics(
            original_length=len(tokens),
            compressed_length=len(serialized_tokens),
            dictionary_tokens=dictionary_tokens,
            dictionary_map=dictionary_map,
            body_tokens=body_tokens,
            candidates_discovered=engine.last_candidates_discovered,
            config=cfg,
        )
        result = CompressionResult(**{**result.__dict__, "metrics": metrics})
        log_metrics(metrics)
        if cfg.metrics_jsonl_path:
            write_metrics_jsonl(cfg.metrics_jsonl_path, metrics)
        cache_stats = cfg.cache_stats
        if cache_stats is None and cfg.cache_stats_source is not None:
            if hasattr(cfg.cache_stats_source, "stats"):
                cache_stats = cfg.cache_stats_source.stats()
        if cfg.combined_metrics_jsonl_path and cache_stats:
            write_combined_metrics_jsonl(
                cfg.combined_metrics_jsonl_path, metrics, cache_stats
            )

    if cfg.verify:
        result.verify(tokens, cfg)

    return result


def compress(
    tokens: TokenSeq, config: CompressionConfig | None = None
) -> CompressionResult:
    cfg = config or CompressionConfig()
    return _compress_internal(tokens, cfg, preferred_candidates=None)


def compress_python_source(
    source: str, config: CompressionConfig | None = None
) -> tuple[list[Token], CompressionResult]:
    cfg = config or CompressionConfig()
    tokens: list[Token]
    ast_candidates: list[Any]
    if cfg.ast_enabled:
        tokens, ast_candidates = discover_ast_candidates(source, cfg)
    else:
        tokens = list(source.split())
        ast_candidates = []
    result = _compress_internal(tokens, cfg, preferred_candidates=ast_candidates)
    return tokens, result


def _expand_token(
    token: Token,
    dictionary_map: dict[Token, list[Token]],
    cfg: CompressionConfig,
    memo: dict[Token, list[Token]],
) -> list[Token]:
    if token in memo:
        return memo[token]
    if token not in dictionary_map:
        return [token]
    expanded: list[Token] = []
    for item in dictionary_map[token]:
        expanded.extend(_expand_token(item, dictionary_map, cfg, memo))
    memo[token] = expanded
    return expanded


def decompress(
    tokens: Sequence[Token], config: CompressionConfig | None = None
) -> list[Token]:
    cfg = config or CompressionConfig()
    if not tokens:
        return []
    idx = 0
    static_dict: dict[Token, tuple[Token, ...]] = {}
    static_id = parse_static_dictionary_marker(tokens[0], cfg)
    if static_id:
        entry = get_static_dictionary(static_id)
        if entry is None:
            raise ValueError("Unknown static dictionary id.")
        static_dict = dict(entry.entries)
        idx = 1
    if idx >= len(tokens) or tokens[idx] != cfg.dict_start_token:
        if static_dict:
            raise ValueError(
                "Compressed sequence does not start with dictionary delimiter."
            )
        return list(tokens)

    try:
        end_idx = tokens.index(cfg.dict_end_token, idx + 1)
    except ValueError as exc:
        raise ValueError(
            "Compressed sequence missing dictionary end delimiter."
        ) from exc

    dict_tokens = tokens[idx + 1 : end_idx]
    body_tokens = tokens[end_idx + 1 :]

    dictionary_map: dict[Token, list[Token]] = {}
    if cfg.dict_length_enabled:
        idx = 0
        while idx < len(dict_tokens):
            meta = dict_tokens[idx]
            if not is_meta_token(meta, cfg):
                raise ValueError("Dictionary entry missing meta-token header.")
            if meta in dictionary_map:
                raise ValueError("Duplicate meta-token in dictionary.")
            if idx + 1 >= len(dict_tokens):
                raise ValueError("Dictionary entry missing length token.")
            entry_length = parse_length_token(dict_tokens[idx + 1], cfg)
            start = idx + 2
            end = start + entry_length
            if end > len(dict_tokens):
                raise ValueError("Dictionary entry length exceeds dictionary bounds.")
            dictionary_map[meta] = list(dict_tokens[start:end])
            idx = end
    else:
        current_meta: Token | None = None
        for token in dict_tokens:
            if is_meta_token(token, cfg):
                if token in dictionary_map:
                    raise ValueError("Duplicate meta-token in dictionary.")
                current_meta = token
                dictionary_map[current_meta] = []
                continue
            if current_meta is None:
                raise ValueError("Dictionary entry missing meta-token header.")
            dictionary_map[current_meta].append(token)

        if current_meta is None and dict_tokens:
            raise ValueError("Dictionary did not contain any meta-tokens.")

    for meta, subseq in dictionary_map.items():
        if not subseq:
            raise ValueError("Empty dictionary entry for meta-token.")

    for static_meta, static_subseq in static_dict.items():
        if static_meta in dictionary_map:
            raise ValueError("Static and dynamic dictionaries share a meta-token.")
        dictionary_map[static_meta] = list(static_subseq)

    decoded: list[Token] = []
    memo: dict[Token, list[Token]] = {}
    idx = 0
    while idx < len(body_tokens):
        token = body_tokens[idx]
        if token in dictionary_map:
            if (
                idx + 1 < len(body_tokens)
                and body_tokens[idx + 1] == cfg.patch_start_token
            ):
                idx += 2
                patches: list[tuple[int, Token]] = []
                while (
                    idx < len(body_tokens) and body_tokens[idx] != cfg.patch_end_token
                ):
                    patch_index = parse_patch_index_token(body_tokens[idx], cfg)
                    if idx + 1 >= len(body_tokens):
                        raise ValueError("Patch entry missing replacement token.")
                    patch_value = body_tokens[idx + 1]
                    patches.append((patch_index, patch_value))
                    idx += 2
                if idx >= len(body_tokens) or body_tokens[idx] != cfg.patch_end_token:
                    raise ValueError("Patch section missing end delimiter.")
                idx += 1
                expanded = list(_expand_token(token, dictionary_map, cfg, memo))
                for patch_index, patch_value in patches:
                    if patch_index < 0 or patch_index >= len(expanded):
                        raise ValueError("Patch index out of bounds.")
                    expanded[patch_index] = patch_value
                decoded.extend(expanded)
                continue
            decoded.extend(_expand_token(token, dictionary_map, cfg, memo))
            idx += 1
            continue
        decoded.append(token)
        idx += 1
    return decoded


def decompress_with_dictionary(
    dictionary: dict[Token, tuple[Token, ...]],
    body_tokens: Sequence[Token],
    config: CompressionConfig | None = None,
) -> list[Token]:
    cfg = config or CompressionConfig()
    dictionary_map: dict[Token, list[Token]] = {
        meta: list(seq) for meta, seq in dictionary.items()
    }
    decoded: list[Token] = []
    memo: dict[Token, list[Token]] = {}
    idx = 0
    body_list = list(body_tokens)
    while idx < len(body_list):
        token = body_list[idx]
        if token in dictionary_map:
            if idx + 1 < len(body_list) and body_list[idx + 1] == cfg.patch_start_token:
                idx += 2
                patches: list[tuple[int, Token]] = []
                while idx < len(body_list) and body_list[idx] != cfg.patch_end_token:
                    patch_index = parse_patch_index_token(body_list[idx], cfg)
                    if idx + 1 >= len(body_list):
                        raise ValueError("Patch entry missing replacement token.")
                    patch_value = body_list[idx + 1]
                    patches.append((patch_index, patch_value))
                    idx += 2
                if idx >= len(body_list) or body_list[idx] != cfg.patch_end_token:
                    raise ValueError("Patch section missing end delimiter.")
                idx += 1
                expanded = list(_expand_token(token, dictionary_map, cfg, memo))
                for patch_index, patch_value in patches:
                    if patch_index < 0 or patch_index >= len(expanded):
                        raise ValueError("Patch index out of bounds.")
                    expanded[patch_index] = patch_value
                decoded.extend(expanded)
                continue
            decoded.extend(_expand_token(token, dictionary_map, cfg, memo))
            idx += 1
            continue
        decoded.append(token)
        idx += 1
    return decoded
