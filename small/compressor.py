"""Core compression and decompression APIs."""

from __future__ import annotations

from typing import Sequence

from .config import CompressionConfig
from .dictionary import build_body_tokens, build_dictionary_tokens
from .ast_python import discover_ast_candidates
from .domain import detect_domain
from .discovery import discover_candidates
from .swap import perform_swaps
from .static_dicts import (
    DOMAIN_TO_STATIC_ID,
    get_static_dictionary,
    parse_static_dictionary_marker,
    static_dictionary_marker,
)
from .types import CompressionResult, Token, TokenSeq
from .utils import is_meta_token, parse_length_token, require_no_reserved_tokens


def _apply_static_dictionary(
    tokens: list[Token],
    static_dict: dict[Token, tuple[Token, ...]],
    config: CompressionConfig,
) -> tuple[list[Token], dict[int, tuple[int, Token]]]:
    if any(token in static_dict for token in tokens):
        raise ValueError("Input sequence contains static meta-tokens.")
    occupied = [False] * len(tokens)
    replacements: dict[int, tuple[int, Token]] = {}
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
                replacements[idx] = (len(subseq), meta)
                for pos in range(idx, idx + len(subseq)):
                    occupied[pos] = True
                idx += len(subseq)
            else:
                idx += 1
    if not replacements:
        return tokens, replacements
    return build_body_tokens(tokens, replacements), replacements


def _select_static_dictionary(tokens: list[Token], config: CompressionConfig) -> str | None:
    if config.static_dictionary_id:
        return config.static_dictionary_id
    if not config.static_dictionary_auto:
        return None
    detection = detect_domain(tokens, config)
    if detection.domain is None or detection.confidence < config.static_dictionary_min_confidence:
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
    static_dict = None
    static_replacements: dict[int, tuple[int, Token]] = {}
    if static_id:
        static_entry = get_static_dictionary(static_id)
        if static_entry is None:
            raise ValueError("Unknown static dictionary id.")
        static_dict = static_entry.entries
        working_tokens, static_replacements = _apply_static_dictionary(working_tokens, static_dict, cfg)

    dictionary_map: dict[Token, tuple[Token, ...]] = {}
    depth_limit = cfg.hierarchical_max_depth if cfg.hierarchical_enabled else 1

    for _ in range(depth_limit):
        candidates = discover_candidates(working_tokens, cfg.max_subsequence_length)
        if preferred_candidates:
            candidates = preferred_candidates + candidates
        if not candidates:
            break
        swap_result = perform_swaps(working_tokens, candidates, cfg)
        if not swap_result.dictionary_map:
            break
        dictionary_map.update(swap_result.dictionary_map)
        working_tokens = build_body_tokens(working_tokens, swap_result.replacements)
        if not cfg.hierarchical_enabled:
            break

    dictionary_tokens = build_dictionary_tokens(dictionary_map, cfg)
    body_tokens = working_tokens
    if static_id:
        marker = static_dictionary_marker(static_id, cfg)
        compressed_tokens = [marker] + dictionary_tokens + body_tokens
    else:
        compressed_tokens = dictionary_tokens + body_tokens

    result = CompressionResult(
        compressed_tokens=compressed_tokens,
        dictionary_tokens=dictionary_tokens,
        body_tokens=body_tokens,
        dictionary_map=dictionary_map,
        meta_tokens_used=tuple(dictionary_map.keys()),
        original_length=len(tokens),
        compressed_length=len(compressed_tokens),
        static_dictionary_id=static_id,
    )

    if cfg.verify:
        roundtrip = decompress(compressed_tokens, cfg)
        if list(roundtrip) != list(tokens):
            raise ValueError("Round-trip verification failed.")

    return result


def compress(tokens: TokenSeq, config: CompressionConfig | None = None) -> CompressionResult:
    cfg = config or CompressionConfig()
    return _compress_internal(tokens, cfg, preferred_candidates=None)


def compress_python_source(source: str, config: CompressionConfig | None = None) -> tuple[list[Token], CompressionResult]:
    cfg = config or CompressionConfig()
    tokens, ast_candidates = discover_ast_candidates(source, cfg) if cfg.ast_enabled else (source.split(), [])
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


def decompress(tokens: Sequence[Token], config: CompressionConfig | None = None) -> list[Token]:
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
        raise ValueError("Compressed sequence does not start with dictionary delimiter.")

    try:
        end_idx = tokens.index(cfg.dict_end_token, idx + 1)
    except ValueError as exc:
        raise ValueError("Compressed sequence missing dictionary end delimiter.") from exc

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

    for meta, subseq in static_dict.items():
        if meta in dictionary_map:
            raise ValueError("Static and dynamic dictionaries share a meta-token.")
        dictionary_map[meta] = list(subseq)

    decoded: list[Token] = []
    memo: dict[Token, list[Token]] = {}
    for token in body_tokens:
        decoded.extend(_expand_token(token, dictionary_map, cfg, memo))
    return decoded
