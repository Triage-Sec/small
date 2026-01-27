"""Core compression and decompression APIs."""

from __future__ import annotations

from typing import Sequence

from .config import CompressionConfig
from .dictionary import build_body_tokens, build_dictionary_tokens
from .ast_python import discover_ast_candidates
from .discovery import discover_candidates
from .swap import perform_swaps
from .types import CompressionResult, Token, TokenSeq
from .utils import is_meta_token, parse_length_token, require_no_reserved_tokens


def _compress_internal(
    tokens: TokenSeq,
    config: CompressionConfig,
    preferred_candidates: list | None = None,
) -> CompressionResult:
    cfg = config or CompressionConfig()
    require_no_reserved_tokens(tokens, cfg)

    working_tokens = list(tokens)
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
    compressed_tokens = dictionary_tokens + body_tokens

    result = CompressionResult(
        compressed_tokens=compressed_tokens,
        dictionary_tokens=dictionary_tokens,
        body_tokens=body_tokens,
        dictionary_map=dictionary_map,
        meta_tokens_used=tuple(dictionary_map.keys()),
        original_length=len(tokens),
        compressed_length=len(compressed_tokens),
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
    if tokens[0] != cfg.dict_start_token:
        raise ValueError("Compressed sequence does not start with dictionary delimiter.")

    try:
        end_idx = tokens.index(cfg.dict_end_token)
    except ValueError as exc:
        raise ValueError("Compressed sequence missing dictionary end delimiter.") from exc

    dict_tokens = tokens[1:end_idx]
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

    decoded: list[Token] = []
    memo: dict[Token, list[Token]] = {}
    for token in body_tokens:
        decoded.extend(_expand_token(token, dictionary_map, cfg, memo))
    return decoded
