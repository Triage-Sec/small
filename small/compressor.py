"""Core compression and decompression APIs."""

from __future__ import annotations

from typing import Sequence

from .config import CompressionConfig
from .dictionary import build_body_tokens, build_dictionary_tokens
from .discovery import discover_candidates
from .swap import perform_swaps
from .types import CompressionResult, Token, TokenSeq
from .utils import is_meta_token, require_no_reserved_tokens


def compress(tokens: TokenSeq, config: CompressionConfig | None = None) -> CompressionResult:
    cfg = config or CompressionConfig()
    require_no_reserved_tokens(tokens, cfg)

    candidates = discover_candidates(tokens, cfg.max_subsequence_length)
    swap_result = perform_swaps(tokens, candidates, cfg)
    dictionary_tokens = build_dictionary_tokens(swap_result.dictionary_map, cfg)
    body_tokens = build_body_tokens(tokens, swap_result.replacements)
    compressed_tokens = dictionary_tokens + body_tokens

    result = CompressionResult(
        compressed_tokens=compressed_tokens,
        dictionary_tokens=dictionary_tokens,
        body_tokens=body_tokens,
        dictionary_map=swap_result.dictionary_map,
        meta_tokens_used=swap_result.meta_tokens_used,
        original_length=len(tokens),
        compressed_length=len(compressed_tokens),
    )

    if cfg.verify:
        roundtrip = decompress(compressed_tokens, cfg)
        if list(roundtrip) != list(tokens):
            raise ValueError("Round-trip verification failed.")

    return result


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

    decoded: list[Token] = []
    for token in body_tokens:
        if token in dictionary_map:
            decoded.extend(dictionary_map[token])
        else:
            decoded.append(token)
    return decoded
