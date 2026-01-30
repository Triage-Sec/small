"""Phase 3: dictionary construction."""

from __future__ import annotations

from .config import CompressionConfig
from .types import Token, TokenSeq
from .utils import is_meta_token, length_token, patch_index_token


def order_dictionary_entries(
    dictionary_map: dict[Token, tuple[Token, ...]], config: CompressionConfig
) -> list[Token]:
    if not dictionary_map:
        return []
    insertion_order = list(dictionary_map.keys())
    deps: dict[Token, set[Token]] = {meta: set() for meta in dictionary_map}
    for meta, subseq in dictionary_map.items():
        for token in subseq:
            if is_meta_token(token, config) and token in dictionary_map:
                deps[meta].add(token)

    ordered: list[Token] = []
    ready = [meta for meta in insertion_order if not deps[meta]]
    while ready:
        meta = ready.pop(0)
        ordered.append(meta)
        for other_meta in insertion_order:
            if meta in deps[other_meta]:
                deps[other_meta].remove(meta)
                if not deps[other_meta]:
                    if other_meta not in ordered and other_meta not in ready:
                        ready.append(other_meta)

    if len(ordered) != len(dictionary_map):
        raise ValueError("Dictionary contains cyclic meta-token dependencies.")
    return ordered


def build_dictionary_tokens(
    dictionary_map: dict[Token, tuple[Token, ...]], config: CompressionConfig
) -> list[Token]:
    tokens: list[Token] = [config.dict_start_token]
    for meta_token in order_dictionary_entries(dictionary_map, config):
        subseq = dictionary_map[meta_token]
        tokens.append(meta_token)
        if config.dict_length_enabled:
            tokens.append(length_token(len(subseq), config))
        tokens.extend(subseq)
    tokens.append(config.dict_end_token)
    return tokens


def build_body_tokens(
    tokens: TokenSeq,
    replacements: dict[int, tuple[int, Token, tuple]],
    config: CompressionConfig,
) -> list[Token]:
    body: list[Token] = []
    idx = 0
    n = len(tokens)
    while idx < n:
        replacement = replacements.get(idx)
        if replacement is None:
            body.append(tokens[idx])
            idx += 1
            continue
        length, meta_token, patches = replacement
        body.append(meta_token)
        if patches:
            body.append(config.patch_start_token)
            for patch_idx, patch_token in patches:
                body.append(patch_index_token(patch_idx, config))
                body.append(patch_token)
            body.append(config.patch_end_token)
        idx += length
    return body
