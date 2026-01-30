"""Python AST-based pattern discovery."""

from __future__ import annotations

import ast
import tokenize
from dataclasses import dataclass
from io import StringIO
from .config import CompressionConfig
from .types import Candidate, Token
from .utils import is_compressible


@dataclass(frozen=True)
class TokenSpan:
    start_line: int
    start_col: int
    end_line: int
    end_col: int


@dataclass(frozen=True)
class TokenWithSpan:
    token: str
    span: TokenSpan


def tokenize_source(source: str) -> list[TokenWithSpan]:
    tokens: list[TokenWithSpan] = []
    reader = StringIO(source).readline
    for tok in tokenize.generate_tokens(reader):
        if tok.type in {
            tokenize.ENCODING,
            tokenize.NL,
            tokenize.NEWLINE,
            tokenize.ENDMARKER,
        }:
            continue
        tokens.append(
            TokenWithSpan(
                token=tok.string,
                span=TokenSpan(
                    start_line=tok.start[0],
                    start_col=tok.start[1],
                    end_line=tok.end[0],
                    end_col=tok.end[1],
                ),
            )
        )
    return tokens


def _node_key(node: ast.AST) -> tuple[object, ...]:
    if isinstance(node, ast.Name):
        return ("Name",)
    if isinstance(node, ast.Constant):
        return ("Const",)
    if isinstance(node, ast.arg):
        return ("arg",)
    fields: list[object] = [node.__class__.__name__]
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            fields.append(
                tuple(_node_key(child) for child in value if isinstance(child, ast.AST))
            )
        elif isinstance(value, ast.AST):
            fields.append(_node_key(value))
        else:
            fields.append((field, type(value).__name__))
    return tuple(fields)


def _node_span(node: ast.AST) -> TokenSpan | None:
    if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
        return None
    return TokenSpan(
        start_line=getattr(node, "lineno"),
        start_col=getattr(node, "col_offset"),
        end_line=getattr(node, "end_lineno"),
        end_col=getattr(node, "end_col_offset"),
    )


def _find_token_range(
    tokens: list[TokenWithSpan], span: TokenSpan
) -> tuple[int, int] | None:
    start_idx = None
    end_idx = None
    for idx, tok in enumerate(tokens):
        if start_idx is None:
            if (tok.span.start_line, tok.span.start_col) >= (
                span.start_line,
                span.start_col,
            ):
                start_idx = idx
        if (tok.span.end_line, tok.span.end_col) <= (span.end_line, span.end_col):
            end_idx = idx
    if start_idx is None or end_idx is None or end_idx < start_idx:
        return None
    return start_idx, end_idx + 1


def discover_ast_candidates(
    source: str, config: CompressionConfig
) -> tuple[list[Token], list[Candidate]]:
    tokens_with_spans = tokenize_source(source)
    tokens: list[Token] = [tok.token for tok in tokens_with_spans]

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return tokens, []

    nodes_by_hash: dict[tuple, list[TokenSpan]] = {}
    for node in ast.walk(tree):
        span = _node_span(node)
        if span is None:
            continue
        key = _node_key(node)
        nodes_by_hash.setdefault(key, []).append(span)

    candidates: dict[tuple[Token, ...], list[int]] = {}
    for spans in nodes_by_hash.values():
        if len(spans) < 2:
            continue
        for span in spans:
            token_range = _find_token_range(tokens_with_spans, span)
            if token_range is None:
                continue
            start, end = token_range
            length = end - start
            if length < 2 or length > config.max_subsequence_length:
                continue
            subseq: tuple[Token, ...] = tuple(tokens[start:end])
            candidates.setdefault(subseq, []).append(start)

    result: list[Candidate] = []
    for subseq, positions in candidates.items():
        positions.sort()
        non_overlapping: list[int] = []
        next_free = -1
        for pos in positions:
            if pos >= next_free:
                non_overlapping.append(pos)
                next_free = pos + len(subseq)
        extra_cost = 1 if config.dict_length_enabled else 0
        if is_compressible(len(subseq), len(non_overlapping), extra_cost=extra_cost):
            result.append(
                Candidate(
                    subsequence=subseq,
                    length=len(subseq),
                    positions=tuple(non_overlapping),
                    priority=config.ast_priority_bonus,
                )
            )

    result.sort(key=lambda cand: cand.length, reverse=True)
    return tokens, result
