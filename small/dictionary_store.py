"""Dictionary store with bidirectional lookup and ordering."""

from __future__ import annotations

from dataclasses import dataclass
from .config import CompressionConfig
from .dictionary import order_dictionary_entries
from .types import Token
from .utils import is_meta_token


@dataclass
class CompressionDictionary:
    meta_to_seq: dict[Token, tuple[Token, ...]]
    seq_to_meta: dict[tuple[Token, ...], Token]
    max_entries: int

    @classmethod
    def empty(cls, max_entries: int) -> "CompressionDictionary":
        return cls(meta_to_seq={}, seq_to_meta={}, max_entries=max_entries)

    def add_entry(
        self, meta: Token, subseq: tuple[Token, ...], config: CompressionConfig
    ) -> None:
        if len(self.meta_to_seq) >= self.max_entries:
            raise ValueError("Dictionary size limit exceeded.")
        if meta in self.meta_to_seq:
            raise ValueError("Meta-token already exists in dictionary.")
        if subseq in self.seq_to_meta:
            raise ValueError("Subsequence already mapped to a meta-token.")
        if any(
            is_meta_token(tok, config) and tok not in self.meta_to_seq for tok in subseq
        ):
            raise ValueError("Subsequence references undefined meta-token.")
        self.meta_to_seq[meta] = subseq
        self.seq_to_meta[subseq] = meta

    def get_subsequence(self, meta: Token) -> tuple[Token, ...] | None:
        return self.meta_to_seq.get(meta)

    def get_meta(self, subseq: tuple[Token, ...]) -> Token | None:
        return self.seq_to_meta.get(subseq)

    def ordered_meta_tokens(self, config: CompressionConfig) -> list[Token]:
        return order_dictionary_entries(self.meta_to_seq, config)
