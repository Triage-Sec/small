"""Tokenizer adapters for common libraries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from .types import Token


class TokenizerAdapter(Protocol):
    def encode(self, text: str) -> list[Token]: ...

    def decode(self, tokens: Iterable[Token]) -> str: ...

    def vocab_size(self) -> int: ...

    def is_special_token(self, token: Token) -> bool: ...


@dataclass
class SimpleWhitespaceTokenizer:
    def encode(self, text: str) -> list[str]:
        return text.split()

    def decode(self, tokens: Iterable[str]) -> str:
        return " ".join(tokens)

    def vocab_size(self) -> int:
        return 0

    def is_special_token(self, token: str) -> bool:
        return False


@dataclass
class HuggingFaceTokenizerAdapter:
    tokenizer: Any

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def decode(self, tokens: Iterable[int]) -> str:
        return self.tokenizer.decode(list(tokens))

    def vocab_size(self) -> int:
        return int(self.tokenizer.vocab_size)

    def is_special_token(self, token: int) -> bool:
        try:
            return bool(
                self.tokenizer.convert_ids_to_tokens([token])[0]
                in self.tokenizer.all_special_tokens
            )
        except Exception:
            return False


@dataclass
class TiktokenAdapter:
    encoder: Any

    def encode(self, text: str) -> list[int]:
        return list(self.encoder.encode(text))

    def decode(self, tokens: Iterable[int]) -> str:
        return self.encoder.decode(list(tokens))

    def vocab_size(self) -> int:
        return int(self.encoder.n_vocab)

    def is_special_token(self, token: int) -> bool:
        if not hasattr(self.encoder, "special_tokens"):
            return False
        return token in getattr(self.encoder, "special_tokens").values()


@dataclass
class SentencePieceAdapter:
    processor: Any

    def encode(self, text: str) -> list[int]:
        return list(self.processor.encode(text, out_type=int))

    def decode(self, tokens: Iterable[int]) -> str:
        return self.processor.decode(list(tokens))

    def vocab_size(self) -> int:
        return int(self.processor.get_piece_size())

    def is_special_token(self, token: int) -> bool:
        if not hasattr(self.processor, "is_unknown"):
            return False
        return bool(self.processor.is_unknown(token))
