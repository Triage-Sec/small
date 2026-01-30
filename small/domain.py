"""Domain detection heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import CompressionConfig
from .types import Token


@dataclass(frozen=True)
class DomainDetection:
    domain: str | None
    confidence: float


def _token_str(token: Token) -> str | None:
    return token.lower() if isinstance(token, str) else None


def detect_domain(
    tokens: Iterable[Token], config: CompressionConfig
) -> DomainDetection:
    tokens_list = list(tokens)
    if not tokens_list:
        return DomainDetection(domain=None, confidence=0.0)
    if not all(isinstance(tok, str) for tok in tokens_list):
        return DomainDetection(domain=None, confidence=0.0)

    # After the isinstance check above, we know all tokens are strings
    str_tokens: list[str] = [str(tok) for tok in tokens_list]
    lowered = [tok.lower() for tok in str_tokens]
    total = len(lowered)

    python_keywords = {
        "def",
        "class",
        "import",
        "return",
        "for",
        "if",
        "elif",
        "else",
        "self",
    }
    json_tokens = {"{", "}", "[", "]", ":", ","}
    policy_tokens = {
        "allow",
        "deny",
        "policy",
        "rule",
        "effect",
        "resource",
        "action",
        "user",
    }

    python_hits = sum(1 for tok in lowered if tok in python_keywords)
    json_hits = sum(1 for tok in lowered if tok in json_tokens)
    policy_hits = sum(1 for tok in lowered if tok in policy_tokens)
    colon_hits = sum(1 for tok in lowered if ":" in tok)

    python_score = (python_hits + colon_hits * 0.2) / total
    json_score = json_hits / total
    policy_score = policy_hits / total

    if python_hits >= 2 and python_score >= 0.08:
        return DomainDetection(
            domain="code-python", confidence=min(1.0, python_score + 0.25)
        )
    if json_hits >= 4 and json_score >= 0.12:
        return DomainDetection(domain="json", confidence=min(1.0, json_score + 0.2))
    if policy_hits >= 3 and policy_score >= 0.1:
        return DomainDetection(
            domain="security-policy", confidence=min(1.0, policy_score + 0.2)
        )

    return DomainDetection(domain=None, confidence=0.0)
