"""Static dictionary registry and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .config import CompressionConfig
from .types import Token


@dataclass(frozen=True)
class StaticDictionary:
    identifier: str
    entries: Mapping[Token, tuple[Token, ...]]


_POLICY_PYTHON_V1 = StaticDictionary(
    identifier="policy-python-v1",
    entries={
        "<SD_0>": ("def", "evaluate(self,"),
        "<SD_1>": ("return", "False"),
        "<SD_2>": ("return", "True"),
        "<SD_3>": ("rule.user", "==", "user"),
        "<SD_4>": ("rule.action", "==", "action"),
        "<SD_5>": ("rule.resource", "==", "resource:"),
        "<SD_6>": ("rule.effect", "==", '"allow":'),
        "<SD_7>": ("rule.effect", "==", '"deny":'),
    },
)

STATIC_DICTIONARIES: dict[str, StaticDictionary] = {
    _POLICY_PYTHON_V1.identifier: _POLICY_PYTHON_V1,
}

DOMAIN_TO_STATIC_ID: dict[str, str] = {
    "code-python": _POLICY_PYTHON_V1.identifier,
    "security-policy": _POLICY_PYTHON_V1.identifier,
}


def static_dictionary_marker(identifier: str, config: CompressionConfig) -> Token:
    return f"{config.static_dictionary_marker_prefix}{identifier}{config.static_dictionary_marker_suffix}"


def parse_static_dictionary_marker(
    token: Token, config: CompressionConfig
) -> str | None:
    if not isinstance(token, str):
        return None
    if not (
        token.startswith(config.static_dictionary_marker_prefix)
        and token.endswith(config.static_dictionary_marker_suffix)
    ):
        return None
    return token[
        len(config.static_dictionary_marker_prefix) : -len(
            config.static_dictionary_marker_suffix
        )
    ]


def get_static_dictionary(identifier: str) -> StaticDictionary | None:
    return STATIC_DICTIONARIES.get(identifier)
