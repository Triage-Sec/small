"""Training data generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from .compressor import compress
from .config import CompressionConfig
from .types import CompressionResult, Token, TokenSeq


@dataclass(frozen=True)
class TrainingExample:
    input_tokens: list[Token]
    target_tokens: list[Token]
    loss_mask: list[int]
    compressed: bool
    compression_result: CompressionResult | None = None


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    config: CompressionConfig
    compress_ratio: float


def build_example(
    prompt_tokens: TokenSeq,
    output_tokens: TokenSeq,
    config: CompressionConfig,
    compress_prompt: bool,
) -> TrainingExample:
    if compress_prompt:
        result = compress(prompt_tokens, config)
        input_tokens = result.serialized_tokens + list(output_tokens)
        loss_mask = [0] * len(result.serialized_tokens) + [1] * len(output_tokens)
        return TrainingExample(
            input_tokens=input_tokens,
            target_tokens=list(output_tokens),
            loss_mask=loss_mask,
            compressed=True,
            compression_result=result,
        )

    input_tokens = list(prompt_tokens) + list(output_tokens)
    loss_mask = [0] * len(prompt_tokens) + [1] * len(output_tokens)
    return TrainingExample(
        input_tokens=input_tokens,
        target_tokens=list(output_tokens),
        loss_mask=loss_mask,
        compressed=False,
        compression_result=None,
    )


def generate_training_examples(
    samples: Iterable[tuple[TokenSeq, TokenSeq]],
    config: CompressionConfig,
    compress_ratio: float = 0.5,
    rng_seed: int | None = None,
) -> list[TrainingExample]:
    rng = random.Random(rng_seed)
    examples: list[TrainingExample] = []
    for prompt_tokens, output_tokens in samples:
        compress_prompt = rng.random() < compress_ratio
        examples.append(
            build_example(prompt_tokens, output_tokens, config, compress_prompt)
        )
    return examples


def build_curriculum(base_config: CompressionConfig) -> list[CurriculumStage]:
    stages: list[CurriculumStage] = []
    stages.append(
        CurriculumStage(
            name="baseline",
            config=CompressionConfig(
                **{
                    **base_config.__dict__,
                    "hierarchical_enabled": False,
                    "max_subsequence_length": 4,
                }
            ),
            compress_ratio=0.4,
        )
    )
    stages.append(
        CurriculumStage(
            name="intermediate",
            config=CompressionConfig(
                **{
                    **base_config.__dict__,
                    "hierarchical_enabled": True,
                    "hierarchical_max_depth": 2,
                    "max_subsequence_length": 6,
                }
            ),
            compress_ratio=0.5,
        )
    )
    stages.append(
        CurriculumStage(
            name="advanced",
            config=CompressionConfig(
                **{
                    **base_config.__dict__,
                    "hierarchical_enabled": True,
                    "hierarchical_max_depth": 3,
                    "max_subsequence_length": base_config.max_subsequence_length,
                }
            ),
            compress_ratio=0.6,
        )
    )
    return stages
