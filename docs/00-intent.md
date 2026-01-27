# Intent and Motivation

## Problem

LLMs process token sequences with quadratic cost in sequence length. External context often injects repeated subsequences, wasting context budget and compute without adding information.

## Foundational Work

Small builds on "Lossless Token Sequence Compression via Meta-Tokens" by Harvill et al. Repeated subsequences are replaced with meta-tokens, and a dictionary maps each meta-token to its original subsequence. Transformer models can learn this representation with fine-tuning.

## Objective

- Improve compression beyond baseline LTSC results.
- Preserve perfect losslessness.
- Keep compression fast and practical.

## Non-Negotiables

- Lossless reconstruction must be verifiable.
- The compression format must be learnable by transformers.
- Efficiency matters; avoid pathological runtime behavior.
