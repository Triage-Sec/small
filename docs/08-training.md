# Model Training Integration

## Training Data Generation

- Compress only the prompt; keep outputs in original tokens.
- Loss is computed only on output tokens.
- Use mixed compression (default 50/50) to preserve uncompressed handling.

`small.training` provides:

- `build_example(prompt, output, config, compress_prompt)`
- `generate_training_examples(samples, config, compress_ratio)`
- `build_curriculum(base_config)`

## Meta-Token Vocabulary Extension

Use `small.vocab.plan_vocab_extension` to list required meta-tokens and special tokens.

For Hugging Face tokenizers, `add_tokens_to_hf_tokenizer` can extend the vocabulary.

## Curriculum

A default curriculum introduces:

- Baseline: no hierarchy, short max subsequence length
- Intermediate: 2-level hierarchy
- Advanced: full hierarchy depth

Adjust `compress_ratio` per stage to balance compressed vs uncompressed learning.

## Domain Dictionary Training

If a static dictionary is used, ensure the training corpus includes those patterns frequently.
Monitor losses on compressed vs uncompressed subsets and rebalance if needed.
