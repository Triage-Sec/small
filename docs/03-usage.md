# Usage

Small operates on token sequences, not raw text. Tokens are represented as hashable values (typically strings or integers).

## High-Level Flow

1) Tokenize input with your model's tokenizer.
2) Compress the token sequence.
3) Feed compressed prompt to the model.
4) Keep the answer uncompressed during fine-tuning.

## Notes

- Compression is lossless and reversible.
- Configure the meta-token prefix and pool size to avoid collisions.
- Use verification in development to catch regressions.
