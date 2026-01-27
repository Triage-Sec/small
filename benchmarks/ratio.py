import argparse
import random
import statistics

from small import CompressionConfig, compress


def generate_tokens(count: int, vocab_size: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    return [f"t{rng.randrange(vocab_size)}" for _ in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="LTSC compression ratio benchmark")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ratios: list[float] = []
    body_ratios: list[float] = []

    for offset in range(args.runs):
        tokens = generate_tokens(args.tokens, args.vocab, args.seed + offset)
        cfg = CompressionConfig(max_subsequence_length=args.max_len, rng_seed=args.seed + offset)
        result = compress(tokens, cfg)
        ratios.append(result.compressed_length / result.original_length)
        body_ratios.append(len(result.body_tokens) / result.original_length)

    mean_ratio = statistics.mean(ratios)
    mean_body_ratio = statistics.mean(body_ratios)
    print(f"Runs: {args.runs}")
    print(f"Tokens: {args.tokens}")
    print(f"Max subsequence length: {args.max_len}")
    print(f"Mean full compression ratio: {mean_ratio:.4f}")
    print(f"Mean body compression ratio: {mean_body_ratio:.4f}")


if __name__ == "__main__":
    main()
