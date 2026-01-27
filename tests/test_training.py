from small import CompressionConfig
from small.training import build_example, generate_training_examples


def test_training_example_masks():
    prompt = ["a", "b", "c", "a", "b", "c"]
    output = ["x", "y"]
    cfg = CompressionConfig(static_dictionary_auto=False, verify=False)

    compressed = build_example(prompt, output, cfg, compress_prompt=True)
    assert compressed.loss_mask[-len(output) :] == [1, 1]
    assert all(value == 0 for value in compressed.loss_mask[: -len(output)])

    uncompressed = build_example(prompt, output, cfg, compress_prompt=False)
    assert uncompressed.loss_mask == [0] * len(prompt) + [1] * len(output)


def test_generate_training_examples_ratio():
    samples = [(["a"] * 4, ["b"]) for _ in range(20)]
    cfg = CompressionConfig(static_dictionary_auto=False, verify=False)
    examples = generate_training_examples(samples, cfg, compress_ratio=0.5, rng_seed=1)
    compressed_count = sum(1 for ex in examples if ex.compressed)
    assert 5 <= compressed_count <= 15
