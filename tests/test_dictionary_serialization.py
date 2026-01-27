from small import CompressionConfig, compress, decompress


def test_dictionary_round_trip():
    tokens = ["x", "y", "x", "y", "x", "y", "z"]
    cfg = CompressionConfig(static_dictionary_auto=False)
    result = compress(tokens, cfg)
    restored = decompress(result.serialized_tokens, cfg)
    assert restored == tokens
