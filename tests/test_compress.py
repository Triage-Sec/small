from small import CompressionConfig, compress, decompress
from small.utils import is_compressible


def test_compressibility_condition():
    assert is_compressible(4, 2)
    assert is_compressible(3, 3)
    assert is_compressible(2, 4)
    assert not is_compressible(4, 1)
    assert not is_compressible(3, 2)
    assert not is_compressible(2, 3)


def test_round_trip_basic():
    tokens = ["a", "b", "c", "a", "b", "c", "a", "b", "c", "z"]
    cfg = CompressionConfig(max_subsequence_length=3, rng_seed=7, verify=True)
    result = compress(tokens, cfg)
    restored = decompress(result.compressed_tokens, cfg)
    assert restored == tokens
    assert len(result.body_tokens) < result.original_length


def test_rejects_reserved_tokens():
    tokens = ["<Dict>", "a", "b"]
    cfg = CompressionConfig()
    try:
        compress(tokens, cfg)
    except ValueError as exc:
        assert "Dictionary delimiter" in str(exc)
    else:
        raise AssertionError("Expected error for reserved tokens.")


def test_dictionary_delimiters_present():
    tokens = ["x", "y", "x", "y", "x", "y", "x", "y"]
    cfg = CompressionConfig(max_subsequence_length=2, rng_seed=11)
    result = compress(tokens, cfg)
    assert result.dictionary_tokens[0] == cfg.dict_start_token
    assert result.dictionary_tokens[-1] == cfg.dict_end_token
