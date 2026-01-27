from small import CompressionConfig, compress, compress_python_source, decompress
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


def test_decompress_rejects_duplicate_meta_tokens():
    cfg = CompressionConfig()
    tokens = [
        cfg.dict_start_token,
        "<MT_1>",
        "<Len:1>",
        "a",
        "<MT_1>",
        "<Len:1>",
        "b",
        cfg.dict_end_token,
        "<MT_1>",
    ]
    try:
        decompress(tokens, cfg)
    except ValueError as exc:
        assert "Duplicate meta-token" in str(exc)
    else:
        raise AssertionError("Expected error for duplicate meta-token.")


def test_decompress_requires_meta_token_header():
    cfg = CompressionConfig()
    tokens = [
        cfg.dict_start_token,
        "a",
        "b",
        cfg.dict_end_token,
        "c",
    ]
    try:
        decompress(tokens, cfg)
    except ValueError as exc:
        assert "Dictionary entry missing meta-token" in str(exc)
    else:
        raise AssertionError("Expected error for missing meta-token header.")


def test_decompress_rejects_empty_dictionary_entry():
    cfg = CompressionConfig()
    tokens = [
        cfg.dict_start_token,
        "<MT_1>",
        "<Len:0>",
        cfg.dict_end_token,
        "x",
    ]
    try:
        decompress(tokens, cfg)
    except ValueError as exc:
        assert "Empty dictionary entry" in str(exc)
    else:
        raise AssertionError("Expected error for empty dictionary entry.")


def test_decompress_requires_end_delimiter():
    cfg = CompressionConfig()
    tokens = [
        cfg.dict_start_token,
        "<MT_1>",
        "<Len:3>",
        "a",
        "b",
        "c",
    ]
    try:
        decompress(tokens, cfg)
    except ValueError as exc:
        assert "missing dictionary end delimiter" in str(exc).lower()
    else:
        raise AssertionError("Expected error for missing dictionary end delimiter.")


def test_decompress_hierarchical_dictionary():
    cfg = CompressionConfig()
    tokens = [
        cfg.dict_start_token,
        "<MT_1>",
        "<Len:2>",
        "a",
        "b",
        "<MT_2>",
        "<Len:2>",
        "<MT_1>",
        "c",
        cfg.dict_end_token,
        "<MT_2>",
    ]
    restored = decompress(tokens, cfg)
    assert restored == ["a", "b", "c"]


def test_compress_python_source_roundtrip():
    source = "def add(x, y):\n    return x + y\n"
    cfg = CompressionConfig(verify=True)
    tokens, result = compress_python_source(source, cfg)
    restored = decompress(result.compressed_tokens, cfg)
    assert restored == tokens


def test_decompress_with_static_dictionary_marker():
    cfg = CompressionConfig()
    tokens = [
        "<StaticDict:policy-python-v1>",
        cfg.dict_start_token,
        cfg.dict_end_token,
        "<SD_0>",
    ]
    restored = decompress(tokens, cfg)
    assert restored == ["def", "evaluate(self,"]


def test_decompress_with_patch_section():
    cfg = CompressionConfig()
    tokens = [
        cfg.dict_start_token,
        "<MT_1>",
        "<Len:3>",
        "a",
        "b",
        "c",
        cfg.dict_end_token,
        "<MT_1>",
        cfg.patch_start_token,
        "<Idx:1>",
        "x",
        cfg.patch_end_token,
    ]
    restored = decompress(tokens, cfg)
    assert restored == ["a", "x", "c"]
