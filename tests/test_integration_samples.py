from small import CompressionConfig, compress, decompress


def test_integration_code_sample():
    tokens = """
    def evaluate(user, action, resource):
        if user == "alice" and action == "read":
            return True
        if user == "bob" and action == "read":
            return False
        return False
    """.split()
    cfg = CompressionConfig(static_dictionary_auto=False)
    result = compress(tokens, cfg)
    restored = decompress(result.serialized_tokens, cfg)
    assert restored == tokens


def test_integration_policy_sample():
    tokens = """
    policy allow read resource repo_1
    policy allow read resource repo_2
    policy deny write resource repo_1
    policy deny write resource repo_2
    """.split()
    cfg = CompressionConfig(static_dictionary_auto=False)
    result = compress(tokens, cfg)
    restored = decompress(result.serialized_tokens, cfg)
    assert restored == tokens
