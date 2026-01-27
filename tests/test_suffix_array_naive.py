from small.suffix_array import build_suffix_array


def naive_suffix_array(tokens):
    suffixes = [(tokens[i:], i) for i in range(len(tokens))]
    suffixes.sort(key=lambda item: item[0])
    return [idx for _, idx in suffixes]


def test_suffix_array_matches_naive():
    tokens = ["b", "a", "b", "a", "c"]
    sa = build_suffix_array(tokens)
    assert sa.suffix_array == naive_suffix_array(tokens)
