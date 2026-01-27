from small.utils import is_compressible


def test_compressibility_cases():
    assert is_compressible(4, 2)
    assert is_compressible(3, 3)
    assert is_compressible(2, 4)
    assert not is_compressible(4, 1)
    assert not is_compressible(3, 2)
    assert not is_compressible(2, 3)
