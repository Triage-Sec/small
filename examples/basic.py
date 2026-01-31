"""Basic usage example for Delta compression."""

from delta import compress, decompress, CompressionConfig


def main():
    # Example 1: Simple compression
    print("=" * 60)
    print("Example 1: Basic Compression")
    print("=" * 60)
    
    tokens = ["the", "quick", "brown", "fox"] * 10
    config = CompressionConfig(verify=True)
    
    result = compress(tokens, config)
    
    print(f"Original length:   {result.original_length} tokens")
    print(f"Compressed length: {result.compressed_length} tokens")
    print(f"Compression ratio: {result.compressed_length / result.original_length:.1%}")
    print(f"Patterns found:    {len(result.dictionary_map)}")
    
    # Verify lossless
    restored = decompress(result.serialized_tokens, config)
    print(f"Lossless:          {restored == tokens}")
    
    # Example 2: Code-like repetition
    print("\n" + "=" * 60)
    print("Example 2: Code Patterns")
    print("=" * 60)
    
    code_tokens = [
        "def", "foo", "(", "x", ")", ":",
        "return", "bar", "(", "x", ")", "+", "1",
        "def", "baz", "(", "x", ")", ":",
        "return", "bar", "(", "x", ")", "+", "2",
        "def", "qux", "(", "x", ")", ":",
        "return", "bar", "(", "x", ")", "+", "3",
    ]
    
    result = compress(code_tokens, config)
    
    print(f"Original length:   {result.original_length} tokens")
    print(f"Compressed length: {result.compressed_length} tokens")
    print(f"Compression ratio: {result.compressed_length / result.original_length:.1%}")
    
    # Show dictionary entries
    if result.dictionary_map:
        print("\nDictionary entries:")
        for meta, seq in result.dictionary_map.items():
            print(f"  {meta} -> {list(seq)}")
    
    # Example 3: Different selection modes
    print("\n" + "=" * 60)
    print("Example 3: Selection Modes Comparison")
    print("=" * 60)
    
    tokens = ["a", "b", "c"] * 20 + ["x", "y"] * 15
    
    for mode in ["greedy", "optimal", "beam"]:
        cfg = CompressionConfig(selection_mode=mode, verify=True)
        result = compress(tokens, cfg)
        print(f"{mode:8s}: {result.original_length} -> {result.compressed_length} "
              f"({result.compressed_length / result.original_length:.1%})")


if __name__ == "__main__":
    main()
