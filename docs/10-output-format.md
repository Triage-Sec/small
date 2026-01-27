# Output Format Specification

## Compressed Sequence Format

The serialized compressed output is:

```
<Dict> ... </Dict> BODY...
```

- The dictionary section begins with `<Dict>` and ends with `</Dict>`.
- Each entry is a meta-token followed by its subsequence (and length token when enabled).
- Entries are ordered so dependencies appear before dependents (hierarchical mode).
- The body is the original sequence with replaced occurrences.

If a static dictionary is used, a marker token precedes the dictionary section:

```
<StaticDict:...> <Dict> ... </Dict> BODY...
```

## Compression Result Object

`CompressionResult` includes:

- `original_tokens`: original sequence (tuple)
- `compressed_tokens`: body-only compressed sequence
- `serialized_tokens`: dictionary + body sequence for model input
- `dictionary`: dictionary object for inspection
- `dictionary_tokens` and `body_tokens` for separate access
- `metrics` when enabled

Use `serialized_tokens` for model input.

## Decompression Interface

- `decompress(serialized_tokens)` parses the dictionary section and expands meta-tokens.
- `decompress_with_dictionary(dictionary_map, body_tokens)` expands when dictionary/body are separate.

Both forms validate structure and handle hierarchical dictionaries.
