# Compression Format

## Dictionary Delimiters

The dictionary uses XML-style delimiters:

- `<Dict>`
- `</Dict>`

These are tokenized according to the target model's tokenizer.

## Dictionary Entries

Each entry is the meta-token immediately followed by its original subsequence. Entries are concatenated without separators. When hierarchical compression is enabled, each entry includes a length token to disambiguate nested meta-token references.

```
<Dict>
META_1 <Len:3> tA tB tC
META_2 tD tE
</Dict>
BODY...
```

## Parsing Assumptions

- Meta-tokens are guaranteed not to appear in original input tokens.
- Subsequence definitions may contain meta-tokens when hierarchical compression is enabled.
- Parsing relies on length tokens when hierarchical compression is enabled.
- Dictionary delimiters are reserved tokens and must not appear in the original input.
- Length tokens are reserved when enabled.
