# Compression Format

## Dictionary Delimiters

The dictionary uses XML-style delimiters:

- `<Dict>`
- `</Dict>`

These are tokenized according to the target model's tokenizer.

## Dictionary Entries

Each entry is the meta-token immediately followed by its original subsequence. Entries are concatenated without separators.

```
<Dict>
META_1 tA tB tC
META_2 tD tE
</Dict>
BODY...
```

## Parsing Assumptions

- Meta-tokens are guaranteed not to appear in original input tokens.
- Subsequence definitions do not contain meta-tokens.
- Parsing relies on recognizing meta-token boundaries inside the dictionary.
