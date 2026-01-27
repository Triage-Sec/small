# Compression Format

## Dictionary Delimiters

The dictionary uses XML-style delimiters:

- `<Dict>`
- `</Dict>`

These are tokenized according to the target model's tokenizer.

## Static Dictionary Marker

When a static dictionary is used, the compressed sequence begins with a single marker token:

```
<StaticDict:policy-python-v1>
<Dict>...</Dict>
BODY...
```

The static dictionary entries are not serialized in the prompt.

## Dictionary Entries

Each entry is the meta-token immediately followed by its original subsequence. Entries are concatenated without separators. When hierarchical compression is enabled, each entry includes a length token to disambiguate nested meta-token references.

```
<Dict>
META_1 <Len:3> tA tB tC
META_2 tD tE
</Dict>
BODY...
```

## Patch Encoding

Fuzzy matching emits a patch section after a meta-token occurrence:

```
META_1 <Patch> <Idx:1> tX </Patch>
```

Each index token is followed by the replacement token for that position.

## Parsing Assumptions

- Meta-tokens are guaranteed not to appear in original input tokens.
- Subsequence definitions may contain meta-tokens when hierarchical compression is enabled.
- Parsing relies on length tokens when hierarchical compression is enabled.
- Dictionary delimiters are reserved tokens and must not appear in the original input.
- Length tokens are reserved when enabled.
- Patch delimiters and index tokens are reserved when fuzzy matching is enabled.
