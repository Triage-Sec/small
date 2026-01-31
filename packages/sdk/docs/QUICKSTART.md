# Quick Start Guide

Get started with Delta LTSC in under 5 minutes. This guide covers the most common use case: compressing prompts before sending them to an LLM API.

## Installation

```bash
npm install @delta-ltsc/sdk
```

## Basic Usage

```typescript
import { compress, decompress, initWasm } from '@delta-ltsc/sdk';

// Initialize WASM (required once, auto-called if you forget)
await initWasm();

// Your token sequence (from any tokenizer)
const tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];

// Compress
const result = await compress(tokens);
console.log(`Compressed: ${result.originalLength} → ${result.compressedLength} tokens`);
console.log(`Ratio: ${(result.compressionRatio * 100).toFixed(1)}%`);

// Decompress when needed
const restored = await decompress(result.serializedTokens);
console.assert(JSON.stringify(tokens) === JSON.stringify(restored));
```

## With LLM APIs

### OpenAI / Anthropic Pattern

```typescript
import { compress, decompress, initWasm } from '@delta-ltsc/sdk';
import { encoding_for_model } from 'tiktoken';

// Setup
await initWasm();
const encoder = encoding_for_model('gpt-4');

// Your prompt
const prompt = `[System instructions repeated many times...]
${systemInstructions}
${contextDocuments}
User: ${userQuery}`;

// Tokenize and compress
const tokens = encoder.encode(prompt);
const compressed = await compress(tokens);

// Use compressed tokens (the model must be fine-tuned to understand the format)
// For inference with a compressed-aware model:
const response = await callModel(compressed.serializedTokens);

// Decompress response if needed
const decompressedResponse = await decompress(response);
const text = encoder.decode(decompressedResponse);
```

### Cost Savings Example

```typescript
const tokens = getPromptTokens(); // 10,000 tokens
const result = await compress(tokens);

// Before: 10,000 tokens × $0.01/1K = $0.10
// After:  7,000 tokens × $0.01/1K = $0.07
// Savings: 30%

console.log(`Tokens saved: ${result.originalLength - result.compressedLength}`);
console.log(`Cost reduction: ${((1 - result.compressionRatio) * 100).toFixed(1)}%`);
```

## Configuration Options

```typescript
const result = await compress(tokens, {
  // Pattern discovery
  minSubsequenceLength: 2,    // Minimum pattern length
  maxSubsequenceLength: 8,    // Maximum pattern length
  
  // Selection algorithm
  selectionMode: 'greedy',    // 'greedy' | 'optimal' | 'beam'
  
  // Hierarchical compression
  hierarchicalEnabled: true,  // Allow patterns of patterns
  hierarchicalMaxDepth: 3,    // Maximum nesting depth
  
  // Verification
  verify: true,               // Round-trip verification (slower but safe)
});
```

## Streaming for Large Inputs

For inputs over 50K tokens, use streaming:

```typescript
import { createStreamingCompressor } from '@delta-ltsc/sdk';

const compressor = await createStreamingCompressor();

// Add chunks as they arrive
for await (const chunk of tokenStream) {
  await compressor.addChunk(chunk);
}

// Finish and get result
const result = await compressor.finish();
```

## Static Dictionaries

Use pre-built dictionaries for domain-specific compression:

```typescript
import { compress, loadStaticDictionary } from '@delta-ltsc/sdk';

// Load a built-in dictionary
const result = await compress(pythonCodeTokens, {
  staticDictionary: 'python-v1',
});

// Available dictionaries: 'python-v1', 'typescript-v1', 'markdown-v1', 'json-v1', 'sql-v1'
```

## Worker Threads (Non-blocking)

For CPU-intensive compression without blocking:

```typescript
import { createWorkerPool } from '@delta-ltsc/sdk';

// Create a pool (defaults to CPU count)
const pool = await createWorkerPool(4);

// Compress in background
const result = await pool.compress(tokens);

// Clean up when done
pool.terminate();
```

## Browser Usage

Works in all modern browsers:

```html
<script type="module">
  import { compress, decompress, initWasm } from 'https://esm.sh/@delta-ltsc/sdk';
  
  await initWasm();
  const result = await compress([1, 2, 3, 1, 2, 3]);
  console.log('Compressed!', result.compressionRatio);
</script>
```

## TypeScript Support

Full TypeScript support with detailed types:

```typescript
import type {
  CompressionConfig,
  CompressionResult,
  TokenSeq,
  StaticDictionaryId,
} from '@delta-ltsc/sdk';

const config: CompressionConfig = {
  maxSubsequenceLength: 10,
  selectionMode: 'optimal',
};

const result: CompressionResult = await compress(tokens, config);
```

## Error Handling

```typescript
try {
  const result = await compress(tokens, { verify: true });
} catch (error) {
  if (error.message.includes('verification failed')) {
    // Round-trip verification failed - compression may be lossy
    console.error('Compression verification failed');
  }
}
```

## Performance Tips

1. **Batch operations**: Compress multiple prompts together if they share common patterns
2. **Use static dictionaries**: Pre-built dictionaries improve compression for domain-specific content
3. **Tune pattern lengths**: Longer patterns = better compression but slower discovery
4. **Use workers**: For inputs > 10K tokens, use worker threads to avoid blocking

## Next Steps

- [Configuration Reference](./CONFIGURATION.md) - All options explained
- [Static Dictionaries](./DICTIONARIES.md) - Creating custom dictionaries
- [ML Features](./ML.md) - Pattern importance and quality prediction
- [API Reference](./API.md) - Complete API documentation
