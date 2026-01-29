/**
 * Browser-based tests for compression using Playwright.
 *
 * These tests verify that the SDK works correctly in real browser environments.
 */

import { test, expect } from '@playwright/test';

test.describe('Browser Compression', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/tests/browser/test.html');
  });

  test('should load SDK in browser', async ({ page }) => {
    const loaded = await page.evaluate(async () => {
      try {
        const sdk = await import('@small-ltsc/sdk');
        return typeof sdk.compress === 'function';
      } catch (e) {
        return false;
      }
    });

    expect(loaded).toBe(true);
  });

  test('should initialize WASM', async ({ page }) => {
    const initialized = await page.evaluate(async () => {
      const { initWasm, isWasmInitialized } = await import('@small-ltsc/sdk');
      await initWasm();
      return isWasmInitialized();
    });

    expect(initialized).toBe(true);
  });

  test('should compress tokens', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const { compress, initWasm } = await import('@small-ltsc/sdk');
      await initWasm();

      const tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
      const result = await compress(tokens, {});

      return {
        originalLength: result.originalLength,
        compressedLength: result.compressedLength,
        ratio: result.compressionRatio,
      };
    });

    expect(result.originalLength).toBe(15);
    expect(typeof result.ratio).toBe('number');
  });

  test('should round-trip compress and decompress', async ({ page }) => {
    const roundTrip = await page.evaluate(async () => {
      const { compress, decompress, initWasm } = await import('@small-ltsc/sdk');
      await initWasm();

      const tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3];
      const compressed = await compress(tokens, {});
      const restored = await decompress(compressed.serializedTokens, {});

      return {
        original: tokens,
        restored: Array.from(restored),
        matches: JSON.stringify(tokens) === JSON.stringify(restored),
      };
    });

    expect(roundTrip.matches).toBe(true);
    expect(roundTrip.restored).toEqual(roundTrip.original);
  });

  test('should handle large inputs', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const { compress, initWasm } = await import('@small-ltsc/sdk');
      await initWasm();

      // Generate 10K tokens with patterns
      const tokens = Array.from({ length: 10000 }, (_, i) => i % 100);
      const result = await compress(tokens, {});

      return {
        originalLength: result.originalLength,
        compressedLength: result.compressedLength,
        ratio: result.compressionRatio,
      };
    });

    expect(result.originalLength).toBe(10000);
    expect(result.ratio).toBeLessThan(1.0); // Should achieve compression
  });

  test('should work with streaming compressor', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const { createStreamingCompressor, initWasm } = await import(
        '@small-ltsc/sdk'
      );
      await initWasm();

      const compressor = await createStreamingCompressor({});

      // Add chunks
      await compressor.addChunk([1, 2, 3, 1, 2, 3]);
      await compressor.addChunk([1, 2, 3, 1, 2, 3]);
      await compressor.addChunk([1, 2, 3]);

      const result = await compressor.finish();

      return {
        originalLength: result.originalLength,
        compressedLength: result.compressedLength,
        isFinished: compressor.isFinished(),
      };
    });

    expect(result.originalLength).toBe(15);
    expect(result.isFinished).toBe(true);
  });

  test('should discover patterns', async ({ page }) => {
    const patterns = await page.evaluate(async () => {
      const { discoverPatterns, initWasm } = await import('@small-ltsc/sdk');
      await initWasm();

      const tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3];
      return discoverPatterns(tokens);
    });

    expect(Array.isArray(patterns)).toBe(true);
  });

  test('should handle configuration options', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const { compress, initWasm } = await import('@small-ltsc/sdk');
      await initWasm();

      const tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3];
      const result = await compress(tokens, {
        maxSubsequenceLength: 4,
        selectionMode: 'greedy',
        hierarchicalEnabled: false,
      });

      return {
        originalLength: result.originalLength,
        hasMetrics: result.metrics !== undefined,
      };
    });

    expect(result.originalLength).toBe(9);
  });
});

test.describe('Browser Error Handling', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/tests/browser/test.html');
  });

  test('should handle empty input', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const { compress, initWasm } = await import('@small-ltsc/sdk');
      await initWasm();

      const result = await compress([], {});
      return {
        originalLength: result.originalLength,
        compressedLength: result.compressedLength,
      };
    });

    expect(result.originalLength).toBe(0);
    expect(result.compressedLength).toBe(0);
  });

  test('should handle single token', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const { compress, initWasm } = await import('@small-ltsc/sdk');
      await initWasm();

      const result = await compress([42], {});
      return {
        originalLength: result.originalLength,
        compressedLength: result.compressedLength,
        ratio: result.compressionRatio,
      };
    });

    expect(result.originalLength).toBe(1);
    expect(result.ratio).toBe(1.0); // No compression possible
  });

  test('should handle unique tokens', async ({ page }) => {
    const result = await page.evaluate(async () => {
      const { compress, initWasm } = await import('@small-ltsc/sdk');
      await initWasm();

      // All unique tokens - no patterns
      const tokens = Array.from({ length: 100 }, (_, i) => i);
      const result = await compress(tokens, {});

      return {
        originalLength: result.originalLength,
        compressedLength: result.compressedLength,
        ratio: result.compressionRatio,
      };
    });

    expect(result.originalLength).toBe(100);
    // Should return original since no compression is beneficial
    expect(result.ratio).toBe(1.0);
  });
});
