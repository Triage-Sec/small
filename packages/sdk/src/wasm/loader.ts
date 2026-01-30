/**
 * WASM module loader - wraps wasm-pack generated web target output.
 *
 * This loader uses the wasm-bindgen generated init() function which properly
 * handles WASM loading across browser and Node.js environments.
 */

// Re-export types - these match the wasm-bindgen generated classes
export interface CompressionResultWasm {
  compression_ratio: number;
  tokens_saved: number;
  original_length: number;
  compressed_length: number;
  getSerializedTokens: () => Uint32Array;
  getDictionaryTokens: () => Uint32Array;
  getBodyTokens: () => Uint32Array;
  getOriginalTokens: () => Uint32Array;
  getStaticDictionaryId: () => string | null;
  free: () => void;
}

export interface StreamingCompressorWasm {
  add_chunk: (tokens: Uint32Array) => void;
  finish: () => CompressionResultWasm;
  memory_usage: () => number;
  free: () => void;
}

// Placeholder types until WASM is loaded
export interface WasmExports {
  compress: (tokens: Uint32Array, config?: unknown) => CompressionResultWasm;
  decompress: (tokens: Uint32Array, config?: unknown) => Uint32Array;
  discover_patterns: (
    tokens: Uint32Array,
    minLength: number,
    maxLength: number
  ) => unknown;
  version: () => string;
  StreamingCompressor: new (config: unknown) => StreamingCompressorWasm;
}

// Track initialization state
let initialized = false;
let initPromise: Promise<void> | null = null;
let wasmExports: WasmExports | null = null;

/**
 * Detect the current runtime environment.
 */
function detectEnvironment(): 'browser' | 'node' {
  if (
    typeof process !== 'undefined' &&
    process.versions &&
    process.versions.node
  ) {
    return 'node';
  }
  return 'browser';
}

/**
 * Initialize the WASM module.
 *
 * This function is idempotent - calling it multiple times will only
 * initialize once.
 *
 * @throws Error if WASM loading fails
 */
export async function initWasm(): Promise<void> {
  if (initialized) {
    return;
  }

  if (initPromise) {
    return initPromise;
  }

  initPromise = (async () => {
    try {
      const env = detectEnvironment();
      
      // Dynamically import the wasm-pack generated module
      // Types are provided by pkg.d.ts stub
      const wasmModule = await import('./pkg/small_ltsc_core.js');
      
      if (env === 'node') {
        // In Node.js, we need to provide the WASM file path/bytes
        const { readFile } = await import('node:fs/promises');
        const { fileURLToPath } = await import('node:url');
        const wasmPath = new URL('./pkg/small_ltsc_core_bg.wasm', import.meta.url);
        const path = fileURLToPath(wasmPath);
        const buffer = await readFile(path);
        await wasmModule.default(buffer);
      } else {
        // In browser, wasm-pack's init() handles loading via import.meta.url
        await wasmModule.default();
      }
      
      // Store exports for getWasm()
      wasmExports = {
        compress: wasmModule.compress,
        decompress: wasmModule.decompress,
        discover_patterns: wasmModule.discover_patterns,
        version: wasmModule.version,
        StreamingCompressor: wasmModule.StreamingCompressor,
      };
      
      initialized = true;
    } catch (error) {
      initPromise = null;
      throw error;
    }
  })();

  return initPromise;
}

/**
 * Initialize from pre-compiled WASM module or Response.
 *
 * @param module - Pre-compiled WebAssembly.Module, Response, or bytes
 */
export async function initWasmFromModule(
  module: WebAssembly.Module | Response | Promise<Response> | BufferSource
): Promise<void> {
  if (initialized) {
    return;
  }

  // Types are provided by pkg.d.ts stub
  const wasmModule = await import('./pkg/small_ltsc_core.js');
  await wasmModule.default(module);
  
  wasmExports = {
    compress: wasmModule.compress,
    decompress: wasmModule.decompress,
    discover_patterns: wasmModule.discover_patterns,
    version: wasmModule.version,
    StreamingCompressor: wasmModule.StreamingCompressor,
  };
  
  initialized = true;
}

/**
 * Initialize from WASM bytes.
 *
 * @param bytes - WASM binary as ArrayBuffer or Uint8Array
 */
export async function initWasmFromBytes(
  bytes: ArrayBuffer | Uint8Array
): Promise<void> {
  return initWasmFromModule(bytes);
}

/**
 * Get the initialized WASM exports.
 *
 * @throws Error if WASM is not initialized
 */
export function getWasm(): WasmExports {
  if (!initialized || !wasmExports) {
    throw new Error(
      'WASM not initialized. Call initWasm() first and await its completion.'
    );
  }
  return wasmExports;
}

/**
 * Check if WASM is initialized.
 */
export function isWasmInitialized(): boolean {
  return initialized;
}

/**
 * Reset the WASM instance (mainly for testing).
 */
export function resetWasm(): void {
  initialized = false;
  initPromise = null;
  wasmExports = null;
}

/**
 * Get WASM module version.
 */
export function getWasmVersion(): string {
  const wasm = getWasm();
  return wasm.version();
}
