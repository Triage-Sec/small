/**
 * Type declarations for wasm-pack generated module.
 * The actual module is generated at build time by wasm-pack.
 */
declare module './pkg/small_ltsc_core.js' {
  export function compress(tokens: Uint32Array, config?: unknown): unknown;
  export function decompress(tokens: Uint32Array, config?: unknown): Uint32Array;
  export function discover_patterns(
    tokens: Uint32Array,
    minLength: number,
    maxLength: number
  ): unknown;
  export function version(): string;
  export class StreamingCompressor {
    constructor(config: unknown);
    add_chunk(tokens: Uint32Array): void;
    finish(): unknown;
    memory_usage(): number;
    free(): void;
  }
  export default function init(
    module?: WebAssembly.Module | BufferSource | Response | Promise<Response>
  ): Promise<void>;
}
