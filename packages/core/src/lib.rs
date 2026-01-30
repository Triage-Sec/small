//! Small LTSC Core - Lossless Token Sequence Compression
//!
//! This is the WebAssembly core for the Small LTSC compression library.
//! It provides high-performance compression and decompression of token sequences
//! using suffix array-based pattern discovery and optimal selection algorithms.
//!
//! # Example (from JavaScript)
//!
//! ```javascript
//! import { initWasm, compress, decompress } from '@small-ltsc/sdk';
//!
//! await initWasm();
//! const tokens = [1, 2, 3, 1, 2, 3, 1, 2, 3];
//! const result = await compress(tokens);
//! const restored = await decompress(result.serializedTokens);
//! ```

pub mod config;
pub mod dictionary;
pub mod discovery;
pub mod selection;
pub mod suffix_array;
#[cfg(feature = "parallel")]
pub mod suffix_array_parallel;
pub mod types;

use config::JsCompressionConfig;
use dictionary::{build_body, build_dictionary, decompress as dict_decompress, parse_dictionary, serialize_result};
use discovery::{deduplicate_candidates, discover_candidates, DiscoveryConfig};
use selection::select_occurrences;
use types::{CompressionConfig, CompressionResult, Token};
use wasm_bindgen::prelude::*;

/// Initialize panic hook for better error messages in WASM.
#[wasm_bindgen(start)]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Compress a token sequence.
///
/// # Arguments
///
/// * `tokens` - The token sequence to compress (Uint32Array from JS)
/// * `config` - Optional configuration (JsValue representing JsCompressionConfig)
///
/// # Returns
///
/// A CompressionResult containing the compressed tokens and metadata.
#[wasm_bindgen]
pub fn compress(tokens: &[u32], config: JsValue) -> Result<CompressionResult, JsValue> {
    let js_config: JsCompressionConfig = if config.is_undefined() || config.is_null() {
        JsCompressionConfig::default()
    } else {
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?
    };

    let compression_config = js_config.merge_with_defaults();
    let next_meta_token = js_config.next_meta_token.unwrap_or(0xFFFF0000);

    compress_internal(tokens, &compression_config, next_meta_token)
}

/// Internal compression implementation.
fn compress_internal(
    tokens: &[Token],
    config: &CompressionConfig,
    next_meta_token: Token,
) -> Result<CompressionResult, JsValue> {
    // Handle empty or small inputs
    if tokens.len() < config.min_subsequence_length * 2 {
        return Ok(CompressionResult::no_compression(tokens.to_vec()));
    }

    // Discovery configuration
    let discovery_config = DiscoveryConfig {
        min_length: config.min_subsequence_length,
        max_length: config.max_subsequence_length,
        min_occurrences: 2,
        extra_cost: if config.dict_length_enabled { 1 } else { 0 },
    };

    // Discover candidate patterns
    let candidates = discover_candidates(tokens, &discovery_config);
    if candidates.is_empty() {
        return Ok(CompressionResult::no_compression(tokens.to_vec()));
    }

    // Deduplicate candidates
    let candidates = deduplicate_candidates(candidates);

    // Select non-overlapping occurrences
    let selection_result = select_occurrences(
        &candidates,
        &config.selection_mode,
        discovery_config.extra_cost,
    );

    if selection_result.selected.is_empty() {
        return Ok(CompressionResult::no_compression(tokens.to_vec()));
    }

    // Build dictionary
    let dict = build_dictionary(&selection_result.selected, config, next_meta_token);
    if dict.entries.is_empty() {
        return Ok(CompressionResult::no_compression(tokens.to_vec()));
    }

    // Build body with replacements
    let body = build_body(tokens, &selection_result.selected, &dict.pattern_to_meta);

    // Check if compression is beneficial
    let compressed_len = dict.tokens.len() + body.len();
    if compressed_len >= tokens.len() {
        return Ok(CompressionResult::no_compression(tokens.to_vec()));
    }

    // Serialize result
    let mut result = serialize_result(&dict, &body, tokens, config);

    // Verify if requested
    if config.verify {
        let restored = dict_decompress(&result.serialized_tokens, &dict.entries, config);
        if restored != tokens {
            return Err(JsValue::from_str(
                "Compression verification failed: round-trip mismatch",
            ));
        }
    }

    // Hierarchical compression
    if config.hierarchical_enabled && config.hierarchical_max_depth > 1 {
        result = apply_hierarchical(result, config, next_meta_token + dict.entries.len() as Token)?;
    }

    Ok(result)
}

/// Apply hierarchical compression passes.
fn apply_hierarchical(
    mut result: CompressionResult,
    config: &CompressionConfig,
    mut next_meta_token: Token,
) -> Result<CompressionResult, JsValue> {
    let min_improvement = 0.02; // 2% improvement threshold

    for _depth in 1..config.hierarchical_max_depth {
        let body = result.body_tokens.clone();
        if body.len() < config.min_subsequence_length * 2 {
            break;
        }

        let discovery_config = DiscoveryConfig {
            min_length: config.min_subsequence_length,
            max_length: config.max_subsequence_length,
            min_occurrences: 2,
            extra_cost: if config.dict_length_enabled { 1 } else { 0 },
        };

        let candidates = discover_candidates(&body, &discovery_config);
        if candidates.is_empty() {
            break;
        }

        let candidates = deduplicate_candidates(candidates);
        let selection_result = select_occurrences(
            &candidates,
            &config.selection_mode,
            discovery_config.extra_cost,
        );

        if selection_result.selected.is_empty() {
            break;
        }

        let new_dict = build_dictionary(&selection_result.selected, config, next_meta_token);
        if new_dict.entries.is_empty() {
            break;
        }

        let new_body = build_body(&body, &selection_result.selected, &new_dict.pattern_to_meta);
        let new_compressed_len = result.dictionary_tokens.len() + new_dict.tokens.len() + new_body.len();

        let improvement = 1.0 - (new_compressed_len as f64 / result.compressed_length as f64);
        if improvement < min_improvement {
            break;
        }

        // Merge dictionaries
        let mut merged_dict_tokens = result.dictionary_tokens.clone();
        // Remove old dict end, add new entries
        if let Some(pos) = merged_dict_tokens.iter().rposition(|&t| t == config.dict_end_token) {
            merged_dict_tokens.truncate(pos);
        }
        // Add new dictionary entries (skip new start token)
        let new_entries_start = new_dict.tokens.iter()
            .position(|&t| t != config.dict_start_token)
            .unwrap_or(0);
        merged_dict_tokens.extend_from_slice(&new_dict.tokens[new_entries_start..]);

        let new_entries_len = new_dict.entries.len();
        // Merge dictionary maps
        let mut merged_map = result.dictionary_map.clone();
        merged_map.extend(new_dict.entries);

        // Update serialized
        let mut serialized = merged_dict_tokens.clone();
        serialized.extend(&new_body);

        result = CompressionResult {
            original_tokens: result.original_tokens,
            serialized_tokens: serialized.clone(),
            dictionary_tokens: merged_dict_tokens,
            body_tokens: new_body,
            dictionary_map: merged_map,
            original_length: result.original_length,
            compressed_length: serialized.len(),
            static_dictionary_id: None,
        };

        next_meta_token += new_entries_len as Token;
    }

    Ok(result)
}

/// Decompress a compressed token sequence.
///
/// # Arguments
///
/// * `tokens` - The compressed token sequence
/// * `config` - Optional configuration
///
/// # Returns
///
/// The original token sequence.
#[wasm_bindgen]
pub fn decompress(tokens: &[u32], config: JsValue) -> Result<Vec<u32>, JsValue> {
    let js_config: JsCompressionConfig = if config.is_undefined() || config.is_null() {
        JsCompressionConfig::default()
    } else {
        serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?
    };

    let compression_config = js_config.merge_with_defaults();

    // Parse dictionary from tokens
    let dictionary = parse_dictionary(tokens, &compression_config);

    // Decompress
    let result = dict_decompress(tokens, &dictionary, &compression_config);

    Ok(result)
}

/// Streaming compressor for large inputs.
#[wasm_bindgen]
pub struct StreamingCompressor {
    chunks: Vec<Vec<Token>>,
    config: CompressionConfig,
    next_meta_token: Token,
}

#[wasm_bindgen]
impl StreamingCompressor {
    /// Create a new streaming compressor.
    #[wasm_bindgen(constructor)]
    pub fn new(config: JsValue) -> Result<StreamingCompressor, JsValue> {
        let js_config: JsCompressionConfig = if config.is_undefined() || config.is_null() {
            JsCompressionConfig::default()
        } else {
            serde_wasm_bindgen::from_value(config).map_err(|e| JsValue::from_str(&e.to_string()))?
        };

        Ok(Self {
            chunks: Vec::new(),
            config: js_config.merge_with_defaults(),
            next_meta_token: js_config.next_meta_token.unwrap_or(0xFFFF0000),
        })
    }

    /// Add a chunk of tokens.
    pub fn add_chunk(&mut self, tokens: &[u32]) {
        self.chunks.push(tokens.to_vec());
    }

    /// Finish streaming and produce compressed result.
    pub fn finish(self) -> Result<CompressionResult, JsValue> {
        // Concatenate all chunks
        let total_len: usize = self.chunks.iter().map(|c| c.len()).sum();
        let mut all_tokens = Vec::with_capacity(total_len);
        for chunk in &self.chunks {
            all_tokens.extend(chunk);
        }

        // Compress the full sequence
        compress_internal(&all_tokens, &self.config, self.next_meta_token)
    }

    /// Get approximate memory usage.
    pub fn memory_usage(&self) -> usize {
        self.chunks.iter().map(|c| c.len() * 4).sum()
    }
}

/// Discover patterns without compressing.
///
/// Useful for analysis and building static dictionaries.
#[wasm_bindgen]
pub fn discover_patterns(
    tokens: &[u32],
    min_length: usize,
    max_length: usize,
) -> Result<JsValue, JsValue> {
    let config = DiscoveryConfig {
        min_length,
        max_length,
        min_occurrences: 2,
        extra_cost: 1,
    };

    let candidates = discover_candidates(tokens, &config);

    // Convert to JS-friendly format
    let result: Vec<serde_json::Value> = candidates
        .iter()
        .map(|c| {
            serde_json::json!({
                "pattern": c.subsequence,
                "length": c.length,
                "positions": c.positions,
                "count": c.positions.len(),
            })
        })
        .collect();

    serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Get version information.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Format version for serialized output.
pub const FORMAT_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_simple() {
        // Use larger input to overcome dictionary overhead
        let pattern = vec![1, 2, 3, 4, 5];
        let tokens: Vec<Token> = pattern.iter().cycle().take(50).cloned().collect();
        let config = CompressionConfig::default();

        let result = compress_internal(&tokens, &config, 0xFFFF0000).unwrap();

        // Should achieve compression with sufficient repetition
        assert!(result.compressed_length < result.original_length);
        assert!(result.compression_ratio() < 1.0);
    }

    #[test]
    fn test_round_trip() {
        let pattern = vec![1, 2, 3, 4, 5];
        let tokens: Vec<Token> = pattern.iter().cycle().take(50).cloned().collect();
        let config = CompressionConfig {
            verify: true,
            ..Default::default()
        };

        let result = compress_internal(&tokens, &config, 0xFFFF0000).unwrap();

        // Parse dictionary and decompress
        let dictionary = parse_dictionary(&result.serialized_tokens, &config);
        let restored = dict_decompress(&result.serialized_tokens, &dictionary, &config);

        assert_eq!(restored, tokens);
    }

    #[test]
    fn test_no_compression_small_input() {
        let tokens = vec![1, 2, 3];
        let config = CompressionConfig::default();

        let result = compress_internal(&tokens, &config, 0xFFFF0000).unwrap();

        // Should return original tokens
        assert_eq!(result.serialized_tokens, tokens);
        assert_eq!(result.compression_ratio(), 1.0);
    }

    #[test]
    fn test_streaming_compressor() {
        let config = CompressionConfig::default();

        let mut compressor = StreamingCompressor {
            chunks: Vec::new(),
            config,
            next_meta_token: 0xFFFF0000,
        };

        compressor.add_chunk(&[1, 2, 3, 1, 2, 3, 1, 2, 3]);
        compressor.add_chunk(&[1, 2, 3, 1, 2, 3]);

        let result = compressor.finish().unwrap();

        // Should compress combined input
        assert!(result.original_length == 15);
    }
}
