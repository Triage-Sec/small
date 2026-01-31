//! Core types for LTSC compression.
//!
//! These types mirror the Python implementation in `delta/types.py`
//! but are optimized for WASM performance.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

/// A token is represented as a u32 for WASM efficiency.
/// In the TypeScript layer, these map to the tokenizer's vocabulary indices.
pub type Token = u32;

/// A sequence of tokens.
pub type TokenSeq = Vec<Token>;

/// A patch represents a position and replacement token for fuzzy matching.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Patch {
    pub index: usize,
    pub token: Token,
}

/// A candidate pattern discovered during compression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// The token subsequence that forms this pattern
    pub subsequence: Vec<Token>,
    /// Length of the subsequence
    pub length: usize,
    /// Positions where this pattern occurs (non-overlapping)
    pub positions: Vec<usize>,
    /// Priority for selection (higher = prefer compression)
    pub priority: i32,
    /// Patches for fuzzy matching (position -> patches)
    pub patches: HashMap<usize, Vec<Patch>>,
}

impl Candidate {
    pub fn new(subsequence: Vec<Token>, positions: Vec<usize>) -> Self {
        let length = subsequence.len();
        Self {
            subsequence,
            length,
            positions,
            priority: 0,
            patches: HashMap::new(),
        }
    }

    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }
}

/// An occurrence of a pattern at a specific position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Occurrence {
    /// Start position in the token sequence
    pub start: usize,
    /// Length of the pattern
    pub length: usize,
    /// The pattern subsequence
    pub subsequence: Vec<Token>,
    /// Priority for selection
    pub priority: i32,
    /// Patches for this specific occurrence
    pub patches: Vec<Patch>,
}

impl Occurrence {
    pub fn end(&self) -> usize {
        self.start + self.length
    }
}

/// Configuration for compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct CompressionConfig {
    /// Minimum pattern length to consider
    pub min_subsequence_length: usize,
    /// Maximum pattern length to consider
    pub max_subsequence_length: usize,
    /// Meta-token prefix string
    #[wasm_bindgen(skip)]
    pub meta_token_prefix: String,
    /// Meta-token suffix string
    #[wasm_bindgen(skip)]
    pub meta_token_suffix: String,
    /// Size of meta-token pool
    pub meta_token_pool_size: usize,
    /// Dictionary start delimiter token ID
    pub dict_start_token: Token,
    /// Dictionary end delimiter token ID
    pub dict_end_token: Token,
    /// Whether to include length tokens in dictionary
    pub dict_length_enabled: bool,
    /// Enable hierarchical compression
    pub hierarchical_enabled: bool,
    /// Maximum hierarchical compression depth
    pub hierarchical_max_depth: usize,
    /// Selection mode: "greedy", "optimal", "beam"
    #[wasm_bindgen(skip)]
    pub selection_mode: String,
    /// Beam width for beam search
    pub beam_width: usize,
    /// Enable round-trip verification
    pub verify: bool,
}

#[wasm_bindgen]
impl CompressionConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[wasm_bindgen(getter)]
    pub fn meta_token_prefix(&self) -> String {
        self.meta_token_prefix.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_meta_token_prefix(&mut self, prefix: String) {
        self.meta_token_prefix = prefix;
    }

    #[wasm_bindgen(getter)]
    pub fn meta_token_suffix(&self) -> String {
        self.meta_token_suffix.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_meta_token_suffix(&mut self, suffix: String) {
        self.meta_token_suffix = suffix;
    }

    #[wasm_bindgen(getter)]
    pub fn selection_mode(&self) -> String {
        self.selection_mode.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_selection_mode(&mut self, mode: String) {
        self.selection_mode = mode;
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            min_subsequence_length: 2,
            max_subsequence_length: 8,
            meta_token_prefix: "<MT_".to_string(),
            meta_token_suffix: ">".to_string(),
            meta_token_pool_size: 500,
            dict_start_token: 0xFFFFFFF0, // Reserved token IDs
            dict_end_token: 0xFFFFFFF1,
            dict_length_enabled: true,
            hierarchical_enabled: true,
            hierarchical_max_depth: 3,
            selection_mode: "greedy".to_string(),
            beam_width: 8,
            verify: false,
        }
    }
}

/// Result of compression operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct CompressionResult {
    /// Original tokens (stored for verification)
    #[wasm_bindgen(skip)]
    pub original_tokens: Vec<Token>,
    /// Fully serialized output (dictionary + body)
    #[wasm_bindgen(skip)]
    pub serialized_tokens: Vec<Token>,
    /// Dictionary tokens only
    #[wasm_bindgen(skip)]
    pub dictionary_tokens: Vec<Token>,
    /// Body tokens only (with meta-token references)
    #[wasm_bindgen(skip)]
    pub body_tokens: Vec<Token>,
    /// Mapping from meta-token to its expansion
    #[wasm_bindgen(skip)]
    pub dictionary_map: HashMap<Token, Vec<Token>>,
    /// Original sequence length
    pub original_length: usize,
    /// Compressed sequence length
    pub compressed_length: usize,
    /// Static dictionary ID if used
    #[wasm_bindgen(skip)]
    pub static_dictionary_id: Option<String>,
}

#[wasm_bindgen]
impl CompressionResult {
    /// Get the compression ratio (compressed/original).
    #[wasm_bindgen(getter)]
    pub fn compression_ratio(&self) -> f64 {
        if self.original_length == 0 {
            return 1.0;
        }
        self.compressed_length as f64 / self.original_length as f64
    }

    /// Get tokens saved by compression.
    #[wasm_bindgen(getter)]
    pub fn tokens_saved(&self) -> i64 {
        self.original_length as i64 - self.compressed_length as i64
    }

    /// Get the serialized tokens as a JS array.
    #[wasm_bindgen(js_name = getSerializedTokens)]
    pub fn get_serialized_tokens(&self) -> Vec<Token> {
        self.serialized_tokens.clone()
    }

    /// Get the dictionary tokens as a JS array.
    #[wasm_bindgen(js_name = getDictionaryTokens)]
    pub fn get_dictionary_tokens(&self) -> Vec<Token> {
        self.dictionary_tokens.clone()
    }

    /// Get the body tokens as a JS array.
    #[wasm_bindgen(js_name = getBodyTokens)]
    pub fn get_body_tokens(&self) -> Vec<Token> {
        self.body_tokens.clone()
    }

    /// Get the original tokens as a JS array.
    #[wasm_bindgen(js_name = getOriginalTokens)]
    pub fn get_original_tokens(&self) -> Vec<Token> {
        self.original_tokens.clone()
    }

    /// Get the static dictionary ID if used.
    #[wasm_bindgen(js_name = getStaticDictionaryId)]
    pub fn get_static_dictionary_id(&self) -> Option<String> {
        self.static_dictionary_id.clone()
    }
}

impl CompressionResult {
    /// Create a new compression result indicating no compression was beneficial.
    pub fn no_compression(tokens: Vec<Token>) -> Self {
        let len = tokens.len();
        Self {
            original_tokens: tokens.clone(),
            serialized_tokens: tokens.clone(),
            dictionary_tokens: Vec::new(),
            body_tokens: tokens,
            dictionary_map: HashMap::new(),
            original_length: len,
            compressed_length: len,
            static_dictionary_id: None,
        }
    }
}

/// Metrics from a compression operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct CompressionMetrics {
    /// Time spent in pattern discovery (ms)
    pub discovery_time_ms: f64,
    /// Time spent in selection (ms)
    pub selection_time_ms: f64,
    /// Time spent in serialization (ms)
    pub serialization_time_ms: f64,
    /// Total compression time (ms)
    pub total_time_ms: f64,
    /// Number of candidates discovered
    pub candidates_discovered: usize,
    /// Number of candidates selected
    pub candidates_selected: usize,
    /// Peak memory usage (bytes, approximate)
    pub peak_memory_bytes: usize,
}

impl Default for CompressionMetrics {
    fn default() -> Self {
        Self {
            discovery_time_ms: 0.0,
            selection_time_ms: 0.0,
            serialization_time_ms: 0.0,
            total_time_ms: 0.0,
            candidates_discovered: 0,
            candidates_selected: 0,
            peak_memory_bytes: 0,
        }
    }
}

/// Check if a pattern is compressible given length and occurrence count.
///
/// Compressibility condition: length * count > 1 + length + count + extra_cost
/// Where extra_cost is 1 if length tokens are enabled.
#[inline]
pub fn is_compressible(length: usize, count: usize, extra_cost: usize) -> bool {
    if length <= 1 || count == 0 {
        return false;
    }
    length * count > 1 + length + count + extra_cost
}

/// Compute minimum occurrence count for a pattern to be compressible.
#[inline]
pub fn min_count_for_compressibility(length: usize, extra_cost: usize) -> usize {
    if length <= 1 {
        return usize::MAX;
    }
    // Solving: length * count > 1 + length + count + extra_cost
    // count * (length - 1) > 1 + length + extra_cost
    // count > (2 + length + extra_cost) / (length - 1)
    let numerator = 2 + length + extra_cost;
    let denominator = length - 1;
    (numerator + denominator - 1) / denominator // Ceiling division
}

/// Compute net token savings for a pattern.
#[inline]
pub fn compute_savings(length: usize, count: usize, extra_cost: usize) -> i64 {
    if count == 0 {
        return 0;
    }
    let original = (length * count) as i64;
    let compressed = (1 + length + count + extra_cost) as i64;
    (original - compressed).max(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_compressible() {
        // Length 2, count 3: 2*3=6 > 1+2+3+1=7? No (6 > 7 is false)
        assert!(!is_compressible(2, 3, 1));
        // Length 2, count 4: 2*4=8 > 1+2+4+1=8? No (8 > 8 is false)
        assert!(!is_compressible(2, 4, 1));
        // Length 2, count 5: 2*5=10 > 1+2+5+1=9? Yes
        assert!(is_compressible(2, 5, 1));
        // Length 3, count 3: 3*3=9 > 1+3+3+1=8? Yes
        assert!(is_compressible(3, 3, 1));
        // Length 1 is never compressible
        assert!(!is_compressible(1, 100, 0));
    }

    #[test]
    fn test_min_count_for_compressibility() {
        // Length 2: need count > (2+2+1)/(2-1) = 5, so min is 6
        assert_eq!(min_count_for_compressibility(2, 1), 5);
        // Length 3: need count > (2+3+1)/(3-1) = 3, so min is 4
        assert_eq!(min_count_for_compressibility(3, 1), 3);
        // Length 8: need count > (2+8+1)/(8-1) = 11/7 ≈ 1.57, so min is 2
        assert_eq!(min_count_for_compressibility(8, 1), 2);
    }

    #[test]
    fn test_compute_savings() {
        // Length 3, count 5, extra 1: 3*5=15, 1+3+5+1=10, savings=5
        assert_eq!(compute_savings(3, 5, 1), 5);
        // Length 2, count 3, extra 1: 2*3=6, 1+2+3+1=7, savings=0 (capped)
        assert_eq!(compute_savings(2, 3, 1), 0);
    }

    #[test]
    fn test_config_default() {
        let config = CompressionConfig::default();
        assert_eq!(config.min_subsequence_length, 2);
        assert_eq!(config.max_subsequence_length, 8);
        assert!(config.hierarchical_enabled);
    }

    #[test]
    fn test_compression_result_no_compression() {
        let tokens = vec![1, 2, 3, 4, 5];
        let result = CompressionResult::no_compression(tokens.clone());
        assert_eq!(result.compression_ratio(), 1.0);
        assert_eq!(result.tokens_saved(), 0);
        assert_eq!(result.serialized_tokens, tokens);
    }
}
