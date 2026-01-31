//! Dictionary serialization and deserialization.
//!
//! Implements the LTSC compression format for dictionary and body tokens.
//! Port of `delta/dictionary.py` and `delta/serialization.py`.

use crate::types::{CompressionConfig, CompressionResult, Occurrence, Token};
use std::collections::HashMap;

/// Default delimiter tokens if not specified in config.
pub const DEFAULT_DICT_START: Token = 0xFFFFFFF0;
pub const DEFAULT_DICT_END: Token = 0xFFFFFFF1;

/// Result of dictionary building.
#[derive(Debug, Clone)]
pub struct Dictionary {
    /// Mapping from meta-token to its expansion
    pub entries: HashMap<Token, Vec<Token>>,
    /// Serialized dictionary tokens
    pub tokens: Vec<Token>,
    /// Meta-token IDs assigned to each pattern
    pub pattern_to_meta: HashMap<Vec<Token>, Token>,
}

/// Build a dictionary from selected occurrences.
///
/// Assigns meta-tokens to each unique pattern and produces the serialized
/// dictionary format: [DICT_START, MT_1, LEN_1, DEF_1..., MT_2, LEN_2, DEF_2..., DICT_END]
pub fn build_dictionary(
    selected: &[Occurrence],
    config: &CompressionConfig,
    next_meta_token: Token,
) -> Dictionary {
    if selected.is_empty() {
        return Dictionary {
            entries: HashMap::new(),
            tokens: Vec::new(),
            pattern_to_meta: HashMap::new(),
        };
    }

    // Collect unique patterns with their occurrence counts
    let mut pattern_counts: HashMap<Vec<Token>, usize> = HashMap::new();
    for occ in selected {
        *pattern_counts.entry(occ.subsequence.clone()).or_default() += 1;
    }

    // Order patterns for serialization (topological sort for hierarchical compression)
    let ordered_patterns = topological_order(&pattern_counts, selected);

    // Assign meta-tokens
    let mut pattern_to_meta: HashMap<Vec<Token>, Token> = HashMap::new();
    let mut meta_counter = next_meta_token;

    for pattern in &ordered_patterns {
        pattern_to_meta.insert(pattern.clone(), meta_counter);
        meta_counter += 1;
    }

    // Build dictionary entries
    let mut entries: HashMap<Token, Vec<Token>> = HashMap::new();
    for (pattern, &meta_token) in &pattern_to_meta {
        entries.insert(meta_token, pattern.clone());
    }

    // Serialize dictionary
    let dict_start = config.dict_start_token;
    let dict_end = config.dict_end_token;

    let mut tokens = Vec::new();
    tokens.push(dict_start);

    for pattern in &ordered_patterns {
        let meta_token = pattern_to_meta[pattern];
        tokens.push(meta_token);

        if config.dict_length_enabled {
            tokens.push(pattern.len() as Token);
        }

        // Serialize definition (may reference earlier meta-tokens)
        let definition = serialize_pattern(pattern, &pattern_to_meta);
        tokens.extend(definition);
    }

    tokens.push(dict_end);

    Dictionary {
        entries,
        tokens,
        pattern_to_meta,
    }
}

/// Serialize a pattern, potentially replacing sub-patterns with meta-tokens.
fn serialize_pattern(
    pattern: &[Token],
    _pattern_to_meta: &HashMap<Vec<Token>, Token>,
) -> Vec<Token> {
    // For now, don't replace sub-patterns in the dictionary
    // (hierarchical compression handles this at a higher level)
    pattern.to_vec()
}

/// Topologically order patterns so dependencies come before dependents.
///
/// This ensures that if pattern A's definition references pattern B's meta-token,
/// pattern B appears first in the dictionary.
fn topological_order(
    pattern_counts: &HashMap<Vec<Token>, usize>,
    _selected: &[Occurrence],
) -> Vec<Vec<Token>> {
    // For now, order by length (shorter first) then by count (higher first)
    // This is a simplified approach; full hierarchical support would need
    // to detect when one pattern's definition references another's meta-token
    let mut patterns: Vec<Vec<Token>> = pattern_counts.keys().cloned().collect();

    patterns.sort_by(|a, b| {
        let len_cmp = a.len().cmp(&b.len());
        if len_cmp != std::cmp::Ordering::Equal {
            return len_cmp;
        }
        // Higher count first
        let count_a = pattern_counts.get(a).copied().unwrap_or(0);
        let count_b = pattern_counts.get(b).copied().unwrap_or(0);
        count_b.cmp(&count_a)
    });

    patterns
}

/// Build body tokens with pattern replacements.
///
/// Replaces selected pattern occurrences with their assigned meta-tokens.
pub fn build_body(
    tokens: &[Token],
    selected: &[Occurrence],
    pattern_to_meta: &HashMap<Vec<Token>, Token>,
) -> Vec<Token> {
    if selected.is_empty() || pattern_to_meta.is_empty() {
        return tokens.to_vec();
    }

    // Sort occurrences by start position
    let mut sorted_selected: Vec<&Occurrence> = selected.iter().collect();
    sorted_selected.sort_by_key(|occ| occ.start);

    let mut body = Vec::with_capacity(tokens.len());
    let mut pos = 0;

    for occ in sorted_selected {
        // Copy tokens before this occurrence
        if pos < occ.start {
            body.extend_from_slice(&tokens[pos..occ.start]);
        }

        // Replace occurrence with meta-token
        if let Some(&meta_token) = pattern_to_meta.get(&occ.subsequence) {
            body.push(meta_token);
        } else {
            // Pattern not found - keep original tokens (shouldn't happen)
            body.extend_from_slice(&tokens[occ.start..occ.start + occ.length]);
        }

        pos = occ.start + occ.length;
    }

    // Copy remaining tokens
    if pos < tokens.len() {
        body.extend_from_slice(&tokens[pos..]);
    }

    body
}

/// Decompress tokens by expanding meta-tokens.
///
/// Iteratively expands all meta-tokens until no more remain.
pub fn decompress(
    tokens: &[Token],
    dictionary: &HashMap<Token, Vec<Token>>,
    config: &CompressionConfig,
) -> Vec<Token> {
    // First, extract body tokens (skip dictionary section)
    let body = extract_body(tokens, config);

    // Iteratively expand meta-tokens
    let mut result = body;
    let max_iterations = 100; // Prevent infinite loops

    for _ in 0..max_iterations {
        let (expanded, changed) = expand_once(&result, dictionary);
        if !changed {
            break;
        }
        result = expanded;
    }

    result
}

/// Extract body tokens from serialized output (after the dictionary section).
fn extract_body(tokens: &[Token], config: &CompressionConfig) -> Vec<Token> {
    let dict_end = config.dict_end_token;

    // Find the end of the dictionary
    if let Some(end_pos) = tokens.iter().position(|&t| t == dict_end) {
        return tokens[end_pos + 1..].to_vec();
    }

    // No dictionary section found - return all tokens
    tokens.to_vec()
}

/// Expand meta-tokens one level.
fn expand_once(tokens: &[Token], dictionary: &HashMap<Token, Vec<Token>>) -> (Vec<Token>, bool) {
    let mut result = Vec::with_capacity(tokens.len() * 2);
    let mut changed = false;

    for &token in tokens {
        if let Some(expansion) = dictionary.get(&token) {
            result.extend(expansion);
            changed = true;
        } else {
            result.push(token);
        }
    }

    (result, changed)
}

/// Parse a serialized token sequence to extract the dictionary mapping.
pub fn parse_dictionary(
    tokens: &[Token],
    config: &CompressionConfig,
) -> HashMap<Token, Vec<Token>> {
    let dict_start = config.dict_start_token;
    let dict_end = config.dict_end_token;

    let mut dictionary = HashMap::new();

    // Find dictionary section
    let start_pos = match tokens.iter().position(|&t| t == dict_start) {
        Some(pos) => pos + 1,
        None => return dictionary,
    };

    let end_pos = match tokens[start_pos..].iter().position(|&t| t == dict_end) {
        Some(pos) => start_pos + pos,
        None => return dictionary,
    };

    // Parse dictionary entries
    let mut pos = start_pos;
    while pos < end_pos {
        let meta_token = tokens[pos];
        pos += 1;

        if pos >= end_pos {
            break;
        }

        let length = if config.dict_length_enabled {
            let len = tokens[pos] as usize;
            pos += 1;
            len
        } else {
            // Without length tokens, we need another way to determine entry length
            // For now, assume fixed-length or delimiter-based (not implemented)
            break;
        };

        if pos + length > end_pos {
            break;
        }

        let definition: Vec<Token> = tokens[pos..pos + length].to_vec();
        dictionary.insert(meta_token, definition);

        pos += length;
    }

    dictionary
}

/// Serialize compression result to final token sequence.
pub fn serialize_result(
    dictionary: &Dictionary,
    body: &[Token],
    original: &[Token],
    _config: &CompressionConfig,
) -> CompressionResult {
    let mut serialized = Vec::with_capacity(dictionary.tokens.len() + body.len());
    serialized.extend(&dictionary.tokens);
    serialized.extend(body);

    CompressionResult {
        original_tokens: original.to_vec(),
        serialized_tokens: serialized.clone(),
        dictionary_tokens: dictionary.tokens.clone(),
        body_tokens: body.to_vec(),
        dictionary_map: dictionary.entries.clone(),
        original_length: original.len(),
        compressed_length: serialized.len(),
        static_dictionary_id: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Occurrence;

    fn default_config() -> CompressionConfig {
        CompressionConfig::default()
    }

    fn make_occurrence(subseq: Vec<Token>, start: usize) -> Occurrence {
        let length = subseq.len();
        Occurrence {
            start,
            length,
            subsequence: subseq,
            priority: 0,
            patches: vec![],
        }
    }

    #[test]
    fn test_build_dictionary_empty() {
        let config = default_config();
        let dict = build_dictionary(&[], &config, 1000);

        assert!(dict.entries.is_empty());
        assert!(dict.tokens.is_empty());
    }

    #[test]
    fn test_build_dictionary_single() {
        let config = default_config();
        let occurrences = vec![
            make_occurrence(vec![1, 2], 0),
            make_occurrence(vec![1, 2], 4),
            make_occurrence(vec![1, 2], 8),
        ];

        let dict = build_dictionary(&occurrences, &config, 1000);

        assert_eq!(dict.entries.len(), 1);
        assert!(dict.pattern_to_meta.contains_key(&vec![1u32, 2u32]));

        // Dictionary format: [START, META, LEN, DEF..., END]
        assert!(dict.tokens.len() >= 5);
        assert_eq!(dict.tokens[0], config.dict_start_token);
        assert_eq!(*dict.tokens.last().unwrap(), config.dict_end_token);
    }

    #[test]
    fn test_build_body_simple() {
        let tokens = vec![1, 2, 3, 4, 1, 2, 5, 6];
        let selected = vec![make_occurrence(vec![1, 2], 0), make_occurrence(vec![1, 2], 4)];

        let mut pattern_to_meta = HashMap::new();
        pattern_to_meta.insert(vec![1u32, 2u32], 1000u32);

        let body = build_body(&tokens, &selected, &pattern_to_meta);

        // Should be: [1000, 3, 4, 1000, 5, 6]
        assert_eq!(body, vec![1000, 3, 4, 1000, 5, 6]);
    }

    #[test]
    fn test_decompress_simple() {
        let config = default_config();
        let mut dictionary = HashMap::new();
        dictionary.insert(1000u32, vec![1u32, 2u32]);

        let serialized = vec![
            config.dict_start_token,
            1000,
            2, // length
            1,
            2, // definition
            config.dict_end_token,
            1000, // meta-token in body
            3,
            4,
            1000,
        ];

        let result = decompress(&serialized, &dictionary, &config);

        // Should expand 1000 -> [1, 2]
        assert_eq!(result, vec![1, 2, 3, 4, 1, 2]);
    }

    #[test]
    fn test_parse_dictionary() {
        let config = default_config();
        let tokens = vec![
            config.dict_start_token,
            1000, // meta-token
            2,    // length
            1,
            2, // definition
            config.dict_end_token,
            1000,
            3,
            4,
        ];

        let dict = parse_dictionary(&tokens, &config);

        assert_eq!(dict.len(), 1);
        assert_eq!(dict.get(&1000), Some(&vec![1u32, 2u32]));
    }

    #[test]
    fn test_round_trip() {
        let config = default_config();
        let original = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
        let selected = vec![
            make_occurrence(vec![1, 2, 3], 0),
            make_occurrence(vec![1, 2, 3], 3),
            make_occurrence(vec![1, 2, 3], 6),
        ];

        // Build dictionary
        let dict = build_dictionary(&selected, &config, 1000);

        // Build body
        let body = build_body(&original, &selected, &dict.pattern_to_meta);

        // Serialize
        let result = serialize_result(&dict, &body, &original, &config);

        // Decompress
        let restored = decompress(&result.serialized_tokens, &dict.entries, &config);

        assert_eq!(restored, original);
    }
}
