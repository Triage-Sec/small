"""Serialization for template-based compression.

Handles encoding templates with slot markers and decoding back to
original token sequences.

Serialization format:
    Dictionary entry: <MT_N><Len:L>frame_token1 frame_token2 <Slot> frame_token3</MT_N>
    Body instance: <MT_N><SlotVal>value1</SlotVal></MT_N>
    
For multi-slot templates:
    Body instance: <MT_N><SlotVal>val1</SlotVal><SlotVal>val2</SlotVal></MT_N>
"""

from __future__ import annotations

from .config import CompressionConfig
from .template_types import (
    FrameElement,
    SlotMarker,
    Template,
    TemplateCandidate,
    TemplateInstance,
)
from .types import Token


# Default serialization tokens
DEFAULT_SLOT_MARKER = "<Slot>"
DEFAULT_SLOT_VAL_START = "<SlotVal>"
DEFAULT_SLOT_VAL_END = "</SlotVal>"
DEFAULT_TEMPLATE_START = "<Template>"
DEFAULT_TEMPLATE_END = "</Template>"


def get_slot_marker_token(config: CompressionConfig) -> str:
    """Get the slot marker token from config or default."""
    return getattr(config, 'template_slot_marker', DEFAULT_SLOT_MARKER)


def get_slot_val_tokens(config: CompressionConfig) -> tuple[str, str]:
    """Get slot value delimiter tokens from config or defaults."""
    start = getattr(config, 'template_slot_val_start', DEFAULT_SLOT_VAL_START)
    end = getattr(config, 'template_slot_val_end', DEFAULT_SLOT_VAL_END)
    return start, end


def serialize_template_frame(
    template: Template | TemplateCandidate,
    config: CompressionConfig,
) -> list[Token]:
    """Serialize a template frame to tokens.
    
    Converts the frame with SlotMarker objects to a token sequence
    with slot marker tokens.
    
    Args:
        template: The template to serialize
        config: Compression configuration
        
    Returns:
        List of tokens representing the frame
    """
    slot_marker = get_slot_marker_token(config)
    result: list[Token] = []
    
    for elem in template.frame:
        if isinstance(elem, SlotMarker):
            result.append(slot_marker)
        else:
            result.append(elem)
    
    return result


def serialize_template_instance(
    instance: TemplateInstance,
    meta_token: Token,
    config: CompressionConfig,
) -> list[Token]:
    """Serialize a template instance (reference + slot values).
    
    Args:
        instance: The template instance to serialize
        meta_token: The meta-token representing this template
        config: Compression configuration
        
    Returns:
        List of tokens for the body
    """
    slot_val_start, slot_val_end = get_slot_val_tokens(config)
    
    result: list[Token] = [meta_token]
    
    for slot_values in instance.slot_values:
        result.append(slot_val_start)
        result.extend(slot_values)
        result.append(slot_val_end)
    
    return result


def serialize_template_to_dict_entry(
    template: Template | TemplateCandidate,
    meta_token: Token,
    config: CompressionConfig,
) -> list[Token]:
    """Serialize a template to a dictionary entry.
    
    Format: <MT_N><Len:L>frame_tokens</MT_N>
    
    Args:
        template: The template to serialize
        meta_token: The assigned meta-token
        config: Compression configuration
        
    Returns:
        List of tokens for the dictionary entry
    """
    frame_tokens = serialize_template_frame(template, config)
    
    result: list[Token] = [meta_token]
    
    # Add length token if enabled
    if config.dict_length_enabled:
        length_token = f"{config.dict_length_prefix}{len(frame_tokens)}{config.dict_length_suffix}"
        result.append(length_token)
    
    result.extend(frame_tokens)
    
    return result


def deserialize_template_instance(
    tokens: list[Token],
    start_pos: int,
    slot_count: int,
    config: CompressionConfig,
) -> tuple[list[tuple[Token, ...]], int] | None:
    """Parse slot values from a serialized template instance.
    
    Args:
        tokens: The token sequence
        start_pos: Position after the meta-token reference
        slot_count: Number of slots to parse
        config: Compression configuration
        
    Returns:
        Tuple of (slot_values, end_position) or None if parsing fails
    """
    slot_val_start, slot_val_end = get_slot_val_tokens(config)
    
    slot_values: list[tuple[Token, ...]] = []
    pos = start_pos
    
    for _ in range(slot_count):
        # Expect slot value start
        if pos >= len(tokens) or tokens[pos] != slot_val_start:
            return None
        pos += 1
        
        # Collect slot value tokens until end marker
        value_tokens: list[Token] = []
        while pos < len(tokens) and tokens[pos] != slot_val_end:
            value_tokens.append(tokens[pos])
            pos += 1
        
        if pos >= len(tokens):
            return None
        
        pos += 1  # Skip end marker
        slot_values.append(tuple(value_tokens))
    
    return slot_values, pos


def expand_template_instance(
    frame: tuple[FrameElement, ...],
    slot_values: list[tuple[Token, ...]],
) -> list[Token]:
    """Expand a template instance to its original tokens.
    
    Args:
        frame: The template frame with slot markers
        slot_values: Values for each slot
        
    Returns:
        The expanded token sequence
    """
    result: list[Token] = []
    slot_index = 0
    
    for elem in frame:
        if isinstance(elem, SlotMarker):
            if slot_index < len(slot_values):
                result.extend(slot_values[slot_index])
            slot_index += 1
        else:
            result.append(elem)
    
    return result


def parse_template_frame(
    tokens: list[Token],
    config: CompressionConfig,
) -> tuple[FrameElement, ...]:
    """Parse a serialized frame back to frame elements.
    
    Args:
        tokens: The serialized frame tokens
        config: Compression configuration
        
    Returns:
        Tuple of frame elements (tokens and SlotMarkers)
    """
    slot_marker = get_slot_marker_token(config)
    
    result: list[FrameElement] = []
    slot_index = 0
    
    for token in tokens:
        if token == slot_marker:
            result.append(SlotMarker(index=slot_index))
            slot_index += 1
        else:
            result.append(token)
    
    return tuple(result)


def compute_template_body_length(
    template: Template | TemplateCandidate,
    config: CompressionConfig,
) -> int:
    """Compute the total body length when using this template.
    
    For each instance: 1 (meta ref) + 2*slot_count (delimiters) + slot_values
    
    Args:
        template: The template
        config: Compression configuration
        
    Returns:
        Total tokens in body for all instances
    """
    total = 0
    slot_count = template.slot_count if hasattr(template, 'slot_count') else len(template.slot_positions)
    
    for instance in template.instances:
        # Meta-token reference
        total += 1
        # Slot value delimiters (start + end for each slot)
        total += 2 * slot_count
        # Slot value tokens
        total += instance.total_slot_tokens()
    
    return total


def compute_template_dict_length(
    template: Template | TemplateCandidate,
    config: CompressionConfig,
) -> int:
    """Compute the dictionary entry length for a template.
    
    Args:
        template: The template
        config: Compression configuration
        
    Returns:
        Number of tokens in dictionary entry
    """
    frame_tokens = serialize_template_frame(template, config)
    
    # Meta-token + length token (if enabled) + frame
    length = 1 + len(frame_tokens)
    if config.dict_length_enabled:
        length += 1
    
    return length


def is_template_meta_token(
    token: Token,
    template_map: dict[Token, tuple[FrameElement, ...]],
) -> bool:
    """Check if a token is a template meta-token."""
    return token in template_map


def build_template_replacement_map(
    templates: list[Template | TemplateCandidate],
    meta_token_start: int,
    config: CompressionConfig,
) -> dict[Token, tuple[FrameElement, ...]]:
    """Build mapping from meta-tokens to template frames.
    
    Args:
        templates: List of templates to assign meta-tokens
        meta_token_start: Starting meta-token ID
        config: Compression configuration
        
    Returns:
        Dict mapping meta-token -> frame tuple
    """
    result: dict[Token, tuple[FrameElement, ...]] = {}
    
    for i, template in enumerate(templates):
        meta_token = f"{config.meta_token_prefix}{meta_token_start + i}{config.meta_token_suffix}"
        result[meta_token] = template.frame
    
    return result
