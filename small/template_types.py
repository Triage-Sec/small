"""Types for template-based compression.

Templates represent parameterized patterns with fixed "frame" tokens
and variable "slot" positions. This enables compression of repeated
structures that differ only in specific positions.

Example:
    Pattern: logger.info("User logged in: {name}")
    Frame: (logger, ., info, (, ", User, logged, in, :, <SLOT>, ", ))
    Instances:
        - Position 0, slots: ("alice",)
        - Position 20, slots: ("bob",)
        - Position 40, slots: ("charlie",)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from .types import Token


# Sentinel class for slot markers in frames
@dataclass(frozen=True)
class SlotMarker:
    """Marker for a variable slot position in a template frame.
    
    Attributes:
        index: The slot index (0-based)
        min_length: Minimum tokens this slot can contain
        max_length: Maximum tokens this slot can contain
    """
    index: int
    min_length: int = 1
    max_length: int = 10
    
    def __repr__(self) -> str:
        return f"<SLOT:{self.index}>"
    
    def __str__(self) -> str:
        return f"<SLOT:{self.index}>"


# Type alias for frame elements (either a token or a slot marker)
FrameElement = Union[Token, SlotMarker]


@dataclass(frozen=True)
class TemplateInstance:
    """A single instance of a template in the input.
    
    Attributes:
        position: Start position in the original token sequence
        slot_values: Tuple of slot value tuples, one per slot
        original_length: Length of this instance in original tokens
    """
    position: int
    slot_values: tuple[tuple[Token, ...], ...]
    original_length: int
    
    def total_slot_tokens(self) -> int:
        """Total number of tokens across all slots."""
        return sum(len(sv) for sv in self.slot_values)


@dataclass(frozen=True)
class Template:
    """A template with fixed frame and variable slots.
    
    The frame contains fixed tokens interspersed with SlotMarker objects
    indicating where variable content appears.
    
    Attributes:
        frame: Tuple of tokens and slot markers defining the pattern
        slot_count: Number of variable slots
        instances: All instances of this template in the input
    """
    frame: tuple[FrameElement, ...]
    slot_count: int
    instances: tuple[TemplateInstance, ...]
    
    def frame_length(self) -> int:
        """Number of fixed tokens in the frame (excluding slots)."""
        return sum(1 for elem in self.frame if not isinstance(elem, SlotMarker))
    
    def frame_tokens(self) -> tuple[Token, ...]:
        """Extract just the fixed tokens from the frame."""
        return tuple(elem for elem in self.frame if not isinstance(elem, SlotMarker))
    
    def slot_positions(self) -> tuple[int, ...]:
        """Indices of slot markers in the frame."""
        return tuple(i for i, elem in enumerate(self.frame) if isinstance(elem, SlotMarker))
    
    def estimated_savings(self, extra_cost: int = 1) -> int:
        """Estimate compression savings from this template.
        
        Original cost: sum of all instance lengths
        Compressed cost: frame + slot markers + all slot values + instance refs
        
        Args:
            extra_cost: Additional cost per dictionary entry (e.g., length token)
            
        Returns:
            Estimated token savings (positive = good compression)
        """
        if not self.instances:
            return 0
            
        # Original cost
        original = sum(inst.original_length for inst in self.instances)
        
        # Compressed cost:
        # - Dictionary entry: 1 (meta token) + frame_length + slot_count (markers) + extra_cost
        # - Body: for each instance: 1 (meta ref) + slot_values
        dict_cost = 1 + self.frame_length() + self.slot_count + extra_cost
        body_cost = sum(
            1 + inst.total_slot_tokens() + self.slot_count  # ref + values + separators
            for inst in self.instances
        )
        
        compressed = dict_cost + body_cost
        return original - compressed


@dataclass(frozen=True)
class TemplateCandidate:
    """A candidate template discovered during mining.
    
    This is the intermediate representation before templates are
    converted to compression candidates.
    
    Attributes:
        frame: The template frame with slot markers
        slot_positions: Positions of slots in the frame
        instances: Discovered instances
        frame_length: Number of fixed tokens
        savings: Estimated compression savings
        priority: Priority for selection (higher = prefer)
    """
    frame: tuple[FrameElement, ...]
    slot_positions: tuple[int, ...]
    instances: tuple[TemplateInstance, ...]
    frame_length: int
    savings: int
    priority: int = 0
    
    @property
    def slot_count(self) -> int:
        """Number of slots in the template."""
        return len(self.slot_positions)
    
    @property
    def instance_count(self) -> int:
        """Number of instances found."""
        return len(self.instances)
    
    def to_template(self) -> Template:
        """Convert to a Template object."""
        return Template(
            frame=self.frame,
            slot_count=self.slot_count,
            instances=self.instances,
        )


@dataclass
class TemplateMatch:
    """Result of matching a template against a token sequence.
    
    Used during template discovery to track potential matches.
    """
    start: int
    end: int
    slot_values: list[list[Token]] = field(default_factory=list)
    score: float = 0.0
    
    @property
    def length(self) -> int:
        return self.end - self.start


def frame_to_string(frame: tuple[FrameElement, ...]) -> str:
    """Convert a frame to a human-readable string representation."""
    parts = []
    for elem in frame:
        if isinstance(elem, SlotMarker):
            parts.append(str(elem))
        else:
            parts.append(repr(elem))
    return " ".join(parts)


def compute_frame_ratio(frame: tuple[FrameElement, ...]) -> float:
    """Compute the ratio of fixed tokens to total frame elements.
    
    A higher ratio means more of the pattern is fixed (better for compression).
    """
    if not frame:
        return 0.0
    fixed = sum(1 for elem in frame if not isinstance(elem, SlotMarker))
    return fixed / len(frame)
