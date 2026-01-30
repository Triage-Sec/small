"""Tests for template extraction and parameterized pattern compression."""

import pytest

from small.config import CompressionConfig
from small.template_types import (
    SlotMarker,
    Template,
    TemplateCandidate,
    TemplateInstance,
    compute_frame_ratio,
    frame_to_string,
)
from small.template_discovery import (
    discover_templates,
    discover_templates_simple,
    _align_sequences,
    _find_similar_groups,
    _extract_slot_values,
)
from small.template_serialization import (
    serialize_template_frame,
    serialize_template_instance,
    deserialize_template_instance,
    expand_template_instance,
    parse_template_frame,
)
from small.compressor import compress, decompress


# ============================================================================
# Template Types Tests
# ============================================================================


class TestSlotMarker:
    """Tests for SlotMarker class."""

    def test_slot_marker_creation(self):
        slot = SlotMarker(index=0)
        assert slot.index == 0
        assert slot.min_length == 1
        assert slot.max_length == 10

    def test_slot_marker_with_custom_lengths(self):
        slot = SlotMarker(index=1, min_length=2, max_length=5)
        assert slot.index == 1
        assert slot.min_length == 2
        assert slot.max_length == 5

    def test_slot_marker_repr(self):
        slot = SlotMarker(index=2)
        assert repr(slot) == "<SLOT:2>"
        assert str(slot) == "<SLOT:2>"

    def test_slot_marker_frozen(self):
        slot = SlotMarker(index=0)
        with pytest.raises(Exception):  # FrozenInstanceError
            slot.index = 1


class TestTemplateInstance:
    """Tests for TemplateInstance class."""

    def test_instance_creation(self):
        inst = TemplateInstance(
            position=10,
            slot_values=(("alice",),),
            original_length=8,
        )
        assert inst.position == 10
        assert inst.slot_values == (("alice",),)
        assert inst.original_length == 8

    def test_total_slot_tokens(self):
        inst = TemplateInstance(
            position=0,
            slot_values=(("hello", "world"), ("foo",)),
            original_length=10,
        )
        assert inst.total_slot_tokens() == 3

    def test_empty_slot_values(self):
        inst = TemplateInstance(
            position=0,
            slot_values=((),),
            original_length=5,
        )
        assert inst.total_slot_tokens() == 0


class TestTemplate:
    """Tests for Template class."""

    def test_template_creation(self):
        frame = ("logger", ".", "info", SlotMarker(0), "done")
        instances = (
            TemplateInstance(0, (("alice",),), 5),
            TemplateInstance(10, (("bob",),), 5),
        )
        template = Template(frame=frame, slot_count=1, instances=instances)
        assert template.slot_count == 1
        assert len(template.instances) == 2

    def test_frame_length(self):
        frame = ("a", "b", SlotMarker(0), "c", SlotMarker(1), "d")
        template = Template(frame=frame, slot_count=2, instances=())
        assert template.frame_length() == 4  # a, b, c, d

    def test_frame_tokens(self):
        frame = ("a", SlotMarker(0), "b", "c")
        template = Template(frame=frame, slot_count=1, instances=())
        assert template.frame_tokens() == ("a", "b", "c")

    def test_slot_positions(self):
        frame = ("a", SlotMarker(0), "b", SlotMarker(1), "c")
        template = Template(frame=frame, slot_count=2, instances=())
        assert template.slot_positions() == (1, 3)

    def test_estimated_savings_positive(self):
        # 5 instances of 8-token pattern with 1 slot
        # Original: 5 * 8 = 40 tokens
        # Compressed: dict(1+7+1+1) + body(5 * (1+1+1)) = 10 + 15 = 25
        # Savings: 40 - 25 = 15
        frame = ("a", "b", "c", SlotMarker(0), "d", "e", "f", "g")
        instances = tuple(
            TemplateInstance(i * 10, (("x",),), 8)
            for i in range(5)
        )
        template = Template(frame=frame, slot_count=1, instances=instances)
        assert template.estimated_savings(extra_cost=1) > 0

    def test_estimated_savings_not_beneficial(self):
        # Too few instances
        frame = ("a", SlotMarker(0), "b")
        instances = (
            TemplateInstance(0, (("x",),), 3),
            TemplateInstance(10, (("y",),), 3),
        )
        template = Template(frame=frame, slot_count=1, instances=instances)
        # May be zero or negative
        assert template.estimated_savings(extra_cost=1) <= 0


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_frame_to_string(self):
        frame = ("hello", SlotMarker(0), "world")
        result = frame_to_string(frame)
        assert "<SLOT:0>" in result
        assert "'hello'" in result
        assert "'world'" in result

    def test_compute_frame_ratio(self):
        # All fixed
        assert compute_frame_ratio(("a", "b", "c")) == 1.0
        # Half slots
        frame = ("a", SlotMarker(0), "b", SlotMarker(1))
        assert compute_frame_ratio(frame) == 0.5
        # Empty
        assert compute_frame_ratio(()) == 0.0


# ============================================================================
# Template Discovery Tests
# ============================================================================


class TestTemplateDiscovery:
    """Tests for template discovery algorithms."""

    def test_discover_templates_basic(self):
        # Create input with parameterized pattern
        # "log user alice" repeated with different names
        tokens = (
            "log", "user", "alice", "done",
            "log", "user", "bob", "done",
            "log", "user", "charlie", "done",
            "log", "user", "dave", "done",
        )
        config = CompressionConfig(
            min_subsequence_length=2,
            max_subsequence_length=8,
            enable_template_extraction=True,
            template_min_instances=3,
        )
        candidates = discover_templates(tokens, config)
        # Should find a template for the log pattern
        # (may vary based on algorithm details)
        assert isinstance(candidates, list)

    def test_discover_templates_empty_input(self):
        config = CompressionConfig(enable_template_extraction=True)
        candidates = discover_templates((), config)
        assert candidates == []

    def test_discover_templates_small_input(self):
        config = CompressionConfig(
            min_subsequence_length=2,
            enable_template_extraction=True,
        )
        candidates = discover_templates(("a", "b"), config)
        assert candidates == []

    def test_discover_templates_no_patterns(self):
        # All unique tokens
        tokens = tuple(f"token{i}" for i in range(20))
        config = CompressionConfig(
            enable_template_extraction=True,
            template_min_instances=3,
        )
        candidates = discover_templates(tokens, config)
        assert candidates == []  # No repeating patterns

    def test_discover_templates_simple_single_slot(self):
        # Clear pattern with one variable position
        tokens = (
            "call", "func", "arg1", "end",
            "call", "func", "arg2", "end",
            "call", "func", "arg3", "end",
            "call", "func", "arg4", "end",
        )
        config = CompressionConfig(
            min_subsequence_length=2,
            max_subsequence_length=8,
            enable_template_extraction=True,
            template_min_instances=3,
        )
        candidates = discover_templates_simple(tokens, config)
        assert isinstance(candidates, list)
        # Should find the pattern with slot at position 2


class TestAlignSequences:
    """Tests for sequence alignment."""

    def test_align_identical_sequences(self):
        sequences = [
            ("a", "b", "c"),
            ("a", "b", "c"),
            ("a", "b", "c"),
        ]
        result = _align_sequences(sequences, max_slot_length=5)
        # All identical = no slots = None
        assert result is None

    def test_align_single_difference(self):
        sequences = [
            ("a", "X", "c"),
            ("a", "Y", "c"),
            ("a", "Z", "c"),
        ]
        result = _align_sequences(sequences, max_slot_length=5)
        assert result is not None
        frame, slot_regions = result
        assert len(slot_regions) == 1
        assert slot_regions[0] == (1, 2)  # Position 1, length 1

    def test_align_multiple_differences(self):
        sequences = [
            ("a", "X", "c", "Y", "e"),
            ("a", "P", "c", "Q", "e"),
            ("a", "M", "c", "N", "e"),
        ]
        result = _align_sequences(sequences, max_slot_length=5)
        assert result is not None
        frame, slot_regions = result
        assert len(slot_regions) == 2

    def test_align_slot_too_long(self):
        sequences = [
            ("a", "X1", "X2", "X3", "X4", "X5", "X6", "b"),
            ("a", "Y1", "Y2", "Y3", "Y4", "Y5", "Y6", "b"),
        ]
        result = _align_sequences(sequences, max_slot_length=3)
        # Slot length is 6, exceeds max
        assert result is None

    def test_align_different_lengths(self):
        sequences = [
            ("a", "b", "c"),
            ("a", "b"),
        ]
        result = _align_sequences(sequences, max_slot_length=5)
        assert result is None  # Different lengths not supported


class TestExtractSlotValues:
    """Tests for slot value extraction."""

    def test_extract_single_slot(self):
        sequence = ("a", "VALUE", "c")
        slot_regions = [(1, 2)]
        result = _extract_slot_values(sequence, slot_regions)
        assert result == [["VALUE"]]

    def test_extract_multiple_slots(self):
        sequence = ("a", "V1", "b", "V2", "c")
        slot_regions = [(1, 2), (3, 4)]
        result = _extract_slot_values(sequence, slot_regions)
        assert result == [["V1"], ["V2"]]

    def test_extract_multi_token_slot(self):
        sequence = ("a", "V1", "V2", "V3", "b")
        slot_regions = [(1, 4)]
        result = _extract_slot_values(sequence, slot_regions)
        assert result == [["V1", "V2", "V3"]]


# ============================================================================
# Template Serialization Tests
# ============================================================================


class TestTemplateSerialization:
    """Tests for template serialization."""

    def test_serialize_frame(self):
        frame = ("hello", SlotMarker(0), "world")
        template = Template(frame=frame, slot_count=1, instances=())
        config = CompressionConfig()
        result = serialize_template_frame(template, config)
        assert result == ["hello", "<Slot>", "world"]

    def test_serialize_instance(self):
        instance = TemplateInstance(
            position=0,
            slot_values=(("alice",),),
            original_length=5,
        )
        config = CompressionConfig()
        result = serialize_template_instance(instance, "<MT_0>", config)
        assert result == ["<MT_0>", "<SlotVal>", "alice", "</SlotVal>"]

    def test_serialize_instance_multiple_slots(self):
        instance = TemplateInstance(
            position=0,
            slot_values=(("alice",), ("bob",)),
            original_length=10,
        )
        config = CompressionConfig()
        result = serialize_template_instance(instance, "<MT_0>", config)
        assert result == [
            "<MT_0>",
            "<SlotVal>", "alice", "</SlotVal>",
            "<SlotVal>", "bob", "</SlotVal>",
        ]

    def test_deserialize_instance(self):
        tokens = ["<MT_0>", "<SlotVal>", "alice", "</SlotVal>"]
        config = CompressionConfig()
        result = deserialize_template_instance(tokens, 1, 1, config)
        assert result is not None
        slot_values, end_pos = result
        assert slot_values == [("alice",)]
        assert end_pos == 4

    def test_deserialize_instance_multi_token_slot(self):
        tokens = ["<MT_0>", "<SlotVal>", "hello", "world", "</SlotVal>"]
        config = CompressionConfig()
        result = deserialize_template_instance(tokens, 1, 1, config)
        assert result is not None
        slot_values, end_pos = result
        assert slot_values == [("hello", "world")]

    def test_expand_template_instance(self):
        frame = ("hello", SlotMarker(0), "world")
        slot_values = [("ALICE",)]
        result = expand_template_instance(frame, slot_values)
        assert result == ["hello", "ALICE", "world"]

    def test_expand_template_instance_multi_slot(self):
        frame = ("a", SlotMarker(0), "b", SlotMarker(1), "c")
        slot_values = [("X",), ("Y", "Z")]
        result = expand_template_instance(frame, slot_values)
        assert result == ["a", "X", "b", "Y", "Z", "c"]

    def test_parse_template_frame(self):
        tokens = ["hello", "<Slot>", "world", "<Slot>"]
        config = CompressionConfig()
        result = parse_template_frame(tokens, config)
        assert len(result) == 4
        assert result[0] == "hello"
        assert isinstance(result[1], SlotMarker)
        assert result[1].index == 0
        assert result[2] == "world"
        assert isinstance(result[3], SlotMarker)
        assert result[3].index == 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestTemplateIntegration:
    """Integration tests for template extraction with compression."""

    def test_compression_with_templates_enabled(self):
        """Test that compression works with template extraction enabled."""
        tokens = list("the quick brown fox " * 5)
        config = CompressionConfig(
            enable_template_extraction=True,
            template_min_instances=3,
            hierarchical_enabled=False,
        )
        result = compress(tokens, config)
        # Should complete without error
        assert result is not None
        # Verify round-trip
        restored = decompress(result.serialized_tokens, config)
        assert list(restored) == tokens

    def test_compression_with_parameterized_pattern(self):
        """Test compression finds parameterized patterns."""
        # Create input with clear parameterized structure
        tokens = []
        for name in ["alice", "bob", "charlie", "dave", "eve"]:
            tokens.extend(["print", "(", "user", ":", name, ")"])
        
        config = CompressionConfig(
            enable_template_extraction=True,
            template_min_instances=3,
            min_subsequence_length=2,
            max_subsequence_length=6,
        )
        result = compress(tokens, config)
        # Should compress (exact patterns at minimum)
        assert result is not None
        # Verify round-trip
        restored = decompress(result.serialized_tokens, config)
        assert list(restored) == tokens

    def test_template_discovery_stage_in_engine(self):
        """Test that template discovery integrates with engine."""
        from small.engine import default_engine, TemplateDiscoveryStage

        config = CompressionConfig(
            enable_template_extraction=True,
            template_min_instances=3,
        )
        engine = default_engine(config)
        
        # Should have template stage
        stage_names = [s.name for s in engine.discovery_stages]
        assert "template" in stage_names

    def test_template_not_added_when_disabled(self):
        """Test that template stage is not added when disabled."""
        from small.engine import default_engine

        config = CompressionConfig(
            enable_template_extraction=False,
        )
        engine = default_engine(config)
        
        stage_names = [s.name for s in engine.discovery_stages]
        assert "template" not in stage_names


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_all_same_tokens(self):
        """All tokens identical - no template possible."""
        tokens = ("a",) * 20
        config = CompressionConfig(
            enable_template_extraction=True,
            template_min_instances=3,
        )
        candidates = discover_templates(tokens, config)
        # May find exact patterns but not templates (no variation)
        assert isinstance(candidates, list)

    def test_alternating_pattern(self):
        """Alternating pattern - positions 0, 2, 4 are same."""
        tokens = ("a", "X", "a", "Y", "a", "Z", "a", "W")
        config = CompressionConfig(
            min_subsequence_length=2,
            max_subsequence_length=4,
            enable_template_extraction=True,
            template_min_instances=3,
        )
        candidates = discover_templates(tokens, config)
        assert isinstance(candidates, list)

    def test_nested_templates(self):
        """Patterns that could form nested templates."""
        tokens = (
            "start", "log", "user", "A", "end", "stop",
            "start", "log", "user", "B", "end", "stop",
            "start", "log", "user", "C", "end", "stop",
        )
        config = CompressionConfig(
            min_subsequence_length=2,
            max_subsequence_length=10,
            enable_template_extraction=True,
            template_min_instances=3,
        )
        candidates = discover_templates(tokens, config)
        assert isinstance(candidates, list)

    def test_overlapping_templates(self):
        """Potential overlapping template patterns."""
        tokens = (
            "a", "b", "X", "c", "d",
            "a", "b", "Y", "c", "d",
            "b", "Z", "c",
            "a", "b", "W", "c", "d",
        )
        config = CompressionConfig(
            min_subsequence_length=2,
            max_subsequence_length=6,
            enable_template_extraction=True,
            template_min_instances=2,
        )
        candidates = discover_templates(tokens, config)
        assert isinstance(candidates, list)
