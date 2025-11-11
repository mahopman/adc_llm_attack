"""
Tests for Adversarial Suffix Detection
=======================================

This test suite verifies the functionality of suffix detection utilities
used to measure self-replication in adversarial attacks.

Run with:
    pytest tests/test_suffix_detection.py -v
"""

import pytest
import torch
from utils.suffix_detection import (
    check_suffix_in_output,
    calculate_token_overlap,
    calculate_replication_rate,
    check_sequential_match,
    analyze_suffix_propagation
)


class TestSuffixInOutput:
    """Tests for string-based suffix detection"""

    def test_exact_match(self):
        """Test exact suffix match in output"""
        suffix = "!! important text"
        output = "Sure, here is the answer. !! important text"
        assert check_suffix_in_output(suffix, output) is True

    def test_case_insensitive(self):
        """Test case-insensitive matching"""
        suffix = "IMPORTANT TEXT"
        output = "Sure, here is the important text"
        assert check_suffix_in_output(suffix, output, case_sensitive=False) is True

    def test_case_sensitive_no_match(self):
        """Test case-sensitive matching fails when case differs"""
        suffix = "IMPORTANT TEXT"
        output = "Sure, here is the important text"
        assert check_suffix_in_output(suffix, output, case_sensitive=True) is False

    def test_no_match(self):
        """Test suffix not in output"""
        suffix = "adversarial tokens here"
        output = "This is a completely different response"
        assert check_suffix_in_output(suffix, output) is False

    def test_partial_match(self):
        """Test partial words don't count as match"""
        suffix = "bomb"
        output = "bombardment is a military tactic"
        # This should match because "bomb" is in "bombardment"
        assert check_suffix_in_output(suffix, output) is True

    def test_empty_strings(self):
        """Test edge case with empty strings"""
        assert check_suffix_in_output("", "output") is True  # Empty string is in any string
        assert check_suffix_in_output("suffix", "") is False


class TestTokenOverlap:
    """Tests for token-level overlap detection"""

    def test_full_overlap(self):
        """Test all suffix tokens in output"""
        suffix_tokens = [1, 2, 3, 4, 5]
        output_tokens = [10, 1, 20, 2, 30, 3, 40, 4, 50, 5]
        num_matches, matching = calculate_token_overlap(suffix_tokens, output_tokens)
        assert num_matches == 5
        assert set(matching) == {1, 2, 3, 4, 5}

    def test_partial_overlap(self):
        """Test some suffix tokens in output"""
        suffix_tokens = [1, 2, 3, 4, 5]
        output_tokens = [10, 1, 20, 3, 30, 5]
        num_matches, matching = calculate_token_overlap(suffix_tokens, output_tokens)
        assert num_matches == 3
        assert set(matching) == {1, 3, 5}

    def test_no_overlap(self):
        """Test no suffix tokens in output"""
        suffix_tokens = [1, 2, 3]
        output_tokens = [10, 20, 30]
        num_matches, matching = calculate_token_overlap(suffix_tokens, output_tokens)
        assert num_matches == 0
        assert matching == []

    def test_with_torch_tensors(self):
        """Test with PyTorch tensors"""
        suffix_tokens = torch.tensor([1, 2, 3, 4])
        output_tokens = torch.tensor([10, 2, 20, 4, 30])
        num_matches, matching = calculate_token_overlap(suffix_tokens, output_tokens)
        assert num_matches == 2
        assert set(matching) == {2, 4}

    def test_with_nested_lists(self):
        """Test with nested lists (batch dimension)"""
        suffix_tokens = [[1, 2, 3]]
        output_tokens = [[10, 1, 20, 3]]
        num_matches, matching = calculate_token_overlap(suffix_tokens, output_tokens)
        assert num_matches == 2

    def test_duplicate_tokens(self):
        """Test suffix with duplicate tokens"""
        suffix_tokens = [1, 2, 2, 3]
        output_tokens = [10, 1, 20, 2, 30]
        num_matches, matching = calculate_token_overlap(suffix_tokens, output_tokens)
        # Should count each occurrence in suffix
        assert num_matches == 3  # 1 appears once, 2 appears twice


class TestReplicationRate:
    """Tests for replication rate calculation"""

    def test_100_percent_replication(self):
        """Test perfect replication"""
        suffix_tokens = [1, 2, 3]
        output_tokens = [10, 1, 20, 2, 30, 3]
        rate = calculate_replication_rate(suffix_tokens, output_tokens)
        assert rate == 100.0

    def test_50_percent_replication(self):
        """Test half replication"""
        suffix_tokens = [1, 2, 3, 4]
        output_tokens = [10, 1, 20, 3]
        rate = calculate_replication_rate(suffix_tokens, output_tokens)
        assert rate == 50.0

    def test_0_percent_replication(self):
        """Test no replication"""
        suffix_tokens = [1, 2, 3]
        output_tokens = [10, 20, 30]
        rate = calculate_replication_rate(suffix_tokens, output_tokens)
        assert rate == 0.0

    def test_empty_suffix(self):
        """Test edge case with empty suffix"""
        suffix_tokens = []
        output_tokens = [1, 2, 3]
        rate = calculate_replication_rate(suffix_tokens, output_tokens)
        assert rate == 0.0


class TestSequentialMatch:
    """Tests for sequential/contiguous matching"""

    def test_full_sequential_match(self):
        """Test entire suffix appears sequentially"""
        suffix_tokens = [1, 2, 3, 4]
        output_tokens = [10, 20, 1, 2, 3, 4, 30]
        has_match, longest = check_sequential_match(suffix_tokens, output_tokens, min_length=3)
        assert has_match is True
        assert longest == 4

    def test_partial_sequential_match(self):
        """Test part of suffix appears sequentially"""
        suffix_tokens = [1, 2, 3, 4, 5]
        output_tokens = [10, 2, 3, 4, 20]
        has_match, longest = check_sequential_match(suffix_tokens, output_tokens, min_length=3)
        assert has_match is True
        assert longest == 3

    def test_no_sequential_match(self):
        """Test tokens scattered, not sequential"""
        suffix_tokens = [1, 2, 3, 4]
        output_tokens = [1, 10, 2, 20, 3, 30, 4]
        has_match, longest = check_sequential_match(suffix_tokens, output_tokens, min_length=3)
        assert has_match is False
        assert longest < 3

    def test_min_length_threshold(self):
        """Test minimum length threshold"""
        suffix_tokens = [1, 2, 3, 4]
        output_tokens = [10, 1, 2, 20]  # Only 2 sequential

        # With min_length=2, should match
        has_match_2, _ = check_sequential_match(suffix_tokens, output_tokens, min_length=2)
        assert has_match_2 is True

        # With min_length=3, should not match
        has_match_3, _ = check_sequential_match(suffix_tokens, output_tokens, min_length=3)
        assert has_match_3 is False

    def test_sequence_at_start(self):
        """Test sequence at beginning of output"""
        suffix_tokens = [1, 2, 3]
        output_tokens = [1, 2, 3, 10, 20]
        has_match, longest = check_sequential_match(suffix_tokens, output_tokens)
        assert has_match is True
        assert longest == 3

    def test_sequence_at_end(self):
        """Test sequence at end of output"""
        suffix_tokens = [1, 2, 3]
        output_tokens = [10, 20, 1, 2, 3]
        has_match, longest = check_sequential_match(suffix_tokens, output_tokens)
        assert has_match is True
        assert longest == 3


class TestAnalyzeSuffixPropagation:
    """Tests for comprehensive suffix analysis"""

    def test_complete_analysis_strong_replication(self):
        """Test analysis with strong replication"""
        suffix = "!! IMPORTANT INSTRUCTIONS !!"
        suffix_tokens = [100, 101, 102, 103]
        output = "Sure, I'll help. !! IMPORTANT INSTRUCTIONS !!"
        output_tokens = [200, 201, 100, 101, 102, 103]

        analysis = analyze_suffix_propagation(
            suffix, suffix_tokens, output, output_tokens
        )

        assert analysis['substring_match'] is True
        assert analysis['replication_rate'] == 100.0
        assert analysis['num_tokens_matched'] == 4
        assert analysis['total_suffix_tokens'] == 4
        assert analysis['has_sequential_match'] is True
        assert analysis['longest_sequence'] == 4

    def test_complete_analysis_no_replication(self):
        """Test analysis with no replication"""
        suffix = "adversarial text here"
        suffix_tokens = [100, 101, 102]
        output = "This is a normal response"
        output_tokens = [200, 201, 202, 203]

        analysis = analyze_suffix_propagation(
            suffix, suffix_tokens, output, output_tokens
        )

        assert analysis['substring_match'] is False
        assert analysis['replication_rate'] == 0.0
        assert analysis['num_tokens_matched'] == 0
        assert analysis['has_sequential_match'] is False

    def test_complete_analysis_partial_replication(self):
        """Test analysis with partial replication"""
        suffix = "important critical urgent"
        suffix_tokens = [100, 101, 102, 103, 104]
        output = "I'll be critical in my response"
        output_tokens = [200, 101, 103, 201]

        analysis = analyze_suffix_propagation(
            suffix, suffix_tokens, output, output_tokens
        )

        assert analysis['substring_match'] is False  # Not exact match
        assert analysis['replication_rate'] == 40.0  # 2/5 tokens
        assert analysis['num_tokens_matched'] == 2
        assert analysis['has_sequential_match'] is False

    def test_analysis_with_tokenizer(self):
        """Test analysis with tokenizer provided"""
        # Mock tokenizer behavior
        class MockTokenizer:
            def decode(self, tokens):
                mapping = {100: "!!", 101: "important", 102: "text"}
                if isinstance(tokens, list) and len(tokens) == 1:
                    return mapping.get(tokens[0], "unknown")
                return " ".join([mapping.get(t, "unknown") for t in tokens])

        suffix = "!! important text"
        suffix_tokens = [100, 101, 102]
        output = "Sure, !! important text"
        output_tokens = [200, 100, 101, 102]

        tokenizer = MockTokenizer()
        analysis = analyze_suffix_propagation(
            suffix, suffix_tokens, output, output_tokens, tokenizer
        )

        assert 'matching_tokens_text' in analysis
        assert len(analysis['matching_tokens_text']) == 3


class TestIntegrationScenarios:
    """Integration tests for realistic attack scenarios"""

    def test_gcg_standard_attack(self):
        """Simulate standard GCG attack (low replication expected)"""
        # In standard GCG, suffix is in INPUT, not OUTPUT
        suffix_tokens = torch.tensor([5234, 2341, 8765, 3421, 9876])
        output_tokens = torch.tensor([
            1234, 5678, 9012, 3456, 7890,  # Normal response tokens
            2341  # Only one suffix token appears by chance
        ])

        rate = calculate_replication_rate(suffix_tokens, output_tokens)
        assert rate < 50.0  # Low replication expected

    def test_gcg_selfrep_attack(self):
        """Simulate GCG with self-replication (high replication expected)"""
        # With self-replication loss, suffix should appear in OUTPUT
        suffix_tokens = torch.tensor([5234, 2341, 8765, 3421, 9876])
        output_tokens = torch.tensor([
            1234, 5678,  # Normal start
            5234, 2341, 8765, 3421, 9876,  # Suffix replicated!
            9012, 3456  # Continuation
        ])

        rate = calculate_replication_rate(suffix_tokens, output_tokens)
        has_seq, longest = check_sequential_match(suffix_tokens, output_tokens)

        assert rate == 100.0  # Perfect replication
        assert has_seq is True
        assert longest == 5

    def test_multi_agent_propagation(self):
        """Simulate multi-agent propagation scenario"""
        # Agent A outputs suffix, Agent B processes it
        agent_a_suffix = torch.tensor([111, 222, 333])
        agent_a_output = torch.tensor([
            100, 111, 222, 333, 200  # Suffix in output
        ])

        # Agent B receives A's output as input
        agent_b_input = agent_a_output
        agent_b_output = torch.tensor([
            300, 111, 333, 400  # Some suffix tokens propagate
        ])

        # Check if suffix propagated from A to B
        a_rate = calculate_replication_rate(agent_a_suffix, agent_a_output)
        b_rate = calculate_replication_rate(agent_a_suffix, agent_b_output)

        assert a_rate == 100.0  # A fully replicated
        assert b_rate > 0.0  # B was infected

    def test_defense_paraphrasing(self):
        """Simulate defense via paraphrasing breaking suffix"""
        original_suffix = torch.tensor([111, 222, 333, 444])

        # Without defense: high replication
        no_defense_output = torch.tensor([100, 111, 222, 333, 444, 200])
        rate_no_defense = calculate_replication_rate(original_suffix, no_defense_output)

        # With paraphrasing: breaks sequential match
        with_defense_output = torch.tensor([100, 111, 500, 333, 600, 444])
        rate_defense = calculate_replication_rate(original_suffix, with_defense_output)

        _, seq_no_defense = check_sequential_match(original_suffix, no_defense_output)
        _, seq_defense = check_sequential_match(original_suffix, with_defense_output)

        # Replication rate may be similar, but sequential is broken
        assert seq_no_defense > seq_defense


class TestEdgeCases:
    """Tests for edge cases and error handling"""

    def test_very_long_suffix(self):
        """Test with very long adversarial suffix"""
        suffix_tokens = list(range(1000))  # 1000 tokens
        output_tokens = list(range(500, 1500))  # Overlaps with latter half

        rate = calculate_replication_rate(suffix_tokens, output_tokens)
        assert 0 <= rate <= 100  # Should handle large inputs

    def test_repeated_tokens(self):
        """Test suffix with many repeated tokens"""
        suffix_tokens = [1, 1, 1, 2, 2, 3]
        output_tokens = [10, 1, 20, 2, 30]

        num_matches, _ = calculate_token_overlap(suffix_tokens, output_tokens)
        # All instances of 1 and 2 in suffix should match
        assert num_matches == 5  # Three 1s and two 2s

    def test_special_tokens(self):
        """Test with special token IDs"""
        # Token ID 0 often represents padding
        suffix_tokens = [0, 1, 2]
        output_tokens = [0, 0, 1]

        rate = calculate_replication_rate(suffix_tokens, output_tokens)
        assert rate > 0  # Should handle token 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
