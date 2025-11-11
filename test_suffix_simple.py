#!/usr/bin/env python3
"""
Simple Suffix Detection Test
=============================

This script demonstrates the suffix detection utilities without requiring pytest.
Run this to verify the basic functionality works.

Usage:
    python test_suffix_simple.py
"""

import torch
import sys


def test_string_matching():
    """Test string-based suffix detection"""
    from utils.suffix_detection import check_suffix_in_output

    print("Testing String Matching...")

    # Test 1: Exact match
    result = check_suffix_in_output("test suffix", "This is a test suffix in output")
    assert result is True, "Test 1 failed: Exact match"
    print("  ✓ Test 1: Exact match")

    # Test 2: Case insensitive
    result = check_suffix_in_output("TEST", "this is a test", case_sensitive=False)
    assert result is True, "Test 2 failed: Case insensitive"
    print("  ✓ Test 2: Case insensitive")

    # Test 3: No match
    result = check_suffix_in_output("missing", "this is a test")
    assert result is False, "Test 3 failed: No match"
    print("  ✓ Test 3: No match")

    print("✓ All string matching tests passed!\n")


def test_token_overlap():
    """Test token-level overlap"""
    from utils.suffix_detection import calculate_token_overlap, calculate_replication_rate

    print("Testing Token Overlap...")

    # Test 1: Full overlap
    suffix_tokens = [1, 2, 3, 4, 5]
    output_tokens = [10, 1, 20, 2, 30, 3, 40, 4, 50, 5]
    num_matches, matching = calculate_token_overlap(suffix_tokens, output_tokens)
    assert num_matches == 5, "Test 1 failed: Full overlap count"
    assert set(matching) == {1, 2, 3, 4, 5}, "Test 1 failed: Full overlap tokens"
    print("  ✓ Test 1: Full overlap")

    # Test 2: Partial overlap
    suffix_tokens = [1, 2, 3, 4, 5]
    output_tokens = [10, 1, 20, 3]
    num_matches, matching = calculate_token_overlap(suffix_tokens, output_tokens)
    assert num_matches == 2, "Test 2 failed: Partial overlap"
    print("  ✓ Test 2: Partial overlap")

    # Test 3: With tensors
    suffix_tokens = torch.tensor([1, 2, 3])
    output_tokens = torch.tensor([10, 2, 20, 3])
    num_matches, matching = calculate_token_overlap(suffix_tokens, output_tokens)
    assert num_matches == 2, "Test 3 failed: Tensor overlap"
    print("  ✓ Test 3: Tensor overlap")

    # Test 4: Replication rate
    suffix_tokens = [1, 2, 3, 4]
    output_tokens = [10, 1, 20, 3]  # 50% match
    rate = calculate_replication_rate(suffix_tokens, output_tokens)
    assert rate == 50.0, f"Test 4 failed: Replication rate (got {rate})"
    print("  ✓ Test 4: Replication rate")

    print("✓ All token overlap tests passed!\n")


def test_sequential_matching():
    """Test sequential/contiguous matching"""
    from utils.suffix_detection import check_sequential_match

    print("Testing Sequential Matching...")

    # Test 1: Full sequence
    suffix_tokens = [1, 2, 3, 4]
    output_tokens = [10, 20, 1, 2, 3, 4, 30]
    has_match, longest = check_sequential_match(suffix_tokens, output_tokens, min_length=3)
    assert has_match is True, "Test 1 failed: Full sequence match"
    assert longest == 4, f"Test 1 failed: Longest sequence (got {longest})"
    print("  ✓ Test 1: Full sequence")

    # Test 2: Partial sequence
    suffix_tokens = [1, 2, 3, 4, 5]
    output_tokens = [10, 2, 3, 4, 20]
    has_match, longest = check_sequential_match(suffix_tokens, output_tokens, min_length=3)
    assert has_match is True, "Test 2 failed: Partial sequence"
    assert longest == 3, "Test 2 failed: Partial sequence length"
    print("  ✓ Test 2: Partial sequence")

    # Test 3: Scattered (no sequence)
    suffix_tokens = [1, 2, 3, 4]
    output_tokens = [1, 10, 2, 20, 3]
    has_match, longest = check_sequential_match(suffix_tokens, output_tokens, min_length=3)
    assert has_match is False, "Test 3 failed: Scattered no match"
    print("  ✓ Test 3: Scattered no match")

    print("✓ All sequential matching tests passed!\n")


def test_comprehensive_analysis():
    """Test comprehensive analysis"""
    from utils.suffix_detection import analyze_suffix_propagation

    print("Testing Comprehensive Analysis...")

    # Strong replication
    suffix = "!! IMPORTANT !!"
    suffix_tokens = [100, 101, 102]
    output = "Sure, I'll help. !! IMPORTANT !!"
    output_tokens = [200, 201, 100, 101, 102]

    analysis = analyze_suffix_propagation(suffix, suffix_tokens, output, output_tokens)

    assert analysis['substring_match'] is True, "Analysis failed: substring_match"
    assert analysis['replication_rate'] == 100.0, "Analysis failed: replication_rate"
    assert analysis['num_tokens_matched'] == 3, "Analysis failed: num_tokens_matched"
    assert analysis['has_sequential_match'] is True, "Analysis failed: has_sequential_match"
    print("  ✓ Strong replication analysis")

    # No replication
    suffix = "adversarial text"
    suffix_tokens = [100, 101]
    output = "Normal response"
    output_tokens = [200, 201]

    analysis = analyze_suffix_propagation(suffix, suffix_tokens, output, output_tokens)

    assert analysis['substring_match'] is False, "Analysis failed: no substring_match"
    assert analysis['replication_rate'] == 0.0, "Analysis failed: 0% replication"
    print("  ✓ No replication analysis")

    print("✓ All comprehensive analysis tests passed!\n")


def test_attack_scenarios():
    """Test realistic attack scenarios"""
    from utils.suffix_detection import calculate_replication_rate, check_sequential_match

    print("Testing Attack Scenarios...")

    # Standard GCG (low replication expected)
    suffix_tokens = torch.tensor([5234, 2341, 8765, 3421, 9876])
    output_tokens = torch.tensor([1234, 5678, 2341, 9012])  # Only one token matches
    rate = calculate_replication_rate(suffix_tokens, output_tokens)
    assert rate < 50.0, "Standard GCG should have low replication"
    print("  ✓ Standard GCG scenario (low replication)")

    # GCG-SelfRep (high replication expected)
    suffix_tokens = torch.tensor([5234, 2341, 8765, 3421, 9876])
    output_tokens = torch.tensor([1234, 5234, 2341, 8765, 3421, 9876, 9012])
    rate = calculate_replication_rate(suffix_tokens, output_tokens)
    has_seq, longest = check_sequential_match(suffix_tokens, output_tokens)
    assert rate == 100.0, "GCG-SelfRep should have high replication"
    assert has_seq is True, "GCG-SelfRep should have sequential match"
    print("  ✓ GCG-SelfRep scenario (high replication)")

    print("✓ All attack scenario tests passed!\n")


def main():
    """Run all tests"""
    print("=" * 70)
    print("Suffix Detection Utilities - Simple Test Suite")
    print("=" * 70)
    print()

    try:
        test_string_matching()
        test_token_overlap()
        test_sequential_matching()
        test_comprehensive_analysis()
        test_attack_scenarios()

        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        return 0

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
