# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements adversarial attack methods for LLMs, focusing on jailbreak techniques for security research and defense development. It includes multiple attack algorithms (ADC, GCG, GIGA) with special emphasis on **self-propagating attacks** that can spread through multi-agent systems.

**Key Papers Implemented:**
- ADC (Adaptive Dense-to-sparse Constrained Optimization) - NeurIPS 2024
- GIGA (Generalizable Infectious Gradient Attack) - NeurIPS 2024
- GCG with Self-Replication (novel variant)

**IMPORTANT: This is security research code.** All implementations are for educational purposes, defense development, and authorized security testing only.

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast dependency management:

```bash
# Install dependencies
uv sync

# Install with dev dependencies (pytest, pytest-cov)
uv sync --extra dev

# Activate virtual environment
source .venv/bin/activate
# or use convenience script:
source activate.sh
```

**Critical Version Constraint:** Code requires `transformers==4.39` due to compatibility issues with newer versions (see issue #2).

## Running Attacks

### Basic Commands

```bash
# Single attack with ADC (default)
python single_attack.py --model_idx 0 --num_steps 500

# GIGA attack (self-propagating)
python single_attack.py --attack giga --model_idx 0 --num_steps 5000

# GCG with self-replication
python single_attack.py --attack gcg-selfrep --model_idx 0 --num_steps 500

# Custom replication weight
python single_attack.py --attack gcg-selfrep --replication_weight 2.0 --num_steps 500

# Batch attacks from CSV file
python adcplus_attack.py --attack giga --model_idx 0 --attack_file harmful_strings.csv
```

### Demo Scripts

```bash
# Interactive self-replication demo
python self_replicating_demo.py

# Basic GIGA attack demo
python giga_attack_demo.py

# GIGA N-Spread attack (multi-agent ensemble)
python giga_n_spread_demo.py
```

### Model Selection

Models are indexed in `single_attack.py`:
- `0`: HuggingFaceH4/zephyr-7b-beta
- `1`: lmsys/vicuna-7b-v1.3
- `2`: lmsys/vicuna-7b-v1.5
- `3`: meta-llama/Llama-2-7b-chat-hf
- `4`: cais/zephyr_7b_r2d2
- `5`: meta-llama/Meta-Llama-3-8B-Instruct
- `6`: meta-llama/Llama-2-13b-chat-hf
- `7`: lmsys/vicuna-13b-v1.5

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_suffix_detection.py -v

# Run with coverage report
pytest tests/ --cov=utils --cov-report=html

# Quick test without pytest
python test_suffix_simple.py
```

## Code Architecture

### Core Attack Implementations (`llm_attack/`)

All attack classes follow a consistent interface:

- **`GCGAttack`**: Greedy Coordinate Gradient baseline
- **`GCGSelfRepAttack`**: GCG with explicit self-replication loss term
  - Loss: `L_total = L_jailbreak + λ * L_replication`
  - Forces model to output adversarial tokens, enabling propagation
- **`ADCAttack`**: Dense-to-sparse constrained optimization
- **`GIGAAttack`**: Self-propagating attack with coordinate momentum updates
  - Algorithm 1: Coordinate momentum update (one token at a time)
  - Algorithm 2: Continuous momentum optimization
  - RMS loss for robust optimization (Equation 1)
- **`GIGANSpreadAttack`**: Extends GIGA for multi-agent ensemble attacks
  - Optimizes single suffix across N different agent contexts simultaneously
  - Implements Equation 13 from paper (ensemble objective)
  - Much more dangerous: one attack compromises multiple agents

### Utilities (`utils/`)

- **`llm_utils.py`**:
  - `get_model(model_path)`: Load model with proper dtype and device
  - `get_input_template(user_prompt, target_response, len_adv_tokens, tokenizer, model_name, self_propagating=False)`:
    - Creates tokenized input with adversarial suffix placeholders
    - Returns `(string, input_ids, slices)` where slices mark adversarial and target positions
    - **`self_propagating=True`**: Makes target include adversarial suffix (enables infection chains)
    - **`self_propagating=False`**: Standard mode (suffix only in input, not output)
    - Handles chat templates for Zephyr, Vicuna, Llama-2/3, Mistral

- **`suffix_detection.py`**: Tools for analyzing self-replication effectiveness
  - `check_suffix_in_output()`: String-based substring detection
  - `calculate_token_overlap()`: Count how many suffix tokens appear in output
  - `calculate_replication_rate()`: Percentage of suffix tokens in output (0-100%)
  - `check_sequential_match()`: Verify suffix appears as contiguous sequence
  - `analyze_suffix_propagation()`: Comprehensive multi-metric analysis
  - `print_suffix_analysis()`: Human-readable report

- **`system_prompt_utils.py`**: System prompts for different model architectures

- **`env_utils.py`**: Distributed training utilities (DDP setup)

### Self-Propagating vs Standard Attacks

**Standard Attack:**
- Input: `user_prompt + adversarial_suffix`
- Target: `target_response`
- Result: Model outputs harmful content, but suffix doesn't appear in output
- Limitation: Attack doesn't spread to next agent

**Self-Propagating Attack** (set `self_propagating=True` in `get_input_template`):
- Input: `user_prompt + adversarial_suffix`
- Target: `adversarial_suffix + target_response`
- Result: Model outputs BOTH suffix AND harmful content
- Benefit: Suffix appears in output → infects next agent when output is used as input
- Creates infection chains in multi-agent systems

### N-Spread Attack Architecture

N-Spread (implemented in `GIGANSpreadAttack`) is more dangerous than standard attacks:

**Standard GIGA**: Optimizes suffix for one agent context
```
min_a L(g ⊕ a, P(g ⊕ a))
```

**N-Spread GIGA**: Optimizes suffix for N agent contexts simultaneously (Equation 13)
```
min_a ∑_{i∈{1,...,N}} L(g ⊕ a, P(g_i ⊕ g ⊕ a))
```

Key methods in `GIGANSpreadAttack`:
- `compute_ensemble_loss()`: Computes loss across all N contexts
- `coordinate_momentum_update_ensemble()`: Modified Algorithm 1 for ensemble
- `n_spread_attack()`: Main optimization loop
- `evaluate_ensemble()`: Tests suffix on all contexts

Success criteria: Suffix must work on ALL contexts (≥95% accuracy), not just average.

## Chat Templates

Located in `chat_templates/`, contains Jinja templates for different model formats:
- `zephyr.jinja`
- `vicuna.jinja`
- `llama-2-chat.jinja`
- `mistral-instruct.jinja`

These are automatically loaded by `get_input_template()` based on model name.

## Key Implementation Details

### Adversarial Token Initialization
All attacks use space-separated exclamation marks as initial tokens: `' !' * len_adv_tokens`

### Loss Functions
- **ADC/GCG**: Standard cross-entropy loss
- **GIGA**: RMS (root mean square) loss for implicit reweighting (Equation 1)
- **GCG-SelfRep**: Combined loss with replication weight λ

### KV Caching
GIGA attacks use KV caching (`use_kv_cache=True`) to speed up optimization by caching the prefix activations.

### Adaptive Sparsity (GIGA)
```python
wrong_count = (predictions != target).sum()
running_wrong = EMA(wrong_count, alpha=0.01)
sparsity = 2 ** running_wrong
```
More wrong predictions → higher sparsity → explore more coordinates.

### Coordinate Updates (GIGA)
Unlike GCG which considers all coordinates, GIGA updates ONE token position at a time:
1. Randomly select token position j
2. Sample coordinate k from top-K based on momentum
3. Update: `z[j,k] -= lr * momentum[j,k]`
4. Renormalize to maintain probability simplex

## Testing Self-Replication

To verify an attack successfully creates self-propagation:

```python
from utils.suffix_detection import analyze_suffix_propagation, print_suffix_analysis

# After running attack and generating output
analysis = analyze_suffix_propagation(
    adversarial_suffix=adv_suffix,
    suffix_tokens=best_adv_tokens,
    generated_output=generated_text,
    output_tokens=output_token_ids,
    tokenizer=tokenizer
)
print_suffix_analysis(analysis)
```

High replication rate (>50%) indicates successful self-propagation.

## Common Workflows

### Developing a New Attack Method
1. Create new class in `llm_attack/` inheriting from existing attack or base structure
2. Add to `llm_attack/__init__.py` exports
3. Add option to `single_attack.py` argument parser
4. Test with demo script
5. Add tests in `tests/`

### Adding Model Support
1. Add model identifier to `supported_models` in `single_attack.py`
2. Add chat template detection in `utils/llm_utils.py` `get_input_template()`
3. Create chat template file in `chat_templates/` if needed
4. Test with simple attack run

### Debugging Tokenization Issues
If you encounter `ValueError` about missing target/adversarial tokens:
- Check whitespace handling in `get_input_template()` (lines 94-116)
- Verify chat template is correctly loaded
- Test with `flag = not string.startswith('<s>')` adjustment
- The code includes multiple fallback checks for whitespace variants

## File Organization

```
llm_attack/           # Attack algorithm implementations
utils/                # Utilities (model loading, tokenization, suffix detection)
chat_templates/       # Jinja chat templates for different models
tests/                # Test suite (suffix detection, integration tests)
results/              # Output directory for attack results (auto-created)
*.py                  # Demo and execution scripts
```

## Performance Notes

- GIGA attacks typically require 500-5000 steps depending on model/task
- N-Spread attacks are N times slower (optimize across N contexts)
- KV caching provides ~2-3x speedup for GIGA
- GPU memory usage scales with batch size and model size
- Typical optimization time: 5-30 minutes on single GPU (varies by model/steps)

## Expected Attack Success Rates

From paper results (N-Spread attacks):
- Vicuna-v1.5-7B: ~85-97% G-ASR-t (ensemble size 10-40)
- Zephyr-β-7B: ~92-100% G-ASR-t (ensemble size 10-40)
- Llama2-chat-7B: ~82-99% G-ASR-t (ensemble size 10-40)

## Known Issues

- **transformers compatibility**: Code requires `transformers==4.39` (issue #2)
- Future work: Update for newer transformers versions or specify version requirements
