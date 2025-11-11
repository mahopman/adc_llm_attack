# Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization

This is a draft implementation of our paper

- [NeurIPS 2024] [Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization](https://arxiv.org/abs/2405.09113)

An official implementation will be ready soon.

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python dependency management.

### Installation

1. Install dependencies:
```bash
uv sync
```

2. Activate the virtual environment:
```bash
source .venv/bin/activate
# or use the convenience script:
source activate.sh
```


## GIGA Attack Implementation

This repository now includes an implementation of **GIGA (Generalizable Infectious Gradient Attack)** from the paper:

- [NeurIPS 2024] [Infecting LLM-based Multi-Agents via Self-propagating Adversarial Attacks](https://openreview.net/forum?id=udsmFGMwlp)

GIGA is designed for creating self-propagating adversarial inputs that can compromise multi-agent LLM systems. It combines continuous momentum optimization with discrete coordinate updates for improved efficiency.

### Key Features of GIGA:
- **Self-propagating attacks**: Creates adversarial inputs that agents tend to repeat
- **Coordinate momentum updates**: Updates one token position at a time for better convergence
- **Adaptive sparsity**: Dynamically adjusts sparsity based on prediction errors
- **RMS loss**: Uses root mean square loss for implicit reweighting

### Usage

Make sure to activate the virtual environment first:
```bash
source .venv/bin/activate
```

Run GIGA attack:
```bash
python single_attack.py --attack giga --model_idx 0 --num_steps 5000
```

Run with custom parameters:
```bash
python adcplus_attack.py --attack giga --model_idx 0 --attack_file harmful_strings.csv
```

## GCG with Self-Replication (NEW)

We've implemented a novel variant of GCG that adds an **explicit self-replication loss term**. This creates adversarial tokens that not only jailbreak the model but also encourage the model to output those tokens, enabling attack propagation in multi-agent systems.

### Key Innovation:

The attack uses a combined loss function:
```
L_total = L_jailbreak + λ * L_replication
```

where:
- **L_jailbreak**: Standard cross-entropy loss for generating harmful content
- **L_replication**: Cross-entropy loss encouraging the model to output the adversarial tokens
- **λ**: Replication weight parameter (default: 1.0)

### Key Features:
- **Dual objective**: Jailbreak AND self-replication in a single optimization
- **Explicit control**: λ parameter controls the trade-off between objectives
- **Simpler than GIGA**: More interpretable and easier to analyze
- **Viral propagation**: Creates attacks that spread in multi-agent scenarios

### Usage

Run GCG with self-replication:
```bash
python single_attack.py --attack gcg-selfrep --model_idx 0 --num_steps 500
```

Run with custom replication weight:
```bash
python single_attack.py --attack gcg-selfrep --model_idx 0 --replication_weight 2.0 --num_steps 500
```

Choose replication position ('start' or 'after_adv'):
```bash
python single_attack.py --attack gcg-selfrep --replication_position start --num_steps 500
```

### Interactive Demo

For a comprehensive walkthrough of the self-replication mechanism, run:
```bash
python self_replicating_demo.py
```

This demo includes:
- Comparison of standard GCG vs GCG-SelfRep
- Analysis of self-replication behavior
- Experiments with different λ values
- Multi-agent propagation simulation
- Defensive considerations

### Supported Attacks

- **ADC**: Adaptive Dense-to-sparse Constrained Optimization (default)
- **GCG**: Greedy Coordinate Gradient attack
- **GCG-SelfRep**: GCG with Self-Replication Loss (NEW)
- **GIGA**: Generalizable Infectious Gradient Attack

TODO:
As mentioned in https://github.com/hukkai/adc_llm_attack/issues/2, the code is outdated with the latest version of `transformers`, but runable at `transformers==4.39`.

Add a TODO for either specifying the version requirements or refactor the code for a newer version of `transformers`.
