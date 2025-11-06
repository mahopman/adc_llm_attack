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

### Supported Attacks

- **ADC**: Adaptive Dense-to-sparse Constrained Optimization (default)
- **GCG**: Greedy Coordinate Gradient attack
- **GIGA**: Generalizable Infectious Gradient Attack (NEW)

TODO:
As mentioned in https://github.com/hukkai/adc_llm_attack/issues/2, the code is outdated with the latest version of `transformers`, but runable at `transformers==4.39`.

Add a TODO for either specifying the version requirements or refactor the code for a newer version of `transformers`.
