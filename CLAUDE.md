# CLAUDE.md

## Project Overview

This repository implements the ADC (Adaptive Dense-to-sparse Constrained) LLM attack from the NeurIPS 2024 paper: "Efficient LLM Jailbreak via Adaptive Dense-to-sparse Constrained Optimization" ([arXiv:2405.09113](https://arxiv.org/abs/2405.09113)).

The project provides adversarial attack methods (ADC and GCG) for testing LLM robustness against jailbreak attempts. This is security research code for authorized testing contexts.

## Directory Structure

```
adc_llm_attack/
├── llm_attack/           # Core attack implementations
│   ├── adc_attack.py     # ADCAttack class - main adaptive dense-to-sparse attack
│   ├── gcg_attack.py     # GCGAttack class - Greedy Coordinate Gradient attack
│   ├── judger.py         # Judges if attack responses are successful
│   └── tools.py          # Utility functions for attacks
├── utils/                # Helper utilities
│   ├── env_utils.py      # Distributed training (DDP) initialization
│   ├── llm_utils.py      # Model loading and input template generation
│   └── system_prompt_utils.py  # System prompts for different models
├── chat_templates/       # Jinja2 chat templates for different model formats
├── advbench/            # Adversarial benchmark datasets
├── single_attack.py     # Main entry point for running attacks
├── adcplus_attack.py    # Enhanced ADC attack (ADC followed by GCG refinement)
└── judger.py            # Standalone judger implementation
```

## Key Commands

### Running a Single Attack
```bash
python single_attack.py --model_idx 0 --attack adc --num_steps 10 --num_adv_tokens 20
```

### Common Arguments
- `--model_idx`: Index into supported_models list (0-7)
- `--attack`: Attack type - `adc` or `gcg`
- `--num_steps`: Number of optimization steps (default: 10)
- `--num_starts`: Number of parallel starts for ADC (default: 1)
- `--num_adv_tokens`: Number of adversarial tokens to optimize (default: 20)
- `--attack_file`: CSV file with attack prompts (default: `harmful_strings.csv`)
- `--launcher`: Distributed training launcher - `none`, `slurm`, or `pytorch`

### Supported Models
```python
supported_models = [
    'HuggingFaceH4/zephyr-7b-beta',      # 0
    'lmsys/vicuna-7b-v1.3',               # 1
    'lmsys/vicuna-7b-v1.5',               # 2
    'meta-llama/Llama-2-7b-chat-hf',      # 3
    'cais/zephyr_7b_r2d2',                # 4
    'meta-llama/Meta-Llama-3-8B-Instruct',# 5
    'meta-llama/Llama-2-13b-chat-hf',     # 6
    'lmsys/vicuna-13b-v1.5',              # 7
]
```

## Architecture Notes

### ADCAttack (`llm_attack/adc_attack.py`)
- Uses soft token optimization with SGD + momentum
- Implements adaptive sparsity based on prediction errors
- Key parameters: `learning_rate=10`, `momentum=0.99`, `buffer_size=64`
- Supports KV-cache for faster inference during evaluation

### Input Template Generation (`utils/llm_utils.py`)
- `get_input_template()`: Constructs attack input with adversarial token slots
- `get_model()`: Loads HuggingFace models in float16
- Chat templates loaded from `chat_templates/` directory

## Known Issues

**Transformers Version**: The code requires `transformers==4.39`. Newer versions of transformers have API changes that break compatibility (see [GitHub Issue #2](https://github.com/hukkai/adc_llm_attack/issues/2)).

## Development Notes

- Results are saved to `./results/{model_name}-{attack}-{attack_file}/`
- Each result saved as `result_{k}.pth` containing: (loss, adv_tokens, steps, time, prompt, generated_text)
- The `Judger` class uses keyword matching to determine attack success
- `adcplus_attack.py` runs ADC first, then falls back to GCG refinement if ADC doesn't converge
