# rBridge: Predicting LLM Reasoning Performance with Small Proxy Models

[![Paper](https://img.shields.io/badge/arXiv-2509.21013-b31b1b.svg)](https://arxiv.org/abs/2509.21013)
[![Dataset](https://img.shields.io/badge/HuggingFace-trillionlabs/rbridge--mask-yellow.svg)](https://huggingface.co/datasets/trillionlabs/rbridge-mask)

This repository contains the implementation of **rBridge**, a method introduced in the paper *[Predicting LLM Reasoning Performance with Small Proxy Model](https://arxiv.org/abs/2509.21013)*.

rBridge enables the use of small proxy models (≤1B parameters) to accurately predict the reasoning performance of much larger language models (7B to 32B+). By aligning proxies with the pre-training objective and the target reasoning task, rBridge reduces dataset ranking costs by over **100x** while maintaining high correlation across multiple reasoning benchmarks.

## Overview

Predicting emergent reasoning capabilities in Large Language Models (LLMs) is traditionally difficult because these behaviors often only appear at scales exceeding 7B parameters. rBridge solves this by:

1. **Weighted NLL Loss:** Computing weighted negative log-likelihood on reasoning traces using token-level importance.
2. **Gold Label Traces:** Using reasoning traces from frontier models as ground truth for alignment.
3. **Cross-Scale Correlation:** Providing a reliable proxy for performance at the 1B to 32B scale.

## Evaluation Modes

rBridge supports two evaluation modes for scoring reasoning traces:

| Mode | Flag | Description | Status |
|------|------|-------------|--------|
| **Mask** | `--mode mask` | Score only tokens inside `<span>` tagged regions | Available |
| **Token Probability** | `--mode token-prob` | Weight each token's log-likelihood by its importance probability (paper method) | Coming soon |

**Mask mode** uses `<span>` tags in reasoning traces to identify key factual/reasoning steps and only computes perplexity on those tokens.

**Token Probability mode** is the method described in the paper — it weights each token's NLL by a learned token-level importance score, providing a continuous weighting rather than binary masking.

## Key Features

* **Efficient Dataset Selection:** Rank pre-training datasets for reasoning tasks without training large models.
* **Cost Reduction:** Achieve predictive accuracy at 100x lower computational cost compared to traditional scaling law baselines.
* **Multi-Benchmark Support:** High correlation demonstrated across multiple reasoning benchmarks (MMLU-Pro, MATH500, GPQA, AIME, etc.).
* **Simple CLI:** Single command evaluation with vLLM backend.

## Installation

```bash
git clone https://github.com/trillionlabs/rbridge.git
cd rbridge
pixi install
pixi run check
```

## Usage

There is a pixi task `rbridge = "python -m rbridge.eval"`, so `pixi run rbridge 
...` is equivalent to `pixi run python -m rbridge.eval ...`

### Quick Start

```bash
pixi run rbridge \
    --model trillionlabs/Tri-0.5B-Base \
    --dataset trillionlabs/rbridge-mask \
    --subsets aime25 \
    --tp 1 \
    --batch-size 32 \
    --output results.json
```

### Full Evaluation

```bash
pixi run rbridge \
    --model trillionlabs/Tri-0.5B-Base \
    --dataset trillionlabs/rbridge-mask \
    --subsets mmlu-pro,math500,gpqa,cqa,bbh,arena-hard,arc,aime25,mmlu,kmmlu \
    --tp 8 \
    --batch-size 32 \
    --output results.json
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Local path or HuggingFace model ID |
| `--dataset` | `trillionlabs/rbridge-mask` | HuggingFace dataset |
| `--subsets` | all | Comma-separated: `mmlu-pro,math500,gpqa,cqa,bbh,arena-hard,arc,aime25,mmlu,kmmlu` |
| `--mode` | `mask` | Evaluation mode: `mask` or `token-prob` (coming soon) |
| `--tp` | 1 | Tensor parallel size |
| `--batch-size` | 32 | Batch size |
| `--max-model-len` | auto | Max sequence length |
| `--no-mask` | false | Score all reasoning tokens (ignores `<span>` tags) |
| `--debug` | false | Print token-level breakdown for first sample |
| `--output` | none | Save results JSON |

### Output

```
Model: trillionlabs/Tri-0.5B-Base

Subset          | Samples | Word PPL | Byte PPL | Bits/Byte
-----------------------------------------------------------------
mmlu-pro        |     601 |     2.34 |     1.89 |    0.9200
gpqa            |     100 |     3.12 |     2.45 |    1.2900
...
-----------------------------------------------------------------
Overall         |    3363 |     2.41 |     1.95 |    0.9600
```

## Token Probability Weighting

> **Coming soon** (`--mode token-prob`)

This is the primary method described in the [paper](https://arxiv.org/abs/2509.21013). Instead of binary masking, each token's log-likelihood is weighted by a continuous importance score derived from the frontier model's token probabilities. Tokens that the frontier model assigns higher probability to are weighted more heavily, providing a smooth, data-driven measure of reasoning importance.

```
NLL_weighted = - sum( w_i * log P(t_i | t_{<i}) )
```

where `w_i` is the token-level importance weight from the frontier model.

## Masked Span Weighting

> **Available** (`--mode mask`)

rBridge computes the **masked log-likelihood** of reasoning traces generated by frontier models. Given a reasoning trace with `<span>` tags marking key reasoning steps:

```
The user is asking about X. <span>The Dane particle is the complete virion of HBV</span>. Now I need...
```

Only the tokens inside `<span>` regions are scored. This is a binary approximation of the token probability method — tokens are either fully included (inside `<span>`) or fully excluded. This focuses evaluation on factual and reasoning content, ignoring filler text. The resulting perplexity metrics (word PPL, byte PPL, bits/byte) correlate strongly with downstream reasoning performance across scales.


## Citation

If you find this work useful in your research, please cite:

```bibtex
@article{koh2025predicting,
  title={Predicting LLM Reasoning Performance with Small Proxy Model},
  author={Koh, Woosung and Suk, Juyoung and Han, Sungjun and Yun, Se-Young and Shin, Jamin},
  journal={arXiv preprint arXiv:2509.21013},
  year={2025}
}
```

## License

This project is licensed under the MIT License
