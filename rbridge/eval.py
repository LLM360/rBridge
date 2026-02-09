#!/usr/bin/env python3
"""RBridge: Evaluate reasoning trace likelihood with masked spans.

Usage:
    python -m rbridge.eval \
        --model /path/to/model \
        --dataset trillionlabs/rbridge-mask \
        --subsets mmlu-pro,math500,gpqa \
        --tp 8 \
        --batch-size 32 \
        --output results.json
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .mask import build_token_mask, extract_span_offsets
from .metrics import aggregate

ALL_SUBSETS = [
    "mmlu-pro", "math500", "gpqa", "cqa", "bbh",
    "arena-hard", "arc", "aime25", "mmlu", "kmmlu",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="RBridge: Reasoning trace likelihood evaluation"
    )
    parser.add_argument("--model", type=str, required=True, help="Model path or HF model ID")
    parser.add_argument("--dataset", type=str, default="trillionlabs/rbridge-mask",
                        help="HuggingFace dataset name")
    parser.add_argument("--subsets", type=str, default=None,
                        help="Comma-separated subset names (default: all)")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for vLLM")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Max model sequence length (default: model's max)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--debug", action="store_true", help="Print debug info for first sample")
    parser.add_argument("--mask", action="store_true", default=True,
                        help="Only score tokens inside <span> tags (default)")
    parser.add_argument("--no-mask", action="store_true",
                        help="Score all reasoning tokens (ignore <span> tags)")
    return parser.parse_args()


def count_words(text: str) -> int:
    return len(text.split())


def count_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


def prepare_samples(
    dataset_name: str,
    subset: str,
    tokenizer: AutoTokenizer,
    use_mask: bool = True,
) -> List[Dict]:
    """Load dataset and prepare samples for evaluation.

    Returns list of dicts with:
        full_text: concatenated question + clean reasoning
        question: original question
        reasoning: original reasoning (with tags)
        clean_reasoning: reasoning with tags stripped
        span_ranges: char-level span offsets in clean_reasoning (or None)
        question_char_len: char length of question prefix in full_text
        word_count: words in scored region
        byte_count: bytes in scored region
    """
    ds = load_dataset(dataset_name, subset, split="test")
    samples = []

    for row in ds:
        question = row["question"]
        reasoning = row["reasoning"]

        clean_reasoning, span_ranges = extract_span_offsets(reasoning)
        full_text = question + clean_reasoning

        # Count words/bytes for the scored region
        if use_mask and span_ranges is not None:
            scored_text = "".join(
                clean_reasoning[s:e] for s, e in span_ranges
            )
        else:
            scored_text = clean_reasoning

        samples.append({
            "full_text": full_text,
            "question": question,
            "reasoning": reasoning,
            "clean_reasoning": clean_reasoning,
            "span_ranges": span_ranges,
            "question_char_len": len(question),
            "word_count": count_words(scored_text),
            "byte_count": count_bytes(scored_text),
        })

    return samples


def compute_loglikelihoods(
    llm: LLM,
    tokenizer: AutoTokenizer,
    samples: List[Dict],
    batch_size: int,
    use_mask: bool = True,
    debug: bool = False,
) -> List[Dict]:
    """Compute (masked) log-likelihoods for all samples using vLLM.

    Uses prompt_logprobs to get per-token log-probabilities in a single
    forward pass, then applies the span mask to sum only relevant tokens.
    """
    # We request prompt_logprobs and generate only 1 token (we don't need output)
    sampling_params = SamplingParams(
        prompt_logprobs=1,
        max_tokens=1,
        temperature=0.0,
    )

    results = []
    texts = [s["full_text"] for s in samples]

    # Process in batches
    for batch_start in range(0, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        batch_texts = texts[batch_start:batch_end]
        batch_samples = samples[batch_start:batch_end]

        outputs = llm.generate(batch_texts, sampling_params, use_tqdm=False)

        for output, sample in zip(outputs, batch_samples):
            prompt_logprobs = output.prompt_logprobs  # list of dicts, one per token

            if prompt_logprobs is None:
                results.append({
                    "loglikelihood": 0.0,
                    "word_count": sample["word_count"],
                    "byte_count": sample["byte_count"],
                })
                continue

            # Get token IDs from the prompt
            token_ids = output.prompt_token_ids

            # Extract per-token logprobs (first token has no logprob)
            token_lls = []
            for i, lp_dict in enumerate(prompt_logprobs):
                if lp_dict is None:
                    token_lls.append(0.0)
                else:
                    # lp_dict maps token_id -> Logprob; get the logprob for the actual token
                    tid = token_ids[i]
                    if tid in lp_dict:
                        token_lls.append(lp_dict[tid].logprob)
                    else:
                        token_lls.append(0.0)

            # Build token-to-char offset mapping for masking
            full_text = sample["full_text"]
            encoding = tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            token_offsets = encoding.get("offset_mapping", [])

            # Determine which tokens to score
            if use_mask and sample["span_ranges"] is not None:
                # Only score tokens inside <span> regions
                char_offset = sample["question_char_len"]
                mask = build_token_mask(token_offsets, sample["span_ranges"], char_offset)
            else:
                # Score all reasoning tokens (skip question prefix)
                question_char_len = sample["question_char_len"]
                mask = []
                for ts, te in token_offsets:
                    mask.append(ts >= question_char_len)

            # Sum masked logprobs (skip index 0 which has no conditioning)
            masked_ll = 0.0
            n_masked = 0
            for i in range(1, min(len(token_lls), len(mask))):
                if mask[i]:
                    masked_ll += token_lls[i]
                    n_masked += 1

            if debug and batch_start == 0 and sample is batch_samples[0]:
                _print_debug(sample, token_ids, token_lls, mask, token_offsets, tokenizer, masked_ll, n_masked)

            results.append({
                "loglikelihood": masked_ll,
                "word_count": sample["word_count"],
                "byte_count": sample["byte_count"],
            })

    return results


def _print_debug(sample, token_ids, token_lls, mask, token_offsets, tokenizer, masked_ll, n_masked):
    """Print debug info for a single sample."""
    print("\n" + "=" * 60)
    print("DEBUG: First sample breakdown")
    print("=" * 60)
    print(f"Question (first 200 chars): {sample['question'][:200]}")
    print(f"Reasoning (first 200 chars): {sample['clean_reasoning'][:200]}")
    if sample["span_ranges"]:
        print(f"Span ranges: {sample['span_ranges'][:5]}{'...' if len(sample['span_ranges']) > 5 else ''}")
    print(f"Total tokens: {len(token_ids)}")
    print(f"Masked tokens scored: {n_masked}")
    print(f"Masked log-likelihood: {masked_ll:.4f}")
    if n_masked > 0:
        print(f"Avg log-prob per masked token: {masked_ll / n_masked:.4f}")

    # Show a few masked tokens
    print("\nSample masked tokens:")
    shown = 0
    for i in range(1, min(len(token_lls), len(mask))):
        if mask[i] and shown < 10:
            tok_text = tokenizer.decode([token_ids[i]])
            print(f"  [{i}] {repr(tok_text):>20s}  ll={token_lls[i]:.4f}")
            shown += 1
    print("=" * 60 + "\n")


def print_results_table(all_results: Dict[str, Dict], model_name: str):
    """Print a formatted results table."""
    print(f"\nModel: {model_name}\n")
    header = f"{'Subset':<15} | {'Samples':>7} | {'Word PPL':>9} | {'Byte PPL':>9} | {'Bits/Byte':>9}"
    sep = "-" * len(header)
    print(header)
    print(sep)

    for subset, metrics in sorted(all_results.items()):
        if subset == "overall":
            continue
        print(
            f"{subset:<15} | {metrics['num_samples']:>7d} | "
            f"{metrics['word_perplexity']:>9.2f} | "
            f"{metrics['byte_perplexity']:>9.2f} | "
            f"{metrics['bits_per_byte']:>9.4f}"
        )

    if "overall" in all_results:
        print(sep)
        m = all_results["overall"]
        print(
            f"{'Overall':<15} | {m['num_samples']:>7d} | "
            f"{m['word_perplexity']:>9.2f} | "
            f"{m['byte_perplexity']:>9.2f} | "
            f"{m['bits_per_byte']:>9.4f}"
        )
    print()


def main():
    args = parse_args()
    use_mask = not args.no_mask

    subsets = args.subsets.split(",") if args.subsets else ALL_SUBSETS
    print("=" * 60)
    print("RBridge: Reasoning Trace Likelihood Evaluation")
    print("=" * 60)
    print(f"Model:   {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Subsets: {', '.join(subsets)}")
    print(f"Mask:    {use_mask}")
    print(f"TP:      {args.tp}")
    print("=" * 60)

    # Initialize tokenizer and vLLM engine
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading vLLM engine...")
    llm_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tp,
        "trust_remote_code": True,
        "enforce_eager": False,
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len
    llm = LLM(**llm_kwargs)

    all_results = {}
    all_sample_results = []  # For overall aggregation

    for subset in subsets:
        print(f"\n--- {subset} ---")

        # Load and prepare
        samples = prepare_samples(args.dataset, subset, tokenizer, use_mask)
        print(f"  Samples: {len(samples)}")

        # Compute log-likelihoods
        sample_results = compute_loglikelihoods(
            llm, tokenizer, samples, args.batch_size,
            use_mask=use_mask,
            debug=(args.debug and subset == subsets[0]),
        )

        # Aggregate
        metrics = aggregate(sample_results)
        all_results[subset] = metrics
        all_sample_results.extend(sample_results)

        print(
            f"  Word PPL: {metrics['word_perplexity']:.2f} | "
            f"Byte PPL: {metrics['byte_perplexity']:.2f} | "
            f"Bits/Byte: {metrics['bits_per_byte']:.4f}"
        )

    # Overall
    if len(subsets) > 1:
        all_results["overall"] = aggregate(all_sample_results)

    # Print table
    print_results_table(all_results, args.model)

    # Save
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "model": args.model,
            "dataset": args.dataset,
            "mask": use_mask,
            "results": {},
        }
        for k, v in all_results.items():
            output_data["results"][k] = {
                "word_perplexity": v["word_perplexity"],
                "byte_perplexity": v["byte_perplexity"],
                "bits_per_byte": v["bits_per_byte"],
                "num_samples": v["num_samples"],
            }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
