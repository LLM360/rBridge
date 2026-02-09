"""Perplexity and bits-per-byte aggregation for RBridge."""

import math
from typing import Dict, List


def aggregate(results: List[Dict]) -> Dict[str, float]:
    """Aggregate per-sample results into summary metrics.

    Each result dict should have:
        loglikelihood: float (sum of log-probs, negative)
        word_count: int
        byte_count: int

    Returns dict with word_perplexity, byte_perplexity, bits_per_byte.
    """
    total_ll = sum(r["loglikelihood"] for r in results)
    total_words = sum(r["word_count"] for r in results)
    total_bytes = sum(r["byte_count"] for r in results)

    metrics = {}

    if total_words > 0:
        metrics["word_perplexity"] = math.exp(-total_ll / total_words)
    else:
        metrics["word_perplexity"] = float("inf")

    if total_bytes > 0:
        metrics["byte_perplexity"] = math.exp(-total_ll / total_bytes)
        metrics["bits_per_byte"] = -total_ll / (total_bytes * math.log(2))
    else:
        metrics["byte_perplexity"] = float("inf")
        metrics["bits_per_byte"] = float("inf")

    metrics["total_loglikelihood"] = total_ll
    metrics["total_words"] = total_words
    metrics["total_bytes"] = total_bytes
    metrics["num_samples"] = len(results)

    return metrics
