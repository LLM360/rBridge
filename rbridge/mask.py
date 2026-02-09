"""Span tag parsing and token-level masking for RBridge."""

import re
from typing import List, Optional, Tuple


def extract_span_offsets(text: str) -> Tuple[str, Optional[List[Tuple[int, int]]]]:
    """Strip <span>...</span> tags and return (clean_text, char-level offset ranges).

    Returns:
        clean_text: Text with all <span>/<\\span> tags removed.
        offset_mapping: List of (start, end) char positions in clean_text that
            were inside <span> tags. None if no spans found.
    """
    pattern = r"<span>(.*?)<[/\\]span>"
    if not re.search(pattern, text):
        return text, None

    offset_mapping = []
    cleaned_parts = []
    last_end = 0
    current_pos = 0

    for match in re.finditer(pattern, text):
        # Add text before this span
        cleaned_parts.append(text[last_end : match.start()])
        current_pos += match.start() - last_end

        # Record span content position in cleaned text
        span_content = match.group(1)
        offset_mapping.append((current_pos, current_pos + len(span_content)))
        cleaned_parts.append(span_content)
        current_pos += len(span_content)
        last_end = match.end()

    # Add remaining text
    cleaned_parts.append(text[last_end:])
    return "".join(cleaned_parts), offset_mapping


def build_token_mask(
    token_offsets: List[Tuple[int, int]],
    span_ranges: List[Tuple[int, int]],
    char_offset: int = 0,
) -> List[bool]:
    """Build a per-token boolean mask from char-level span ranges.

    A token is True (included) if any part of it overlaps a span range.

    Args:
        token_offsets: Per-token (start, end) char positions from the tokenizer.
        span_ranges: Char-level (start, end) ranges from extract_span_offsets,
            adjusted by char_offset.
        char_offset: Character offset to add to span_ranges (e.g., length of
            the question prefix).

    Returns:
        List of booleans, one per token.
    """
    adjusted = [(s + char_offset, e + char_offset) for s, e in span_ranges]
    mask = []
    for ts, te in token_offsets:
        hit = any(min(te, ce) > max(ts, cs) for cs, ce in adjusted)
        mask.append(hit)
    return mask
