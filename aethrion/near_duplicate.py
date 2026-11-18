"""Bounded near-duplicate detection for BASE corpus split safety.

This is a conservative leakage gate, not a semantic plagiarism detector. It
uses deterministic token shingles and exact Jaccard verification. Ambiguous
high-cost inputs fail closed instead of silently skipping comparisons.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_SHINGLE_SIZE = 5
_MIN_TOKENS = 12
_MIN_SIMILARITY = 0.85
_MAX_POSTING = 500
_MAX_CANDIDATES_PER_RECORD = 2_000
_MAX_COMPARISONS = 250_000


def _shingles(value: str) -> set[tuple[str, ...]]:
    tokens = _TOKEN_RE.findall(value.casefold())
    if len(tokens) < _MIN_TOKENS:
        return set()
    return {
        tuple(tokens[index:index + _SHINGLE_SIZE])
        for index in range(len(tokens) - _SHINGLE_SIZE + 1)
    }


def find_cross_group_near_duplicates(
    records: list[dict[str, Any]],
    *,
    minimum_similarity: float = _MIN_SIMILARITY,
) -> list[dict[str, Any]]:
    """Return cross-group near-duplicate pairs; raise if bounded audit is inconclusive."""
    if not isinstance(minimum_similarity, (int, float)) or not 0.0 < minimum_similarity <= 1.0:
        raise ValueError("minimum_similarity must be in (0, 1].")

    postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
    signatures: list[set[tuple[str, ...]]] = []
    groups: list[str] = []
    matches: list[dict[str, Any]] = []
    comparisons = 0

    for index, record in enumerate(records):
        text = record.get("text")
        group_id = record.get("group_id")
        if not isinstance(text, str) or not isinstance(group_id, str) or not group_id:
            raise ValueError("Near-duplicate audit requires text and non-empty group_id strings.")
        current = _shingles(text)
        signatures.append(current)
        groups.append(group_id)
        if not current:
            continue

        candidates: set[int] = set()
        for shingle in sorted(current):
            prior = postings.get(shingle, [])
            if len(prior) > _MAX_POSTING:
                raise ValueError(
                    "Near-duplicate audit exceeded a common-shingle posting bound; "
                    "human review or a separately approved audit is required."
                )
            candidates.update(prior)
            if len(candidates) > _MAX_CANDIDATES_PER_RECORD:
                raise ValueError(
                    "Near-duplicate audit exceeded its per-record comparison bound; "
                    "refusing to skip candidates."
                )

        for previous in sorted(candidates):
            if groups[previous] == group_id:
                continue
            comparisons += 1
            if comparisons > _MAX_COMPARISONS:
                raise ValueError(
                    "Near-duplicate audit exceeded its global comparison bound; "
                    "refusing to skip candidates."
                )
            earlier = signatures[previous]
            union_size = len(current | earlier)
            if not union_size:
                continue
            similarity = len(current & earlier) / union_size
            if similarity >= minimum_similarity:
                matches.append({
                    "left_index": previous,
                    "right_index": index,
                    "similarity": round(similarity, 6),
                })

        for shingle in sorted(current):
            postings[shingle].append(index)

    return matches
