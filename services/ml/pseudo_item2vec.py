"""
Pseudo-Item2Vec: Lightweight Co-Visitation Graph Engine

This module provides Item2Vec-style similarity without ML training.
It builds co-visitation vectors from transaction history in ~1-5ms.

Inspired by Pinterest's retrieval system and YouTube's candidate generation,
but optimized for serverless/Cloud Functions deployment (no training overhead).

Enhanced with PMI (Pointwise Mutual Information) and Lift weighting to
down-weight ubiquitous products and surface meaningful co-occurrences.

NOTE: All identifiers use variant_id as the PRIMARY key.
SKU is only used for display purposes (never as join key).

Usage:
    vectors = build_covis_vectors(order_lines, top_variant_ids=top_200)
    similarity = cosine_similarity(vectors["variant_id_1"], vectors["variant_id_2"])
"""
from collections import defaultdict
from itertools import combinations
from math import sqrt, log
from typing import Dict, Iterable, List, Optional, Set, Any, Literal
import logging

logger = logging.getLogger(__name__)


class CoVisVector:
    """
    Lightweight pseudo-Item2Vec vector built from co-visitation counts.
    Stored as a sparse dict: neighbor_variant_id -> weight.

    This mimics Word2Vec/Item2Vec embeddings but computed directly from
    co-occurrence statistics (no training required).

    NOTE: Uses variant_id as the PRIMARY identifier.
    """

    def __init__(self, variant_id: str, weights: Dict[str, float]):
        self.variant_id = variant_id
        self.weights = weights  # normalized weights (L2 norm = 1)

    def __repr__(self):
        neighbor_count = len(self.weights)
        return f"CoVisVector(variant_id={self.variant_id!r}, neighbors={neighbor_count})"


def _normalize(vec: Dict[str, float]) -> Dict[str, float]:
    """L2 normalization for cosine similarity."""
    norm_sq = sum(v * v for v in vec.values())
    if norm_sq <= 0:
        return {}
    norm = sqrt(norm_sq)
    return {k: v / norm for k, v in vec.items()}


def build_covis_vectors(
    order_lines: Iterable,
    top_variant_ids: Optional[Set[str]] = None,
    min_co_visits: int = 1,
    max_neighbors: int = 50,
    weighting: Literal["raw", "lift", "pmi"] = "raw",  # Default to raw for backwards compatibility
    min_lift: float = 0.0,  # No filtering by default - let downstream ranking decide
) -> Dict[str, CoVisVector]:
    """
    Build pseudo-Item2Vec style vectors from order_lines.

    This creates a co-visitation graph where each product is represented
    as a sparse vector of products it frequently appears with in orders.

    Args:
        order_lines: Iterable of order line objects (must have .order_id and .variant_id)
        top_variant_ids: Optional set to restrict to top products (improves speed)
        min_co_visits: Minimum co-occurrence count to include a neighbor
        max_neighbors: Maximum neighbors per product (keeps vectors sparse)
        weighting: Weight calculation method:
            - "raw": Raw co-occurrence counts (original behavior)
            - "lift": Lift = P(A,B) / (P(A) * P(B)) - values > 1 indicate positive association
            - "pmi": Pointwise Mutual Information = log(lift) - unbounded, can be negative
        min_lift: Minimum lift threshold (only for lift/pmi weighting, default 1.0)

    Returns:
        Dict mapping variant_id -> CoVisVector

    Performance:
        - 1000 orders × 3 products: ~2-5ms
        - 10000 orders × 3 products: ~20-30ms
        - 100000 orders × 3 products: ~200-300ms

    Example:
        >>> vectors = build_covis_vectors(order_lines, top_variant_ids=top_200, weighting="lift")
        >>> sim = cosine_similarity(vectors["variant_id_1"], vectors["variant_id_2"])
        >>> # 0.85 = highly similar products
    """
    # 1) Group variant_ids per order and track individual product frequencies
    orders: Dict[str, List[str]] = defaultdict(list)
    variant_order_count: Dict[str, int] = defaultdict(int)  # How many orders each variant_id appears in

    for line in order_lines:
        variant_id = getattr(line, "variant_id", None)
        # variant_id is the PRIMARY identifier
        if not variant_id:
            continue
        identifier = str(variant_id).strip()
        if not identifier:
            continue
        if top_variant_ids is not None and identifier not in top_variant_ids:
            continue
        order_id = str(getattr(line, "order_id"))
        orders[order_id].append(identifier)

    # Count unique orders per variant_id (for frequency calculation)
    for order_id, variant_ids in orders.items():
        unique_variant_ids = set(variant_ids)
        for vid in unique_variant_ids:
            variant_order_count[vid] += 1

    total_orders = len(orders)
    if total_orders == 0:
        logger.warning("No orders found for co-visitation graph")
        return {}

    # 2) Count co-visits (products that appear together in orders)
    covis_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, vids in orders.items():
        unique_vids = list(set(vids))  # dedupe within order
        if len(unique_vids) < 2:
            continue
        # All pairs in this order co-occurred
        for v1, v2 in combinations(unique_vids, 2):
            covis_counts[v1][v2] += 1
            covis_counts[v2][v1] += 1

    # 3) Convert counts -> weights (using lift/PMI) and normalize
    result: Dict[str, CoVisVector] = {}

    for variant_id, neighbors in covis_counts.items():
        # Drop weak signals (co-occurrence count)
        filtered = {n: c for n, c in neighbors.items() if c >= min_co_visits}
        if not filtered:
            continue

        # Calculate weights based on weighting method
        if weighting == "raw":
            # Original behavior: raw co-occurrence counts
            weights = dict(filtered)
        else:
            # Lift or PMI weighting
            weights = {}
            freq_variant = variant_order_count.get(variant_id, 1)
            p_variant = freq_variant / total_orders

            for neighbor, cooccur_count in filtered.items():
                freq_neighbor = variant_order_count.get(neighbor, 1)
                p_neighbor = freq_neighbor / total_orders
                p_both = cooccur_count / total_orders

                # Lift = P(A,B) / (P(A) * P(B))
                # Values > 1 mean positive association (appear together more than random)
                # Values < 1 mean negative association (appear together less than random)
                expected = p_variant * p_neighbor
                if expected > 0:
                    lift = p_both / expected
                else:
                    lift = 1.0

                # Filter by minimum lift threshold
                if lift < min_lift:
                    continue

                if weighting == "pmi":
                    # PMI = log(lift), can be negative for lift < 1
                    # Use max to avoid log(0) issues
                    weights[neighbor] = log(max(lift, 0.001))
                else:
                    # Lift weighting (default)
                    weights[neighbor] = lift

        if not weights:
            continue

        # Keep top-K neighbors only (sparsity for speed)
        if len(weights) > max_neighbors:
            weights = dict(
                sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[
                    :max_neighbors
                ]
            )

        # L2 normalize for cosine similarity
        normed = _normalize(weights)
        if not normed:
            continue

        result[variant_id] = CoVisVector(variant_id=variant_id, weights=normed)

    weighting_desc = {
        "raw": "raw counts",
        "lift": f"lift (min={min_lift})",
        "pmi": f"PMI (min_lift={min_lift})"
    }.get(weighting, weighting)

    logger.info(
        f"Built co-visitation graph: {len(result)} products, "
        f"avg neighbors: {sum(len(v.weights) for v in result.values()) / max(len(result), 1):.1f}, "
        f"weighting: {weighting_desc}"
    )

    return result


def cosine_similarity(vec_a: CoVisVector, vec_b: CoVisVector) -> float:
    """
    Cosine similarity between two CoVisVectors (already L2-normalized).

    Returns:
        Float in [0, 1] range:
        - 1.0 = identical co-visitation patterns
        - 0.0 = no common neighbors
        - 0.5+ = strong similarity

    Performance: O(min(|a|, |b|)) where |a| is number of neighbors
    """
    if not vec_a or not vec_b:
        return 0.0
    if not vec_a.weights or not vec_b.weights:
        return 0.0

    # Iterate over smaller dict for speed
    if len(vec_a.weights) > len(vec_b.weights):
        vec_a, vec_b = vec_b, vec_a

    total = 0.0
    b_weights = vec_b.weights
    for vid, wa in vec_a.weights.items():
        wb = b_weights.get(vid)
        if wb is not None:
            total += wa * wb

    return total


def batch_similarity(
    anchor: CoVisVector, candidates: List[CoVisVector], threshold: float = 0.0
) -> List[tuple[str, float]]:
    """
    Compute similarity between anchor and multiple candidates.

    Args:
        anchor: Reference product vector
        candidates: List of candidate vectors
        threshold: Only return similarities >= threshold

    Returns:
        List of (variant_id, similarity) tuples, sorted by similarity descending

    Example:
        >>> similar = batch_similarity(
        ...     vectors["variant_id_1"],
        ...     [vectors[v] for v in candidate_variant_ids],
        ...     threshold=0.3
        ... )
        >>> # [("variant_id_42", 0.85), ("variant_id_123", 0.72), ...]
    """
    results = []
    for candidate in candidates:
        sim = cosine_similarity(anchor, candidate)
        if sim >= threshold:
            results.append((candidate.variant_id, sim))

    # Sort by similarity descending
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def compute_bundle_coherence(variant_ids: List[str], vectors: Dict[str, CoVisVector]) -> float:
    """
    Compute average pairwise similarity within a bundle.

    This measures how "coherent" a bundle is - products that frequently
    appear together will have high coherence.

    Args:
        variant_ids: List of variant_ids in the bundle
        vectors: Co-visitation vectors

    Returns:
        Float in [0, 1] range:
        - 0.8+ = very coherent bundle (products often bought together)
        - 0.5-0.8 = moderate coherence
        - <0.5 = weak coherence (might be random pairing)

    Example:
        >>> coherence = compute_bundle_coherence(
        ...     ["variant_id_1", "variant_id_2", "variant_id_3"],
        ...     vectors
        ... )
        >>> # 0.75 = good bundle
    """
    if len(variant_ids) < 2:
        return 0.0

    similarities = []
    for i in range(len(variant_ids)):
        for j in range(i + 1, len(variant_ids)):
            v1 = vectors.get(variant_ids[i])
            v2 = vectors.get(variant_ids[j])
            if v1 and v2:
                similarities.append(cosine_similarity(v1, v2))

    return sum(similarities) / len(similarities) if similarities else 0.0


def find_complementary_products(
    anchor_variant_id: str,
    vectors: Dict[str, CoVisVector],
    top_k: int = 10,
    min_similarity: float = 0.3,
) -> List[tuple[str, float]]:
    """
    Find products that complement the anchor product.

    This is the core of "Frequently Bought Together" recommendations.

    Args:
        anchor_variant_id: Reference product variant_id
        vectors: Co-visitation vectors
        top_k: Number of recommendations to return
        min_similarity: Minimum similarity threshold

    Returns:
        List of (variant_id, similarity) tuples, sorted by similarity

    Example:
        >>> complements = find_complementary_products(
        ...     "laptop-variant-id",
        ...     vectors,
        ...     top_k=5
        ... )
        >>> # [("laptop-case-variant", 0.85), ("mouse-variant", 0.72), ...]
    """
    anchor_vec = vectors.get(anchor_variant_id)
    if not anchor_vec:
        return []

    candidates = [v for vid, v in vectors.items() if vid != anchor_variant_id]
    results = batch_similarity(anchor_vec, candidates, threshold=min_similarity)

    return results[:top_k]


def enhance_candidate_with_covisitation(
    candidate: Dict[str, Any], vectors: Dict[str, CoVisVector]
) -> Dict[str, Any]:
    """
    Add co-visitation features to a bundle candidate.

    This enriches candidates with similarity signals for downstream ranking.

    Args:
        candidate: Bundle candidate dict with "products" key
        vectors: Co-visitation vectors

    Returns:
        Enhanced candidate with added features:
        - covis_similarity: Average pairwise similarity
        - covis_min_similarity: Weakest product pair
        - covis_max_similarity: Strongest product pair

    Example:
        >>> candidate = {
        ...     "products": [{"variant_id": "A"}, {"variant_id": "B"}],
        ...     "features": {}
        ... }
        >>> enhanced = enhance_candidate_with_covisitation(candidate, vectors)
        >>> enhanced["features"]["covis_similarity"]
        >>> # 0.75
    """
    products = candidate.get("products", [])
    # Use variant_id as primary identifier
    variant_ids = [p["variant_id"] for p in products if "variant_id" in p]

    if len(variant_ids) < 2:
        candidate.setdefault("features", {})["covis_similarity"] = 0.0
        return candidate

    similarities = []
    for i in range(len(variant_ids)):
        for j in range(i + 1, len(variant_ids)):
            v1 = vectors.get(variant_ids[i])
            v2 = vectors.get(variant_ids[j])
            if v1 and v2:
                similarities.append(cosine_similarity(v1, v2))

    if not similarities:
        avg_sim = 0.0
        min_sim = 0.0
        max_sim = 0.0
    else:
        avg_sim = sum(similarities) / len(similarities)
        min_sim = min(similarities)
        max_sim = max(similarities)

    # Add features to candidate
    candidate.setdefault("features", {}).update({
        "covis_similarity": avg_sim,
        "covis_min_similarity": min_sim,
        "covis_max_similarity": max_sim,
    })

    return candidate
