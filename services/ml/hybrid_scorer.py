"""
Hybrid Scoring System: LLM Semantic + Transactional Signals

Combines:
- α × LLM semantic similarity (product understanding)
- β × Transactional lift (real purchase behavior)
- γ × Business signals (margin, inventory, pricing)

Dynamically adjusts weights based on data availability.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Weights for hybrid scoring model"""
    alpha: float  # LLM semantic similarity weight
    beta: float   # Transactional lift weight
    gamma: float  # Business signal weight

    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = self.alpha + self.beta + self.gamma
        if not (0.99 <= total <= 1.01):  # Allow small floating point errors
            logger.warning(f"Scoring weights don't sum to 1.0: α={self.alpha}, β={self.beta}, γ={self.gamma}, total={total}")


class HybridScorer:
    """
    Hybrid scoring engine that combines LLM + transactional + business signals

    Philosophy: "LLM drives, data validates"
    - Small stores (< 300 txns): LLM leads (α=0.6)
    - Medium stores (300-1200 txns): Balanced (α=0.4)
    - Large stores (> 1200 txns): Data leads (α=0.2)
    """

    def __init__(self):
        """Initialize the hybrid scorer with default configuration"""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Weight tiers based on transaction volume
        self.weight_tiers = {
            "small": ScoringWeights(alpha=0.6, beta=0.2, gamma=0.2),   # < 300 txns: Trust LLM
            "medium": ScoringWeights(alpha=0.4, beta=0.4, gamma=0.2),  # 300-1200 txns: Balanced
            "large": ScoringWeights(alpha=0.2, beta=0.6, gamma=0.2),   # > 1200 txns: Trust data
        }

        # Thresholds for tier classification
        self.tier_thresholds = {
            "small_max": 300,
            "medium_max": 1200,
        }

        self.logger.debug("HybridScorer initialized with weight tiers: %s", self.weight_tiers)

    def get_weights_for_dataset(self, transaction_count: int) -> ScoringWeights:
        """
        Dynamically determine scoring weights based on transaction count

        Args:
            transaction_count: Number of transactions in 60-day window

        Returns:
            ScoringWeights with optimal α, β, γ values
        """
        try:
            # Validate transaction_count
            if not isinstance(transaction_count, (int, float)) or transaction_count < 0:
                logger.warning(
                    f"HYBRID_SCORER: Invalid transaction_count={transaction_count}, using 0"
                )
                transaction_count = 0

            # Determine tier based on thresholds
            if transaction_count < self.tier_thresholds["small_max"]:
                tier = "small"
            elif transaction_count < self.tier_thresholds["medium_max"]:
                tier = "medium"
            else:
                tier = "large"

            weights = self.weight_tiers[tier]

            logger.info(
                f"HYBRID_SCORER: Dynamic weights selected | "
                f"transaction_count={transaction_count}, tier={tier}, "
                f"α={weights.alpha} (LLM), β={weights.beta} (transactional), γ={weights.gamma} (business)"
            )

            return weights

        except Exception as e:
            logger.error(
                f"HYBRID_SCORER: Error selecting weights: {e} | "
                f"Falling back to balanced weights (α=0.4, β=0.4, γ=0.2)",
                exc_info=True
            )
            # Fallback to medium tier weights
            return self.weight_tiers.get("medium", ScoringWeights(alpha=0.4, beta=0.4, gamma=0.2))

    def score_bundle(
        self,
        bundle_products: List[str],
        llm_similarity: float,
        transactional_lift: float,
        business_signals: Dict[str, float],
        transaction_count: int
    ) -> Dict[str, any]:
        """
        Score a bundle using hybrid approach

        Args:
            bundle_products: List of SKUs in bundle
            llm_similarity: LLM semantic similarity score (0-1)
            transactional_lift: Association rule lift (typically 1.0-10.0)
            business_signals: Dict of business metrics (margin, inventory_velocity, etc.)
            transaction_count: Number of transactions (for weight selection)

        Returns:
            Dict with final score and breakdown
        """
        weights = self.get_weights_for_dataset(transaction_count)

        # Normalize signals to 0-1 range
        llm_score = self._normalize_llm_similarity(llm_similarity)
        transactional_score = self._normalize_transactional_lift(transactional_lift)
        business_score = self._compute_business_score(business_signals)

        # Hybrid score
        final_score = (
            weights.alpha * llm_score +
            weights.beta * transactional_score +
            weights.gamma * business_score
        )

        return {
            "final_score": float(final_score),
            "breakdown": {
                "llm_score": float(llm_score),
                "llm_weight": weights.alpha,
                "llm_contribution": float(weights.alpha * llm_score),

                "transactional_score": float(transactional_score),
                "transactional_weight": weights.beta,
                "transactional_contribution": float(weights.beta * transactional_score),

                "business_score": float(business_score),
                "business_weight": weights.gamma,
                "business_contribution": float(weights.gamma * business_score),
            },
            "weights_used": {
                "alpha": weights.alpha,
                "beta": weights.beta,
                "gamma": weights.gamma,
                "transaction_count": transaction_count
            }
        }

    def _normalize_llm_similarity(self, similarity: float) -> float:
        """
        Normalize LLM cosine similarity to 0-1 range

        Cosine similarity is already 0-1 for embeddings, but we apply
        a slight transformation to spread out the distribution
        """
        # Clip to 0-1 range
        similarity = max(0.0, min(1.0, similarity))

        # Apply power transformation to spread out mid-range values
        # This makes 0.7 similarity more distinct from 0.8
        return float(similarity ** 0.8)

    def _normalize_transactional_lift(self, lift: float) -> float:
        """
        Normalize transactional lift to 0-1 range

        Lift typically ranges from 1.0 (no association) to 10+ (strong association)
        We use log transformation to handle the wide range
        """
        if lift <= 1.0:
            return 0.0  # No positive association

        # Log transformation: lift of 2.0 → 0.5, lift of 4.0 → 0.75, lift of 10.0 → 0.92
        normalized = np.log1p(lift - 1.0) / np.log1p(9.0)  # Normalize to 0-1

        return float(min(1.0, normalized))

    def _compute_business_score(self, signals: Dict[str, float]) -> float:
        """
        Compute business score from various business signals

        Signals may include:
        - margin: Product profit margin (0-1, higher is better)
        - inventory_velocity: How fast product sells (0-1, higher is better)
        - discount_attractiveness: How good the discount is (0-1, higher is better)
        - price_competitiveness: How competitive pricing is (0-1, higher is better)
        """
        if not signals:
            return 0.5  # Neutral score if no signals

        # Simple average of all business signals
        # Could be made more sophisticated with weighted average
        values = [v for v in signals.values() if isinstance(v, (int, float))]

        if not values:
            return 0.5

        return float(np.mean(values))

    def rank_candidates(
        self,
        candidates: List[Dict],
        transaction_count: int
    ) -> List[Dict]:
        """
        Rank bundle candidates using hybrid scoring

        Args:
            candidates: List of bundle candidate dicts with fields:
                - products: List of SKUs
                - llm_similarity: LLM similarity score
                - transactional_lift: Association lift (optional)
                - business_signals: Dict of business metrics (optional)
            transaction_count: Number of transactions for weight selection

        Returns:
            Ranked list of candidates with scores
        """
        try:
            logger.info(
                f"HYBRID_SCORER: Ranking candidates | "
                f"candidate_count={len(candidates) if candidates else 0}, "
                f"transaction_count={transaction_count}"
            )

            scored_candidates = []

            for idx, candidate in enumerate(candidates if candidates else []):
                try:
                    # Extract signals
                    llm_sim = candidate.get('llm_similarity', 0.5)
                    trans_lift = candidate.get('transactional_lift', 1.0)
                    business_sigs = candidate.get('business_signals', {})

                    # Score the bundle
                    score_result = self.score_bundle(
                        bundle_products=candidate.get('products', []),
                        llm_similarity=llm_sim,
                        transactional_lift=trans_lift,
                        business_signals=business_sigs,
                        transaction_count=transaction_count
                    )

                    # Add score to candidate
                    candidate['hybrid_score'] = score_result['final_score']
                    candidate['score_breakdown'] = score_result['breakdown']
                    candidate['weights_used'] = score_result['weights_used']

                    scored_candidates.append(candidate)

                except Exception as e:
                    logger.warning(
                        f"HYBRID_SCORER: Error scoring candidate {idx}: {e} | "
                        f"Skipping candidate"
                    )
                    continue

            # Sort by final score descending
            scored_candidates.sort(key=lambda x: x.get('hybrid_score', 0), reverse=True)

            if scored_candidates:
                top_score = scored_candidates[0].get('hybrid_score', 0)
                bottom_score = scored_candidates[-1].get('hybrid_score', 0)
                logger.info(
                    f"HYBRID_SCORER: Ranking complete | "
                    f"ranked_candidates={len(scored_candidates)}, "
                    f"score_range=[{bottom_score:.3f}, {top_score:.3f}]"
                )
            else:
                logger.warning("HYBRID_SCORER: No candidates successfully scored")

            return scored_candidates

        except Exception as e:
            logger.error(
                f"HYBRID_SCORER: Error ranking candidates: {e} | "
                f"Returning empty list",
                exc_info=True
            )
            return []

    def merge_candidates_from_sources(
        self,
        llm_candidates: List[Dict],
        transactional_candidates: List[Dict],
        transaction_count: int
    ) -> List[Dict]:
        """
        Merge candidates from LLM and transactional sources, removing duplicates

        Args:
            llm_candidates: Candidates generated by LLM semantic similarity
            transactional_candidates: Candidates from association rules
            transaction_count: Number of transactions

        Returns:
            Merged and scored list of unique candidates
        """
        # Create lookup by product set
        seen_product_sets = set()
        merged = []

        def get_product_set_key(products: List[str]) -> str:
            """Create unique key for product set (sorted tuple)"""
            return "|".join(sorted(products))

        # Add all candidates, tracking seen product sets
        for candidate in llm_candidates + transactional_candidates:
            products = candidate.get('products', [])
            if not products:
                continue

            key = get_product_set_key(products)
            if key in seen_product_sets:
                continue  # Skip duplicate

            seen_product_sets.add(key)
            merged.append(candidate)

        # Rank the merged candidates
        return self.rank_candidates(merged, transaction_count)


# Global instance
hybrid_scorer = HybridScorer()
