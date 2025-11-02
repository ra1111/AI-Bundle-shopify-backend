"""
Dynamic Objective Scorer V2: Data-Driven Priority Selection

Replaces hard-coded tiers with continuous scoring based on:
- Store maturity metrics (transactions, SKUs, diversity)
- Signal strength per objective (data availability)
- Confidence intervals (statistical reliability)
- Time budget constraints (SLO compliance)
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
import time

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveScore:
    """Score for a single objective with confidence and metadata"""
    objective: str
    benefit_score: float  # 0-1, higher = more beneficial
    confidence: float     # 0-1, higher = more reliable data
    priority: float       # benefit_score × confidence
    data_volume: int      # Number of relevant data points
    metadata: Dict        # Additional scoring details


class ContinuousObjectiveScorer:
    """
    Score objectives continuously based on store data characteristics.

    Philosophy: "Let the data choose the objectives, not hard-coded tiers"
    """

    def __init__(self):
        # Time budget for combo selection (12s out of 20s SLO)
        self.target_budget_ms = 12000

        # Rolling average cost per combo (updated from metrics)
        self.avg_cost_per_combo_ms = 3000  # Conservative default

        # Maximum objectives to consider
        self.max_objectives = 6

    def score_all_objectives(
        self,
        transactions: List[Dict],
        catalog: List[Dict],
        order_lines: List[Dict],
        metrics: Optional[Dict] = None
    ) -> List[ObjectiveScore]:
        """
        Score all objectives based on store data.

        Returns ranked list of ObjectiveScore objects.
        """
        try:
            logger.info(
                f"CONTINUOUS_SCORER: Starting objective scoring | "
                f"transactions={len(transactions) if transactions else 0}, "
                f"catalog={len(catalog) if catalog else 0}, "
                f"order_lines={len(order_lines) if order_lines else 0}"
            )

            scores = []

            # Compute store-level metrics once
            store_metrics = self._compute_store_metrics(transactions, catalog, order_lines)

            logger.debug(
                f"CONTINUOUS_SCORER: Store metrics computed | "
                f"n_transactions={store_metrics.get('n_transactions')}, "
                f"n_skus={store_metrics.get('n_skus')}, "
                f"n_categories={store_metrics.get('n_categories')}"
            )

            # Score each objective
            scores.append(self._score_margin_guard(store_metrics))
            scores.append(self._score_clear_slow_movers(store_metrics))
            scores.append(self._score_increase_aov(store_metrics))
            scores.append(self._score_new_launch(store_metrics))
            scores.append(self._score_seasonal_promo(store_metrics))
            scores.append(self._score_category_bundle(store_metrics))
            scores.append(self._score_gift_box(store_metrics))
            scores.append(self._score_subscription_push(store_metrics))

            # Sort by priority (benefit × confidence) descending
            scores.sort(key=lambda x: x.priority, reverse=True)

            logger.info(
                f"CONTINUOUS_SCORER: Objective scoring complete | "
                f"objectives_scored={len(scores)}, "
                f"top_3=[{', '.join([f'{s.objective}={s.priority:.3f}' for s in scores[:3]])}]"
            )

            return scores

        except Exception as e:
            logger.error(
                f"CONTINUOUS_SCORER: Error scoring objectives: {e} | "
                f"Returning empty score list",
                exc_info=True
            )
            return []  # Safe fallback: empty list

    def select_objectives_with_budget(
        self,
        objective_scores: List[ObjectiveScore],
        rolling_cost_ms: Optional[float] = None
    ) -> List[str]:
        """
        Select top objectives that fit within time budget.

        Args:
            objective_scores: Ranked list of ObjectiveScore
            rolling_cost_ms: Recent average ms per combo (from metrics)

        Returns:
            List of objective names to evaluate
        """
        try:
            logger.info(
                f"CONTINUOUS_SCORER: Budget-aware objective selection started | "
                f"input_objectives={len(objective_scores) if objective_scores else 0}, "
                f"rolling_cost_ms={rolling_cost_ms}, "
                f"current_avg_cost={self.avg_cost_per_combo_ms}ms"
            )

            # Update cost estimate if provided
            if rolling_cost_ms:
                old_cost = self.avg_cost_per_combo_ms
                self.avg_cost_per_combo_ms = rolling_cost_ms
                logger.info(
                    f"CONTINUOUS_SCORER: Updated cost estimate | "
                    f"{old_cost}ms -> {rolling_cost_ms}ms"
                )

            # Calculate max combos within budget
            # Each objective gets 1-2 bundle types, so avg 1.5 combos per objective
            avg_combos_per_objective = 1.5
            cost_per_objective = self.avg_cost_per_combo_ms * avg_combos_per_objective

            max_objectives = max(2, min(
                self.max_objectives,
                int(self.target_budget_ms / cost_per_objective)
            ))

            logger.info(
                f"CONTINUOUS_SCORER: Budget calculation | "
                f"target_budget={self.target_budget_ms}ms, "
                f"cost_per_objective={cost_per_objective:.0f}ms, "
                f"max_objectives={max_objectives}"
            )

            # Select top N objectives by priority
            selected = []
            skipped = []

            for score in objective_scores[:max_objectives]:
                # Only include if priority is meaningful (> 0.1)
                if score.priority > 0.1:
                    selected.append(score.objective)
                    logger.info(
                        f"CONTINUOUS_SCORER: Selected objective | "
                        f"objective={score.objective}, "
                        f"priority={score.priority:.3f}, benefit={score.benefit_score:.3f}, "
                        f"confidence={score.confidence:.3f}, data_volume={score.data_volume}"
                    )
                else:
                    skipped.append((score.objective, score.priority))
                    logger.debug(
                        f"CONTINUOUS_SCORER: Skipped low-priority objective | "
                        f"objective={score.objective}, priority={score.priority:.3f} (threshold=0.1)"
                    )

            if skipped:
                logger.info(
                    f"CONTINUOUS_SCORER: Skipped {len(skipped)} low-priority objectives: "
                    f"{[f'{obj}={pri:.3f}' for obj, pri in skipped]}"
                )

            logger.info(
                f"CONTINUOUS_SCORER: Budget-aware selection complete | "
                f"selected={len(selected)}/{len(objective_scores)} objectives, "
                f"budget={self.target_budget_ms}ms, "
                f"estimated_cost={len(selected) * cost_per_objective:.0f}ms"
            )

            return selected

        except Exception as e:
            logger.error(
                f"CONTINUOUS_SCORER: Error in budget-aware selection: {e} | "
                f"Falling back to top 2 objectives",
                exc_info=True
            )
            # Safe fallback: return top 2 objectives if available
            if objective_scores and len(objective_scores) >= 2:
                return [objective_scores[0].objective, objective_scores[1].objective]
            elif objective_scores and len(objective_scores) == 1:
                return [objective_scores[0].objective]
            else:
                return []

    def _compute_store_metrics(
        self,
        transactions: List[Dict],
        catalog: List[Dict],
        order_lines: List[Dict]
    ) -> Dict:
        """Compute store-level metrics for objective scoring"""

        try:
            n_transactions = len(transactions) if transactions else 0
            n_skus = len(catalog) if catalog else 0
            n_order_lines = len(order_lines) if order_lines else 0

            logger.debug(
                f"CONTINUOUS_SCORER: Computing store metrics | "
                f"n_transactions={n_transactions}, n_skus={n_skus}, n_order_lines={n_order_lines}"
            )

            # Product-level metrics
            sku_to_product = {p.get('sku'): p for p in catalog if p.get('sku')}

            # Margin statistics
            margins = []
            high_margin_count = 0
            for p in catalog:
                margin = p.get('margin', 0) or 0
                if margin > 0:
                    margins.append(margin)
                    if margin > 0.4:  # >40% margin
                        high_margin_count += 1

            margin_variance = float(np.var(margins)) if margins else 0
            margin_iqr = float(stats.iqr(margins)) if len(margins) > 2 else 0
            pct_high_margin = high_margin_count / max(n_skus, 1)

            # Velocity / slow mover analysis
            sku_sales = {}
            for line in order_lines:
                sku = line.get('sku')
                if sku:
                    sku_sales[sku] = sku_sales.get(sku, 0) + line.get('quantity', 1)

            if sku_sales:
                velocities = list(sku_sales.values())
                velocity_median = float(np.median(velocities))
                slow_movers = sum(1 for v in velocities if v < velocity_median * 0.5)
                pct_slow_movers = slow_movers / max(len(velocities), 1)
            else:
                velocity_median = 0
                pct_slow_movers = 0
                slow_movers = 0

            # New product analysis (created in last 30 days)
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=30)
            new_skus = 0
            for p in catalog or []:
                created = p.get('created_at')
                if created and isinstance(created, datetime) and created > cutoff_date:
                    new_skus += 1
            pct_new_skus = new_skus / max(n_skus, 1)

            # AOV and attach rate
            if transactions:
                cart_sizes = [t.get('line_item_count', 1) for t in transactions]
                avg_items_per_order = float(np.mean(cart_sizes))
                items_per_order_std = float(np.std(cart_sizes))
            else:
                avg_items_per_order = 1.0
                items_per_order_std = 0.0

            # Category diversity
            categories = set()
            for p in catalog or []:
                cat = p.get('product_category') or p.get('category')
                if cat:
                    categories.add(cat)
            n_categories = len(categories)

            metrics = {
                'n_transactions': n_transactions,
                'n_skus': n_skus,
                'n_order_lines': n_order_lines,
                'n_categories': n_categories,

                # Margin metrics
                'margin_variance': margin_variance,
                'margin_iqr': margin_iqr,
                'pct_high_margin': pct_high_margin,
                'high_margin_count': high_margin_count,

                # Velocity metrics
                'pct_slow_movers': pct_slow_movers,
                'slow_mover_count': slow_movers,
                'velocity_median': velocity_median,

                # New product metrics
                'pct_new_skus': pct_new_skus,
                'new_sku_count': new_skus,

                # AOV metrics
                'avg_items_per_order': avg_items_per_order,
                'items_per_order_std': items_per_order_std,

                # Category metrics
                'category_diversity': n_categories / max(n_skus, 1),
            }

            logger.debug(
                f"CONTINUOUS_SCORER: Store metrics computed successfully | "
                f"n_transactions={n_transactions}, n_skus={n_skus}, "
                f"margin_iqr={margin_iqr:.3f}, pct_slow_movers={pct_slow_movers:.3f}"
            )

            return metrics

        except Exception as e:
            logger.error(
                f"CONTINUOUS_SCORER: Error computing store metrics: {e} | "
                f"Returning default metrics with zeros",
                exc_info=True
            )
            # Return safe default metrics (all zeros)
            return {
                'n_transactions': 0,
                'n_skus': 0,
                'n_order_lines': 0,
                'n_categories': 0,
                'margin_variance': 0.0,
                'margin_iqr': 0.0,
                'pct_high_margin': 0.0,
                'high_margin_count': 0,
                'pct_slow_movers': 0.0,
                'slow_mover_count': 0,
                'velocity_median': 0.0,
                'pct_new_skus': 0.0,
                'new_sku_count': 0,
                'avg_items_per_order': 1.0,
                'items_per_order_std': 0.0,
                'category_diversity': 0.0,
            }

    def _score_margin_guard(self, metrics: Dict) -> ObjectiveScore:
        """Score margin_guard objective"""

        # Benefit: High if margin variance is high (opportunity to protect margins)
        margin_z = min(1.0, metrics['margin_iqr'] / 0.3)  # Normalize to 0-1
        high_margin_z = metrics['pct_high_margin']

        benefit = 0.6 * margin_z + 0.4 * high_margin_z

        # Confidence: Based on data volume
        confidence = self._wilson_confidence(
            metrics['high_margin_count'],
            metrics['n_skus']
        )

        return ObjectiveScore(
            objective='margin_guard',
            benefit_score=float(benefit),
            confidence=float(confidence),
            priority=float(benefit * confidence),
            data_volume=metrics['high_margin_count'],
            metadata={
                'margin_iqr': metrics['margin_iqr'],
                'pct_high_margin': metrics['pct_high_margin']
            }
        )

    def _score_clear_slow_movers(self, metrics: Dict) -> ObjectiveScore:
        """Score clear_slow_movers objective"""

        # Benefit: High if many slow movers exist
        slow_mover_z = metrics['pct_slow_movers']

        benefit = slow_mover_z

        # Confidence: Based on transaction volume
        confidence = self._wilson_confidence(
            metrics['slow_mover_count'],
            metrics['n_skus']
        )

        return ObjectiveScore(
            objective='clear_slow_movers',
            benefit_score=float(benefit),
            confidence=float(confidence),
            priority=float(benefit * confidence),
            data_volume=metrics['slow_mover_count'],
            metadata={
                'pct_slow_movers': metrics['pct_slow_movers'],
                'velocity_median': metrics['velocity_median']
            }
        )

    def _score_increase_aov(self, metrics: Dict) -> ObjectiveScore:
        """Score increase_aov objective"""

        # Benefit: High if attach rate can be improved (low avg items per order)
        # Inverse relationship: lower items per order = higher opportunity
        attach_opportunity = max(0, (3.0 - metrics['avg_items_per_order']) / 3.0)

        # Also high if there's variance (some orders big, some small)
        variance_z = min(1.0, metrics['items_per_order_std'] / 2.0)

        benefit = 0.7 * attach_opportunity + 0.3 * variance_z

        # Confidence: Based on transaction count
        confidence = min(1.0, metrics['n_transactions'] / 100.0)

        return ObjectiveScore(
            objective='increase_aov',
            benefit_score=float(benefit),
            confidence=float(confidence),
            priority=float(benefit * confidence),
            data_volume=metrics['n_transactions'],
            metadata={
                'avg_items_per_order': metrics['avg_items_per_order'],
                'items_std': metrics['items_per_order_std']
            }
        )

    def _score_new_launch(self, metrics: Dict) -> ObjectiveScore:
        """Score new_launch objective"""

        # Benefit: High if there are new SKUs to promote
        new_sku_z = min(1.0, metrics['pct_new_skus'] * 5)  # 20% new = full score

        benefit = new_sku_z

        # Confidence: Only confident if we have meaningful new SKU count
        confidence = self._wilson_confidence(
            metrics['new_sku_count'],
            max(10, metrics['new_sku_count'])  # Need at least a few new products
        )

        return ObjectiveScore(
            objective='new_launch',
            benefit_score=float(benefit),
            confidence=float(confidence),
            priority=float(benefit * confidence),
            data_volume=metrics['new_sku_count'],
            metadata={
                'pct_new_skus': metrics['pct_new_skus'],
                'new_sku_count': metrics['new_sku_count']
            }
        )

    def _score_seasonal_promo(self, metrics: Dict) -> ObjectiveScore:
        """Score seasonal_promo objective"""

        # Lower priority, moderate benefit
        benefit = 0.4

        # Confidence based on transaction count
        confidence = min(0.8, metrics['n_transactions'] / 50.0)

        return ObjectiveScore(
            objective='seasonal_promo',
            benefit_score=float(benefit),
            confidence=float(confidence),
            priority=float(benefit * confidence),
            data_volume=metrics['n_transactions'],
            metadata={}
        )

    def _score_category_bundle(self, metrics: Dict) -> ObjectiveScore:
        """Score category_bundle objective"""

        # Benefit: High if category diversity is high
        diversity_z = min(1.0, metrics['category_diversity'] * 2)  # 50% diversity = full

        benefit = diversity_z * 0.5  # Medium priority even at full diversity

        # Confidence based on category count
        confidence = min(1.0, metrics['n_categories'] / 5.0)

        return ObjectiveScore(
            objective='category_bundle',
            benefit_score=float(benefit),
            confidence=float(confidence),
            priority=float(benefit * confidence),
            data_volume=metrics['n_categories'],
            metadata={'category_diversity': metrics['category_diversity']}
        )

    def _score_gift_box(self, metrics: Dict) -> ObjectiveScore:
        """Score gift_box objective"""

        # Lower priority, small benefit
        benefit = 0.3

        # Confidence based on SKU count
        confidence = min(0.7, metrics['n_skus'] / 20.0)

        return ObjectiveScore(
            objective='gift_box',
            benefit_score=float(benefit),
            confidence=float(confidence),
            priority=float(benefit * confidence),
            data_volume=metrics['n_skus'],
            metadata={}
        )

    def _score_subscription_push(self, metrics: Dict) -> ObjectiveScore:
        """Score subscription_push objective"""

        # Low priority by default (would need subscription SKU detection)
        benefit = 0.2
        confidence = 0.5

        return ObjectiveScore(
            objective='subscription_push',
            benefit_score=float(benefit),
            confidence=float(confidence),
            priority=float(benefit * confidence),
            data_volume=0,
            metadata={}
        )

    def _wilson_confidence(self, successes: int, total: int) -> float:
        """
        Compute confidence using Wilson score interval width.

        Returns 1 - interval_width (so narrow interval = high confidence)
        """
        if total == 0:
            return 0.0

        p = successes / total
        n = total

        # Wilson score with 95% confidence (z=1.96)
        z = 1.96

        denominator = 1 + z**2 / n
        center = (p + z**2 / (2*n)) / denominator

        margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / denominator

        # Confidence is inverse of interval width
        interval_width = 2 * margin
        confidence = max(0.0, 1.0 - interval_width)

        return float(confidence)


# Global instance
continuous_objective_scorer = ContinuousObjectiveScorer()
