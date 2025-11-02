# Continuous Optimization Design: Data-Driven Bundle Generation

## Philosophy

**"Let the data choose the objectives, not hard-coded tiers"**

Replace fixed tier thresholds with continuous, confidence-weighted scoring that adapts to each store's unique characteristics in real-time.

---

## 1. Continuous Objective Scoring

### Current Approach (Fixed Tiers)
```python
if txn_count < 50:
    objectives = ['margin_guard', 'increase_aov']  # Hard-coded
elif txn_count < 200:
    objectives = ['margin_guard', 'clear_slow_movers', 'increase_aov']
else:
    objectives = ['margin_guard', 'clear_slow_movers', 'increase_aov', 'new_launch']
```

### New Approach (Continuous Scoring)
```python
# Compute benefit score for each objective
S_margin_guard = margin_variance_z + pct_high_margin_sku
S_clear_slow = pct_slow_movers + inventory_age_z
S_increase_aov = (3.0 - avg_items_per_order) / 3.0 + items_std_z
S_new_launch = pct_new_sku_30d * 5  # 20% new = 1.0

# Compute confidence using Wilson score interval
conf_margin = 1 - wilson_ci_width(high_margin_count, total_skus)
conf_slow = 1 - wilson_ci_width(slow_mover_count, total_skus)
conf_aov = min(1.0, txn_count / 100)
conf_new = 1 - wilson_ci_width(new_sku_count, max(10, new_sku_count))

# Priority = Benefit × Confidence
priority_margin = S_margin_guard * conf_margin
priority_slow = S_clear_slow * conf_slow
priority_aov = S_increase_aov * conf_aov
priority_new = S_new_launch * conf_new

# Rank and select top objectives
objectives = top_k_by_priority(all_objectives, k=budget_aware_k)
```

### Key Metrics Per Objective

| Objective | Benefit Score | Confidence Factor | Data Required |
|-----------|---------------|-------------------|---------------|
| **margin_guard** | `margin_iqr/0.3 + pct_high_margin` | Wilson CI on high-margin SKUs | Product margins |
| **clear_slow_movers** | `pct_slow_movers` | Wilson CI on slow-mover count | Sales velocity |
| **increase_aov** | `(3 - avg_items)/3 + std_items/2` | `min(1, txns/100)` | Transaction history |
| **new_launch** | `pct_new_sku_30d * 5` | Wilson CI on new SKU count | Product created_at |
| **seasonal_promo** | `0.4` (static) | `min(0.8, txns/50)` | Transaction count |
| **category_bundle** | `category_diversity * 0.5` | `min(1, n_categories/5)` | Category metadata |
| **gift_box** | `0.3` (static) | `min(0.7, skus/20)` | SKU count |
| **subscription_push** | `0.2` (static) | `0.5` (low) | Subscription tags |

---

## 2. Time-Budget Driven Selection

### Objective: Never Exceed SLO

**Target**: 12s out of 20s SLO (60% budget for Phase 3)

```python
target_budget_ms = 12_000  # 12 seconds
avg_cost_per_combo_ms = rolling_mean(last_50_runs)  # e.g., 3000ms

# Each objective gets 1.5 combos on average (1-2 bundle types)
avg_combos_per_objective = 1.5
cost_per_objective = avg_cost_per_combo_ms * avg_combos_per_objective

# Max objectives that fit within budget
max_objectives = clamp(
    floor(target_budget_ms / cost_per_objective),
    min=2,    # Always try at least 2
    max=6     # Cap at 6 to avoid quality dilution
)
```

### Rolling Cost Estimation

Track combo execution time in metrics:
```python
metrics["phase_timings"]["combos"] = [
    {"objective": "margin_guard", "type": "FBT", "duration_ms": 2834},
    {"objective": "margin_guard", "type": "FIXED", "duration_ms": 3102},
    {"objective": "increase_aov", "type": "MIX_MATCH", "duration_ms": 2945},
    ...
]

# Update rolling average (exponential moving average)
alpha = 0.2  # Weight for new observations
new_avg = alpha * current_run_avg + (1 - alpha) * historical_avg
```

### Adaptive Behavior

| Store Evolution | Avg Cost/Combo | Max Objectives | Total Combos | Total Time |
|----------------|----------------|----------------|--------------|------------|
| **Week 1** (10 txns) | 1500ms | 5 | 8 | 12s |
| **Week 4** (50 txns) | 2500ms | 3 | 5 | 12.5s |
| **Week 12** (200 txns) | 3500ms | 2 | 4 | 14s |
| **After optimization** (200 txns, faster) | 2000ms | 4 | 6 | 12s |

System **automatically scales up** as it gets faster!

---

## 3. Cold-Start Rescue

### Problem: Tiny Datasets (< 10 txns)

Current: Skip ML phase → 0 bundles → poor UX

### Solution: Semantic Bundling

Use LLM embeddings to create curated bundles without transaction data:

```python
def cold_start_rescue(catalog, embeddings, limit=4):
    """Generate semantic bundles for cold-start scenarios"""

    bundles = []

    # Strategy 1: Price-banded complementaries
    for anchor_product in top_k_by_price(catalog, k=10):
        # Find semantically similar products in compatible price range
        similar = find_similar_with_price_filter(
            anchor_product,
            embeddings,
            min_similarity=0.4,
            max_similarity=0.7,  # Complementary, not identical
            price_ratio_range=(0.5, 2.0)  # Within 2x price
        )

        if len(similar) >= 1:
            bundles.append({
                'type': 'FBT',
                'objective': 'increase_aov',
                'products': [anchor_product] + similar[:2],
                'cold_start': True,
                'generation_method': 'semantic_price_banded'
            })

    # Strategy 2: Category compatibility
    for category in catalog_categories:
        products_in_cat = filter_by_category(catalog, category)
        if len(products_in_cat) >= 3:
            # Pick top 3 by price diversity
            diverse_set = select_price_diverse(products_in_cat, k=3)
            bundles.append({
                'type': 'MIX_MATCH',
                'objective': 'increase_aov',
                'products': diverse_set,
                'cold_start': True,
                'generation_method': 'category_diversity'
            })

    # Strategy 3: Margin-optimized sets
    high_margin = filter_by_margin(catalog, min_margin=0.4)
    if len(high_margin) >= 2:
        bundles.append({
            'type': 'FIXED',
            'objective': 'margin_guard',
            'products': high_margin[:3],
            'cold_start': True,
            'generation_method': 'high_margin_curation'
        })

    return scored_and_ranked(bundles)[:limit]
```

### Cold-Start Guardrails

- **Category compatibility**: Don't bundle shoes + electronics
- **Price ratio**: Keep within [0.5×, 2×] of anchor
- **Margin floor**: Ensure bundle margin > 20%
- **Inventory check**: Only include in-stock items
- **Mark as cold_start**: Track separately in analytics

---

## 4. Diversity & Quality Gates

### MMR (Maximal Marginal Relevance) for Diversity

Avoid generating 5 near-identical bundles:

```python
def select_diverse_bundles(candidates, k=10, lambda_param=0.7):
    """Select k diverse bundles using MMR"""

    selected = []
    remaining = candidates.copy()

    # Start with highest-scored bundle
    selected.append(max(remaining, key=lambda b: b['score']))
    remaining.remove(selected[0])

    while len(selected) < k and remaining:
        # For each candidate, compute MMR score
        mmr_scores = []
        for candidate in remaining:
            # Relevance: hybrid score
            relevance = candidate['score']

            # Diversity: min similarity to already selected
            max_similarity = max(
                product_set_similarity(candidate['products'], s['products'])
                for s in selected
            )

            # MMR: balance relevance and diversity
            mmr = lambda_param * relevance - (1 - lambda_param) * max_similarity
            mmr_scores.append((candidate, mmr))

        # Pick highest MMR
        best = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(best)
        remaining.remove(best)

    return selected
```

### Policy Gates (Pre-Selection Filters)

```python
def apply_policy_gates(bundles):
    """Filter bundles through business policy gates"""

    valid_bundles = []

    for bundle in bundles:
        # Gate 1: Inventory availability
        if not all_products_in_stock(bundle['products']):
            log_rejection(bundle, reason="out_of_stock")
            continue

        # Gate 2: Margin floor
        bundle_margin = compute_bundle_margin(bundle)
        if bundle_margin < 0.15:  # 15% minimum
            log_rejection(bundle, reason="margin_too_low")
            continue

        # Gate 3: Min support for association rules
        if bundle.get('generation_method') == 'association_rules':
            if bundle.get('support', 0) < 0.01:  # 1% min support
                log_rejection(bundle, reason="low_support")
                continue

        # Gate 4: Price sanity
        if not validate_price_ladder(bundle):
            log_rejection(bundle, reason="invalid_pricing")
            continue

        # Gate 5: Category compatibility (for FBT/FIXED)
        if bundle['type'] in ['FBT', 'FIXED']:
            if not validate_category_compatibility(bundle['products']):
                log_rejection(bundle, reason="incompatible_categories")
                continue

        valid_bundles.append(bundle)

    return valid_bundles
```

---

## 5. Enhanced Hybrid Scoring (Per-Category α/β)

### Current: Store-Level α/β

```python
if txn_count < 300:
    alpha, beta = 0.6, 0.2  # LLM leads
elif txn_count < 1200:
    alpha, beta = 0.4, 0.4  # Balanced
else:
    alpha, beta = 0.2, 0.6  # Data leads
```

### New: Category-Level α/β

```python
def get_alpha_beta_for_bundle(bundle, store_txns, category_txns):
    """Compute α/β based on anchor product's category maturity"""

    anchor_sku = bundle['products'][0]
    category = get_category(anchor_sku)

    # Category-specific transaction count
    cat_txn_count = count_transactions_with_category(category_txns, category)

    # Smooth transition using sigmoid
    def sigmoid_alpha(x, midpoint=300, steepness=0.01):
        return 0.6 / (1 + np.exp(steepness * (x - midpoint)))

    alpha = sigmoid_alpha(cat_txn_count)
    beta = 0.8 - alpha  # Complementary
    gamma = 0.2  # Business signals constant

    return alpha, beta, gamma
```

### Benefits

- **Mature categories** (electronics with 500 txns) use data-heavy scoring even in a young store
- **New categories** (just launched accessories with 10 txns) use LLM-heavy scoring even in a mature store
- **Smooth transitions** instead of hard thresholds

---

## 6. Simple Orchestrator (Implementation)

```python
async def generate_bundles_continuous(csv_upload_id: str):
    """
    Continuous, budget-aware bundle generation.

    Philosophy: Let data choose objectives, budget choose count.
    """

    # 1. Load data
    transactions = await load_transactions(csv_upload_id)
    catalog = await load_catalog(csv_upload_id)
    order_lines = await load_order_lines(csv_upload_id)

    # 2. Score all objectives (continuous)
    scorer = ContinuousObjectiveScorer()
    objective_scores = scorer.score_all_objectives(
        transactions, catalog, order_lines
    )

    # 3. Get rolling cost estimate
    rolling_cost_ms = await get_rolling_avg_cost_per_combo(csv_upload_id)

    # 4. Select objectives within time budget
    selected_objectives = scorer.select_objectives_with_budget(
        objective_scores,
        rolling_cost_ms
    )

    # 5. Build combo list with intelligent type mapping
    combos = []
    for obj_name in selected_objectives:
        bundle_types = get_bundle_types_for_objective(obj_name)  # 1-2 types
        for btype in bundle_types:
            combos.append((obj_name, btype))

    # 6. Cold-start rescue if needed
    if len(combos) == 0 and len(transactions) < 10:
        logger.info("Cold-start rescue: generating semantic bundles")
        embeddings = await llm_embedding_engine.get_embeddings_batch(catalog)
        return cold_start_rescue(catalog, embeddings, limit=4)

    # 7. Generate candidates for each combo
    results = []
    start_time = time.time()

    for obj, btype in combos:
        # Check time budget
        elapsed_ms = (time.time() - start_time) * 1000
        if elapsed_ms > scorer.target_budget_ms:
            logger.warning(f"Budget exceeded at {elapsed_ms}ms, stopping early")
            break

        # Generate candidates
        candidates = await generate_candidates(
            objective=obj,
            bundle_type=btype,
            transactions=transactions,
            catalog=catalog,
            embeddings=embeddings
        )

        # Score with category-aware α/β
        for cand in candidates:
            alpha, beta, gamma = get_alpha_beta_for_bundle(
                cand, transactions, category_txns
            )
            cand['hybrid_score'] = hybrid_score(
                cand, alpha, beta, gamma
            )

        results.extend(candidates)

    # 8. Apply policy gates
    valid_bundles = apply_policy_gates(results)

    # 9. Select diverse bundles with MMR
    final_bundles = select_diverse_bundles(
        valid_bundles,
        k=10,
        lambda_param=0.7  # 70% relevance, 30% diversity
    )

    # 10. Log metrics
    await log_generation_metrics(
        csv_upload_id,
        selected_objectives=selected_objectives,
        objective_scores=objective_scores,
        combos_attempted=len(combos),
        candidates_generated=len(results),
        bundles_after_gates=len(valid_bundles),
        final_bundle_count=len(final_bundles),
        duration_ms=(time.time() - start_time) * 1000
    )

    return final_bundles
```

---

## 7. Measurement & Observability

### Logs to Capture

```python
{
  "csv_upload_id": "...",
  "objective_selection": {
    "margin_guard": {"benefit": 0.72, "confidence": 0.85, "priority": 0.61, "rank": 1},
    "clear_slow": {"benefit": 0.65, "confidence": 0.78, "priority": 0.51, "rank": 2},
    "increase_aov": {"benefit": 0.58, "confidence": 0.92, "priority": 0.53, "rank": 3},
    "new_launch": {"benefit": 0.21, "confidence": 0.35, "priority": 0.07, "rank": 7}
  },
  "time_budget": {
    "target_ms": 12000,
    "rolling_cost_per_combo_ms": 2834,
    "max_objectives": 3,
    "actual_duration_ms": 11245
  },
  "combos_evaluated": [
    {"objective": "margin_guard", "type": "FBT", "candidates": 12, "duration_ms": 2834},
    {"objective": "margin_guard", "type": "FIXED", "candidates": 8, "duration_ms": 3102},
    {"objective": "increase_aov", "type": "MIX_MATCH", "candidates": 15, "duration_ms": 2945}
  ],
  "policy_gates": {
    "total_candidates": 35,
    "rejected_out_of_stock": 5,
    "rejected_low_margin": 3,
    "rejected_low_support": 2,
    "rejected_price_invalid": 1,
    "passed": 24
  },
  "diversity_selection": {
    "candidates_after_gates": 24,
    "mmr_lambda": 0.7,
    "final_bundle_count": 10
  }
}
```

### Offline Metrics

- **HitRate@K**: How often does the generated bundle appear in actual purchases?
- **NDCG@K**: Normalized Discounted Cumulative Gain (ranking quality)
- **Holdout validation**: Reserve 20% of data, measure prediction accuracy

### Online Metrics

- **Attach rate lift**: % increase in bundle add-to-cart vs baseline
- **AOV lift**: Average order value increase
- **Conversion rate**: Bundle view → purchase
- **Per-objective performance**: Track which objectives drive most revenue

---

## 8. Expected Performance

| Store Type | Txns | Old Combos | New Combos | Old Time | New Time | Quality Impact |
|------------|------|------------|------------|----------|----------|----------------|
| **Cold-start** | 0 | 0 (failed) | 4 (semantic) | 5s | 8s | +∞ (0→4 bundles) |
| **Tiny** | 10 | 4 (fixed) | 2-3 (adaptive) | 15s | 6-9s | Same (data-driven) |
| **Small** | 50 | 6 (fixed) | 3-4 (adaptive) | 25s | 10-14s | Better (confidence-weighted) |
| **Medium** | 200 | 8 (fixed) | 4-5 (adaptive) | 35s | 12-18s | Better (category-aware α/β) |
| **Large** | 1000 | 8 (fixed) | 5-6 (adaptive) | 40s | 15-22s | Better (mature signals) |

### Key Improvements

1. **Always within budget**: 12-22s vs 15-40s (and sometimes 330s timeout)
2. **Adaptive to growth**: Automatically explores more as pipeline gets faster
3. **Better quality**: Confidence weighting prevents low-signal objectives
4. **Cold-start works**: 4 semantic bundles vs 0 bundles
5. **Per-category intelligence**: Mature categories use data even in young stores

---

## 9. Implementation Roadmap

### Phase 1: Continuous Scoring (This PR)
- ✅ Implement `ContinuousObjectiveScorer`
- ✅ Wilson confidence intervals
- ✅ Benefit × Confidence priority

### Phase 2: Time-Budget Selection (Next PR)
- Add rolling cost tracking
- Implement budget-aware objective selection
- Add early termination on budget exceeded

### Phase 3: Cold-Start Rescue (Next PR)
- Implement semantic bundling
- Add price-banded complementaries
- Add category compatibility checks

### Phase 4: Diversity & Gates (Next PR)
- Implement MMR selection
- Add policy gates (inventory, margin, support)
- Add price sanity validation

### Phase 5: Category-Aware α/β (Next PR)
- Compute per-category transaction counts
- Implement sigmoid α/β transition
- Add category-level metrics

### Phase 6: Observability (Next PR)
- Enhanced logging
- Metrics dashboard
- A/B testing framework

---

## 10. Summary

**From**: Hard-coded tiers (Tiny/Small/Medium/Large)
**To**: Continuous, confidence-weighted, budget-aware selection

**Benefits**:
- ✅ Adapts to each store uniquely
- ✅ Never exceeds SLO (12s target)
- ✅ Works in cold-start scenarios
- ✅ Improves quality with data growth
- ✅ Automatically scales with optimization

**Philosophy**: "Let the data choose the objectives, the budget choose the count, and the confidence ensure quality"
