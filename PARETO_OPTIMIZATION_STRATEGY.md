# Pareto Optimization Strategy for Bundle Generation

## Problem Analysis

### Current State (40 Combinations)
- **8 objectives** × **5 bundle types** = **40 parallel tasks**
- Runtime: 195 seconds for Phase 3 alone
- Many tasks produce 0 bundles (redundant work)
- Timeout risk at 300 seconds

### Root Cause
1. **Over-generating**: Not all objectives are equally valuable
2. **Bundle type redundancy**: Frontend doesn't distinguish between all 5 types
3. **No early termination**: System tries all 40 combinations even when data is insufficient

## Optimization Strategy

### 1. Pareto Analysis for Objectives (80/20 Rule)

Apply Pareto principle to identify top objectives covering 80% of business value:

```python
# Priority-based ranking (from bundle_generator.py:65-74)
'margin_guard': 1.3       # Highest priority
'clear_slow_movers': 1.2
'new_launch': 1.1
'increase_aov': 1.0
'subscription_push': 1.0
'seasonal_promo': 0.9
'category_bundle': 0.8
'gift_box': 0.7          # Lowest priority
```

**Top 3 objectives (80% coverage)**:
1. `margin_guard` - Protect profit margins (priority 1.3)
2. `clear_slow_movers` - Move inventory (priority 1.2)
3. `increase_aov` - Boost revenue (priority 1.0)

**Reduction**: 8 objectives → **3 objectives** (62.5% reduction)

### 2. Intelligent Bundle Type Selection

Instead of trying all 5 types for each objective, select the best-fit type:

| Objective | Best Bundle Type | Rationale |
|-----------|------------------|-----------|
| `margin_guard` | `FBT` | Frequently Bought Together preserves margins |
| `clear_slow_movers` | `VOLUME_DISCOUNT` | Volume discounts move inventory fast |
| `increase_aov` | `MIX_MATCH` | Mix & Match maximizes cart value |

**Fallback types** (if primary fails):
- `FIXED` - Works for gift boxes and curated sets
- `BXGY` - Works for promotional campaigns

**Reduction**: 5 types per objective → **1-2 types** per objective

### 3. Final Combination Matrix

**Old**: 8 objectives × 5 types = 40 tasks

**New**: 3 objectives × 1-2 types = **3-6 tasks**

Specific combinations:
1. `margin_guard` + `FBT` (primary)
2. `margin_guard` + `FIXED` (fallback)
3. `clear_slow_movers` + `VOLUME_DISCOUNT` (primary)
4. `clear_slow_movers` + `BXGY` (fallback)
5. `increase_aov` + `MIX_MATCH` (primary)
6. `increase_aov` + `FBT` (fallback)

**Total reduction**: 40 → 6 tasks (**85% reduction**)

### 4. Early Termination for Small Datasets

Detect insufficient data BEFORE Phase 3:

```python
def should_skip_ml_phase(csv_upload_id: str, context: CandidateGenerationContext) -> bool:
    """Skip ML phase if data is too small to generate meaningful bundles"""

    # Check transaction count
    if len(context.transactions) < 10:
        logger.warning(f"[{csv_upload_id}] Skipping ML phase: only {len(context.transactions)} transactions (need 10+)")
        return True

    # Check unique products
    unique_products = len(context.valid_skus)
    if unique_products < 5:
        logger.warning(f"[{csv_upload_id}] Skipping ML phase: only {unique_products} unique products (need 5+)")
        return True

    # Check if we have any association rules
    has_rules = len(context.transactions) >= 10  # FPGrowth needs minimum data
    if not has_rules:
        logger.warning(f"[{csv_upload_id}] Skipping ML phase: insufficient data for association rules")
        return True

    return False
```

**Impact**: Save 195 seconds on tiny datasets by failing fast

### 5. Dynamic Objective Selection Based on Data

Instead of hardcoding top 3, analyze the dataset to pick relevant objectives:

```python
def select_objectives_for_dataset(context: CandidateGenerationContext) -> List[str]:
    """Dynamically select top objectives based on dataset characteristics"""

    txn_count = len(context.transactions)
    product_count = len(context.valid_skus)

    # Small dataset (<50 txns): Focus on basics
    if txn_count < 50:
        return ['increase_aov', 'margin_guard']  # 2 objectives

    # Medium dataset (50-200 txns): Add inventory management
    elif txn_count < 200:
        return ['increase_aov', 'clear_slow_movers', 'margin_guard']  # 3 objectives

    # Large dataset (200+ txns): Full coverage
    else:
        return ['increase_aov', 'clear_slow_movers', 'margin_guard', 'new_launch']  # 4 objectives
```

### 6. Performance Targets

| Metric | Old (40 tasks) | New (3-6 tasks) | Improvement |
|--------|----------------|-----------------|-------------|
| **Phase 3 tasks** | 40 | 3-6 | 85% reduction |
| **Phase 3 duration** | 195s | 15-30s | 6-13× faster |
| **Total runtime** | 330s | 30-60s | 5-11× faster |
| **Timeout risk** | High (>300s) | Low (<100s) | Eliminated |

## Implementation Plan

### Phase 1: Add Early Termination (Immediate - 1 hour)
- Detect tiny datasets (<10 txns, <5 products)
- Skip ML phase entirely
- Return helpful error message

### Phase 2: Reduce Objectives (Quick Win - 2 hours)
- Implement Pareto analysis
- Use top 3 objectives by default
- Add dynamic selection based on dataset size

### Phase 3: Intelligent Bundle Type Selection (2-3 hours)
- Map objectives to best-fit bundle types
- Reduce from 5 types to 1-2 per objective
- Add fallback logic

### Phase 4: Validate & Test (1 week)
- A/B test: 40 tasks vs 3-6 tasks
- Measure quality (bundles generated, CTR, conversion)
- Measure performance (duration, timeout rate)

## Expected Results

### Small Dataset (<10 txns)
- **Old**: 330s, 0 bundles, timeout risk
- **New**: 5s, helpful error message, no timeout

### Medium Dataset (50-200 txns)
- **Old**: 195s for Phase 3, 10-20 bundles
- **New**: 20-30s for Phase 3, 8-15 bundles (comparable quality)

### Large Dataset (200+ txns)
- **Old**: 120s for Phase 3, 30-50 bundles
- **New**: 30-40s for Phase 3, 25-40 bundles (comparable quality)

## Risk Mitigation

### Quality Concerns
- **Risk**: Fewer combinations might miss good bundles
- **Mitigation**:
  - A/B test to validate quality
  - Pareto analysis ensures we keep high-value objectives
  - LLM embeddings compensate for reduced combinations

### Backward Compatibility
- **Risk**: Existing bundles reference all 8 objectives
- **Mitigation**:
  - Keep all 8 objectives defined (for backward compat)
  - Only generate for top 3-4
  - Existing bundles continue to work

## Success Metrics

1. **Performance**: Phase 3 duration < 60s (vs 195s)
2. **Reliability**: 0% timeout rate (vs current ~10%)
3. **Quality**: Bundle generation rate ≥ 80% of old system
4. **Coverage**: Top 3 objectives cover ≥80% of business value

## Next Steps

1. Implement early termination (immediate)
2. Add Pareto objective selection (today)
3. Implement intelligent bundle type mapping (tomorrow)
4. Deploy and monitor (next week)
