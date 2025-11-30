# Bundle Generation Flows - Complete E2E Documentation

## Overview
The system has **TWO** distinct bundle generation paths:
1. **Quick-Start Path** - Fast preview for first-time installations (~40-120s)
2. **Full ML Path** - Comprehensive ML-powered generation (5-10+ minutes)

---

## 🚀 QUICK-START PATH (First-Time Install Only)

### **Trigger Conditions**
```python
# From: routers/bundle_recommendations.py:266-311

QUICK_START_ENABLED = True  # Feature flag
is_first_install = not shop_sync_status.initial_sync_completed
has_existing_quick_start = quick_start_bundle_count > 0

# Quick-start runs if:
if is_first_install and QUICK_START_ENABLED and not has_existing_quick_start:
    # Run quick-start
```

**Detection Logic:**
- Check `shop_sync_status.initial_sync_completed`
- If `False` OR no record exists → First-time install
- Check for existing bundles with `discount_reference LIKE '__quick_start_%'`
- If bundles exist → Skip quick-start

---

### **Quick-Start Flow (E2E)**

#### **Configuration**
```python
max_products: 50          # Limit to top 50 products by sales
max_bundles: 10           # Generate only 10 preview bundles
timeout_seconds: 120      # Hard 2-minute timeout
```

#### **Phase 1: Data Loading & Filtering (10-25% progress)**
```python
1. Load ALL order lines from database
   └─ Query: SELECT * FROM order_lines WHERE csv_upload_id = ?

2. Early Exit Checks:
   ├─ If order_lines < 10 → Exit with 0 bundles
   └─ If unique_skus < 2 → Exit with 0 bundles

3. Product Selection:
   ├─ Count sales volume per SKU
   ├─ Select TOP 50 products by quantity sold
   └─ Filter order_lines to only include top products

Output: filtered_order_lines (subset of original)
```

#### **Phase 2: Simple Objective Scoring (25-40% progress)**
```python
1. Load catalog snapshots
   └─ Query: SELECT * FROM catalog_snapshot WHERE csv_upload_id = ?

2. Score ONLY 2 objectives (fast):
   ├─ increase_aov
   └─ clear_slow_movers

3. Simple heuristic scoring:
   product_scores[sku] = {
       base: 0.5,
       + 0.3 if is_slow_mover,
       + 0.2 if is_high_margin
   }

Output: product_scores = {sku: score}
```

#### **Phase 2.5: Co-Visitation Graph (40-60% progress)**
```python
1. Build lightweight similarity vectors:
   └─ build_covis_vectors(filtered_order_lines)

2. For each product, find products bought together:
   └─ Uses co-occurrence in same orders
   └─ Creates pseudo-embeddings (Item2Vec-style)

Output: covis_vectors = {sku: [similar_skus]}
```

#### **Phase 3: Multi-Type Bundle Generation (60-70% progress)**
```python
1. Allocate bundle targets:
   ├─ FBT (Frequently Bought Together): 3-5 bundles
   ├─ BOGO (Buy X Get Y): 2-3 bundles
   └─ Volume Discount: 1-2 bundles
   Total: max 10 bundles

2. Generate FBT bundles:
   ├─ Use covis_vectors to find related products
   ├─ Create 2-3 item bundles
   └─ Score by product_scores + co-visit frequency

3. Generate BOGO bundles:
   ├─ Pair high-margin with slow-movers
   └─ Simple discount: Buy 1 Get 1 at 50% off

4. Generate Volume bundles:
   ├─ Use top-selling products
   └─ Discount: Buy 3+ get 10-20% off

5. Combine and cap at max_bundles (10)

Output: recommendations = [bundle1, bundle2, ...]
```

#### **Phase 4: Persistence (70-100% progress)**
```python
1. Mark bundles with special flag:
   └─ discount_reference = '__quick_start_{uuid}'

2. Save to database:
   └─ INSERT INTO bundle_recommendations (...)

3. Update progress to 100% "completed"

4. Trigger notification:
   └─ notify_partial_ready() - User gets preview notification

5. Schedule FULL generation in background:
   └─ pipeline_scheduler.schedule(_run_full_generation_after_quickstart())
   └─ This runs separately after quick-start completes
```

---

### **Quick-Start Summary**
| Phase | Duration | Output |
|-------|----------|--------|
| Phase 1: Data Load | 5-10s | Top 50 products, filtered orders |
| Phase 2: Scoring | 2-5s | Product scores (2 objectives) |
| Phase 2.5: Co-vis | 3-8s | Similarity graph |
| Phase 3: Generation | 10-20s | 10 preview bundles |
| Phase 4: Save | 2-5s | Bundles persisted |
| **TOTAL** | **40-120s** | **~10 bundles** |

**Key Features:**
- ✅ Simplified scoring (2 objectives vs 8)
- ✅ Limited products (50 vs unlimited)
- ✅ Fast co-visitation (no embeddings)
- ✅ Fixed bundle count (10)
- ✅ Hard timeout (120s)
- ✅ Immediate user notification

---

## 🧠 FULL ML PIPELINE (Normal Path)

### **Trigger Conditions**
```python
# Runs in these scenarios:
1. Quick-start completed → Auto-scheduled in background
2. Quick-start failed → Falls back to full pipeline
3. NOT first-time install → Always uses full pipeline
4. Manual regeneration request
```

---

### **Full Pipeline Flow (E2E)**

#### **Phase 1: Data Mapping & Enrichment (5-25% progress)**
```python
Feature Flag: enable_data_mapping = True

1. Load order lines:
   └─ Query: SELECT * FROM order_lines WHERE csv_upload_id = ?

2. Enrich with variant data:
   ├─ For each order line SKU:
   │   ├─ Query: SELECT * FROM variants WHERE sku = ? AND csv_upload_id = ?
   │   ├─ Add product_id, variant_id, inventory data
   │   └─ Add flags: is_slow_mover, is_new_launch, is_seasonal, is_high_margin
   └─ Updates order_lines records with enrichment

3. Load full catalog snapshot:
   └─ Query: SELECT * FROM catalog_snapshot WHERE csv_upload_id = ?

Output:
- enriched_order_lines (with variant data)
- catalog_map = {sku: catalog_snapshot}
- Metrics: resolved_variants, unresolved_skus, total_order_lines
```

#### **Phase 2: Objective Scoring (30-45% progress)**
```python
Feature Flag: enable_objective_scoring = False (currently disabled)

IF ENABLED (not running in your case):
1. Load all 8 business objectives:
   ├─ increase_aov
   ├─ clear_slow_movers
   ├─ boost_new_launches
   ├─ maximize_margin
   ├─ seasonal_promotions
   ├─ customer_acquisition
   ├─ cross_category_bundling
   └─ volume_upsell

2. Compute objective flags for each product:
   └─ Updates catalog_snapshot with objective scores

IF DISABLED (current state):
- Skip this phase
- Progress: "Objective scoring skipped (disabled)"
- Checkpoint: phase_2_objective_scoring_skipped

Output: Updated catalog with objective flags (if enabled)
```

#### **Phase 3: ML Candidate Generation (50-70% progress)**
```python
Feature Flag: enable_ml_candidates = True

1. Prepare ML Context:
   └─ context = CandidateGenerator.prepare_context(csv_upload_id)

   Context includes:
   ├─ order_lines (enriched)
   ├─ catalog_snapshot
   ├─ orders (transaction data)
   ├─ variants (product details)
   └─ shop_info

2. Dataset Profile & Pareto Optimization:
   ├─ Analyze dataset size (order_lines, unique_skus)
   ├─ Select objectives dynamically based on data
   └─ Reduce from 8 objectives to 2-4 (for small datasets)

3. PARALLEL ML Generation (Concurrency: 3-5 tasks):

   For EACH (objective × bundle_type) combination:
   └─ generate_objective_bundles(objective, bundle_type)

   Example combinations:
   ├─ (increase_aov, FBT)
   ├─ (increase_aov, BXGY)
   ├─ (clear_slow_movers, VOLUME_DISCOUNT)
   ├─ (maximize_margin, MIX_MATCH)
   └─ ... (2-4 objectives × 5 bundle types = 10-20 parallel tasks)

4. ML Candidate Generation Process:

   A. Load ML Context Data:
      ├─ Get order history
      ├─ Get catalog with enrichment
      └─ Get product embeddings (if available)

   B. Try Multiple ML Tiers (FallbackLadder):

      Tier 1: Association Rules (if sufficient data)
      ├─ Requires: 50+ order lines
      ├─ Uses: Apriori/FP-Growth algorithm
      ├─ Finds: Products frequently bought together
      └─ Duration: 4-8s

      Tier 2: Adaptive Relaxation
      ├─ Relax constraints if Tier 1 fails
      ├─ Lower min_support, min_confidence
      └─ Duration: 4-6s

      Tier 3: Smoothed Co-Occurrence
      ├─ Count products in same orders
      ├─ Apply Laplace smoothing
      └─ Duration: 3-5s

      Tier 4: Item-Item Similarity
      ├─ Compute product embeddings (text + metadata)
      ├─ Use cosine similarity
      └─ Duration: 2-3s

      Tier 5: Heuristic Rules
      ├─ Same category bundling
      ├─ Price-tier matching
      └─ Duration: 1-2s

      Tier 6: Popularity-Based (CURRENTLY USED FOR SMALL DATA)
      ├─ Top-selling products
      ├─ Simple pairing by sales rank
      └─ Duration: 1-2s ✅ FAST - Works for small datasets

      Tier 7: Cold-Start Content
      ├─ Use product metadata only
      ├─ Category, tags, vendor matching
      └─ Duration: 1-2s

   C. Generate Bundle Candidates:
      ├─ For each tier, generate N candidates
      ├─ Score candidates by ML model
      ├─ Filter by business rules (min price, max items)
      └─ Return top K candidates per objective

5. Aggregate Results:
   ├─ Collect all candidates from parallel tasks
   ├─ Total: 50-200 candidates (before dedup)
   └─ Checkpoint: phase_3_candidates_completed

Output: all_recommendations = [candidate1, candidate2, ...]
```

**Important: FallbackLadder Optimization**
```python
# Recent optimization (from your commit):
# Reversed tier order for small datasets!

OLD ORDER (Slow → Fast):
Tier 1 (Association) → Tier 2 → ... → Tier 7 (Cold-start)
Result: Try expensive tiers first, waste 18-29s

NEW ORDER (Fast → Slow):
Tier 7 (Cold-start) → Tier 6 (Popularity) → ... → Tier 1 (Association)
Result: Find candidates in 2-4s, early exit! ✅

For small datasets (< 50 orders):
- Tier 6 (Popularity) succeeds in ~2s
- Skip remaining expensive tiers
- 5-9× speedup!
```

#### **Phase 4: Deduplication (75-80% progress)**
```python
Feature Flag: enable_deduplication = True

1. Identify duplicate bundles:
   ├─ Same SKU combination (order-independent)
   ├─ Same bundle type
   └─ Same discount structure

2. Keep best version:
   ├─ Score by: ML confidence + business objective alignment
   └─ Remove duplicates

3. Typical reduction:
   ├─ Input: 100-200 candidates
   └─ Output: 30-80 unique bundles

Output: unique_recommendations
```

#### **Phase 5a: Enterprise Optimization (80-85% progress)**
```python
Feature Flag: enable_enterprise_optimization = False (disabled)

IF ENABLED:
1. Portfolio-level optimization:
   ├─ Maximize total AOV across all bundles
   ├─ Balance bundle types (FBT, BXGY, etc.)
   └─ Ensure category coverage

2. Constraint management:
   ├─ Inventory constraints
   ├─ Margin requirements
   └─ Business rules

IF DISABLED: Skip this phase

Output: optimized_recommendations (if enabled)
```

#### **Phase 5b: Bayesian Pricing (85-90% progress)**
```python
Feature Flag: enable_bayesian_pricing = False (disabled)

IF ENABLED:
1. For each bundle, compute optimal discount:
   ├─ Historical conversion data
   ├─ Price elasticity model
   └─ Bayesian inference

2. Adjust discount percentages:
   ├─ Input: Fixed 10-20% discounts
   └─ Output: Optimized 8-25% discounts

IF DISABLED: Use default discount rules

Output: priced_recommendations (if enabled)
```

#### **Phase 6: Weighted Ranking (90-95% progress)**
```python
Feature Flag: enable_weighted_ranking = False (disabled)

IF ENABLED:
1. Score each bundle:
   ├─ ML confidence: 40%
   ├─ Business objective alignment: 30%
   ├─ Profitability: 20%
   └─ Inventory availability: 10%

2. Rank bundles by composite score

3. Select top N bundles (e.g., top 30)

IF DISABLED: Keep all bundles from dedup

Output: ranked_recommendations (if enabled)
```

#### **Phase 7: AI Copy Generation (95-98% progress)**
```python
Feature Flag: enable_ai_copy = True (usually enabled)

1. For EACH bundle, generate marketing copy:

   A. Call OpenAI GPT-4:
      ├─ Input: Bundle SKUs, prices, product titles
      ├─ Prompt: "Create compelling bundle title and description"
      └─ Output: title, description, call_to_action

   B. Generate bundle title:
      └─ Example: "Complete Home Office Bundle - Save 20%"

   C. Generate description:
      └─ Example: "Get everything you need for your home office..."

   D. Generate call-to-action:
      └─ Example: "Add to Cart & Save $50"

2. Rate limiting:
   ├─ Max 5 concurrent API calls
   └─ Retry on failures

Output: recommendations_with_copy
```

#### **Phase 8: Explainability (98-99% progress)**
```python
Feature Flag: enable_explainability = False (disabled)

IF ENABLED:
1. For each bundle, generate explanation:
   ├─ "Why these products?"
   ├─ "Based on 50 customers who bought X"
   └─ "Popular in your store"

IF DISABLED: Skip

Output: recommendations_with_explanations (if enabled)
```

#### **Phase 9: Final Persistence (99-100% progress)**
```python
1. Save ALL bundles to database:
   └─ INSERT INTO bundle_recommendations (...)

2. Update CSV upload status:
   └─ UPDATE csv_uploads SET status = 'completed'

3. Record final metrics:
   └─ UPDATE csv_uploads SET bundle_generation_metrics = {...}

4. Send completion notification:
   └─ notify_bundle_ready(csv_upload_id)

5. Progress: 100% "Bundle generation complete"

Output: Final bundle recommendations in database
```

---

## 📊 Full Pipeline Summary

| Phase | Feature Flag | Duration | Output |
|-------|--------------|----------|--------|
| **Phase 1: Enrichment** | `enable_data_mapping` | 10-30s | Enriched order lines |
| **Phase 2: Scoring** | `enable_objective_scoring` | 0s (disabled) | Skipped |
| **Phase 3: ML Candidates** | `enable_ml_candidates` | 60-300s | 50-200 candidates |
| **Phase 4: Dedup** | `enable_deduplication` | 5-15s | 30-80 unique |
| **Phase 5a: Optimization** | `enable_enterprise_optimization` | 0s (disabled) | Skipped |
| **Phase 5b: Pricing** | `enable_bayesian_pricing` | 0s (disabled) | Skipped |
| **Phase 6: Ranking** | `enable_weighted_ranking` | 0s (disabled) | Skipped |
| **Phase 7: AI Copy** | `enable_ai_copy` | 30-120s | Bundles with copy |
| **Phase 8: Explainability** | `enable_explainability` | 0s (disabled) | Skipped |
| **Phase 9: Save** | Always | 5-10s | Persisted bundles |
| **TOTAL** | | **5-10 minutes** | **30-80 bundles** |

---

## 🔄 Current State (Your Run)

### **Quick-Start (07:56-07:57)**
```
✅ Phase 1: Loaded 26 order lines
✅ Phase 2: Scored top 17 products
✅ Phase 2.5: Built co-visitation graph
✅ Phase 3: Generated bundles
❌ Phase 4: Saved 0 bundles (insufficient data/variety)
⏱️ Duration: 42.5s
```

### **Full Pipeline (08:00-Current)**
```
✅ Phase 1: Enriched 26 variants (08:00:25)
✅ Phase 2: Skipped (objective scoring disabled) (08:00:40)
🔄 Phase 3: Generating ML candidates (08:00:50-Current)
   └─ Using FallbackLadder Tier 6 (Popularity-based)
   └─ Processing embeddings for similarity
   └─ Generating candidates for 2-4 objectives
⏳ Phase 4-9: Pending...
```

**Current Phase 3 Details:**
- Running for ~5 minutes (normal for ML phase)
- Embedding cache queries = Active ML processing
- Using optimized FallbackLadder (fast → slow order)
- Should complete within next 2-5 minutes

---

## 🎯 Key Differences: Quick-Start vs Full

| Aspect | Quick-Start | Full ML Pipeline |
|--------|-------------|------------------|
| **Trigger** | First-time install only | Always (or after quick-start) |
| **Products** | Top 50 by sales | ALL products |
| **Objectives** | 2 simple objectives | 8 comprehensive objectives |
| **ML Tiers** | Co-visitation only | 7-tier fallback ladder |
| **Embeddings** | No embeddings | Full embeddings + similarity |
| **Bundle Count** | Fixed 10 bundles | 30-80 bundles |
| **Timeout** | 120s hard limit | 600s+ soft limit |
| **Optimization** | None | Dedup + ranking + pricing |
| **AI Copy** | No AI copy | GPT-4 generated copy |
| **Duration** | 40-120s | 5-10 minutes |
| **Purpose** | Fast preview | Comprehensive results |

---

## 📝 Configuration Reference

```python
# Quick-Start Settings (routers/bundle_recommendations.py)
QUICK_START_ENABLED = True
QUICK_START_MAX_PRODUCTS = 50
QUICK_START_MAX_BUNDLES = 10
QUICK_START_TIMEOUT_SECONDS = 120

# Full Pipeline Settings (services/bundle_generator.py)
enable_data_mapping = True              # ✅ Enabled
enable_objective_scoring = False        # ❌ Disabled
enable_ml_candidates = True             # ✅ Enabled
enable_deduplication = True             # ✅ Enabled
enable_enterprise_optimization = False  # ❌ Disabled
enable_bayesian_pricing = False         # ❌ Disabled
enable_weighted_ranking = False         # ❌ Disabled
enable_explainability = False           # ❌ Disabled

max_time_budget_seconds = 600           # 10-minute hard timeout
soft_timeout_seconds = 300              # 5-minute soft warning
phase3_concurrency_limit = 3            # Parallel ML tasks
max_total_attempts = 1000               # Max bundle generation attempts
```

---

## 🚨 Decision Flow: Which Path Runs?

```
User triggers bundle generation
    │
    ├─ Check: Is first-time install?
    │   └─ Query: shop_sync_status.initial_sync_completed
    │
    ├─ YES (first install)
    │   │
    │   ├─ Check: QUICK_START_ENABLED?
    │   │   │
    │   │   ├─ YES
    │   │   │   │
    │   │   │   ├─ Check: Existing quick-start bundles?
    │   │   │   │   │
    │   │   │   │   ├─ NO → RUN QUICK-START
    │   │   │   │   │   ├─ Success with bundles → Notify user
    │   │   │   │   │   ├─ Success with 0 bundles → Skip notification
    │   │   │   │   │   └─ Failure → Fall through to FULL PIPELINE
    │   │   │   │   │
    │   │   │   │   └─ Then: SCHEDULE FULL PIPELINE in background
    │   │   │   │
    │   │   │   └─ YES → SKIP quick-start, RUN FULL PIPELINE
    │   │   │
    │   │   └─ NO → RUN FULL PIPELINE only
    │   │
    │   └─ NO (returning user) → RUN FULL PIPELINE only
    │
    └─ End
```

---

## 💡 Pro Tips

1. **For first installs**: You get BOTH paths
   - Quick-start gives preview in ~1 minute
   - Full pipeline gives comprehensive results in ~10 minutes

2. **Current feature flags**: Minimal setup
   - Only 3 phases enabled (Enrichment, ML, Dedup)
   - Faster but less optimized results
   - Enable more flags for production quality

3. **FallbackLadder optimization**: Now smart!
   - Tries fast tiers first for small datasets
   - 5-9× speedup on stores with < 50 orders
   - Your commit made this happen! 🎉

4. **Monitoring**: Check logs for phase progress
   - `phase_1_enrichment_completed`
   - `phase_2_objective_scoring_skipped`
   - `phase_3_candidates_started` ← Current
   - `phase_3_candidates_completed` ← Next
   - `phase_4_deduplication_completed`
   - etc.

---

**Generated**: 2025-11-30
**Version**: Based on latest codebase with FallbackLadder optimization
