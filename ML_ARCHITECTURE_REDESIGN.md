# ML Architecture Redesign: LLM-First Approach

## Executive Summary

**Goal**: Replace slow/ineffective ML models with fast LLM-driven approach while improving quality

**Key Insight**: LLMs in driver's seat, transactional data as validation signal

---

## Current Architecture Analysis

### Existing ML Components & Their Issues

| Component | Location | Purpose | Status | Issue |
|-----------|----------|---------|--------|-------|
| **item2vec** | `candidate_generator.py:45` | Product similarity | ❌ DISABLED | 60-120s training, cold start |
| **Objective Scorer** | `objectives.py:28` | Tag slow movers | ❌ DISABLED | 60s timeout, sequential |
| **FPGrowth/Apriori** | `candidate_generator.py:334` | ✅ Association rules | ✅ ACTIVE | Works well |
| **BayesianPricing** | `pricing.py` | Discount optimization | ✅ ACTIVE | Unknown perf |
| **WeightedRanker** | `ranker.py` | Bundle scoring | ✅ ACTIVE | Unknown perf |
| **OptimizationEngine** | `ml/optimization_engine.py` | Pareto optimization | ✅ ACTIVE | 2-5s |
| **Explainability** | `explainability.py` | Why bundles? | ❌ DISABLED | 5-15s |
| **AICopyGenerator** | `ai_copy_generator.py` | Marketing copy | ✅ ACTIVE | 10-30s batched |

### Current Pipeline Flow (SEQUENTIAL)

```
Phase 1: Data Mapping (enrichment)
   ↓
Phase 2: Objective Scoring (DISABLED - was 60s)
   ↓
Phase 3: ML Candidate Generation (PARALLEL: 40 tasks)
   ├─ FPGrowth association rules ✅
   ├─ item2vec embeddings ❌ (DISABLED - was 60-120s)
   └─ Top pair mining ✅
   ↓
Phase 4: Deduplication
   ↓
Phase 5: Enterprise Optimization (Pareto, constraints)
   ↓
Phase 6: Explainability (DISABLED - was 5-15s)
   ↓
Phase 7: Bayesian Pricing
   ↓
Phase 8: AI Copy Generation (batched)
   ↓
Phase 9: Storage
```

**Problem**: Sequential phases, some slow ML models, no LLM semantic understanding

---

## Proposed LLM-First Architecture

### Core Principle: "LLM Drives, Data Validates"

**60-Day Transaction Volumes**:
- Small stores: 60-300 transactions → **LLM-heavy** (α=0.7)
- Medium stores: 300-1200 transactions → **Balanced** (α=0.5)
- Large stores: 1200+ transactions → **Data-heavy** (α=0.3)

### New Hybrid Scoring Model

```python
final_score = α × LLM_signal + β × transactional_signal + γ × business_signal

where:
  α = LLM weight (semantic similarity, objectives)
  β = Transactional weight (association lift, co-purchase)
  γ = Business weight (margin, inventory, pricing)
```

**Dynamic Weight Adjustment**:
```python
def get_weights(txn_count: int) -> Tuple[float, float, float]:
    if txn_count < 300:      # Small store
        return (0.6, 0.2, 0.2)  # LLM leads
    elif txn_count < 1200:   # Medium store
        return (0.4, 0.4, 0.2)  # Balanced
    else:                    # Large store
        return (0.2, 0.6, 0.2)  # Data leads
```

---

## Redesigned Pipeline: Parallel + LLM-First

### New Architecture (PARALLEL where possible)

```
Phase 0: Context Preparation (PARALLEL)
   ├─ Load catalog + transactions
   ├─ Generate LLM embeddings (batch 100 products in 2s)
   └─ Compute association rules (reuse from upload)
   ↓
Phase 1: LLM-Driven Candidate Generation (PARALLEL: 8 objectives × 5 types)
   ├─ LLM identifies semantic bundles
   ├─ Validates with transactional lift
   └─ Scores with hybrid model
   ↓
Phase 2: Multi-Objective Optimization (PARALLEL)
   ├─ Pareto optimization
   ├─ Pricing optimization
   └─ Deduplication
   ↓
Phase 3: LLM Copy + Explainability (PARALLEL batches)
   ├─ Generate marketing copy
   └─ Generate explanations
   ↓
Phase 4: Storage
```

**Expected Performance**:
- Phase 0: 2-3 seconds (was 60-120s with item2vec)
- Phase 1: 3-5 seconds (parallel LLM inference)
- Phase 2: 2-3 seconds
- Phase 3: 5-10 seconds (batched)
- **Total: 12-21 seconds** (vs 360+ before)

---

## ML Models to Replace/Introduce

### 1. REPLACE: item2vec → LLM Embeddings

**Old (item2vec)**:
- Training time: 60-120 seconds
- Cold start: Fails for new products
- Understanding: Behavioral only

**New (LLM Embeddings)**:
- Training time: 0 seconds (pre-trained)
- Cold start: Works immediately
- Understanding: Semantic + behavioral hybrid
- Cost: ~$0.001 per 100 products

```python
# services/ml/llm_embeddings.py (NEW)
class LLMEmbeddingEngine:
    async def get_product_embedding(self, product: Dict) -> np.ndarray:
        """Generate semantic embedding using OpenAI"""
        text = f"{product['title']} {product['category']} {product['description'][:100]}"
        response = await openai.embeddings.create(
            model="text-embedding-3-small",  # $0.02/1M tokens
            input=text
        )
        return np.array(response.data[0].embedding)

    async def find_similar_products(self, target_sku, catalog, top_k=20):
        """Find semantically similar products"""
        target_emb = await self.get_cached_embedding(target_sku)
        similarities = [
            (p['sku'], cosine_similarity(target_emb, await self.get_cached_embedding(p['sku'])))
            for p in catalog
        ]
        return sorted(similarities, key=lambda x: x[1], reverse=True)[:top_k]
```

### 2. REPLACE: Objective Scorer → LLM Classifier

**Old (Objective Scorer)**:
- Method: Velocity calculations, DB queries
- Time: 60+ seconds
- Accuracy: Requires lots of data

**New (LLM Objective Classifier)**:
- Method: Zero-shot classification
- Time: 1-2 seconds (batch 100 products)
- Accuracy: Works with product metadata alone

```python
# services/ml/llm_objective_scorer.py (NEW)
class LLMObjectiveScorer:
    async def classify_products(self, products: List[Dict]) -> Dict[str, List[str]]:
        """Classify products into objectives using LLM"""
        prompt = f"""Classify these products into objectives:

Products: {json.dumps([p['title'] for p in products[:50]])}

Objectives:
- slow_mover: Products that seem seasonal, niche, or less popular
- high_margin: Premium products, luxury items
- new_launch: Products with "new" or recent release dates
- seasonal: Holiday, seasonal products
- gift_worthy: Gift boxes, sets, bundles

Return JSON: {{"slow_mover": [...], "high_margin": [...], ...}}
"""
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Fast + cheap
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
```

### 3. INTRODUCE: LLM Bundle Reasoner

**Purpose**: Explain WHY bundles make sense (replace Explainability engine)

```python
# services/ml/llm_bundle_reasoner.py (NEW)
class LLMBundleReasoner:
    async def explain_bundle(self, bundle: Dict, context: Dict) -> str:
        """Generate explanation for why bundle makes sense"""
        products = bundle['products']
        objective = bundle['objective']
        lift = context.get('association_lift', 0)

        prompt = f"""Explain why this bundle makes business sense:

Products: {', '.join([p['title'] for p in products])}
Objective: {objective}
Co-purchase lift: {lift:.2f}x
Customer segment: {context.get('segment', 'general')}

Provide 2-sentence explanation focusing on customer value and business objective.
"""
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

### 4. ENHANCE: Pricing Optimizer with LLM Insights

**Old (BayesianPricing)**:
- Method: Statistical modeling
- Inputs: Price history, margins
- Output: Discount %

**New (LLM-Enhanced Pricing)**:
- Method: LLM suggests, Bayesian validates
- Inputs: Product data + market context
- Output: Discount % with reasoning

```python
# services/ml/llm_pricing_advisor.py (NEW)
class LLMPricingAdvisor:
    async def suggest_discount(self, bundle: Dict, context: Dict) -> Dict:
        """LLM suggests discount, Bayesian pricing validates"""

        # 1. LLM suggests discount range
        llm_suggestion = await self._get_llm_discount_suggestion(bundle, context)

        # 2. Bayesian pricing computes optimal discount
        bayesian_optimal = await self.bayesian_engine.compute_optimal_discount(bundle)

        # 3. Hybrid decision
        α = 0.4  # LLM weight (market context)
        β = 0.6  # Bayesian weight (data-driven)

        final_discount = α * llm_suggestion + β * bayesian_optimal

        return {
            "discount_pct": final_discount,
            "llm_suggestion": llm_suggestion,
            "bayesian_optimal": bayesian_optimal,
            "reasoning": await self._explain_discount(bundle, final_discount)
        }
```

### 5. INTRODUCE: LLM Demand Forecaster (Optional)

**Purpose**: Predict bundle appeal before generating

```python
# services/ml/llm_demand_forecaster.py (NEW)
class LLMDemandForecaster:
    async def predict_bundle_appeal(self, bundle: Dict, context: Dict) -> float:
        """Predict if bundle will resonate with customers"""
        prompt = f"""Rate bundle appeal 0-10:

Products: {', '.join([p['title'] for p in bundle['products']])}
Target segment: {context.get('segment', 'general')}
Season: {context.get('season', 'any')}
Price point: ${bundle.get('total_price', 0)}

Consider:
- Product complementarity
- Price-value perception
- Segment fit
- Seasonality

Return JSON: {{"score": X.X, "reasoning": "..."}}
"""
        # ... LLM inference ...
        return score
```

---

## Model Orchestration: Parallel vs Sequential

### Current (SUBOPTIMAL): Mostly Sequential

```
prepare_context() [BLOCKING 60-120s]
  ↓
40 parallel tasks [each waits for embeddings]
  ↓
deduplication [BLOCKING]
  ↓
optimization [BLOCKING]
  ↓
AI copy [BATCHED but sequential batches]
```

### Proposed (OPTIMAL): Maximum Parallelization

```
┌─ LLM embeddings (2s) ──┐
├─ Association rules (0s - cached) ─┤
├─ LLM objective classification (1s) ─┤
└─ Product metadata enrichment (1s) ──┘
         ↓ (ALL PARALLEL)
   Context Ready (2-3s)
         ↓
┌─ Objective 1 × Type 1 ─┐
├─ Objective 1 × Type 2 ─┤
├─ ... (40 tasks) ... ─┤  ← ALL PARALLEL
└─ Objective 8 × Type 5 ─┘
         ↓ (3-5s)
   Candidates Ready
         ↓ (PARALLEL)
┌─ Deduplication ──┐
├─ Pareto optimization ─┤
└─ Pricing optimization ─┘
         ↓ (2-3s)
   Optimized Bundles
         ↓ (PARALLEL BATCHES)
┌─ Batch 1: AI copy + explanation ─┐
├─ Batch 2: AI copy + explanation ─┤
└─ Batch N: AI copy + explanation ─┘
         ↓ (5-10s)
   Final Bundles
```

**Key Changes**:
1. **Parallel context prep**: All context loading happens simultaneously
2. **Parallel optimization**: Dedup + Pareto + Pricing run together
3. **Parallel AI generation**: Copy + Explanation in same LLM call (1 API call instead of 2)

---

## Implementation Roadmap

### Phase 1: LLM Embeddings (IMMEDIATE)
- Replace item2vec with OpenAI embeddings
- Expected: 60-120s → 2s speedup
- Files: Create `services/ml/llm_embeddings.py`

### Phase 2: LLM Objective Scorer (QUICK WIN)
- Replace velocity-based scorer with LLM classifier
- Expected: 60s → 1s speedup
- Files: Create `services/ml/llm_objective_scorer.py`

### Phase 3: Hybrid Scoring (QUALITY BOOST)
- Implement α × LLM + β × transactional scoring
- Expected: Better bundles for small stores
- Files: Update `services/ml/candidate_generator.py`

### Phase 4: Parallel Optimization (SPEED BOOST)
- Run dedup + Pareto + pricing in parallel
- Expected: 6-8s → 2-3s speedup
- Files: Update `services/bundle_generator.py`

### Phase 5: LLM Reasoner (QUALITY)
- Replace explainability engine with LLM
- Expected: Better explanations, faster
- Files: Create `services/ml/llm_bundle_reasoner.py`

---

## Testing Strategy: Signal Comparison

### A/B Test Setup

**Control Group**: Current FPGrowth-only approach
**Treatment Groups**:
- Group A: LLM-only (α=1.0, β=0.0)
- Group B: Transactional-only (α=0.0, β=1.0)
- Group C: Hybrid small-store (α=0.7, β=0.3)
- Group D: Hybrid medium-store (α=0.4, β=0.6)
- Group E: Hybrid large-store (α=0.2, β=0.8)

**Metrics**:
- Bundle click-through rate
- Add-to-cart rate
- Purchase conversion
- Average order value lift
- Customer satisfaction (qualitative)

### Expected Results

| Store Size | Best α/β | Why |
|------------|----------|-----|
| Small (60-300 txns) | α=0.7, β=0.3 | Limited transaction data, LLM semantic understanding wins |
| Medium (300-1200 txns) | α=0.4, β=0.6 | Good balance, transactional signal strengthens |
| Large (1200+ txns) | α=0.2, β=0.8 | Rich transaction data, trust the numbers |

---

## Cost Analysis

### Current Costs
- item2vec training: $0 (self-hosted)
- Objective scoring: $0 (self-hosted)
- AI copy: ~$0.10 per 50 bundles (GPT-3.5)

### New Costs (LLM-First)
- LLM embeddings: ~$0.001 per 100 products (one-time per upload)
- LLM objective scoring: ~$0.01 per 50 products (one-time per upload)
- AI copy + explanation: ~$0.15 per 50 bundles (combined call)

**Total**: ~$0.20 per bundle generation run
**Acceptable**: Given 10-30× speedup and quality improvements

---

## Quality vs Speed Tradeoffs

| Approach | Speed | Quality (Small) | Quality (Large) | Cost |
|----------|-------|-----------------|-----------------|------|
| **Current (FPGrowth only)** | Slow (360s) | Low | High | $0 |
| **LLM-only** | Fast (15s) | Very High | Low | $0.20 |
| **Hybrid (proposed)** | Fast (15s) | Very High | Very High | $0.20 |

**Conclusion**: Hybrid approach wins on all dimensions

---

## Next Steps

1. **Implement LLM Embeddings** (2-3 hours)
   - Create `services/ml/llm_embeddings.py`
   - Replace item2vec calls in `candidate_generator.py`
   - Add caching to DB

2. **Implement Hybrid Scoring** (2-3 hours)
   - Update `candidate_generator.py` with α/β weighting
   - Dynamic weight selection based on transaction count

3. **Test Signal Comparison** (1 week)
   - Run A/B tests on real stores
   - Collect CTR, conversion metrics
   - Validate optimal α/β weights

4. **Deploy Parallel Optimization** (2-3 hours)
   - Refactor Phase 5 to run dedup/Pareto/pricing in parallel
   - Expected 4-5s savings

5. **Optional: Add LLM Reasoner** (if needed)
   - Only if explainability becomes critical again

---

## Summary

**Key Decisions**:
1. ✅ LLM in driver's seat (especially for small/medium stores)
2. ✅ Transactional data validates and boosts (especially for large stores)
3. ✅ Dynamic α/β weighting based on data availability
4. ✅ Maximum parallelization of all pipeline stages
5. ✅ Test signals empirically (A/B testing)

**Expected Improvements**:
- **Speed**: 360s → 12-21s (17-30× faster)
- **Quality**: Better bundles for small stores (LLM semantic understanding)
- **Cost**: ~$0.20 per run (acceptable for value delivered)
- **Cold Start**: Zero-shot works for new products immediately

**Philosophy**: "Let LLMs do what they do best (understanding semantics), let transaction data do what it does best (validating behavior), combine them intelligently based on data availability"
