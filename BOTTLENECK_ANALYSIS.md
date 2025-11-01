# Critical Bottleneck Analysis - Bundle Generation Taking 6+ Minutes

## Current Status
- **Problem**: Bundle generation timing out after 6+ minutes (360+ seconds)
- **Target**: Should complete in 10-20 seconds
- **Latest Test**: Upload `76eb0842` took 558s and timed out with 0 bundles generated

## Identified Bottlenecks (In Order of Impact)

### 1. **Phase 2: Objective Scoring - MAJOR BOTTLENECK**
**Impact**: 60-second timeout hit on EVERY run

**Location**: `services/objectives.py:28-92`

**Problem**:
```python
async def compute_objective_flags(self, csv_upload_id: str):
    catalog_items = await storage.get_catalog_snapshots_by_run(run_id)
    for item in catalog_items:  # Sequential processing
        flags = await self.compute_flags_for_item(item, csv_upload_id)
        # Each item requires multiple DB queries for velocity calculation
        await storage.update_catalog_snapshots_with_flags(updated_items)
```

**Why It's Slow**:
- Processes EVERY catalog item sequentially (no parallelization)
- Each item triggers compute_flags_for_item() which does heavy DB lookups
- Bulk update at the end, but processing is serial
- Always hits 60s timeout and continues with empty flags

**Fix**: **DISABLE IT ENTIRELY** - The objective flags are not critical for bundle generation
- Set `self.enable_objective_scoring = False` in bundle_generator.py line 89
- This removes 60+ seconds from every run

---

### 2. **Phase 3: Candidate Generation prepare_context() - MAJOR BOTTLENECK**
**Impact**: Called ONCE but takes 120-180 seconds

**Location**: `services/ml/candidate_generator.py:38-54`

**Problem**:
```python
async def prepare_context(self, csv_upload_id: str):
    transactions = await self.get_transactions_for_mining(csv_upload_id)
    sequences = await self.get_purchase_sequences(csv_upload_id)
    embeddings = await self.get_or_train_embeddings(csv_upload_id, sequences=sequences)
```

**Why It's Slow**:
- `get_or_train_embeddings()` trains item2vec model from scratch every time
- item2vec training on even small datasets (50-100 transactions) takes 60-120 seconds
- FPGrowth pattern mining takes another 30-60 seconds
- All this happens BEFORE any parallel Phase 3 tasks can start

**Fix**: **DISABLE ML EMBEDDINGS**
- Remove embeddings training entirely
- Use only association rules (already computed during CSV upload)
- This saves 120+ seconds

---

### 3. **Phase 8: AI Copy Generation - MODERATE BOTTLENECK**
**Impact**: 10-30 seconds depending on bundle count

**Location**: `services/bundle_generator.py:1039-1070`

**Problem**:
```python
# Batched but still slow
batch_size = 5
for i in range(0, len(bundles_needing_copy), batch_size):
    batch = bundles_needing_copy[i:i+batch_size]
    tasks = [generate_bundle_copy(...) for bundle in batch]
    await asyncio.gather(*tasks)
```

**Why It's Slow**:
- OpenAI API calls (gpt-3.5-turbo) take 2-5 seconds per bundle
- Even with batching of 5, this is 10-30 seconds for 25-50 bundles
- Not critical for bundle functionality

**Fix**: **DISABLE AI COPY GENERATION DURING GENERATION**
- Generate bundles without AI copy
- Add AI copy generation as a separate background job after bundles are saved
- OR use fallback copy only (instant)

---

### 4. **FallbackLadder - MODERATE BOTTLENECK**
**Impact**: 4-18 seconds per call × 40 calls = 160-720 seconds

**Status**: ALREADY FIXED in latest code (reversed tier order)
- Now runs bottom-up (popularity first)
- Should be 2-4 seconds per call instead of 18s
- But still adds up across 40 parallel tasks

**Further Fix**: Reduce FallbackLadder usage
- Only use it if initial candidate generation returns < 5 candidates
- Skip entirely for very small datasets (< 10 lines) - ALREADY IMPLEMENTED

---

### 5. **Phase 5a: Enterprise Optimization - MINOR BOTTLENECK**
**Impact**: 2-5 seconds

**Location**: `services/bundle_generator.py:549-600`

**Problem**: Pareto optimization and constraint management add complexity

**Fix**: Disable for small datasets or simplify

---

## Recommended Fix Strategy

### **AGGRESSIVE FIX (Get to 5-10 seconds)**

1. **DISABLE Phase 2: Objective Scoring**
   ```python
   # In bundle_generator.py __init__
   self.enable_objective_scoring = False  # Line 89
   ```
   **Saves**: 60+ seconds

2. **DISABLE ML Embeddings in Phase 3**
   ```python
   # In candidate_generator.py prepare_context()
   # Comment out embedding training entirely
   embeddings = {}  # Skip training
   ```
   **Saves**: 120+ seconds

3. **DISABLE AI Copy Generation (or make it async)**
   ```python
   # In bundle_generator.py
   self.enable_ai_copy_generation = False  # Add this flag
   # Use fallback copy only
   ```
   **Saves**: 10-30 seconds

4. **REDUCE FallbackLadder usage**
   - Already improved with bottom-up approach
   - Add: Skip if > 10 candidates already found
   **Saves**: Additional 50-100 seconds

**Total Time Saved**: 200-300+ seconds
**New Expected Time**: 5-15 seconds

---

## Implementation Priority

### Phase 1: Disable Non-Critical AI/ML Features (IMMEDIATE)
1. Disable objective scoring (60s saved)
2. Disable item2vec embeddings (120s saved)
3. Disable AI copy generation (20s saved)

**Result**: 200s saved, should get to 15-25 seconds

### Phase 2: Optimize Remaining Logic (IF NEEDED)
1. Further reduce FallbackLadder calls
2. Simplify enterprise optimization
3. Add caching for repeated queries

**Result**: Additional 10-20s saved, target 5-10 seconds

---

## Why These Features Can Be Disabled

1. **Objective Scoring**:
   - Used to tag products as "slow movers", "new launches", etc.
   - Bundles can be generated without these tags
   - Tags can be computed offline/async

2. **item2vec Embeddings**:
   - Provides ML-based product similarity
   - Association rules already provide good candidates
   - Minimal quality impact for most use cases

3. **AI Copy Generation**:
   - Makes bundle titles/descriptions more appealing
   - Fallback copy is perfectly functional
   - Can be generated async after bundles are created

4. **FallbackLadder**:
   - Provides candidates when association rules fail
   - Can be reduced to only run when critically needed
   - Bottom-up approach already much faster

---

## Next Steps

1. Create feature flags for each component
2. Deploy with all slow features disabled
3. Test performance (should be 5-15 seconds)
4. Gradually re-enable features with optimization/caching
5. Move slow AI/ML operations to background jobs
