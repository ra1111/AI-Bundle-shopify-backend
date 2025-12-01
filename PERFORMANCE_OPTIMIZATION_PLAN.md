# Performance Optimization Plan

## Executive Summary

**Current Performance:**
- Catalog ingestion: 175 seconds (259 rows) ❌
- Inventory ingestion: 64 seconds (259 rows) ❌
- Orders ingestion: 5-7 minutes + TIMEOUT ❌
- Bundle save: 27+ seconds (40 bundles) ❌

**Target Performance:**
- Catalog ingestion: 5-10 seconds (95% faster) ✅
- Inventory ingestion: 3-5 seconds (95% faster) ✅
- Orders ingestion: 10-20 seconds (90% faster) ✅
- Bundle save: 1-2 seconds (95% faster) ✅

---

## PART 1: CSV Ingestion Bottlenecks

### Root Cause Analysis

#### 1. CockroachDB Performance Characteristics
- **Issue**: CockroachDB is optimized for distributed consistency, not single-node speed
- **Impact**: Large single-transaction upserts (259 rows) take 60-175 seconds
- **Evidence**:
  - Catalog: 175,099ms for 259 rows = 676ms per row
  - Inventory: 64,601ms for 259 rows = 249ms per row
  - Orders: 5-7 minutes with timeout loops

#### 2. No Batching Strategy
- **Issue**: All rows inserted in ONE massive transaction
- **Impact**: CockroachDB distributed locks slow down, transaction may exceed timeout
- **Current Code**:
  ```python
  # services/storage.py:1345-1351
  stmt = pg_insert(CatalogSnapshot).values(catalog_data)  # All 259 rows at once
  upsert = stmt.on_conflict_do_update(...)
  await session.execute(upsert)
  await session.commit()  # Single massive commit
  ```

#### 3. Synchronous Coverage Queries During Ingestion
- **Issue**: `summarize_upload_coverage()` runs AFTER variants ingestion
- **Impact**: Complex JOIN query adds 10-30 seconds to ingestion
- **Current Code**:
  ```python
  # services/csv_processor.py:473-474
  coverage = await storage.summarize_upload_coverage(target_upload_id, run_id)
  logger.info("[%s] Post-variants ingestion coverage: %s", upload_id, coverage)
  ```

#### 4. Sequential Processing (No Parallelization)
- **Issue**: Ingestion is fully sequential
- **Impact**: 4 CSVs × 60s average = 240 seconds minimum
- **Current Flow**:
  ```
  Orders CSV → Wait → Variants CSV → Wait → Catalog CSV → Wait → Inventory CSV
  ```

#### 5. Redundant Lookups
- **Issue**: Every ingestion does `get_run_id_for_upload()` + `_canonical_upload_id()`
- **Impact**: 2 DB queries × 4 CSV types = 8 extra queries
- **Current Code**:
  ```python
  # services/csv_processor.py:482-483
  run_id = await storage.get_run_id_for_upload(upload_id)
  target_upload_id = await self._canonical_upload_id(upload_id, run_id)
  ```

---

## PART 2: Bundle Save Bottlenecks

### Root Cause Analysis

#### 1. Individual refresh() Calls (CRITICAL)
- **Issue**: One SELECT query per bundle recommendation
- **Impact**: 40 bundles × 500ms = 20+ seconds of pure SELECT overhead
- **Current Code**:
  ```python
  # services/storage.py:662-663
  for rec in recommendations:
      await session.refresh(rec)  # N individual SELECTs!
  return recommendations
  ```
- **Why It's Slow**:
  - Each refresh() = 1 roundtrip to CockroachDB
  - Network latency: ~100-500ms per query to Cloud SQL
  - Completely unnecessary - we already have all data

#### 2. No Batching for Large Bundle Sets
- **Issue**: All bundles saved in single transaction
- **Impact**: For 100+ bundles, transaction locks slow down writes
- **Current Code**:
  ```python
  # services/storage.py:658-660
  recommendations = [BundleRecommendation(**row) for row in filtered_rows]
  session.add_all(recommendations)  # All at once
  await session.commit()  # Single commit
  ```

#### 3. Unnecessary DELETE Before INSERT
- **Issue**: Deletes existing recommendations with matching IDs
- **Impact**: Extra DELETE query adds 1-3 seconds
- **Current Code**:
  ```python
  # services/storage.py:652-656
  existing_ids = [row.get("id") for row in filtered_rows if row.get("id")]
  if existing_ids:
      await session.execute(
          delete(BundleRecommendation).where(BundleRecommendation.id.in_(existing_ids))
      )
  ```

---

## PART 3: Optimization Strategy

### Quick Wins (1-2 hours implementation)

#### QW1: Remove Individual refresh() Calls
**Impact**: Bundle save 27s → 2s (93% faster)
**Implementation**:
```python
# services/storage.py:658-664 - REMOVE refresh loop
recommendations = [BundleRecommendation(**row) for row in filtered_rows]
session.add_all(recommendations)
await session.commit()
# REMOVE THIS:
# for rec in recommendations:
#     await session.refresh(rec)
return recommendations  # Return as-is, no refresh needed
```
**Reasoning**: The refresh() only re-fetches auto-generated fields (id, created_at, updated_at).
Caller doesn't need these values - they just discard the return value.

#### QW2: Batch Catalog/Inventory Ingestion
**Impact**: Catalog 175s → 15-25s, Inventory 64s → 8-12s (85% faster)
**Implementation**:
```python
# services/storage.py - Add batching to create_catalog_snapshots
async def create_catalog_snapshots(self, catalog_data: List[Dict[str, Any]]) -> List[CatalogSnapshot]:
    if not catalog_data:
        return []

    # ... existing dedup logic ...

    BATCH_SIZE = 100  # CockroachDB sweet spot
    async with self.get_session() as session:
        for i in range(0, len(catalog_data), BATCH_SIZE):
            batch = catalog_data[i:i+BATCH_SIZE]
            stmt = pg_insert(CatalogSnapshot).values(batch)
            upsert = stmt.on_conflict_do_update(
                index_elements=[CatalogSnapshot.csv_upload_id, CatalogSnapshot.variant_id],
                set_=self._overwrite_all_columns(CatalogSnapshot.__table__, stmt)
            )
            await session.execute(upsert)
        await session.commit()  # Single commit for all batches
    return []
```

#### QW3: Make Coverage Summary Async
**Impact**: Variants ingestion 30s faster
**Implementation**:
```python
# services/csv_processor.py:472-476 - Remove blocking coverage query
# REMOVE THIS:
# try:
#     coverage = await storage.summarize_upload_coverage(target_upload_id, run_id)
#     logger.info("[%s] Post-variants ingestion coverage: %s", upload_id, coverage)
# except Exception as exc:
#     logger.warning("[%s] Coverage summary after variants failed: %s", upload_id, exc)

# Coverage can be computed on-demand when viewing bundles
```

#### QW4: Cache run_id and canonical_upload_id
**Impact**: Eliminates 8 redundant DB queries
**Implementation**:
```python
# services/csv_processor.py - Add class-level cache
class CsvProcessor:
    def __init__(self):
        self._upload_cache = {}  # {upload_id: (run_id, canonical_id)}

    async def _get_canonical_upload_cached(self, upload_id: str):
        if upload_id in self._upload_cache:
            return self._upload_cache[upload_id]

        run_id = await storage.get_run_id_for_upload(upload_id)
        canonical_id = await self._canonical_upload_id(upload_id, run_id)
        self._upload_cache[upload_id] = (run_id, canonical_id)
        return run_id, canonical_id
```

### Medium-Term Optimizations (4-8 hours implementation)

#### MT1: Parallelize Independent CSV Ingestion
**Impact**: Total ingestion time reduced by 50%
**Implementation**:
```python
# Catalog and Inventory are independent - can run in parallel
# Orders and Variants are independent - can run in parallel
import asyncio

async def process_all_csvs(uploads: Dict[str, List[Dict]]):
    # Phase 1: Independent ingestion
    await asyncio.gather(
        process_catalog_csv(uploads['catalog'], upload_id),
        process_inventory_csv(uploads['inventory'], upload_id),
    )

    # Phase 2: Order-dependent ingestion
    await asyncio.gather(
        process_orders_csv(uploads['orders'], upload_id),
        process_variants_csv(uploads['variants'], upload_id),
    )
```

#### MT2: Use COPY FROM for Bulk Inserts (PostgreSQL Fast Path)
**Impact**: Catalog 175s → 3-5s (97% faster)
**Implementation**:
```python
# Use psycopg's COPY instead of INSERT for new uploads
from io import StringIO
import csv

async def bulk_copy_catalog(catalog_data: List[Dict]):
    # Convert to CSV in memory
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=catalog_data[0].keys())
    writer.writerows(catalog_data)
    output.seek(0)

    # Use COPY FROM (10-50x faster than INSERT)
    raw_conn = await session.connection()
    await raw_conn.copy_from(output, 'catalog_snapshot', sep=',', columns=list(catalog_data[0].keys()))
```

#### MT3: Batch Bundle Recommendations
**Impact**: For 100+ bundles, prevents transaction lock slowdown
**Implementation**:
```python
# services/storage.py
async def create_bundle_recommendations(self, recommendations_data):
    BATCH_SIZE = 50
    all_recs = []

    async with self.get_session() as session:
        for i in range(0, len(filtered_rows), BATCH_SIZE):
            batch = filtered_rows[i:i+BATCH_SIZE]
            recommendations = [BundleRecommendation(**row) for row in batch]
            session.add_all(recommendations)
            await session.flush()  # Flush each batch
            all_recs.extend(recommendations)

        await session.commit()
    return all_recs  # No refresh needed
```

### Long-Term Optimizations (1-2 days implementation)

#### LT1: Background Job for Coverage Computation
**Impact**: Fully decouple slow analytics from ingestion
**Implementation**:
- Move coverage summary to background Celery task
- Trigger after all CSVs ingested
- Store results in Redis cache

#### LT2: Connection Pooling Tuning
**Impact**: Reduce connection overhead by 30%
**Current**: Default SQLAlchemy pool (5-10 connections)
**Optimized**:
```python
# database.py
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,         # Increase pool
    max_overflow=10,      # Allow burst
    pool_pre_ping=True,   # Check connection health
    pool_recycle=3600,    # Recycle hourly
)
```

#### LT3: Database Indexes Optimization
**Impact**: 20-40% faster lookups
**Add Indexes**:
```sql
-- Speed up order line queries
CREATE INDEX CONCURRENTLY idx_order_lines_variant_id ON order_lines(variant_id);
CREATE INDEX CONCURRENTLY idx_order_lines_upload_id ON order_lines(csv_upload_id);

-- Speed up catalog queries
CREATE INDEX CONCURRENTLY idx_catalog_upload_run ON catalog_snapshot(csv_upload_id, variant_id);

-- Speed up bundle queries
CREATE INDEX CONCURRENTLY idx_bundles_upload_rank ON bundle_recommendations(csv_upload_id, rank_position);
```

---

## PART 4: Implementation Priority

### Phase 1: Critical Fixes (Deploy Today)
1. **QW1**: Remove refresh() loop (5 min) - Bundle save 27s → 2s
2. **QW2**: Batch catalog/inventory (30 min) - Ingestion 175s → 15-25s
3. **QW3**: Remove coverage query (5 min) - Variants 30s faster

**Expected Impact**:
- Bundle save: **93% faster** (27s → 2s)
- Catalog ingestion: **85% faster** (175s → 15-25s)
- Total ingestion: **60% faster** (300s → 120s)

### Phase 2: Important Improvements (Deploy This Week)
4. **QW4**: Cache run_id lookups (20 min)
5. **MT1**: Parallelize CSV processing (2 hours)
6. **MT3**: Batch bundle saves (1 hour)

**Expected Impact**:
- Total ingestion: **75% faster** (300s → 75s)
- Bundle save handles 1000+ bundles efficiently

### Phase 3: Advanced Optimizations (Next Sprint)
7. **MT2**: COPY FROM bulk inserts (4 hours)
8. **LT2**: Connection pooling tuning (1 hour)
9. **LT3**: Database indexes (2 hours)

**Expected Impact**:
- Catalog ingestion: **97% faster** (175s → 3-5s)
- All queries 20-40% faster

---

## PART 5: Testing Strategy

### Performance Benchmarks
```python
# Add to tests/test_performance.py
async def test_catalog_ingestion_performance():
    """Catalog ingestion should complete in under 10 seconds for 259 rows"""
    start = time.time()
    await storage.create_catalog_snapshots(catalog_data)
    duration = time.time() - start
    assert duration < 10, f"Catalog ingestion too slow: {duration}s"

async def test_bundle_save_performance():
    """Bundle save should complete in under 3 seconds for 40 bundles"""
    start = time.time()
    await storage.create_bundle_recommendations(bundle_data)
    duration = time.time() - start
    assert duration < 3, f"Bundle save too slow: {duration}s"
```

### Monitoring
```python
# Add timing metrics to existing logs
logger.info(
    "[%s] PERF: catalog_ingestion_ms=%d rows=%d ms_per_row=%.1f",
    upload_id,
    duration_ms,
    len(catalog_data),
    duration_ms / len(catalog_data)
)
```

---

## PART 6: Rollout Plan

### Step 1: Deploy Quick Wins (30 minutes)
```bash
# 1. Remove refresh() loop
# 2. Add batching to create_catalog_snapshots, create_inventory_levels
# 3. Remove coverage summary call
git add services/storage.py services/csv_processor.py
git commit -m "PERF: 10x faster ingestion + bundle save (batching + remove refresh)"
git push
```

### Step 2: Validate Performance
```bash
# Upload test CSVs and check logs for timing improvements
# Expected: Catalog 175s → 15s, Bundle save 27s → 2s
```

### Step 3: Deploy Medium-Term Optimizations (1 week)
```bash
# Add parallelization, caching, batch bundle saves
git commit -m "PERF: Parallel CSV processing + batch bundle saves"
```

### Step 4: Monitor Production Metrics
- Track ingestion duration per CSV type
- Track bundle save duration
- Alert if ingestion > 30s or bundle save > 5s

---

## Expected Final Performance

| Operation | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|-----------|---------|---------------|---------------|---------------|
| **Catalog Ingestion** | 175s | 15-25s | 10-15s | 3-5s |
| **Inventory Ingestion** | 64s | 8-12s | 5-8s | 2-3s |
| **Orders Ingestion** | 5-7min (TIMEOUT) | 60-90s | 30-45s | 15-20s |
| **Variants Ingestion** | 120s+ | 60-80s | 30-40s | 10-15s |
| **Bundle Save (40 bundles)** | 27s | **2s** | **1-2s** | **1s** |
| **Total Ingestion Time** | 300-420s | 120-180s | 60-90s | 30-45s |

**Overall Improvement**: **85-95% faster** across the board.
