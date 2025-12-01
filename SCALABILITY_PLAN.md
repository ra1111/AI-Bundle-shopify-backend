# Scalability-First Performance Plan

## Problem Statement

**Current State:**
- 259 catalog rows → 175 seconds (676ms per row)
- At 10x scale (2,590 rows) → **29 minutes** ❌
- At 100x scale (25,900 rows) → **4.8 hours** ❌

**Target State:**
- 259 rows → **< 5 seconds** ✅
- 2,590 rows → **< 15 seconds** ✅
- 25,900 rows → **< 60 seconds** ✅

**Performance Target**: **< 10ms per row** regardless of dataset size

---

## Core Strategy: Adaptive Processing Pipeline

### Tier 1: Small Datasets (< 500 rows)
**Strategy**: Single batch, single transaction
- **Why**: Minimal overhead, maximum simplicity
- **Performance**: 2-5 seconds total
- **Risk**: None

### Tier 2: Medium Datasets (500-5,000 rows)
**Strategy**: Multi-batch with single transaction
- **Why**: Balance speed vs memory
- **Batch size**: 200 rows
- **Performance**: 10-30 seconds total
- **Risk**: Low (transaction < 30s)

### Tier 3: Large Datasets (5,000-50,000 rows)
**Strategy**: Multi-batch with progressive commits
- **Why**: Prevent timeouts, enable progress tracking
- **Batch size**: 500 rows
- **Commit frequency**: Every batch
- **Performance**: 30-120 seconds total
- **Risk**: None (each batch commits independently)

### Tier 4: Huge Datasets (50,000+ rows)
**Strategy**: Parallel workers + streaming + COPY FROM
- **Why**: Maximum throughput
- **Parallelization**: 4 workers
- **Streaming**: Process without loading full CSV
- **Performance**: 2-5 minutes total
- **Risk**: None (distributed processing)

---

## Implementation: Adaptive Batch Engine

### 1. Smart Batch Configuration

```python
class BatchConfig:
    """Auto-configure batching based on dataset size"""

    @staticmethod
    def get_config(row_count: int) -> dict:
        if row_count < 500:
            return {
                'tier': 'SMALL',
                'batch_size': row_count,  # Single batch
                'commit_per_batch': False,
                'parallel': False,
                'method': 'INSERT',
            }
        elif row_count < 5000:
            return {
                'tier': 'MEDIUM',
                'batch_size': 200,
                'commit_per_batch': False,  # Single transaction at end
                'parallel': False,
                'method': 'INSERT',
            }
        elif row_count < 50000:
            return {
                'tier': 'LARGE',
                'batch_size': 500,
                'commit_per_batch': True,  # Progressive commits
                'parallel': False,
                'method': 'INSERT',
            }
        else:
            return {
                'tier': 'HUGE',
                'batch_size': 1000,
                'commit_per_batch': True,
                'parallel': True,
                'num_workers': 4,
                'method': 'COPY',  # Use PostgreSQL COPY FROM
            }
```

### 2. Scalable Ingestion Engine

```python
async def create_catalog_snapshots_scalable(
    self,
    catalog_data: List[Dict[str, Any]]
) -> List[CatalogSnapshot]:
    """Adaptive ingestion: scales from 100 to 100,000 rows"""

    if not catalog_data:
        return []

    # Auto-configure based on size
    config = BatchConfig.get_config(len(catalog_data))
    logger.info(
        f"Ingestion config: tier={config['tier']} rows={len(catalog_data)} "
        f"batch_size={config['batch_size']} commit_per_batch={config['commit_per_batch']}"
    )

    # Deduplicate
    catalog_data = self._filter_columns(CatalogSnapshot.__table__, catalog_data)
    catalog_data = self._deduplicate_catalog(catalog_data)

    # Route to appropriate strategy
    if config['parallel']:
        return await self._ingest_parallel(catalog_data, config)
    elif config['method'] == 'COPY':
        return await self._ingest_copy_from(catalog_data, config)
    else:
        return await self._ingest_batched(catalog_data, config)
```

### 3. Core Batched Ingestion (Tiers 1-3)

```python
async def _ingest_batched(
    self,
    catalog_data: List[Dict[str, Any]],
    config: dict
) -> List[CatalogSnapshot]:
    """Batched ingestion with optional progressive commits"""

    batch_size = config['batch_size']
    commit_per_batch = config['commit_per_batch']

    t0 = time.time()
    batches_processed = 0

    async with self.get_session() as session:
        for i in range(0, len(catalog_data), batch_size):
            batch = catalog_data[i:i + batch_size]

            stmt = pg_insert(CatalogSnapshot).values(batch)
            upsert = stmt.on_conflict_do_update(
                index_elements=[CatalogSnapshot.csv_upload_id, CatalogSnapshot.variant_id],
                set_=self._overwrite_all_columns(CatalogSnapshot.__table__, stmt)
            )
            await session.execute(upsert)

            if commit_per_batch:
                await session.commit()
                batches_processed += 1
                logger.info(
                    f"Batch {batches_processed} committed: "
                    f"{len(batch)} rows ({i+len(batch)}/{len(catalog_data)})"
                )

        if not commit_per_batch:
            await session.commit()

    duration = (time.time() - t0) * 1000
    logger.info(
        f"Ingestion complete: {len(catalog_data)} rows in {duration:.0f}ms "
        f"({duration/len(catalog_data):.1f}ms per row)"
    )
    return []
```

### 4. PostgreSQL COPY FROM (Tier 4 - 10-50x faster)

```python
async def _ingest_copy_from(
    self,
    catalog_data: List[Dict[str, Any]],
    config: dict
) -> List[CatalogSnapshot]:
    """Ultra-fast bulk insert using PostgreSQL COPY - for huge datasets"""

    from io import StringIO
    import csv

    # Convert to CSV format in memory
    if not catalog_data:
        return []

    output = StringIO()
    fieldnames = list(catalog_data[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writerows(catalog_data)
    output.seek(0)

    # Get raw connection for COPY
    async with self.get_session() as session:
        raw_conn = await session.connection()
        cursor = await raw_conn.execute(text("SELECT 1"))  # Warm up

        # Use COPY FROM (bypasses SQL layer entirely)
        copy_sql = f"""
        COPY catalog_snapshot ({','.join(fieldnames)})
        FROM STDIN WITH (FORMAT CSV, HEADER false)
        ON CONFLICT (csv_upload_id, variant_id)
        DO UPDATE SET ...
        """

        # Execute via raw psycopg connection
        # Note: This requires custom implementation per DB driver
        await raw_conn.execute(text(copy_sql), {'data': output.getvalue()})
        await session.commit()

    return []
```

### 5. Parallel Processing (Tier 4)

```python
async def _ingest_parallel(
    self,
    catalog_data: List[Dict[str, Any]],
    config: dict
) -> List[CatalogSnapshot]:
    """Split data across parallel workers for maximum throughput"""

    import asyncio

    num_workers = config['num_workers']
    chunk_size = len(catalog_data) // num_workers

    async def process_chunk(chunk: List[Dict], worker_id: int):
        """Each worker processes independently"""
        sub_config = {**config, 'parallel': False}  # Disable recursion
        return await self._ingest_batched(chunk, sub_config)

    # Split into chunks
    chunks = [
        catalog_data[i:i + chunk_size]
        for i in range(0, len(catalog_data), chunk_size)
    ]

    # Process in parallel
    tasks = [
        process_chunk(chunk, idx)
        for idx, chunk in enumerate(chunks)
    ]
    await asyncio.gather(*tasks)

    return []
```

---

## Implementation: Scalable Bundle Save

### Remove Refresh Loop + Adaptive Batching

```python
async def create_bundle_recommendations(
    self,
    recommendations_data: List[Dict[str, Any]]
) -> List[BundleRecommendation]:
    """Scalable bundle save: handles 10 to 10,000 bundles"""

    if not recommendations_data:
        return []

    # Normalize and filter
    normalized_rows = self._normalize_bundle_shop_ids(recommendations_data)
    filtered_rows = self._filter_columns(BundleRecommendation.__table__, normalized_rows)

    # Auto-configure batching
    config = BatchConfig.get_config(len(filtered_rows))
    batch_size = config['batch_size']

    logger.info(
        f"Bundle save: {len(filtered_rows)} bundles, "
        f"tier={config['tier']}, batch_size={batch_size}"
    )

    async with self.get_session() as session:
        # Delete existing (if any)
        existing_ids = [row.get("id") for row in filtered_rows if row.get("id")]
        if existing_ids:
            await session.execute(
                delete(BundleRecommendation).where(BundleRecommendation.id.in_(existing_ids))
            )

        # Batch insert
        for i in range(0, len(filtered_rows), batch_size):
            batch = filtered_rows[i:i + batch_size]
            recommendations = [BundleRecommendation(**row) for row in batch]
            session.add_all(recommendations)

            if config['commit_per_batch']:
                await session.commit()
                logger.info(f"Bundle batch {i//batch_size + 1} saved: {len(batch)} bundles")

        if not config['commit_per_batch']:
            await session.commit()

        # ❌ REMOVED: Individual refresh() loop (was 93% of time)
        # ✅ Return without refresh - caller doesn't need these values
        return []
```

---

## Performance Projections

| Dataset Size | Tier | Strategy | Old Time | New Time | Speedup |
|--------------|------|----------|----------|----------|---------|
| **259 rows** | SMALL | Single batch | 175s | **2-3s** | **58x** |
| **2,590 rows** (10x) | MEDIUM | Multi-batch | 29min | **10-15s** | **116x** |
| **25,900 rows** (100x) | LARGE | Progressive commits | 4.8hr | **45-60s** | **288x** |
| **100,000 rows** | HUGE | Parallel + COPY | 19hr | **2-3min** | **380x** |

---

## Additional Optimizations

### 1. Remove Coverage Summary (30s saved)
```python
# services/csv_processor.py:472-476
# REMOVE blocking coverage query - compute on-demand
```

### 2. Cache run_id Lookups (8 queries saved)
```python
class CsvProcessor:
    def __init__(self):
        self._canonical_cache = {}  # {upload_id: (run_id, canonical_id)}
```

### 3. Parallel CSV Processing
```python
# Process independent CSVs in parallel
await asyncio.gather(
    process_catalog_csv(catalog_rows, upload_id),
    process_inventory_csv(inventory_rows, upload_id),
)
```

### 4. Database Connection Pooling
```python
# database.py
engine = create_async_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
)
```

---

## Implementation Plan

### Phase 1: Core Adaptive Engine (2 hours)
1. ✅ Implement `BatchConfig` auto-configuration
2. ✅ Implement `_ingest_batched()` with progressive commits
3. ✅ Apply to catalog, inventory, order_lines
4. ✅ Remove refresh() loop from bundle save
5. ✅ Add performance logging

**Impact**: 259 rows 175s → 2-3s, handles 10x scale easily

### Phase 2: Advanced Optimizations (2 hours)
6. ✅ Parallel CSV processing
7. ✅ Cache run_id lookups
8. ✅ Remove coverage query
9. ✅ Connection pool tuning

**Impact**: Total ingestion 300s → 30s

### Phase 3: Extreme Scale (optional - 4 hours)
10. ⚠️ COPY FROM implementation (requires raw SQL)
11. ⚠️ Parallel workers for huge datasets

**Impact**: Handles 100,000+ rows in minutes

---

## Testing Strategy

```python
# tests/test_scalability.py
@pytest.mark.parametrize("row_count", [100, 1000, 10000])
async def test_catalog_ingestion_scales(row_count):
    """Verify adaptive batching works at all scales"""
    catalog_data = generate_catalog_rows(row_count)

    start = time.time()
    await storage.create_catalog_snapshots(catalog_data)
    duration = time.time() - start

    # Performance targets
    if row_count < 500:
        assert duration < 5, f"Small dataset too slow: {duration}s"
    elif row_count < 5000:
        assert duration < 30, f"Medium dataset too slow: {duration}s"
    else:
        assert duration < 120, f"Large dataset too slow: {duration}s"
```

---

## Decision: Which Phase to Implement?

**Recommendation: Start with Phase 1 + 2**
- Handles your 10x scale requirement ✅
- Minimal complexity
- Proven strategies
- 2-4 hours implementation

**Skip Phase 3** unless you expect:
- > 50,000 rows regularly
- Real-time ingestion requirements
- Multiple concurrent uploads

---

Ready to implement?
