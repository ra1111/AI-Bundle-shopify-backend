# Bundle API Log Analysis & Fixes

**Date:** 2025-11-04
**Branch:** `claude/debug-bundle-api-logs-011CUoWBSq3iUEXnTghEb4gU`
**Analysis Period:** 20:37:27 - 20:38:14

## Summary

Analyzed Google Cloud Run logs for the bundle-api service and identified critical issues causing errors and performance problems. Applied fixes to resolve feature flag type mismatches and configuration errors.

---

## Issues Identified

### 1. Feature Flag Type Mismatch (CRITICAL - FIXED)

**Error Message:**
```
Error getting flag bundling.staged_thresholds: unhashable type: 'list'
```

**Root Cause:**
- `FeatureFlagsManager.get_flag()` method declared `default: bool = False`
- `bundle_generator.py` line 826 called with list default: `get_flag("bundling.staged_thresholds", [3, 5, 10, 20, 40])`
- Lists are unhashable in Python and caused failures in caching/comparison operations
- Multiple configuration flags were missing from the `default_flags` dictionary

**Impact:**
- Feature flag lookups failed
- Bundle generation pipeline initialization errors
- Fallback to incorrect default values

**Fix Applied:**
- Updated `get_flag()` signature from `default: bool = False` to `default: Any = False`
- Updated `set_flag()`, `_persist_flag()`, and `_record_flag_change()` to accept `Any` type
- Updated `FeatureFlag` dataclass to accept `Any` type for value field
- Added missing configuration flags to `default_flags`:
  ```python
  # Staged publishing configuration
  "bundling.staged_publish_enabled": True,
  "bundling.staged_thresholds": [3, 5, 10, 20, 40],
  "bundling.staged_hard_cap": 40,
  "bundling.staged_prefer_high_score": True,
  "bundling.staged_cycle_interval_seconds": 0,
  "bundling.staged.soft_guard_seconds": 45,
  "bundling.staged.wave_batch_size": 10,
  "bundling.staged.backpressure_queue_threshold": 0,
  "bundling.staged.backpressure_cooldown_waves": 1,

  # Finalization concurrency configuration
  "bundling.finalize.concurrent_tracks.copy": 3,
  "bundling.finalize.concurrent_tracks.pricing": 2,

  # Data mapping configuration
  "data_mapping.cache_ttl_seconds": 1800,
  "data_mapping.target_chunk_p95_ms": 800,
  ```

**Files Modified:**
- `services/feature_flags.py` (lines 15-20, 140-161, 190-218, 343-351, 405-419)

---

### 2. Frequent Database ROLLBACK Operations (OBSERVED)

**Pattern:**
Multiple ROLLBACK operations observed at:
- 20:37:51.965
- 20:37:56.818
- 20:37:58.517
- 20:38:00.218
- 20:38:02.918
- 20:38:04.417
- 20:38:08.018 & 20:38:08.219
- 20:38:12.319 & 20:38:12.454

**Possible Causes:**
1. **Transaction Timeouts:** Long-running transactions exceeding timeout limits
2. **Connection Pool Exhaustion:** Limited connection pool (pool_cap=20) with high concurrency
3. **Deadlocks:** Multiple concurrent operations competing for database resources
4. **Retry Logic:** Application-level retries triggering ROLLBACK on failures

**Evidence from Logs:**
- `DataMapper mapping start | concurrency_limit=1 (pool_cap=20 default=25)` - Very restrictive concurrency
- Multiple `BEGIN (implicit)` followed by `ROLLBACK` without `COMMIT`
- Query caching indicators: `[cached since X.Xs ago]` suggests repeated queries

**Recommendations:**
1. Review database transaction isolation levels
2. Increase connection pool size if needed
3. Add connection pool monitoring and alerting
4. Implement circuit breaker pattern for database operations
5. Review query performance and add missing indexes
6. Consider increasing `concurrency_limit` in DataMapper after testing

---

### 3. Multiple Pipeline Restarts (OBSERVED)

**Pattern:**
Bundle generation pipeline started multiple times in short succession:
- 20:37:52: "BUNDLE GENERATION PIPELINE STARTED"
- 20:37:57: "BUNDLE GENERATION PIPELINE STARTED" (5 seconds later)
- 20:37:59: "BUNDLE GENERATION PIPELINE STARTED" (2 seconds later)

Each start included:
- "Phase 1: Data Mapping & Enrichment - STARTED"
- "Soft deadline set to 270.0s"
- "Started pipeline tracking"

**Possible Causes:**
1. **Timeout/Retry Logic:** Pipeline timing out and being retried by orchestration
2. **Duplicate Requests:** Multiple concurrent requests triggering same upload_id processing
3. **Health Check Failures:** Service restarts due to health check timeouts
4. **Memory/Resource Constraints:** Cloud Run instance restarting due to resource limits

**Evidence:**
- All restarts for same upload: `aa4e2edb-7d06-4b82-a42f-2c72585043ee`
- Same run_id: `0659c5c9-b7cb-4e2a-a2f9-34265805cc6c`
- Shop: `rahular1.myshopify.com`
- Soft deadline: 270s (4.5 minutes)

**Recommendations:**
1. Add request deduplication/idempotency checks
2. Implement distributed locking for upload processing
3. Review timeout configuration (currently 300s max, 270s soft)
4. Add metrics for pipeline completion rates
5. Monitor Cloud Run instance lifecycle and restart patterns

---

## Log Metrics

### Database Activity
- **Total Query Executions:** 100+ within 47-second window
- **Cached Queries:** High percentage with "cached since X.Xs ago" messages
- **Query Types:** Primarily SELECT on variants, order_lines, csv_uploads, generation_progress
- **Transaction Pattern:** Multiple short-lived transactions with frequent ROLLBACKs

### Data Mapping Performance
- **Batch Size:** 100 items per chunk
- **Concurrency Limit:** 1 (very restrictive, down from default 25)
- **Pool Capacity:** 20 connections
- **Processing Pattern:** Sequential chunk processing with 100ms+ per chunk

### Upload Details
- **Upload ID:** `aa4e2edb-7d06-4b82-a42f-2c72585043ee`
- **Run ID:** `0659c5c9-b7cb-4e2a-a2f9-34265805cc6c`
- **Shop:** `rahular1.myshopify.com`
- **Phase:** Data Mapping & Enrichment
- **Status:** In Progress (repeated restarts)

---

## Verification Steps

To verify the fixes:

1. **Feature Flag Fix:**
   ```python
   # Test that list values work correctly
   from services.feature_flags import feature_flags

   value = feature_flags.get_flag("bundling.staged_thresholds", [3, 5, 10, 20, 40])
   assert value == [3, 5, 10, 20, 40]
   assert isinstance(value, list)
   ```

2. **Monitor Logs:**
   - Check for absence of "unhashable type: 'list'" errors
   - Verify bundle generation starts successfully
   - Monitor ROLLBACK frequency

3. **Database Monitoring:**
   - Track connection pool utilization
   - Monitor transaction duration
   - Alert on excessive ROLLBACKs

---

## Next Steps

### Immediate (Completed)
- [x] Fix feature flag type mismatch
- [x] Add missing configuration flags
- [x] Document findings

### Short Term (Recommended)
- [ ] Review database connection pool settings
- [ ] Add request deduplication for bundle generation
- [ ] Implement distributed locking for upload processing
- [ ] Add performance monitoring dashboards
- [ ] Review and optimize SQL queries

### Long Term (Suggested)
- [ ] Implement circuit breaker pattern
- [ ] Add comprehensive error tracking (Sentry/similar)
- [ ] Performance testing with realistic data volumes
- [ ] Auto-scaling configuration review
- [ ] Consider read replicas for heavy SELECT queries

---

## References

- Feature Flags PR-8: Dynamic feature control system
- Data Mapper: services/data_mapper.py
- Bundle Generator: services/bundle_generator.py
- Feature Flags: services/feature_flags.py
