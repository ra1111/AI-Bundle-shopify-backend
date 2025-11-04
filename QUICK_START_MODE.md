# Quick-Start Mode for First-Time Installations

## Overview

Quick-Start Mode is an optimized bundle generation pipeline designed specifically for **first-time merchant installations**. It ensures new merchants see bundle recommendations within **seconds to 2 minutes**, providing an excellent onboarding experience while the comprehensive analysis runs in the background.

## Problem Solved

Previously, all merchants (new and existing) used the same bundle generation pipeline with:
- 20-minute soft watchdog timeout
- Processing all products
- Full ML pipeline with 4-8 objectives
- Extensive optimization phases

This meant **new merchants had to wait up to 20 minutes** before seeing any bundle recommendations, leading to:
- Poor first impression
- Merchant drop-off during onboarding
- Reduced engagement with the app

## Solution: Two-Tier Architecture

### 🚀 Fast Path: Quick-Start Mode (First-Time Installs)
- **Target**: 30 seconds - 2 minutes
- **Scope**: Top 50 products by sales volume
- **Objectives**: 2 high-priority objectives (increase_aov, clear_slow_movers)
- **Output**: 10 preview bundles
- **Method**: Simple co-occurrence analysis (no heavy ML)
- **Triggers**: Automatically for merchants where `ShopSyncStatus.initial_sync_completed = False`

### 🔄 Full Pipeline (Background Follow-Up)
- **Target**: 20 minutes (existing soft watchdog)
- **Scope**: All products
- **Objectives**: 4-8 business objectives with Pareto optimization
- **Output**: 50+ comprehensive bundles
- **Method**: Full v2 pipeline with ML, pricing optimization, ranking
- **Triggers**: Queued automatically after quick-start completes

### 📊 Regular Path (Existing Merchants)
- Uses the full pipeline as before
- No changes to existing merchant experience
- Ideal for cron jobs during off-hours

## Technical Architecture

### 1. Detection: Is First-Time Install?

```python
# services/storage.py
async def is_first_time_install(self, shop_id: str) -> bool:
    """
    Returns True if:
    - No sync status exists for the shop
    - Sync status exists but initial_sync_completed is False

    Returns False if initial_sync_completed is True (regular user)
    """
    status = await session.get(ShopSyncStatus, shop_id)
    if not status:
        return True
    return not status.initial_sync_completed
```

### 2. Quick-Start Bundle Generation

```python
# services/bundle_generator.py
async def generate_quick_start_bundles(
    self,
    csv_upload_id: str,
    max_products: int = 50,      # Limit products
    max_bundles: int = 10,       # Limit bundles
    timeout_seconds: int = 120   # 2-minute hard timeout
) -> Dict[str, Any]:
    """
    Optimized fast-path pipeline:
    - Limit to top 50 products by sales volume
    - Simple co-occurrence analysis (no ML)
    - 10% discount on all bundles
    - 2-minute deadline
    """
```

#### Quick-Start Pipeline Phases

1. **Phase 1: Data Loading (10-40% progress)**
   - Load order lines
   - Count sales by SKU
   - Select top N products
   - Filter order lines to top products only

2. **Phase 2: Simple Scoring (40-70% progress)**
   - Use only 2 objectives: `increase_aov`, `clear_slow_movers`
   - Simple flag-based scoring (no complex ML)
   - Score based on `is_slow_mover` and `is_high_margin` flags

3. **Phase 3: Bundle Generation (70-90% progress)**
   - Build co-occurrence matrix from orders
   - Count SKU pairs bought together
   - Sort by frequency
   - Take top N pairs as bundles
   - Apply simple 10% discount

4. **Phase 4: Persistence (90-100% progress)**
   - Save bundles with `discount_reference = "__quick_start_{upload_id}__"`
   - Mark upload as completed
   - Notify merchant

### 3. Orchestration: generate_bundles_background

```python
# routers/bundle_recommendations.py
async def generate_bundles_background(csv_upload_id, resume_only=False):
    # Acquire shop lock
    shop_id = lock_context["shop_id"]

    # Check if first-time install
    is_first_install = await storage.is_first_time_install(shop_id)

    if is_first_install and QUICK_START_ENABLED:
        # FAST PATH: Quick-Start Mode
        generation_result = await generator.generate_quick_start_bundles(
            csv_upload_id,
            max_products=QUICK_START_MAX_PRODUCTS,
            max_bundles=QUICK_START_MAX_BUNDLES,
            timeout_seconds=QUICK_START_TIMEOUT_SECONDS
        )

        # Mark upload completed
        await storage.safe_mark_upload_completed(csv_upload_id)

        # Notify merchant (preview bundles ready)
        await notify_partial_ready(csv_upload_id, metrics)

        # Queue full generation in background
        pipeline_scheduler.schedule(
            _run_full_generation_after_quickstart(csv_upload_id, shop_id)
        )

        return  # Exit early

    # NORMAL PATH: Full Pipeline
    generation_result = await generator.generate_bundle_recommendations(csv_upload_id)
    # ... continue with full pipeline
```

### 4. Background Full Generation

```python
def _run_full_generation_after_quickstart(csv_upload_id, shop_id):
    """
    Runs after quick-start preview is shown to merchant.

    - Waits 5 seconds to avoid overload
    - Runs full v2 pipeline (20-minute soft watchdog)
    - Replaces quick-start bundles with comprehensive results
    - Marks shop sync as completed
    - Notifies merchant when full bundles are ready
    """
```

## Configuration

All quick-start settings are configurable via environment variables:

```bash
# Enable/disable quick-start mode
QUICK_START_ENABLED=true                 # Default: true

# Quick-start timeout
QUICK_START_TIMEOUT_SECONDS=120          # Default: 120 (2 minutes)

# Product and bundle limits
QUICK_START_MAX_PRODUCTS=50              # Default: 50
QUICK_START_MAX_BUNDLES=10               # Default: 10
```

To disable quick-start mode:
```bash
QUICK_START_ENABLED=false
```

## Database Schema

### ShopSyncStatus Table

```python
class ShopSyncStatus(Base):
    __tablename__ = "shop_sync_status"

    shop_id: str                            # Primary key
    initial_sync_completed: bool = False    # Key field for detection
    last_sync_started_at: datetime
    last_sync_completed_at: datetime
    created_at: datetime
    updated_at: datetime
```

**Lifecycle**:
1. **First install**: No record exists → `is_first_time_install()` returns `True`
2. **Quick-start completes**: Upload marked completed, but `initial_sync_completed` stays `False`
3. **Full generation completes**: `mark_shop_sync_completed()` sets `initial_sync_completed = True`
4. **Subsequent runs**: `is_first_time_install()` returns `False` → normal pipeline

## Flow Diagram

```
┌─────────────────────────────────────────┐
│  Merchant Installs App & Uploads Data   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  generate_bundles_background() called   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│  Check: is_first_time_install(shop_id)? │
└──────────────┬──────────────────────────┘
               │
         ┌─────┴─────┐
         │           │
      YES│           │NO
         │           │
         ▼           ▼
┌──────────────┐  ┌──────────────────────┐
│ QUICK-START  │  │   NORMAL PIPELINE    │
│   MODE       │  │                      │
│ (2 minutes)  │  │  (20 min watchdog)   │
└──────┬───────┘  └──────────────────────┘
       │
       ▼
┌────────────────────────────────────┐
│  10 Preview Bundles Generated      │
│  (30s - 2 min)                     │
└──────┬─────────────────────────────┘
       │
       ▼
┌────────────────────────────────────┐
│  Merchant Sees Bundles Immediately │
└──────┬─────────────────────────────┘
       │
       ▼
┌────────────────────────────────────┐
│  Queue Full Generation (Background)│
└──────┬─────────────────────────────┘
       │
       ▼ (5s delay)
┌────────────────────────────────────┐
│  Full V2 Pipeline Runs             │
│  (All products, all objectives)    │
└──────┬─────────────────────────────┘
       │
       ▼
┌────────────────────────────────────┐
│  50+ Comprehensive Bundles Ready   │
│  Replace Quick-Start Bundles       │
└──────┬─────────────────────────────┘
       │
       ▼
┌────────────────────────────────────┐
│  Mark initial_sync_completed=True  │
│  Merchant Now "Regular User"       │
└────────────────────────────────────┘
```

## Bundle Identification

Quick-start bundles are tagged with a special `discount_reference`:

```python
rec = {
    "discount_reference": f"__quick_start_{csv_upload_id}__",
    # ... other fields
}
```

This allows:
- Easy identification of quick-start vs. full bundles
- Cleanup before full generation
- Debugging and monitoring

## Monitoring & Observability

### Logs

Quick-start mode adds detailed logging:

```
[upload_123] 🚀 QUICK-START MODE ACTIVATED for first-time install
  Shop: shop-abc
  Max products: 50
  Max bundles: 10
  Timeout: 120s
  Full generation will be queued for background processing

[upload_123] Quick-start: Selected top 48 products
[upload_123] Quick-start: Analyzing 48 products…
[upload_123] Quick-start: Generating bundles…

[upload_123] ✅ Quick-start completed successfully
  Bundles: 10
  Duration: 45.32s

[upload_123] 📋 Scheduling full bundle generation in background
  This will run with normal timeout (1200s watchdog)

[upload_123] 🔄 Starting full bundle generation after quick-start
  Shop: shop-abc
  Mode: Comprehensive v2 pipeline
  Timeout: 1200s soft watchdog

[upload_123] ✅ Full generation complete after quick-start
  Total bundles: 67
  Processing time: 347.21s

[upload_123] Marked shop shop-abc initial sync as completed
```

### Metrics

Quick-start metrics are tracked separately:

```python
metrics = {
    "quick_start_mode": True,
    "max_products_limit": 50,
    "max_bundles_limit": 10,
    "timeout_seconds": 120,
    "total_recommendations": 10,
    "processing_time_ms": 45320,
    "phase_timings": {
        "phase1_data_loading_ms": 12400,
        "phase2_scoring_ms": 8900,
        "phase3_generation_ms": 21100,
        "phase4_persistence_ms": 2920
    },
    "products_analyzed": 48,
    "orders_analyzed": 234
}
```

## Performance Targets

| Metric | Quick-Start | Full Pipeline |
|--------|-------------|---------------|
| **Target Duration** | 30s - 2 min | 5 - 20 min |
| **Products Analyzed** | Top 50 | All (1000s) |
| **Bundles Generated** | 10 preview | 50+ comprehensive |
| **Objectives Used** | 2 | 4-8 |
| **ML Complexity** | Simple co-occurrence | Full v2 pipeline |
| **Merchant Wait Time** | < 2 minutes | No wait (background) |

## Use Cases

### First-Time Merchant (New Install)
1. Merchant installs app, uploads catalog + orders
2. Quick-start activates automatically
3. **Within 2 minutes**: Merchant sees 10 preview bundles
4. Merchant explores bundles, gains confidence in app
5. **In background**: Full generation runs (5-20 minutes)
6. **Later**: Merchant gets 50+ comprehensive bundles
7. Future uploads use normal pipeline (merchant is now "regular")

### Regular Merchant (Existing User)
1. Merchant uploads new data
2. System detects `initial_sync_completed = True`
3. Full pipeline runs normally (20-minute watchdog)
4. No change from previous behavior

### Cron Jobs / Scheduled Updates
1. System runs scheduled bundle refresh
2. Shop already marked as completed
3. Full pipeline runs (can take 20 minutes, no user waiting)
4. Merchant sees updated bundles next time they log in

## Fallback Behavior

If quick-start fails for any reason, the system **gracefully falls back** to the full pipeline:

```python
try:
    generation_result = await generator.generate_quick_start_bundles(...)
except asyncio.TimeoutError:
    logger.warning("Quick-start timed out, falling back to full pipeline")
    # Falls through to normal generation
except Exception as e:
    logger.error("Quick-start failed, falling back to full pipeline")
    # Falls through to normal generation
```

This ensures merchants **always** get bundles, even if quick-start encounters issues.

## Testing

### Test First-Time Install
1. Create new shop record (or clear existing `ShopSyncStatus`)
2. Upload CSV data
3. Trigger bundle generation
4. Verify:
   - Quick-start activates
   - Bundles appear within 2 minutes
   - Full generation queued
   - `initial_sync_completed` set to `True` after full generation

### Test Regular Merchant
1. Use shop with `initial_sync_completed = True`
2. Upload CSV data
3. Trigger bundle generation
4. Verify:
   - Quick-start skipped
   - Full pipeline runs normally

### Test Quick-Start Disable
1. Set `QUICK_START_ENABLED=false`
2. Upload CSV for new shop
3. Verify:
   - Quick-start skipped
   - Full pipeline runs for all merchants

## Future Enhancements

1. **Adaptive Limits**: Adjust max_products/max_bundles based on catalog size
2. **Quality Metrics**: Track quick-start bundle acceptance rate vs. full bundles
3. **A/B Testing**: Compare merchant engagement with/without quick-start
4. **Smart Scheduling**: Delay full generation to off-peak hours
5. **Incremental Updates**: Only regenerate bundles for changed products

## Summary

Quick-Start Mode transforms the first-time merchant experience:

| Before | After |
|--------|-------|
| Wait 20 minutes for bundles | See 10 bundles in < 2 minutes |
| All-or-nothing generation | Preview now, comprehensive later |
| Poor onboarding experience | Excellent first impression |
| Same flow for all merchants | Optimized for first-time vs. regular |

This ensures new merchants see value immediately while maintaining comprehensive bundle quality for all users.
