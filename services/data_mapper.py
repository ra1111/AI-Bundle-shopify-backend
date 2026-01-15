"""
Data Mapping Service
Handles comprehensive linking of OrderLine.sku → Variant → Product → Inventory
"""
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from decimal import Decimal
from datetime import datetime, timedelta
import time
import contextlib
import unicodedata
from collections import deque

from services.feature_flags import feature_flags

from services.storage import storage, update_csv_upload_status, summarize_upload_coverage

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind
except ImportError:  # pragma: no cover - optional dependency
    trace = None
    SpanKind = None

logger = logging.getLogger(__name__)

class DataMapper:
    """Enhanced data mapping service for OrderLine → Variant → Product → Inventory links"""
    
    def __init__(self):
        self.resolved_variant_cache: Dict[str, str] = {}
        self.unresolved_sku_cache: Dict[str, datetime] = {}
        self._last_unresolved_log_at: Optional[datetime] = None
        self._variant_map_by_scope: Dict[str, Dict[str, Any]] = {}
        self._variant_id_map_by_scope: Dict[str, Dict[str, Any]] = {}
        self._inventory_map_by_scope: Dict[str, Dict[str, Any]] = {}
        self._catalog_map_by_scope: Dict[str, Dict[str, Any]] = {}
        self._product_meta_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry: Dict[str, float] = {}
        self._tuning_state: Dict[str, Dict[str, Any]] = {}
        self._tracer = trace.get_tracer(__name__) if trace else None

    def _scope_key(self, csv_upload_id: str, run_id: Optional[str] = None) -> str:
        return f"run:{run_id}" if run_id else f"upload:{csv_upload_id}"

    @contextlib.contextmanager
    def _start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        if not self._tracer or not SpanKind:
            yield None
            return
        with self._tracer.start_as_current_span(name, kind=SpanKind.INTERNAL) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    def _maybe_expire_scope(self, scope: str) -> None:
        ttl_seconds = feature_flags.get_flag("data_mapping.cache_ttl_seconds", 1800) or 1800
        last = self._cache_expiry.get(scope)
        now = time.time()
        if last and now - last > ttl_seconds:
            self._variant_map_by_scope.pop(scope, None)
            self._variant_id_map_by_scope.pop(scope, None)
            self._inventory_map_by_scope.pop(scope, None)
            self._catalog_map_by_scope.pop(scope, None)
            keys_to_drop = [k for k in self._product_meta_cache if k.startswith(f"{scope}::")]
            for key in keys_to_drop:
                self._product_meta_cache.pop(key, None)
            self._cache_expiry.pop(scope, None)

    def _get_tuning_parameters(
        self,
        scope: str,
        default_batch: int,
        default_concurrency: int,
    ) -> Tuple[int, int]:
        state = self._tuning_state.get(scope)
        if not state:
            state = {
                "batch_size": max(1, default_batch),
                "concurrency": max(1, default_concurrency),
                "latency_history": deque(maxlen=20),
            }
            self._tuning_state[scope] = state
            logger.info(
                "DataMapper tuning state initialized | scope=%s batch_size=%d concurrency=%d",
                scope,
                state["batch_size"],
                state["concurrency"],
            )
        return state["batch_size"], state["concurrency"]

    def _update_tuning_state(
        self,
        scope: str,
        chunk_durations: List[float],
        default_batch: int,
        default_concurrency: int,
    ) -> None:
        if not chunk_durations:
            return
        state = self._tuning_state.get(scope)
        if not state:
            return
        durations_ms = sorted(d * 1000 for d in chunk_durations if d is not None)
        if not durations_ms:
            return
        index_95 = max(0, int(len(durations_ms) * 0.95) - 1)
        p95 = durations_ms[index_95]
        state["latency_history"].append(p95)
        target_ms = feature_flags.get_flag("data_mapping.target_chunk_p95_ms", 800) or 800
        original_batch = state["batch_size"]
        original_concurrency = state["concurrency"]
        if p95 > target_ms * 1.5:
            state["concurrency"] = max(1, int(state["concurrency"] * 0.8))
            state["batch_size"] = max(10, int(state["batch_size"] * 0.8))
        elif p95 < target_ms * 0.7:
            state["concurrency"] = min(default_concurrency, state["concurrency"] + 1)
            state["batch_size"] = min(500, int(state["batch_size"] * 1.1))
        state["batch_size"] = max(10, min(state["batch_size"], 1000))
        state["concurrency"] = max(1, min(state["concurrency"], default_concurrency))
        if (state["batch_size"], state["concurrency"]) != (original_batch, original_concurrency):
            logger.info(
                "DataMapper tuning adjusted | scope=%s p95_ms=%.1f batch=%d->%d concurrency=%d->%d",
                scope,
                p95,
                original_batch,
                state["batch_size"],
                original_concurrency,
                state["concurrency"],
            )
        else:
            logger.debug(
                "DataMapper tuning stable | scope=%s p95_ms=%.1f batch=%d concurrency=%d",
                scope,
                p95,
                state["batch_size"],
                state["concurrency"],
            )

    async def resolve_variant_from_sku(self, sku: str, csv_upload_id: str) -> Optional[str]:
        """Resolve variant_id from SKU with caching"""
        if sku in self.resolved_variant_cache:
            return self.resolved_variant_cache[sku]

        scope_key = self._scope_key(csv_upload_id)
        scoped_variants = self._variant_map_by_scope.get(scope_key)
        if scoped_variants:
            cached_variant = scoped_variants.get(sku)
            if cached_variant and getattr(cached_variant, "variant_id", None):
                variant_id = cached_variant.variant_id
                self.resolved_variant_cache[sku] = variant_id
                return variant_id

        ttl_seconds = feature_flags.get_flag("data_mapping.unresolved_cache_ttl_seconds", 600) or 0
        cached_at = self.unresolved_sku_cache.get(sku)
        if cached_at:
            if ttl_seconds > 0:
                age = (datetime.utcnow() - cached_at).total_seconds()
                if age > ttl_seconds:
                    self.unresolved_sku_cache.pop(sku, None)
                else:
                    return None
            else:
                return None

        # Try to find variant by SKU
        variant = await storage.get_variant_by_sku(sku, csv_upload_id)
        if variant:
            variant_id = variant.variant_id
            self.resolved_variant_cache[sku] = variant_id
            return variant_id

        # Cache unresolved SKUs to avoid repeated lookups
        self.unresolved_sku_cache[sku] = datetime.utcnow()
        logger.warning(f"Could not resolve variant_id for SKU: {sku}")
        return None

    async def enrich_order_lines_with_variants(
        self,
        csv_upload_id: str,
        catalog_upload_id: Optional[str] = None,
        variants_upload_id: Optional[str] = None,
        inventory_upload_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Enrich order lines with variant mappings and compute metrics.

        Args:
            csv_upload_id: The orders upload ID
            catalog_upload_id: Optional separate catalog upload ID (for Quickstart mode)
            variants_upload_id: Optional separate variants upload ID (for Quickstart mode)
            inventory_upload_id: Optional separate inventory upload ID (for Quickstart mode)
        """
        import traceback

        logger.info(f"[{csv_upload_id}] 🔄 DataMapper.enrich_order_lines_with_variants STARTED")
        if catalog_upload_id or variants_upload_id or inventory_upload_id:
            logger.info(
                f"[{csv_upload_id}] Using separate upload IDs: "
                f"catalog={catalog_upload_id}, variants={variants_upload_id}, inventory={inventory_upload_id}"
            )
        
        metrics = {
            "total_order_lines": 0,
            "order_line_count": 0,
            "resolved_variants": 0,
            "unresolved_skus": 0,
            "missing_inventory": 0,
            "enriched_lines": 0,
            "variant_count": 0,
            "catalog_count": 0,
            "blocked_no_variants_or_catalog": False,
            "missing_catalog_variants": 0,
            "catalog_coverage_ratio": 0.0,
        }

        unresolved_samples: List[str] = []
        timing_enabled = feature_flags.get_flag("data_mapping.log_timings", True)
        overall_start = time.perf_counter() if timing_enabled else None
        fetch_start = time.perf_counter() if timing_enabled else None
        fetch_duration = 0.0
        prefetch_duration = 0.0
        mapping_duration = 0.0
        
        try:
            logger.info(f"[{csv_upload_id}] 📂 Step 1: Resolving run_id...")
            # Resolve run_id to correlate across files
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            logger.info(f"[{csv_upload_id}] ✅ run_id resolved: {run_id or 'None (using upload_id)'}")

            # Optional: backfill missing order_line.sku values from variants when available.
            if feature_flags.get_flag("data_mapping.backfill_order_line_skus", True):
                backfilled = await storage.backfill_order_line_skus_from_variants(csv_upload_id, run_id)
                if backfilled:
                    logger.info(
                        "[%s] Backfilled %d order_lines.sku from variants (run=%s)",
                        csv_upload_id,
                        backfilled,
                        run_id,
                    )
                    metrics["order_line_skus_backfilled"] = backfilled

            # Snapshot coverage counts up front for easier debugging and metrics.
            try:
                coverage = await summarize_upload_coverage(csv_upload_id, run_id)
                metrics["order_count"] = coverage.get("orders", 0)
                metrics["order_line_count"] = coverage.get("order_lines", 0)
                metrics["variant_count"] = coverage.get("variants", 0)
                metrics["catalog_count"] = coverage.get("catalog", 0)
                metrics["missing_catalog_variants"] = coverage.get("missing_catalog_variants", 0)
                metrics["catalog_coverage_ratio"] = coverage.get("catalog_coverage_ratio", 0.0)
                if metrics["catalog_coverage_ratio"] < 1.0:
                    logger.warning(
                        "[%s] Catalog coverage incomplete (ratio=%.3f missing=%d variants=%d catalog=%d)",
                        csv_upload_id,
                        metrics["catalog_coverage_ratio"],
                        metrics["missing_catalog_variants"],
                        metrics["variant_count"],
                        metrics["catalog_count"],
                    )
            except Exception as exc:
                logger.warning(
                    "[%s] Coverage summary failed (run=%s): %s",
                    csv_upload_id,
                    run_id,
                    exc,
                )
                coverage = {}
            
            # Get all order lines for this upload (or entire run if available)
            logger.info(f"[{csv_upload_id}] 📦 Step 2: Fetching order lines...")
            with self._start_span(
                "data_mapping.fetch_order_lines",
                {"csv_upload_id": csv_upload_id, "run_id": run_id or ""},
            ):
                if run_id:
                    order_lines = await storage.get_order_lines_by_run(run_id)
                else:
                    order_lines = await storage.get_order_lines_by_upload(csv_upload_id)

            if timing_enabled and fetch_start is not None:
                fetch_duration = time.perf_counter() - fetch_start
            
            logger.info(
                f"[{csv_upload_id}] ✅ Order lines fetched in {fetch_duration * 1000:.0f}ms\n"
                f"  Count: {len(order_lines)}"
            )

            metrics["total_order_lines"] = len(order_lines)
            metrics["order_line_count"] = len(order_lines)
            
            if not order_lines:
                logger.warning(f"[{csv_upload_id}] ⚠️ No order lines found - returning early")
                return {
                    "metrics": metrics,
                    "enriched_lines": []
                }

            logger.info(f"[{csv_upload_id}] 🗂️ Step 3: Prefetching catalog/variant/inventory data...")
            with self._start_span(
                "data_mapping.prefetch",
                {
                    "csv_upload_id": csv_upload_id,
                    "run_id": run_id or "",
                    "order_line_count": len(order_lines),
                },
            ):
                prefetch_start = time.perf_counter() if timing_enabled else None
                prefetch_data = await self._prefetch_data(
                    csv_upload_id,
                    run_id,
                    order_lines,
                    catalog_upload_id=catalog_upload_id,
                    variants_upload_id=variants_upload_id,
                    inventory_upload_id=inventory_upload_id,
                )
                if timing_enabled and prefetch_start is not None:
                    prefetch_duration = time.perf_counter() - prefetch_start
            
            logger.info(
                f"[{csv_upload_id}] ✅ Prefetch complete in {prefetch_duration * 1000:.0f}ms\n"
                f"  Variants: {len(prefetch_data.get('variants_by_sku', {}))}\n"
                f"  Catalog: {len(prefetch_data.get('catalog_map', {}))}"
            )

            # Hard-stop if the upload/run has no variant or catalog rows. This indicates ingestion never ran.
            if feature_flags.get_flag("data_mapping.block_on_missing_variant_catalog", True):
                counts = await storage.count_variant_and_catalog_records(csv_upload_id, run_id)
                variant_count = counts.get("variant_count", metrics.get("variant_count", 0))
                catalog_count = counts.get("catalog_count", metrics.get("catalog_count", 0))
                metrics["variant_count"] = variant_count
                metrics["catalog_count"] = catalog_count
                if variant_count == 0 or catalog_count == 0:
                    metrics["blocked_no_variants_or_catalog"] = True
                    error_message = (
                        f"Phase 1 blocked: variants={variant_count}, catalog={catalog_count} "
                        f"for upload={csv_upload_id}"
                    )
                    try:
                        await update_csv_upload_status(
                            csv_upload_id=csv_upload_id,
                            status="blocked_missing_catalog",
                            error_message=error_message,
                            extra_metrics=metrics,
                        )
                    except Exception as exc:
                        logger.warning(
                            "[%s] Failed to mark upload as blocked_missing_catalog: %s",
                            csv_upload_id,
                            exc,
                        )
                    logger.error(
                        "[%s] 🚫 Blocking Phase 1: no variants/catalog found (variant_count=%d, catalog_count=%d, run_id=%s)",
                        csv_upload_id,
                        variant_count,
                        catalog_count,
                        run_id,
                    )
                    return {"metrics": metrics, "enriched_lines": []}

            scope = prefetch_data.get("scope", self._scope_key(csv_upload_id, run_id))
            default_batch = feature_flags.get_flag("data_mapping.prefetch_batch_size", 100) or 100
            default_concurrency = feature_flags.get_flag("data_mapping.concurrent_map_limit", 25) or 25
            try:
                default_batch = int(default_batch)
            except (TypeError, ValueError):
                default_batch = 100
            try:
                default_concurrency = int(default_concurrency)
            except (TypeError, ValueError):
                default_concurrency = 25
            batch_size, tuned_concurrency = self._get_tuning_parameters(scope, default_batch, default_concurrency)
            pool_cap = max(1, int(default_concurrency * 0.8))
            concurrency_limit = max(1, min(tuned_concurrency, default_concurrency, pool_cap))
            concurrency_enabled = feature_flags.get_flag("data_mapping.concurrent_mapping", True)
            if not concurrency_enabled:
                concurrency_limit = 1
            semaphore = asyncio.Semaphore(concurrency_limit) if concurrency_enabled else None
            logger.info(
                "DataMapper mapping start | upload=%s scope=%s batch_size=%d concurrency_limit=%d (pool_cap=%d default=%d)",
                csv_upload_id,
                scope,
                batch_size,
                concurrency_limit,
                pool_cap,
                default_concurrency,
            )

            mapping_start = time.perf_counter() if timing_enabled else None
            results: List[Dict[str, Any]] = []
            chunk_durations: List[float] = []
            enumerated_lines = list(enumerate(order_lines))

            for chunk_index in range(0, len(enumerated_lines), batch_size):
                chunk = enumerated_lines[chunk_index:chunk_index + batch_size]
                span_attrs = {
                    "csv_upload_id": csv_upload_id,
                    "chunk_index": chunk_index // batch_size,
                    "chunk_size": len(chunk),
                    "concurrency_limit": concurrency_limit,
                }
                with self._start_span("data_mapping.map_chunk", span_attrs):
                    chunk_start_time = time.perf_counter() if timing_enabled else None

                    async def map_line(item: Tuple[int, Any]) -> Dict[str, Any]:
                        idx, order_line = item
                        if semaphore:
                            async with semaphore:
                                return await self._map_order_line(idx, order_line, csv_upload_id, run_id, prefetch_data)
                        return await self._map_order_line(idx, order_line, csv_upload_id, run_id, prefetch_data)

                    if concurrency_enabled:
                        chunk_results = await asyncio.gather(*[map_line(item) for item in chunk])
                    else:
                        chunk_results = []
                        for item in chunk:
                            chunk_results.append(await map_line(item))

                    if timing_enabled and chunk_start_time is not None:
                        chunk_durations.append(time.perf_counter() - chunk_start_time)

                    results.extend(chunk_results)
                    if timing_enabled and chunk_start_time is not None:
                        elapsed_ms = (time.perf_counter() - chunk_start_time) * 1000
                        logger.info(
                            "DataMapper chunk processed | upload=%s chunk=%d size=%d concurrency=%d elapsed_ms=%.1f",
                            csv_upload_id,
                            span_attrs["chunk_index"],
                            span_attrs["chunk_size"],
                            concurrency_limit,
                            elapsed_ms,
                        )

            if timing_enabled and mapping_start is not None:
                mapping_duration = time.perf_counter() - mapping_start

            self._update_tuning_state(scope, chunk_durations, default_batch, default_concurrency)
            self._cache_expiry[scope] = time.time()

            results.sort(key=lambda r: r["index"])

            enriched_lines: List[Dict[str, Any]] = []

            for result in results:
                local_metrics = result.get("metrics", {})
                metrics["resolved_variants"] += local_metrics.get("resolved_variants", 0)
                metrics["unresolved_skus"] += local_metrics.get("unresolved_skus", 0)
                metrics["missing_inventory"] += local_metrics.get("missing_inventory", 0)
                metrics["enriched_lines"] += local_metrics.get("enriched_lines", 0)

                sample = result.get("unresolved_sample")
                if sample and len(unresolved_samples) < 10:
                    unresolved_samples.append(sample)

                enriched_line = result.get("enriched_line")
                if enriched_line:
                    enriched_lines.append(enriched_line)
            
            # Store enriched data for later use
            await self.persist_enriched_mappings(csv_upload_id, enriched_lines)
            
            if metrics["unresolved_skus"]:
                self._log_unresolved_summary(csv_upload_id, metrics["unresolved_skus"], unresolved_samples)

            if timing_enabled and overall_start is not None:
                total_duration = time.perf_counter() - overall_start
                logger.info(
                    "Data mapping timings | upload=%s total=%.2fs fetch=%.2fs prefetch=%.2fs map=%.2fs lines=%d",
                    csv_upload_id,
                    total_duration,
                    fetch_duration,
                    prefetch_duration,
                    mapping_duration,
                    len(order_lines),
                )

            logger.info(f"Data mapping completed: {metrics}")
            return {
                "metrics": metrics,
                "enriched_lines": enriched_lines
            }
            
        except Exception as e:
            overall_duration = (time.perf_counter() - overall_start) * 1000 if overall_start else 0
            logger.error(
                f"[{csv_upload_id}] ❌ DataMapper.enrich_order_lines_with_variants FAILED!\n"
                f"  Duration: {overall_duration:.0f}ms\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Error message: {str(e)}\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            raise

    async def _prefetch_data(
        self,
        csv_upload_id: str,
        run_id: Optional[str],
        _order_lines: List[Any],
        catalog_upload_id: Optional[str] = None,
        variants_upload_id: Optional[str] = None,
        inventory_upload_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Prefetch catalog, variant, and inventory data to minimize per-row I/O.

        Args:
            csv_upload_id: The orders upload ID
            run_id: The run ID correlating all uploads
            _order_lines: Order lines to prefetch for
            catalog_upload_id: Optional separate catalog upload ID (fixes Quickstart bug)
            variants_upload_id: Optional separate variants upload ID (fixes Quickstart bug)
            inventory_upload_id: Optional separate inventory upload ID (fixes Quickstart bug)
        """
        scope = self._scope_key(csv_upload_id, run_id)
        self._maybe_expire_scope(scope)
        if not feature_flags.get_flag("data_mapping.prefetch_enabled", True):
            self._cache_expiry[scope] = time.time()
            return {
                "scope": scope,
                "variants_by_sku": self._variant_map_by_scope.get(scope, {}),
                "variants_by_id": self._variant_id_map_by_scope.get(scope, {}),
                "inventory_map": self._inventory_map_by_scope.get(scope, {}),
                "catalog_map": self._catalog_map_by_scope.get(scope, {}),
            }

        tasks: Dict[str, asyncio.Task] = {}

        # Use separate upload IDs if provided (Quickstart mode fix), otherwise fall back to run_id or csv_upload_id
        effective_variants_upload_id = variants_upload_id or csv_upload_id
        effective_catalog_upload_id = catalog_upload_id or csv_upload_id
        effective_inventory_upload_id = inventory_upload_id or csv_upload_id

        if scope not in self._variant_map_by_scope or scope not in self._variant_id_map_by_scope:
            tasks["variant_maps"] = asyncio.create_task(
                storage.get_variant_maps_by_run(run_id) if run_id and not variants_upload_id else storage.get_variant_maps(effective_variants_upload_id)
            )

        if scope not in self._inventory_map_by_scope:
            tasks["inventory_map"] = asyncio.create_task(
                storage.get_inventory_levels_map_by_run(run_id) if run_id and not inventory_upload_id else storage.get_inventory_levels_map(effective_inventory_upload_id)
            )

        if scope not in self._catalog_map_by_scope:
            tasks["catalog_map"] = asyncio.create_task(
                storage.get_catalog_snapshots_map_by_variant_and_run(run_id)
                if run_id and not catalog_upload_id
                else storage.get_catalog_snapshots_map_by_variant(effective_catalog_upload_id)
            )

        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for key, result in zip(tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.warning(
                        "DataMapper prefetch %s failed for upload=%s run=%s: %s",
                        key,
                        csv_upload_id,
                        run_id,
                        result,
                    )
                    continue
                if key == "variant_maps":
                    by_sku, by_id = result
                    self._variant_map_by_scope[scope] = by_sku
                    self._variant_id_map_by_scope[scope] = by_id
                elif key == "inventory_map":
                    self._inventory_map_by_scope[scope] = self._build_inventory_data_map(result)
                elif key == "catalog_map":
                    self._catalog_map_by_scope[scope] = result

        logger.info(
            "DataMapper prefetch summary | scope=%s variants=%d inventory=%d catalog=%d tasks_started=%d",
            scope,
            len(self._variant_map_by_scope.get(scope, {})),
            len(self._inventory_map_by_scope.get(scope, {})),
            len(self._catalog_map_by_scope.get(scope, {})),
            len(tasks),
        )

        catalog_map = self._catalog_map_by_scope.get(scope, {})
        if catalog_map:
            for variant_id, snapshot in catalog_map.items():
                if not variant_id:
                    continue
                cache_key = self._product_cache_key(scope, variant_id)
                if cache_key not in self._product_meta_cache:
                    self._product_meta_cache[cache_key] = self._convert_snapshot_to_product_data(snapshot)

        self._cache_expiry[scope] = time.time()

        return {
            "scope": scope,
            "variants_by_sku": self._variant_map_by_scope.get(scope, {}),
            "variants_by_id": self._variant_id_map_by_scope.get(scope, {}),
            "inventory_map": self._inventory_map_by_scope.get(scope, {}),
            "catalog_map": catalog_map,
        }

    async def _map_order_line(
        self,
        index: int,
        line: Any,
        csv_upload_id: str,
        run_id: Optional[str],
        prefetch_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Process a single order line with optional prefetched data."""
        metrics = {
            "resolved_variants": 0,
            "unresolved_skus": 0,
            "missing_inventory": 0,
            "enriched_lines": 0,
        }
        scope = prefetch_data.get("scope", self._scope_key(csv_upload_id, run_id))
        base_variants_by_sku = prefetch_data.get("variants_by_sku", {})
        base_variants_by_id = prefetch_data.get("variants_by_id", {})
        base_inventory_map = prefetch_data.get("inventory_map", {})
        catalog_map = prefetch_data.get("catalog_map", {})

        scope_variant_map = self._variant_map_by_scope.setdefault(scope, base_variants_by_sku)
        scope_variant_id_map = self._variant_id_map_by_scope.setdefault(scope, base_variants_by_id)
        scope_inventory_map = self._inventory_map_by_scope.setdefault(scope, base_inventory_map)

        sku = getattr(line, "sku", None)
        line_vid = getattr(line, "variant_id", None)

        # Skip synthetic no-sku-* placeholders - treat them as missing SKUs
        if sku and sku.startswith("no-sku-"):
            sku = None

        resolved_variant_obj = None
        resolved_variant_id: Optional[str] = None

        # UNIFORM JOIN KEY: Use variant_id as the sole join key (not SKU)
        # variant_id is the stable, always-present Shopify identifier
        if not line_vid:
            # No variant_id = cannot resolve (Shopify always provides variant_id)
            metrics["unresolved_skus"] += 1
            return {
                "index": index,
                "metrics": metrics,
                "unresolved_sample": sku or "no_variant_id",
                "enriched_line": None,
            }

        # Lookup by variant_id only
        if run_id:
            resolved_variant_obj = scope_variant_id_map.get(line_vid)
            if not resolved_variant_obj and feature_flags.get_flag("data_mapping.enable_run_scope_fallback", True):
                resolved_variant_obj = await storage.get_variant_by_id_run(line_vid, run_id)
            if not resolved_variant_obj:
                resolved_variant_obj = await storage.get_variant_by_id(line_vid, csv_upload_id)
        else:
            resolved_variant_obj = scope_variant_id_map.get(line_vid)
            if not resolved_variant_obj:
                resolved_variant_obj = await storage.get_variant_by_id(line_vid, csv_upload_id)
            if not resolved_variant_obj and line_vid:
                resolved_variant_obj = await storage.get_variant_by_id(line_vid, csv_upload_id)

        # Use variant_id from resolved object or from line
        if resolved_variant_obj:
            resolved_variant_id = getattr(resolved_variant_obj, "variant_id", None)
        if not resolved_variant_id:
            resolved_variant_id = line_vid

        if not resolved_variant_id:
            metrics["unresolved_skus"] += 1
            return {
                "index": index,
                "metrics": metrics,
                "unresolved_sample": line_vid or sku or "unknown",
                "enriched_line": None,
            }

        metrics["resolved_variants"] += 1

        # Cache by variant_id (primary key)
        if resolved_variant_obj:
            scope_variant_id_map.setdefault(resolved_variant_id, resolved_variant_obj)
            # Also cache by SKU if available (for backward compat display only)
            variant_sku = getattr(resolved_variant_obj, "sku", None)
            if variant_sku:
                scope_variant_map.setdefault(variant_sku, resolved_variant_obj)

        # Get product data using variant_id (not SKU)
        product_data = await self._get_product_data(
            resolved_variant_id,
            csv_upload_id,
            scope,
            catalog_map,
        )

        inventory_item_id = getattr(resolved_variant_obj, "inventory_item_id", None) if resolved_variant_obj else None
        inventory_data = await self._get_inventory_data(
            inventory_item_id,
            resolved_variant_id,
            csv_upload_id,
            run_id,
            scope,
            scope_inventory_map,
        )

        if inventory_item_id and inventory_data:
            scope_inventory_map[inventory_item_id] = inventory_data

        if inventory_data is None:
            metrics["missing_inventory"] += 1
            inventory_data = {"available_total": -1, "location_count": 0}
            stock_available = -1
        else:
            stock_available = inventory_data.get("available_total", 0)

        # Get SKU from variant for display purposes (not used as join key)
        display_sku = getattr(resolved_variant_obj, "sku", None) if resolved_variant_obj else sku

        enriched_line = {
            "order_line_id": line.id,
            "variant_id": resolved_variant_id,  # Primary join key
            "sku": display_sku,  # Display metadata only (not used for joins)
            "inventory_item_id": inventory_item_id,
            "product_data": product_data,
            "inventory_data": inventory_data,
            "line_data": {
                "quantity": getattr(line, "quantity", None),
                "unit_price": getattr(line, "unit_price", None),
                "line_total": getattr(line, "line_total", None),
                "name": getattr(line, "name", None),
                "category": getattr(line, "category", None),
                "brand": getattr(line, "brand", None),
            },
            "stock_available": stock_available,
        }

        metrics["enriched_lines"] += 1

        return {
            "index": index,
            "metrics": metrics,
            "unresolved_sample": None,
            "enriched_line": enriched_line,
        }

    def _product_cache_key(self, scope: str, variant_id: str) -> str:
        """Generate cache key using variant_id (uniform join key)."""
        return f"{scope}::{variant_id}"

    def _normalize_text(self, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return unicodedata.normalize("NFKC", str(value)).strip()

    def _convert_snapshot_to_product_data(self, snapshot: Any) -> Dict[str, Any]:
        if snapshot is None:
            return {}
        return {
            "product_id": getattr(snapshot, "product_id", None),
            "product_title": self._normalize_text(getattr(snapshot, "product_title", None)),
            "product_type": self._normalize_text(getattr(snapshot, "product_type", None)),
            "vendor": self._normalize_text(getattr(snapshot, "vendor", None)),
            "tags": getattr(snapshot, "tags", None),
            "product_status": getattr(snapshot, "product_status", None),
            "created_at": getattr(snapshot, "product_created_at", None),
            "published_at": getattr(snapshot, "product_published_at", None),
            "variant_title": self._normalize_text(getattr(snapshot, "variant_title", None)),
            "price": getattr(snapshot, "price", None),
            "compare_at_price": getattr(snapshot, "compare_at_price", None),
        }

    def _build_inventory_data_map(self, level_map: Dict[str, List[Any]]) -> Dict[str, Dict[str, Any]]:
        inventory_data: Dict[str, Dict[str, Any]] = {}
        for item_id, levels in level_map.items():
            inventory_data[item_id] = self._convert_inventory_levels(levels)
        return inventory_data

    def _convert_inventory_levels(self, levels: Any) -> Dict[str, Any]:
        if isinstance(levels, dict):
            return levels
        if not levels:
            return {"available_total": -1, "location_count": 0, "locations": []}
        total_available = 0
        locations = []
        for level in levels:
            available = getattr(level, "available", None)
            if available is None:
                available = 0
            total_available += available
            locations.append(
                {
                    "location_id": getattr(level, "location_id", None),
                    "available": available,
                }
            )
        return {
            "available_total": total_available,
            "location_count": len(levels),
            "locations": locations,
        }

    async def _get_product_data(
        self,
        variant_id: str,
        csv_upload_id: str,
        scope: str,
        catalog_map: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Get product data using variant_id as the uniform join key."""
        if not variant_id:
            return None

        # Check cache by variant_id
        cache_key = self._product_cache_key(scope, variant_id)
        cached = self._product_meta_cache.get(cache_key)
        if cached:
            return cached

        # Lookup in catalog_map by variant_id (catalog_map is keyed by variant_id now)
        snapshot = catalog_map.get(variant_id)
        if snapshot:
            product_data = self._convert_snapshot_to_product_data(snapshot)
            self._product_meta_cache[cache_key] = product_data
            return product_data

        # Database fallback
        product_data = await self.get_product_data_for_variant(variant_id, csv_upload_id)
        if product_data:
            self._product_meta_cache[cache_key] = product_data
        return product_data

    async def _get_inventory_data(
        self,
        inventory_item_id: Optional[str],
        variant_id: Optional[str],
        csv_upload_id: str,
        run_id: Optional[str],
        scope: str,
        inventory_map: Dict[str, Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if inventory_item_id:
            cached = inventory_map.get(inventory_item_id)
            if cached:
                return cached

        levels_data = None

        if inventory_item_id and run_id:
            levels = await storage.get_inventory_levels_by_item_id_run(inventory_item_id, run_id)
            if levels:
                levels_data = self._convert_inventory_levels(levels)
        elif inventory_item_id:
            levels = await storage.get_inventory_levels_by_item_id(inventory_item_id, csv_upload_id)
            if levels:
                levels_data = self._convert_inventory_levels(levels)

        if levels_data and inventory_item_id:
            inventory_map[inventory_item_id] = levels_data
            self._inventory_map_by_scope.setdefault(scope, inventory_map)[inventory_item_id] = levels_data
            return levels_data

        if variant_id:
            return await self.get_inventory_data_for_variant(variant_id, csv_upload_id)

        return None

    def reset_unresolved_cache(self) -> None:
        """Clear cached unresolved SKUs so new catalog data can be re-evaluated."""
        self.unresolved_sku_cache.clear()

    def reset_resolved_cache(self) -> None:
        """Clear resolved cache, typically when catalog variants are refreshed."""
        self.resolved_variant_cache.clear()

    def _log_unresolved_summary(self, upload_id: str, count: int, samples: List[str]) -> None:
        now = datetime.utcnow()
        if self._last_unresolved_log_at and (now - self._last_unresolved_log_at).total_seconds() < 30:
            return
        self._last_unresolved_log_at = now
        sample_preview = ", ".join(samples) if samples else "(none captured)"
        logger.warning(
            "DataMapper unresolved SKUs remain after enrichment: upload=%s count=%s sample=%s",
            upload_id,
            count,
            sample_preview
        )
    
    async def get_product_data_for_variant(self, variant_id: str, csv_upload_id: str) -> Optional[Dict[str, Any]]:
        """Get product data for a variant"""
        try:
            # ARCHITECTURE: Catalog is keyed by variant_id (primary key)
            # Use variant_id directly for lookup instead of querying variant table first
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            catalog_map = await storage.get_catalog_snapshots_map_by_variant(csv_upload_id)
            if not catalog_map and run_id:
                catalog_map = await storage.get_catalog_snapshots_map_by_variant_and_run(run_id)
            if not catalog_map:
                return None

            product_data = catalog_map.get(variant_id)
            if not product_data:
                return None
            
            return {
                "product_id": product_data.product_id,
                "product_title": self._normalize_text(product_data.product_title),
                "product_type": self._normalize_text(product_data.product_type),
                "vendor": self._normalize_text(product_data.vendor),
                "tags": product_data.tags,
                "product_status": product_data.product_status,
                "created_at": product_data.product_created_at,
                "published_at": product_data.product_published_at,
                "variant_title": self._normalize_text(product_data.variant_title),
                "price": product_data.price,
                "compare_at_price": product_data.compare_at_price
            }
            
        except Exception as e:
            logger.warning(f"Error getting product data for variant {variant_id}: {e}")
            return None
    
    async def get_inventory_data_for_variant(self, variant_id: str, csv_upload_id: str) -> Optional[Dict[str, Any]]:
        """Get inventory data for a variant"""
        try:
            variant = await storage.get_variant_by_id(variant_id, csv_upload_id)
            if not variant or not variant.inventory_item_id:
                return None
            
            inventory_levels = await storage.get_inventory_levels_by_item_id(
                variant.inventory_item_id, csv_upload_id
            )
            
            if not inventory_levels:
                return {"available_total": -1, "location_count": 0}  # Not tracked
            
            total_available = sum(level.available for level in inventory_levels)
            
            return {
                "available_total": total_available,
                "location_count": len(inventory_levels),
                "locations": [
                    {
                        "location_id": level.location_id,
                        "available": level.available
                    } for level in inventory_levels
                ]
            }
            
        except Exception as e:
            logger.warning(f"Error getting inventory data for variant {variant_id}: {e}")
            return None
    
    async def persist_enriched_mappings(self, csv_upload_id: str, enriched_lines: List[Dict[str, Any]]) -> None:
        """Persist enriched mappings for later use in bundle generation"""
        try:
            # Store in catalog_snapshot for fast access during bundle generation
            catalog_entries = []
            
            for line in enriched_lines:
                if line.get("variant_id") and line.get("product_data"):
                    product_data = line["product_data"]
                    inventory_data = line.get("inventory_data", {})
                    
                    # Compute velocity and other signals
                    velocity = await self.compute_velocity_for_variant(
                        line["variant_id"], csv_upload_id
                    )
                    
                    # Use actual SKU only, not synthetic no-sku-* placeholders
                    raw_sku = line.get("sku", "")
                    actual_sku = raw_sku if raw_sku and not str(raw_sku).startswith("no-sku-") else ""

                    catalog_entry = {
                        "variant_id": line["variant_id"],
                        "csv_upload_id": csv_upload_id,
                        "sku": actual_sku,
                        "product_id": product_data["product_id"],
                        "product_title": product_data["product_title"],
                        "product_type": product_data["product_type"],
                        "vendor": product_data["vendor"],
                        "tags": product_data["tags"],
                        "product_status": product_data["product_status"],
                        "product_created_at": product_data.get("created_at"),
                        "product_published_at": product_data.get("published_at"),
                        "variant_title": product_data["variant_title"],
                        "price": product_data["price"],
                        "compare_at_price": product_data["compare_at_price"],
                        "inventory_item_id": line.get("inventory_item_id") or "",
                        "available_total": inventory_data.get("available_total", -1),
                        "last_inventory_update": datetime.now(),
                        # Note: objective flags and velocity are computed/updated elsewhere
                    }
                    
                    catalog_entries.append(catalog_entry)
            
            if catalog_entries:
                await storage.create_catalog_snapshots(catalog_entries)
                logger.info(f"Persisted {len(catalog_entries)} catalog entries")
            
        except Exception as e:
            logger.error(f"Error persisting enriched mappings: {e}")
            raise
    
    async def compute_velocity_for_variant(self, variant_id: str, csv_upload_id: str) -> Decimal:
        """Compute sales velocity for variant (units sold per day)"""
        try:
            # Get sales data for last 60 days
            sales_data = await storage.get_variant_sales_data(variant_id, csv_upload_id, days=60)
            if not sales_data:
                return Decimal('0')
            
            total_quantity = sum(sale.quantity for sale in sales_data if sale.quantity is not None)
            days_period = 60
            
            return Decimal(str(total_quantity)) / Decimal(str(days_period))
            
        except Exception as e:
            logger.warning(f"Error computing velocity for variant {variant_id}: {e}")
            return Decimal('0')
    
    def get_resolution_metrics(self) -> Dict[str, Any]:
        """Get data mapping resolution metrics"""
        return {
            "resolved_variants_cached": len(self.resolved_variant_cache),
            "unresolved_skus_cached": len(self.unresolved_sku_cache),
            "cache_hit_rate": len(self.resolved_variant_cache) / max(1, 
                len(self.resolved_variant_cache) + len(self.unresolved_sku_cache))
        }
