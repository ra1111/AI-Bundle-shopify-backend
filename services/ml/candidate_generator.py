"""
ML Candidate Generator Service
Implements LLM embeddings (replacing item2vec) and FPGrowth algorithm for better candidate generation
"""
from typing import List, Dict, Any, Optional, Tuple, Set, Iterable
import asyncio
import logging
import os
import numpy as np
from decimal import Decimal
from collections import defaultdict, Counter
from itertools import combinations
import hashlib
import json
from dataclasses import dataclass

from services.storage import storage
from services.ml.llm_embeddings import llm_embedding_engine
from services.ml.hybrid_scorer import hybrid_scorer

logger = logging.getLogger(__name__)


@dataclass
class CandidateGenerationContext:
    """Context for candidate generation.

    NOTE: All identifiers use variant_id as the PRIMARY key.
    SKU is only stored for display purposes (varid_to_display_sku map).
    """
    run_id: Optional[str]
    valid_variant_ids: Set[str]  # Set of valid variant_ids (primary key)
    varid_to_display_sku: Dict[str, str]  # Maps variant_id -> display SKU (for UI only)
    transactions: List[Set[str]]  # Sets of variant_ids
    sequences: List[List[str]]  # Lists of variant_ids
    embeddings: Dict[str, np.ndarray]  # variant_id -> embedding
    catalog_products: Dict[str, Dict[str, Any]]  # variant_id -> product data
    catalog_subset: List[Dict[str, Any]]
    embedding_targets: Set[str]  # Set of variant_ids to embed
    variant_id_frequency: Dict[str, int]  # variant_id -> frequency count
    llm_candidate_target: int
    llm_only: bool = False
    covis_vectors: Optional[Dict[str, Any]] = None  # MODERN: Co-visitation graph (pseudo-Item2Vec)

class CandidateGenerator:
    """Advanced candidate generation using ML techniques"""
    
    def __init__(self):
        self.item_embeddings = {}
        self.embedding_dim = 64
        self.min_frequency = 2
        self.use_fpgrowth = True  # Feature flag for FPGrowth vs Apriori
        self.small_store_relax_threshold = 30  # Allow looser filtering when catalog is tiny
        self.max_embedding_targets = int(os.getenv("MAX_EMBED_TARGETS", "400"))
        self.min_embedding_targets = int(os.getenv("MIN_EMBED_TARGETS", "75"))
        self.anchor_prefetch_extra = int(os.getenv("EMBED_ANCHOR_EXTRA", "50"))

        # Adaptive thresholds for different store sizes
        # Small: <50 orders, Medium: 50-500, Large: 500+
        self.store_size_thresholds = {
            "small": {"max_orders": 50, "min_support": 0.01, "min_transactions": 3, "min_frequency": 1},
            "medium": {"max_orders": 500, "min_support": 0.02, "min_transactions": 5, "min_frequency": 2},
            "large": {"max_orders": float("inf"), "min_support": 0.05, "min_transactions": 10, "min_frequency": 2},
        }

    def _get_store_tier(self, order_count: int) -> str:
        """Determine store tier based on order count"""
        if order_count < self.store_size_thresholds["small"]["max_orders"]:
            return "small"
        elif order_count < self.store_size_thresholds["medium"]["max_orders"]:
            return "medium"
        else:
            return "large"

    def _get_adaptive_min_support(self, transaction_count: int) -> float:
        """Calculate adaptive min_support based on transaction count"""
        tier = self._get_store_tier(transaction_count)
        base_support = self.store_size_thresholds[tier]["min_support"]

        # For very small stores, ensure at least 1 co-occurrence is enough
        # min_support * transaction_count >= 1
        min_viable_support = 1.0 / max(transaction_count, 1)

        # Use the higher of base_support or min_viable_support
        # This ensures we don't require more co-occurrences than possible
        adaptive_support = max(base_support, min_viable_support)

        logger.info(
            f"Adaptive min_support | tier={tier} orders={transaction_count} "
            f"base={base_support:.3f} min_viable={min_viable_support:.3f} final={adaptive_support:.3f}"
        )
        return adaptive_support

    def _get_adaptive_min_transactions(self, transaction_count: int) -> int:
        """Get minimum transaction threshold based on store size"""
        tier = self._get_store_tier(transaction_count)
        return self.store_size_thresholds[tier]["min_transactions"]
    
    async def prepare_context(self, csv_upload_id: str) -> CandidateGenerationContext:
        """Prefetch expensive data needed by candidate generation.

        NOTE: All identifiers use variant_id as PRIMARY key.
        """
        import time
        start_time = time.time()

        run_id = await storage.get_run_id_for_upload(csv_upload_id)
        valid_variant_ids = await self.get_valid_variant_ids_for_csv(csv_upload_id)
        varid_to_display_sku = await self.get_variantid_to_display_sku_map(csv_upload_id)
        raw_transactions = await self.get_transactions_for_mining(csv_upload_id)
        raw_sequences = await self.get_purchase_sequences(csv_upload_id)

        transactions = self._normalize_transactions(raw_transactions, varid_to_display_sku, valid_variant_ids)
        sequences = self._normalize_sequences(raw_sequences, varid_to_display_sku, valid_variant_ids)
        variant_id_frequency = self._compute_variant_id_frequency(transactions)

        # NEW: Generate LLM embeddings (replaces item2vec)
        # Expected time: 2-3 seconds (vs 60-120s for item2vec)
        embeddings: Dict[str, np.ndarray] = {}
        catalog_products: List[Dict[str, Any]] = []
        catalog_map: Dict[str, Dict[str, Any]] = {}  # Keyed by variant_id
        catalog_subset: List[Dict[str, Any]] = []
        embedding_targets: Set[str] = set()
        try:
            # Get catalog products for embedding generation
            catalog_entries = await storage.get_catalog_snapshots_by_run(run_id)
            catalog_products = self._materialize_catalog_products(catalog_entries)
            # Key by variant_id (primary key), not SKU
            catalog_map = {item["variant_id"]: item for item in catalog_products if item.get("variant_id")}
            embedding_targets = set(
                self._choose_embedding_targets(
                    valid_variant_ids=valid_variant_ids,
                    catalog_map=catalog_map,
                    variant_id_frequency=variant_id_frequency,
                )
            )
            catalog_subset = [
                catalog_map[variant_id]
                for variant_id in embedding_targets
                if variant_id in catalog_map
            ]

            if not catalog_subset and catalog_products:
                fallback_subset = catalog_products[: min(self.max_embedding_targets, len(catalog_products))]
                catalog_subset = fallback_subset
                embedding_targets = {item["variant_id"] for item in fallback_subset if item.get("variant_id")}

            with_variant_id = sum(1 for item in catalog_products if item.get("variant_id"))
            logger.info(
                "[%s] Preparing LLM embeddings | catalog_items=%d with_variant_id=%d",
                csv_upload_id,
                len(catalog_products),
                with_variant_id,
            )

            logger.info(
                "[%s] LLM embeddings targets | shortlist=%d (max=%d) unique_catalog=%d",
                csv_upload_id,
                len(catalog_subset),
                self.max_embedding_targets,
                len(catalog_map),
            )

            embeddings = await llm_embedding_engine.get_embeddings_batch(catalog_subset, use_cache=True)

            missing_targets = [
                variant_id
                for variant_id in embedding_targets
                if variant_id not in embeddings and variant_id in catalog_map
            ]
            if missing_targets:
                supplemental_products = [catalog_map[variant_id] for variant_id in missing_targets if variant_id in catalog_map]
                if supplemental_products:
                    supplemental = await llm_embedding_engine.get_embeddings_batch(
                        supplemental_products, use_cache=True
                    )
                    embeddings.update(supplemental)

            logger.info(f"LLM embeddings generated in {time.time() - start_time:.2f}s")
            logger.info(
                "[%s] Embedding result | embeddings_available=%d",
                csv_upload_id,
                len(embeddings),
            )

        except Exception as e:
            logger.warning(f"Failed to generate LLM embeddings: {e}. Continuing without embeddings.")
            embeddings = {}
            catalog_products = catalog_products or []
            catalog_map = catalog_map or {}
            catalog_subset = catalog_subset or []
            embedding_targets = embedding_targets or set()

        # MODERN ML: Build co-visitation graph (pseudo-Item2Vec)
        # This provides semantic similarity without ML training (~1-5ms)
        covis_vectors = None
        try:
            from services.ml.pseudo_item2vec import build_covis_vectors

            # Get order lines for co-visitation analysis
            order_lines = await storage.get_order_lines(csv_upload_id)

            if order_lines and embedding_targets:
                covis_start = time.time()
                covis_vectors = build_covis_vectors(
                    order_lines=order_lines,
                    top_variant_ids=embedding_targets,  # Limit to same variant_ids as embeddings
                    min_co_visits=1,
                    max_neighbors=50,
                )
                covis_duration = (time.time() - covis_start) * 1000
                logger.info(
                    "[%s] Built co-visitation graph | vectors=%d duration_ms=%.1f",
                    csv_upload_id,
                    len(covis_vectors),
                    covis_duration,
                )
            else:
                logger.info(f"[{csv_upload_id}] Skipping co-visitation graph (no order lines or embedding targets)")

        except Exception as e:
            logger.warning(f"Failed to build co-visitation graph: {e}. Continuing without covis features.")
            covis_vectors = None

        context = CandidateGenerationContext(
            run_id=run_id,
            valid_variant_ids=valid_variant_ids,
            varid_to_display_sku=varid_to_display_sku,
            transactions=transactions,
            sequences=sequences,
            embeddings=embeddings or {},
            catalog_products=catalog_map,
            catalog_subset=catalog_subset,
            embedding_targets=embedding_targets,
            variant_id_frequency=dict(variant_id_frequency),
            llm_candidate_target=20,
            covis_vectors=covis_vectors,  # MODERN: Add co-visitation vectors
        )
        logger.info(
            "[%s] Candidate context prepared | txns=%d sequences=%d embeddings=%d catalog_subset=%d targets=%d",
            csv_upload_id,
            len(context.transactions),
            len(context.sequences),
            len(context.embeddings),
            len(context.catalog_subset),
            len(context.embedding_targets),
        )
        return context

    def _normalize_transactions(
        self,
        transactions: Optional[List[Set[str]]],
        varid_to_display_sku: Dict[str, str],
        valid_variant_ids: Set[str],
    ) -> List[Set[str]]:
        """Normalize transactions to use variant_id as primary key."""
        if not transactions:
            logger.warning("No transactions provided for normalization")
            return []
        normalized: List[Set[str]] = []
        total_items = 0
        mapped_items = 0
        filtered_items = 0
        retained_items = 0

        # DEBUG: Log sample raw items and valid_variant_ids to understand mismatch
        if transactions:
            sample_raw = list(transactions[0])[:3] if transactions[0] else []
            logger.warning(f"DEBUG _normalize_transactions | sample_raw_items={sample_raw} | valid_variant_ids_count={len(valid_variant_ids) if valid_variant_ids else 0} | varid_map_count={len(varid_to_display_sku)}")
            sample_valid = list(valid_variant_ids)[:5] if valid_variant_ids else []
            logger.warning(f"DEBUG _normalize_transactions | sample_valid_variant_ids={sample_valid}")
            sample_varid = list(varid_to_display_sku.items())[:3] if varid_to_display_sku else []
            logger.warning(f"DEBUG _normalize_transactions | sample_varid_to_display_sku={sample_varid}")

        for raw_items in transactions:
            cast_items: Set[str] = set()
            for raw in raw_items:
                total_items += 1
                resolved = varid_to_display_sku.get(raw, raw)
                if not resolved:
                    continue
                sku = str(resolved).strip()
                if not sku:
                    continue
                mapped_items += 1
                if valid_variant_ids and sku not in valid_variant_ids:
                    filtered_items += 1
                    # DEBUG: Log first few filtered items to understand mismatch
                    if filtered_items <= 3:
                        logger.warning(f"DEBUG filtered | raw={raw} resolved={resolved} sku={sku} type={type(sku)}")
                    continue
                retained_items += 1
                cast_items.add(sku)
            if len(cast_items) >= 2:
                normalized.append(cast_items)

        # Log detailed stats to help debug transaction filtering issues
        logger.info(
            "Transactions normalized | input=%d retained=%d | items: total=%d mapped=%d filtered=%d retained=%d | valid_identifiers=%d varid_map=%d",
            len(transactions),
            len(normalized),
            total_items,
            mapped_items,
            filtered_items,
            retained_items,
            len(valid_variant_ids) if valid_variant_ids else 0,
            len(varid_to_display_sku),
        )
        return normalized

    def _normalize_sequences(
        self,
        sequences: Optional[List[List[str]]],
        varid_to_display_sku: Dict[str, str],
        valid_variant_ids: Set[str],
    ) -> List[List[str]]:
        if not sequences:
            return []
        normalized: List[List[str]] = []
        for raw_seq in sequences:
            seq: List[str] = []
            for raw in raw_seq:
                resolved = varid_to_display_sku.get(raw, raw)
                if not resolved:
                    continue
                sku = str(resolved).strip()
                if not sku:
                    continue
                if valid_variant_ids and sku not in valid_variant_ids:
                    continue
                seq.append(sku)
            if len(seq) >= 2:
                normalized.append(seq)
        logger.debug(
            "Sequences normalized | input=%d retained=%d",
            len(sequences),
            len(normalized),
        )
        return normalized

    @staticmethod
    def _compute_variant_id_frequency(transactions: Optional[List[Set[str]]]) -> Counter:
        ctr: Counter = Counter()
        if not transactions:
            return ctr
        for tx in transactions:
            for vid in tx:
                ctr[vid] += 1
        logger.debug(
            "variant_id frequency computed | transactions=%d distinct_variant_ids=%d",
            len(transactions),
            len(ctr),
        )
        return ctr

    def _materialize_catalog_products(self, catalog_entries: Optional[List[Any]]) -> List[Dict[str, Any]]:
        """Materialize catalog entries using variant_id as the uniform identifier."""
        products: List[Dict[str, Any]] = []
        if not catalog_entries:
            return products
        for entry in catalog_entries:
            variant_id = getattr(entry, "variant_id", None)
            if not variant_id:
                continue

            variant_id_str = str(variant_id).strip()
            if not variant_id_str:
                continue

            # Get SKU for display only (not used as join key)
            sku = getattr(entry, "sku", None)
            display_sku = str(sku).strip() if sku else None

            product: Dict[str, Any] = {
                "sku": variant_id_str,  # "sku" field actually stores variant_id for uniform joins
                "variant_id": variant_id_str,  # Explicit variant_id field
                "display_sku": display_sku,  # Actual SKU for display only
                "title": getattr(entry, "product_title", None) or getattr(entry, "title", "") or getattr(entry, "variant_title", ""),
                "product_type": getattr(entry, "product_type", None) or getattr(entry, "category", None),
                "product_category": getattr(entry, "product_type", None) or getattr(entry, "category", None),
                "brand": getattr(entry, "brand", None),
                "vendor": getattr(entry, "vendor", None),
                "description": getattr(entry, "description", None) or getattr(entry, "product_description", None),
                "tags": getattr(entry, "tags", None),
                "available_total": getattr(entry, "available_total", None),
                "is_slow_mover": getattr(entry, "is_slow_mover", False),
                "is_new_launch": getattr(entry, "is_new_launch", False),
                "is_seasonal": getattr(entry, "is_seasonal", False),
                "is_high_margin": getattr(entry, "is_high_margin", False),
            }
            price = getattr(entry, "price", None)
            compare_at = getattr(entry, "compare_at_price", None)
            if price is not None:
                try:
                    product["price"] = float(price)
                except (TypeError, ValueError) as e:
                    logger.debug(f"Failed to convert price to float for variant_id {variant_id_str}: {price} - {e}")
            if compare_at is not None:
                try:
                    product["compare_at_price"] = float(compare_at)
                except (TypeError, ValueError) as e:
                    logger.debug(f"Failed to convert compare_at_price to float for variant_id {variant_id_str}: {compare_at} - {e}")
            products.append(product)
        logger.debug(
            "Materialized catalog entries | input=%d usable=%d",
            len(catalog_entries),
            len(products),
        )
        return products

    def _choose_embedding_targets(
        self,
        *,
        valid_variant_ids: Set[str],
        catalog_map: Dict[str, Dict[str, Any]],
        variant_id_frequency: Counter,
    ) -> List[str]:
        """Choose variant_ids for embedding generation.

        NOTE: catalog_map is keyed by variant_id (primary key).
        """
        prioritized: List[str] = []
        if variant_id_frequency:
            prioritized.extend(
                [vid for vid, _ in variant_id_frequency.most_common(self.max_embedding_targets + self.anchor_prefetch_extra)]
            )
        flagged = [
            vid
            for vid, meta in catalog_map.items()
            if meta.get("is_slow_mover") or meta.get("is_new_launch") or meta.get("is_seasonal") or meta.get("is_high_margin")
        ]
        prioritized.extend(flagged)
        if len(valid_variant_ids) <= self.max_embedding_targets:
            prioritized.extend(valid_variant_ids)
        deduped: List[str] = []
        seen: Set[str] = set()
        for vid in prioritized:
            vid_key = (vid or "").strip()
            if not vid_key or vid_key in seen or vid_key not in catalog_map:
                continue
            deduped.append(vid_key)
            seen.add(vid_key)
            if len(deduped) >= self.max_embedding_targets:
                break
        if len(deduped) < self.min_embedding_targets:
            for vid in catalog_map.keys():
                if vid in seen:
                    continue
                deduped.append(vid)
                seen.add(vid)
                if len(deduped) >= self.min_embedding_targets or len(deduped) >= self.max_embedding_targets:
                    break
        logger.info(
            "Embedding targets finalized | distinct=%d frequency_prioritized=%d flagged=%d valid_variant_ids=%d",
            len(deduped),
            len(variant_id_frequency),
            len(flagged),
            len(valid_variant_ids),
        )
        return deduped

    def _derive_similarity_seed_variant_ids(
        self,
        apriori_candidates: List[Dict[str, Any]],
        fpgrowth_candidates: List[Dict[str, Any]],
        top_pair_candidates: List[Dict[str, Any]],
        *,
        context: Optional[CandidateGenerationContext],
        objective: str,
    ) -> List[str]:
        seeds: Set[str] = set()
        varid_to_display_sku = context.varid_to_display_sku if context else {}
        valid_variant_ids = context.valid_variant_ids if context else set()

        def _resolve(identifier: Any) -> Optional[str]:
            value = varid_to_display_sku.get(identifier, identifier)
            if not value:
                return None
            sku = str(value).strip()
            if not sku:
                return None
            if valid_variant_ids and sku not in valid_variant_ids:
                return None
            return sku

        for candidate_group in (apriori_candidates, fpgrowth_candidates, top_pair_candidates):
            for candidate in candidate_group or []:
                for prod in candidate.get("products", []):
                    sku = _resolve(prod)
                    if sku:
                        seeds.add(sku)

        if context and context.embedding_targets:
            seeds.update(context.embedding_targets)

        objective_variant_ids = []
        if context and context.catalog_products:
            objective_variant_ids = self._objective_specific_variant_ids(context.catalog_products, objective)
            seeds.update(objective_variant_ids)

        frequency = context.variant_id_frequency if context else {}

        ordered: List[str] = []
        seen: Set[str] = set()

        def _append_candidates(candidates: Iterable[str]) -> None:
            for sku in candidates:
                if not sku or sku in seen or sku not in seeds:
                    continue
                ordered.append(sku)
                seen.add(sku)
                if len(ordered) >= self.max_embedding_targets:
                    return

        if context and context.embedding_targets:
            _append_candidates(context.embedding_targets)

        freq_sorted = sorted(seeds, key=lambda sku: (-frequency.get(sku, 0), sku))
        _append_candidates(freq_sorted)
        _append_candidates(objective_variant_ids)

        if len(ordered) < self.min_embedding_targets and context and context.catalog_products:
            _append_candidates(context.catalog_products.keys())

        logger.info(
            "Similarity seed SKUs derived | seeds=%d ordered=%d objective=%s",
            len(seeds),
            len(ordered),
            objective,
        )
        return ordered[: self.max_embedding_targets]

    async def _ensure_embeddings_for_seed_variant_ids(
        self,
        embeddings: Dict[str, np.ndarray],
        catalog_map: Dict[str, Dict[str, Any]],
        seed_variant_ids: Set[str],
    ) -> Dict[str, np.ndarray]:
        missing = [
            vid
            for vid in seed_variant_ids
            if vid not in embeddings and vid in catalog_map
        ]
        if not missing:
            return embeddings
        products = [catalog_map[vid] for vid in missing if vid in catalog_map]
        if not products:
            return embeddings
        supplemental = await llm_embedding_engine.get_embeddings_batch(products, use_cache=True)
        embeddings.update(supplemental)
        logger.debug(
            "Seed embedding supplementation | requested=%d fetched=%d total_embeddings=%d",
            len(missing),
            len(supplemental),
            len(embeddings),
        )
        return embeddings

    def _prioritize_catalog_subset(
        self,
        catalog_subset: List[Dict[str, Any]],
        seed_variant_ids: List[str],
        catalog_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Prioritize catalog products for embedding generation.

        NOTE: catalog_map is keyed by variant_id (primary key).
        """
        if not catalog_subset and catalog_map:
            catalog_subset = list(catalog_map.values())

        seed_set = {vid for vid in seed_variant_ids if vid}
        prioritized: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for vid in seed_variant_ids:
            product = catalog_map.get(vid)
            if not product:
                continue
            if vid in seen:
                continue
            prioritized.append(product)
            seen.add(vid)
            if len(prioritized) >= self.max_embedding_targets:
                break

        if len(prioritized) < self.max_embedding_targets:
            for product in catalog_subset:
                vid = product.get("variant_id")
                if not vid or vid in seen:
                    continue
                prioritized.append(product)
                seen.add(vid)
                if len(prioritized) >= self.max_embedding_targets:
                    break

        if len(prioritized) < self.max_embedding_targets and catalog_map:
            for vid, product in catalog_map.items():
                if vid in seen:
                    continue
                prioritized.append(product)
                seen.add(vid)
                if len(prioritized) >= self.max_embedding_targets:
                    break

        logger.debug(
            "Catalog subset prioritised | seed_count=%d prioritized=%d",
            len(seed_variant_ids),
            len(prioritized),
        )
        return prioritized[: self.max_embedding_targets]

    def _objective_specific_variant_ids(
        self,
        catalog_map: Dict[str, Dict[str, Any]],
        objective: str,
    ) -> List[str]:
        """Get variant_ids matching the objective flags.

        NOTE: catalog_map is keyed by variant_id (primary key).
        """
        objective = (objective or "").lower()
        if not catalog_map:
            return []
        if objective == "clear_slow_movers":
            return [vid for vid, meta in catalog_map.items() if meta.get("is_slow_mover")]
        if objective == "new_launch":
            return [vid for vid, meta in catalog_map.items() if meta.get("is_new_launch")]
        if objective == "seasonal_promo":
            return [vid for vid, meta in catalog_map.items() if meta.get("is_seasonal")]
        if objective == "margin_guard":
            return [vid for vid, meta in catalog_map.items() if meta.get("is_high_margin")]
        return []
    
    async def generate_candidates(
        self,
        csv_upload_id: str,
        bundle_type: str,
        objective: str,
        context: Optional[CandidateGenerationContext] = None,
    ) -> Dict[str, Any]:
        """Generate bundle candidates using both Apriori and item2vec"""
        metrics = {
            "apriori_candidates": 0,
            "item2vec_candidates": 0,
            "fpgrowth_candidates": 0,
            "top_pair_candidates": 0,
            "total_unique_candidates": 0,
            "generation_method": "hybrid",
            "invalid_variant_id_candidates_filtered": 0,
            "store_tier": "unknown",
            "order_count": 0,
        }
        
        try:
            run_id = context.run_id if context else await storage.get_run_id_for_upload(csv_upload_id)

            # Determine store tier for adaptive thresholds
            transaction_count = len(context.transactions) if context and context.transactions else 0
            store_tier = self._get_store_tier(transaction_count)
            tier_config = self.store_size_thresholds[store_tier]
            metrics["store_tier"] = store_tier
            metrics["order_count"] = transaction_count
            logger.info(
                f"[{csv_upload_id}] Store tier detection | orders={transaction_count} "
                f"tier={store_tier} min_support={tier_config['min_support']} "
                f"min_transactions={tier_config['min_transactions']} min_frequency={tier_config['min_frequency']}"
            )

            # CRITICAL FIX: Preload valid SKUs to prevent infinite loop
            logger.info(f"Preloading valid SKUs for scope: run_id={run_id} csv_upload_id={csv_upload_id}")
            valid_variant_ids = context.valid_variant_ids if context else await self.get_valid_variant_ids_for_csv(csv_upload_id)
            # Map variant_id -> sku so rule products expressed as variant_ids can be validated against catalog
            varid_to_display_sku = context.varid_to_display_sku if context else await self.get_variantid_to_display_sku_map(csv_upload_id)
            logger.info(f"Found {len(valid_variant_ids)} valid SKUs for prefiltering")

            # VOLUME BUNDLE SPECIAL HANDLING
            # VOLUME bundles are single-product quantity discounts, NOT pairs
            # Use dedicated generator that picks top-selling products
            if bundle_type == "VOLUME":
                logger.info(f"[{csv_upload_id}] Using VOLUME-specific candidate generation (single-product)")
                volume_candidates = await self.generate_volume_candidates(
                    csv_upload_id,
                    context=context,
                    limit=10,
                )
                metrics["generation_method"] = "volume_top_sellers"
                metrics["volume_candidates"] = len(volume_candidates)
                metrics["total_unique_candidates"] = len(volume_candidates)
                logger.info(
                    f"[{csv_upload_id}] VOLUME candidates generated | count={len(volume_candidates)}"
                )
                return {
                    "candidates": volume_candidates,
                    "metrics": metrics
                }

            llm_only_mode = bool(getattr(context, "llm_only", False))
            if llm_only_mode:
                metrics["generation_method"] = "llm_only"
                logger.info(f"[{csv_upload_id}] LLM-only candidate generation enabled (sparse dataset)")

            # OPTIMIZATION: Parallel candidate generation (3-4x faster)
            # Run apriori, fpgrowth, and top-pair in parallel instead of sequentially
            apriori_candidates = []
            item2vec_candidates = []
            fpgrowth_candidates = []
            top_pair_candidates = []

            if not llm_only_mode:
                logger.info("[%s] Starting parallel candidate generation", csv_upload_id)
                parallel_tasks = {}

                # Task 1: Traditional Apriori rules
                async def get_apriori():
                    rules = (
                        await storage.get_association_rules_by_run(run_id)
                        if run_id else await storage.get_association_rules(csv_upload_id)
                    )
                    return self.convert_rules_to_candidates(rules, bundle_type)

                parallel_tasks['apriori'] = asyncio.create_task(get_apriori())

                # Task 2: FPGrowth algorithm
                if self.use_fpgrowth:
                    transactions = context.transactions if context else None
                    parallel_tasks['fpgrowth'] = asyncio.create_task(
                        self.generate_fpgrowth_candidates(csv_upload_id, bundle_type, transactions=transactions)
                    )

                # Task 3: Top pair mining (moved here for parallel execution)
                transactions_for_pairs = context.transactions if context else None
                parallel_tasks['top_pairs'] = asyncio.create_task(
                    self.generate_top_pair_candidates(csv_upload_id, bundle_type, transactions=transactions_for_pairs)
                )

                # Execute all transactional sources in parallel
                results = await asyncio.gather(*parallel_tasks.values(), return_exceptions=True)
                result_map = dict(zip(parallel_tasks.keys(), results))

                # Unpack results
                apriori_candidates = result_map.get('apriori', []) if not isinstance(result_map.get('apriori'), Exception) else []
                if isinstance(result_map.get('apriori'), Exception):
                    logger.error(f"Apriori generation failed: {result_map.get('apriori')}")

                fpgrowth_candidates = result_map.get('fpgrowth', []) if not isinstance(result_map.get('fpgrowth'), Exception) else []
                if isinstance(result_map.get('fpgrowth'), Exception):
                    logger.error(f"FPGrowth generation failed: {result_map.get('fpgrowth')}")

                top_pair_candidates = result_map.get('top_pairs', []) if not isinstance(result_map.get('top_pairs'), Exception) else []
                if isinstance(result_map.get('top_pairs'), Exception):
                    logger.error(f"Top pair generation failed: {result_map.get('top_pairs')}")

                metrics["apriori_candidates"] = len(apriori_candidates)
                metrics["fpgrowth_candidates"] = len(fpgrowth_candidates)
                metrics["top_pair_candidates"] = len(top_pair_candidates)

                logger.info(
                    "[%s] Parallel candidate generation complete | apriori=%d fpgrowth=%d top_pairs=%d",
                    csv_upload_id,
                    len(apriori_candidates),
                    len(fpgrowth_candidates),
                    len(top_pair_candidates),
                )
            else:
                metrics["apriori_candidates"] = 0
                metrics["fpgrowth_candidates"] = 0
                logger.info("[%s] Transactional candidate sources skipped (LLM-only mode)", csv_upload_id)
            
            # 3. LLM embeddings (semantic similarity) - REPLACES item2vec
            embeddings = context.embeddings if context else {}
            llm_target = getattr(context, "llm_candidate_target", 20) if context else 20
            catalog_map = context.catalog_products if context else {}
            catalog_subset = context.catalog_subset if context else []
            llm_candidates = []
            if embeddings:
                try:
                    if not catalog_map or not catalog_subset:
                        fallback_catalog = (
                            await storage.get_catalog_snapshots_by_run(run_id)
                            if run_id
                            else await storage.get_catalog_snapshots_by_upload(csv_upload_id)
                        )
                        fallback_products = self._materialize_catalog_products(fallback_catalog)
                        catalog_map = {item["variant_id"]: item for item in fallback_products if item.get("variant_id")}
                        catalog_subset = fallback_products[: min(self.max_embedding_targets, len(fallback_products))]

                    seed_variant_ids = self._derive_similarity_seed_variant_ids(
                        apriori_candidates,
                        fpgrowth_candidates,
                        top_pair_candidates,
                        context=context,
                        objective=objective,
                    )
                    embeddings = await self._ensure_embeddings_for_seed_variant_ids(
                        embeddings,
                        catalog_map,
                        set(seed_variant_ids),
                    )
                    prioritized_catalog = self._prioritize_catalog_subset(
                        catalog_subset,
                        seed_variant_ids,
                        catalog_map,
                    )
                    if context is not None:
                        context.catalog_products = catalog_map
                        context.catalog_subset = prioritized_catalog
                        context.embeddings = embeddings

                    orders_count = (
                        len(context.transactions)
                        if context and context.transactions is not None
                        else None
                    )
                    llm_candidates = await llm_embedding_engine.generate_candidates_by_similarity(
                        csv_upload_id=csv_upload_id,
                        bundle_type=bundle_type,
                        objective=objective,
                        catalog=prioritized_catalog,
                        embeddings=embeddings,
                        num_candidates=llm_target,
                        orders_count=orders_count,
                        seed_variant_ids=seed_variant_ids if seed_variant_ids else None,
                    )
                    metrics["llm_candidates_seeded"] = len(seed_variant_ids)
                    logger.info(
                        "[%s] Generated %d LLM-based candidates (target=%d seed_variant_ids=%d catalog_subset=%d)",
                        csv_upload_id,
                        len(llm_candidates),
                        llm_target,
                        len(seed_variant_ids),
                        len(prioritized_catalog),
                    )
                    if llm_candidates:
                        for candidate in llm_candidates:
                            candidate.setdefault("objective", objective)
                            candidate.setdefault("bundle_type", bundle_type)
                            candidate.setdefault("generation_method", "llm_similarity")
                            sources = candidate.get("generation_sources")
                            if not sources:
                                candidate["generation_sources"] = ["llm_similarity"]
                            elif isinstance(sources, list) and "llm_similarity" not in sources:
                                candidate["generation_sources"] = sources + ["llm_similarity"]
                except Exception as e:
                    logger.warning(f"Failed to generate LLM candidates: {e}")
            else:
                if llm_only_mode:
                    logger.warning(f"[{csv_upload_id}] LLM-only mode requested but no embeddings available")
                else:
                    logger.info(f"[{csv_upload_id}] Skipping LLM similarity candidates (no embeddings)")

            metrics["llm_candidates"] = len(llm_candidates)
            if embeddings and not llm_candidates:
                logger.info(
                    "[%s] LLM embeddings produced zero candidates | catalog_with_embeddings=%d objective=%s bundle_type=%s",
                    csv_upload_id,
                    len(embeddings),
                    objective,
                    bundle_type,
                )
            metrics["item2vec_candidates"] = 0  # Deprecated

            # NOTE: top_pair_candidates already generated in parallel above (line 561-563)
            # No need to regenerate sequentially

            # 4. Combine and deduplicate candidates (using LLM + transactional)
            all_candidates = self.combine_candidates(
                apriori_candidates, fpgrowth_candidates, llm_candidates, top_pair_candidates
            )
            logger.info(
                "[%s] Candidate generation breakdown | mode=%s apriori=%d fpgrowth=%d llm=%d top_pair=%d combined=%d",
                csv_upload_id,
                "llm_only" if llm_only_mode else "hybrid",
                len(apriori_candidates),
                len(fpgrowth_candidates),
                len(llm_candidates),
                len(top_pair_candidates),
                len(all_candidates),
            )

            # CRITICAL FIX: Filter out candidates with invalid products (allow variant_id by mapping to sku)
            original_count = len(all_candidates)
            all_candidates = self.filter_candidates_by_valid_variant_ids(
                all_candidates, valid_variant_ids, varid_to_display_sku, csv_upload_id=csv_upload_id
            )
            invalid_filtered = original_count - len(all_candidates)
            metrics["invalid_variant_id_candidates_filtered"] = invalid_filtered

            if invalid_filtered > 0:
                logger.warning(f"Candidates: filtered_invalid={invalid_filtered} remaining={len(all_candidates)} valid_variant_ids={len(valid_variant_ids)} varid_map={len(varid_to_display_sku)}")

            # 5.5. MODERN ML: Enrich candidates with co-visitation similarity
            # This adds Item2Vec-style features without training overhead
            if all_candidates and context and hasattr(context, 'covis_vectors') and context.covis_vectors:
                logger.info(f"[{csv_upload_id}] Enriching {len(all_candidates)} candidates with co-visitation features")
                from services.ml.pseudo_item2vec import enhance_candidate_with_covisitation

                enriched_count = 0
                for candidate in all_candidates:
                    try:
                        enhance_candidate_with_covisitation(candidate, context.covis_vectors)
                        enriched_count += 1
                    except Exception as e:
                        logger.debug(f"Failed to enrich candidate with covis features: {e}")
                        # Continue without covis features for this candidate

                logger.info(f"[{csv_upload_id}] Co-visitation enrichment complete: {enriched_count}/{len(all_candidates)} candidates enriched")
            elif all_candidates:
                logger.info(f"[{csv_upload_id}] Skipping co-visitation enrichment (no vectors in context)")

            # 6. HYBRID SCORING: Combine LLM semantic + transactional signals
            transaction_count = len(context.transactions) if context and context.transactions else 0
            all_candidates = hybrid_scorer.rank_candidates(all_candidates, transaction_count)

            metrics["total_unique_candidates"] = len(all_candidates)
            metrics["hybrid_scoring"] = {
                "transaction_count": transaction_count,
                "weights": hybrid_scorer.get_weights_for_dataset(transaction_count).__dict__
            }

            logger.info(
                f"Candidates: scope upload_id={csv_upload_id} run_id={run_id} "
                f"apriori={len(apriori_candidates)} fpgrowth={len(fpgrowth_candidates)} llm={len(llm_candidates)} "
                f"unique_after_filter={len(all_candidates)} hybrid_scored=True"
            )

            # 7. Add source information for explainability
            for candidate in all_candidates:
                candidate["generation_sources"] = self.identify_sources(
                    candidate,
                    apriori_candidates,
                    fpgrowth_candidates,
                    llm_candidates,
                    top_pair_candidates
                )

            return {
                "candidates": all_candidates,
                "metrics": metrics
            }
            
        except Exception as e:
            logger.error(f"Error generating candidates: {e}")
            return {"candidates": [], "metrics": metrics}

    async def get_variantid_to_display_sku_map(self, csv_upload_id: str) -> Dict[str, str]:
        """Build an identity mapping from variant_id to variant_id (uniform join key).

        NOTE: This function name is legacy. It now returns variant_id → variant_id
        since we use variant_id uniformly as the join key, not SKU.
        """
        try:
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            if run_id:
                snapshots = await storage.get_catalog_snapshots_by_run(run_id)
            else:
                snapshots = await storage.get_catalog_snapshots_by_upload(csv_upload_id)

            mapping = {}
            for s in snapshots:
                vid = getattr(s, 'variant_id', None)
                if vid:
                    vid_str = str(vid)
                    # Identity mapping: variant_id → variant_id (uniform join key)
                    mapping[vid_str] = vid_str

            logger.info(f"[{csv_upload_id}] Built variant_id identity map: {len(mapping)} variants")
            return mapping
        except Exception as e:
            logger.warning(f"Error building variant_id map: {e}")
            return {}
    
    async def get_valid_variant_ids_for_csv(self, csv_upload_id: str) -> Set[str]:
        """Get all valid variant_ids for the given CSV upload ID.

        NOTE: This function name is legacy. It now returns variant_ids (uniform join key), not SKUs.
        """
        try:
            # Resolve run_id to aggregate across all related uploads
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            if run_id:
                catalog_entries = await storage.get_catalog_snapshots_by_run(run_id)
            else:
                catalog_entries = await storage.get_catalog_snapshots_by_upload(csv_upload_id)

            valid_variant_ids = set()

            for entry in catalog_entries:
                vid = getattr(entry, 'variant_id', None)
                if vid:
                    vid_str = str(vid).strip()
                    if vid_str:
                        valid_variant_ids.add(vid_str)

            logger.info(f"[{csv_upload_id}] Valid variant_ids: {len(catalog_entries)} total -> {len(valid_variant_ids)} valid")
            return valid_variant_ids
        except Exception as e:
            logger.warning(f"Error getting valid SKUs for {csv_upload_id}: {e}")
            return set()

    async def generate_top_pair_candidates(
        self,
        csv_upload_id: str,
        bundle_type: str,
        limit: int = 20,
        transactions: Optional[List[Set[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Ensure the strongest SKU pairs are always available as candidates."""
        if transactions is None:
            try:
                run_id = await storage.get_run_id_for_upload(csv_upload_id)
            except Exception:
                run_id = None

            if run_id:
                order_lines = await storage.get_order_lines_by_run(run_id)
            else:
                order_lines = await storage.get_order_lines(csv_upload_id)

            order_variant_map: Dict[str, Set[str]] = defaultdict(set)
            for line in order_lines:
                order_id = getattr(line, "order_id", None)
                sku = getattr(line, "sku", None)
                variant_id = getattr(line, "variant_id", None)

                # ALWAYS use variant_id as primary identifier (uniform join key)
                # variant_id is stable and always present from Shopify
                # SKU is for display only and may be synthetic (no-sku-*)
                if variant_id:
                    identifier = str(variant_id).strip()
                elif sku and str(sku).strip() and not str(sku).startswith('no-sku-'):
                    # Fallback to real SKU only if variant_id is missing
                    identifier = str(sku).strip()
                else:
                    continue

                if not order_id or not identifier:
                    continue
                # Only filter GraphQL IDs which are internal Shopify references
                if identifier.startswith("gid://"):
                    continue
                order_variant_map[order_id].add(identifier)

            transactions = [skus for skus in order_variant_map.values() if len(skus) >= 2]

        transaction_count = len(transactions)
        if transaction_count == 0:
            logger.info(f"Top pair mining skipped for {csv_upload_id}: no multi-item transactions")
            return []

        item_counts: Dict[str, int] = defaultdict(int)
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)

        for skus in transactions:
            for sku in skus:
                item_counts[sku] += 1
            for a, b in combinations(sorted(skus), 2):
                pair_counts[(a, b)] += 1

        if not pair_counts:
            logger.info(f"Top pair mining found no frequent pairs for {csv_upload_id}")
            return []

        candidates: List[Dict[str, Any]] = []
        for (sku_a, sku_b), pair_count in sorted(pair_counts.items(), key=lambda item: item[1], reverse=True):
            item_count_a = item_counts.get(sku_a, 0)
            item_count_b = item_counts.get(sku_b, 0)
            if item_count_a == 0 or item_count_b == 0:
                continue

            support = pair_count / transaction_count
            confidence = max(pair_count / item_count_a, pair_count / item_count_b)
            support_a = item_count_a / transaction_count
            support_b = item_count_b / transaction_count
            expected_support = support_a * support_b if support_a > 0 and support_b > 0 else 0
            lift = support / expected_support if expected_support > 0 else 1.0

            candidate = {
                "products": [sku_a, sku_b],
                "support": float(support),
                "confidence": float(confidence),
                "lift": float(lift),
                "bundle_type": bundle_type,
                "generation_method": "top_pair_mining",
                "generation_sources": ["top_pair_mining"],
            }
            candidates.append(candidate)

            if len(candidates) >= limit:
                break

        logger.info(
            f"Top pair mining for {csv_upload_id}: transactions={transaction_count} "
            f"pairs_considered={len(pair_counts)} returned={len(candidates)}"
        )
        return candidates

    async def generate_volume_candidates(
        self,
        csv_upload_id: str,
        context: Optional[CandidateGenerationContext] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate single-product VOLUME bundle candidates.

        VOLUME bundles are quantity-based discounts on single products:
        - Buy 2+, get 5% off
        - Buy 3+, get 10% off
        - Buy 5+, get 15% off

        Selection criteria:
        1. High sales velocity (variant_id_frequency) - if available
        2. Valid price (> 0)
        3. Sufficient inventory (available_total > 5) - if available

        FALLBACK: When context data is missing, fetches catalog data directly
        and generates VOLUME bundles from catalog products (catalog-driven mode).

        Returns single-product candidates (NOT pairs like FBT).

        Performance: O(n) where n = number of valid variant_ids
        Expected time: <100ms for catalogs up to 10,000 products
        """
        import time
        start_time = time.time()
        logger.info(f"[{csv_upload_id}] VOLUME candidate generation - STARTED | limit={limit}")

        candidates: List[Dict[str, Any]] = []

        # Get frequency data (sales velocity)
        variant_id_frequency = context.variant_id_frequency if context else {}
        catalog_map = context.catalog_products if context else {}
        valid_variant_ids = context.valid_variant_ids if context else set()

        # CATALOG FALLBACK: When context data is missing, fetch catalog directly
        # This ensures VOLUME bundles are ALWAYS generated (catalog-driven mode)
        if not catalog_map or not valid_variant_ids:
            logger.info(f"[{csv_upload_id}] VOLUME candidates: Context missing, fetching catalog directly (fallback mode)")
            try:
                run_id = await storage.get_run_id_for_upload(csv_upload_id)
                if run_id:
                    catalog_entries = await storage.get_catalog_snapshots_by_run(run_id)
                else:
                    catalog_entries = await storage.get_catalog_snapshots_by_upload(csv_upload_id)

                # Build catalog_map and valid_variant_ids from catalog entries
                catalog_map = {}
                valid_variant_ids = set()
                for entry in catalog_entries:
                    vid = getattr(entry, 'variant_id', None)
                    if not vid:
                        continue
                    vid_str = str(vid).strip()
                    if not vid_str:
                        continue

                    # Extract product data from catalog entry
                    price = getattr(entry, 'price', 0)
                    try:
                        price = float(price) if price else 0
                    except (TypeError, ValueError):
                        price = 0

                    # Skip products without valid price
                    if price <= 0:
                        continue

                    valid_variant_ids.add(vid_str)
                    catalog_map[vid_str] = {
                        "variant_id": vid_str,
                        "title": getattr(entry, 'title', '') or getattr(entry, 'product_title', '') or '',
                        "price": price,
                        "available_total": getattr(entry, 'available_total', 0) or getattr(entry, 'inventory_quantity', 0) or 0,
                        "sku": getattr(entry, 'sku', '') or '',
                        "product_gid": getattr(entry, 'product_gid', '') or '',
                    }

                logger.info(
                    f"[{csv_upload_id}] VOLUME fallback: Loaded {len(catalog_map)} products from catalog "
                    f"(valid_variant_ids={len(valid_variant_ids)})"
                )
            except Exception as e:
                logger.warning(f"[{csv_upload_id}] VOLUME fallback: Failed to load catalog: {e}")
                # Return empty but don't fail - other bundle types may still work
                return []

        if not catalog_map and not valid_variant_ids:
            logger.warning(f"[{csv_upload_id}] VOLUME candidates: No catalog data available even after fallback")
            return []

        # Determine if we're in catalog-driven mode (no sales data)
        is_catalog_driven = not variant_id_frequency or sum(variant_id_frequency.values()) == 0

        # Build scored list of products for VOLUME bundles
        scored_products: List[tuple] = []

        for variant_id in valid_variant_ids:
            product = catalog_map.get(variant_id, {})
            frequency = variant_id_frequency.get(variant_id, 0)

            # Get price - skip products without valid price
            price = product.get("price", 0)
            try:
                price = float(price) if price else 0
            except (TypeError, ValueError):
                price = 0

            if price <= 0:
                continue

            # Get inventory level
            available = product.get("available_total", 0)
            try:
                available = int(available) if available else 0
            except (TypeError, ValueError):
                available = 0

            # Score calculation differs based on mode:
            # - Data-driven: prioritize high frequency (sales), then inventory
            # - Catalog-driven: prioritize price (higher priced items benefit more from volume discounts)
            if is_catalog_driven:
                # Catalog-driven: price is key factor (normalized to 0-100 range)
                score = min(100, price / 10.0) + available * 0.01
            else:
                # Data-driven: sales frequency is key factor
                score = frequency * 10 + available * 0.1

            scored_products.append((variant_id, score, frequency, available, price, product))

        # Sort by score descending
        scored_products.sort(key=lambda x: x[1], reverse=True)

        # Take top N products
        if not scored_products:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.warning(f"[{csv_upload_id}] VOLUME candidates: No valid products with price > 0 | elapsed={elapsed_ms:.0f}ms")
            return []

        # Pre-compute max frequency to avoid repeated iteration
        top_products = scored_products[:limit]
        max_frequency = max((f for _, _, f, _, _, _ in top_products), default=1)
        total_frequency = sum(variant_id_frequency.values()) or 1

        for variant_id, score, frequency, available, price, product in top_products:
            # Confidence calculation:
            # - Data-driven: 0.3 baseline + up to 0.5 based on sales velocity
            # - Catalog-driven: 0.4 baseline (reasonable starting point for volume bundles)
            if is_catalog_driven:
                # Catalog-driven: flat confidence based on price tier
                confidence = min(0.6, 0.4 + min(0.2, price / 1000.0))
                generation_method = "volume_catalog_fallback"
            else:
                # Data-driven: confidence based on sales velocity
                confidence = min(0.8, 0.3 + (frequency / max(1, max_frequency) * 0.5))
                generation_method = "volume_top_sellers"

            candidate = {
                "products": [variant_id],  # Single product for VOLUME
                "bundle_type": "VOLUME",
                "generation_method": generation_method,
                "generation_sources": [generation_method],
                "confidence": confidence,
                "lift": 1.15,  # Volume discounts typically have modest lift
                "support": min(0.5, frequency / total_frequency * 10) if not is_catalog_driven else 0.1,
                "volume_metadata": {
                    "sales_frequency": frequency,
                    "available_inventory": available,
                    "unit_price": price,
                    "product_title": product.get("title", ""),
                    "catalog_driven": is_catalog_driven,
                },
            }
            candidates.append(candidate)

        elapsed_ms = (time.time() - start_time) * 1000
        mode_label = "CATALOG-DRIVEN" if is_catalog_driven else "DATA-DRIVEN"
        logger.info(
            f"[{csv_upload_id}] VOLUME candidate generation - COMPLETED in {elapsed_ms:.0f}ms | "
            f"mode={mode_label} scored_products={len(scored_products)} returned={len(candidates)}"
        )

        return candidates

    def filter_candidates_by_valid_variant_ids(
        self,
        candidates: List[Dict[str, Any]],
        valid_variant_ids: Set[str],
        varid_to_display_sku: Dict[str, str],
        csv_upload_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filter out candidates containing invalid products.
        Accept either SKU directly or variant_id that can be mapped to a SKU via varid_to_display_sku.
        """
        if not valid_variant_ids:
            logger.warning("No valid SKUs available for filtering")
            return candidates
        
        filtered_candidates = []
        scope = csv_upload_id or "unknown_upload"
        small_store = len(valid_variant_ids) <= self.small_store_relax_threshold
        if small_store:
            logger.info(
                "[%s] Small catalog detected (valid_variant_ids=%d). Relaxed candidate filtering enabled.",
                scope,
                len(valid_variant_ids),
            )
        filtered_reason_counts: Dict[str, int] = defaultdict(int)
        relaxed_reason_counts: Dict[str, int] = defaultdict(int)
        relaxed_candidates = 0
        for candidate in candidates:
            products = candidate.get("products", [])
            bundle_type = candidate.get("bundle_type", "")
            failure_reason = None
            failure_product = None
            normalized: List[str] = []
            all_valid = True

            # VOLUME bundles are single-product with quantity tiers, others need 2+ products
            min_products_required = 1 if bundle_type == "VOLUME" else 2

            if not isinstance(products, list) or len(products) < min_products_required:
                all_valid = False
                failure_reason = f"insufficient products (need {min_products_required}, got {len(products) if isinstance(products, list) else 0})"
            else:
                for p in products:
                    if p is None:
                        all_valid = False
                        failure_reason = "null product entry"
                        failure_product = p
                        break

                    p_str = str(p).strip()
                    if not p_str:
                        all_valid = False
                        failure_reason = "empty identifier"
                        failure_product = p
                        break

                    if p_str in valid_variant_ids:
                        normalized.append(p_str)
                        continue

                    mapped = varid_to_display_sku.get(p_str)
                    if mapped and mapped in valid_variant_ids:
                        normalized.append(mapped)
                        continue

                    all_valid = False
                    failure_product = p_str
                    if mapped and mapped not in valid_variant_ids:
                        failure_reason = f"variant resolved to SKU {mapped} not present in catalog"
                    else:
                        failure_reason = "unmapped product (not SKU or variant)"
                    break

            if all_valid and len(normalized) >= min_products_required:
                try:
                    candidate["products"] = normalized
                except (TypeError, AttributeError) as e:
                    # Candidate dict might be immutable or frozen
                    logger.debug(f"Could not update candidate products (using original): {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error updating candidate products: {e}")
                filtered_candidates.append(candidate)
            else:
                sources = candidate.get("generation_sources") or []
                if isinstance(sources, str):
                    sources = [sources]
                source = candidate.get("generation_method") or ",".join(sources)
                source = source or "unknown"
                source_lower = source.lower()
                is_llm = "llm" in source_lower or any("llm" in str(s).lower() for s in sources)

                reason_key = failure_reason or "unspecified"
                if small_store:
                    relaxed_reason_counts[reason_key] += 1
                    relaxed_candidates += 1
                    fallback_products = []
                    for original in products:
                        original_str = str(original).strip() if original else ""
                        if original_str in valid_variant_ids:
                            fallback_products.append(original_str)
                        else:
                            mapped = varid_to_display_sku.get(original_str)
                            fallback_products.append(mapped if mapped else original_str or original)
                    try:
                        candidate["products"] = fallback_products
                    except (TypeError, AttributeError) as e:
                        # Candidate dict might be immutable or frozen
                        logger.debug(f"Could not update candidate with fallback products (using original): {e}")
                    except Exception as e:
                        logger.warning(f"Unexpected error updating candidate with fallback products: {e}")
                    logger.info(
                        "[%s] Retained candidate for small catalog | source=%s reason=%s offending=%s products=%s",
                        scope,
                        source,
                        reason_key,
                        failure_product,
                        products,
                    )
                    filtered_candidates.append(candidate)
                    continue

                filtered_reason_counts[reason_key] += 1
                log_fn = logger.warning if is_llm else logger.debug
                log_fn(
                    "[%s] Filtered candidate | source=%s bundle_type=%s offending=%s reason=%s products=%s",
                    scope,
                    source,
                    candidate.get("bundle_type"),
                    failure_product,
                    failure_reason or "unspecified",
                    products,
                )
        
        if filtered_reason_counts:
            logger.info(
                "[%s] Candidate filter summary | dropped=%d reasons=%s",
                scope,
                sum(filtered_reason_counts.values()),
                dict(filtered_reason_counts),
            )
        if relaxed_reason_counts:
            logger.info(
                "[%s] Candidate filter relaxed summary | relaxed=%d threshold=%d reasons=%s",
                scope,
                relaxed_candidates,
                self.small_store_relax_threshold,
                dict(relaxed_reason_counts),
            )

        return filtered_candidates
    
    async def generate_fpgrowth_candidates(
        self,
        csv_upload_id: str,
        bundle_type: str,
        transactions: Optional[List[Set[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate candidates using FPGrowth algorithm"""
        try:
            # Get transaction data (order baskets)
            if transactions is None:
                transactions = await self.get_transactions_for_mining(csv_upload_id)

            transaction_count = len(transactions)
            min_transactions = self._get_adaptive_min_transactions(transaction_count)

            if transaction_count < min_transactions:
                tier = self._get_store_tier(transaction_count)
                logger.info(
                    f"Insufficient transactions for FPGrowth | count={transaction_count} "
                    f"min_required={min_transactions} tier={tier}"
                )
                return []

            # Use adaptive min_support based on store size
            adaptive_min_support = self._get_adaptive_min_support(transaction_count)

            # Build FP-tree and mine frequent itemsets
            frequent_itemsets = self.fpgrowth_mining(transactions, min_support=adaptive_min_support)
            
            # Convert frequent itemsets to bundle candidates
            candidates = []
            for itemset, support in frequent_itemsets.items():
                if len(itemset) >= 2:  # Only multi-item bundles
                    candidate = {
                        "products": list(itemset),
                        "support": support,
                        "confidence": 0.0,  # FPGrowth doesn't compute confidence directly
                        "lift": 1.0,  # Will be computed later if needed
                        "bundle_type": bundle_type,
                        "generation_method": "fpgrowth",
                        "itemset_size": len(itemset)
                    }
                    candidates.append(candidate)
            
            # Sort by support and limit results
            candidates.sort(key=lambda x: x["support"], reverse=True)
            return candidates[:100]  # Limit to top 100 candidates
            
        except Exception as e:
            logger.warning(f"Error in FPGrowth generation: {e}")
            return []
    
    def fpgrowth_mining(self, transactions: List[Set[str]], min_support: float) -> Dict[frozenset, float]:
        """Optimized FPGrowth implementation using mlxtend library"""
        try:
            from mlxtend.frequent_patterns import fpgrowth
            from mlxtend.preprocessing import TransactionEncoder
            import pandas as pd

            if not transactions or len(transactions) == 0:
                logger.debug("FPGrowth: No transactions to process")
                return {}

            # Convert set transactions to list format for TransactionEncoder
            transactions_list = [list(txn) for txn in transactions]

            # Use TransactionEncoder to create one-hot encoded DataFrame
            te = TransactionEncoder()
            te_array = te.fit_transform(transactions_list)
            df = pd.DataFrame(te_array, columns=te.columns_)

            # Use optimized FPGrowth implementation (10-20x faster than naive approach)
            frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)

            # Convert to expected format
            result = {}
            for _, row in frequent_itemsets.iterrows():
                itemset = frozenset(row['itemsets'])
                if len(itemset) > 1:  # Only multi-item sets for bundles
                    result[itemset] = float(row['support'])

            logger.info(f"FPGrowth mining complete: {len(result)} itemsets (min_support={min_support})")
            return result

        except ImportError:
            logger.warning("mlxtend not installed, falling back to naive FPGrowth implementation")
            # Fallback to original implementation if mlxtend not available
            return self._fpgrowth_mining_fallback(transactions, min_support)

        except Exception as e:
            logger.error(f"Error in FPGrowth mining: {e}", exc_info=True)
            return {}

    def _fpgrowth_mining_fallback(self, transactions: List[Set[str]], min_support: float) -> Dict[frozenset, float]:
        """Fallback: Simple FPGrowth implementation (used if mlxtend unavailable)"""
        # Count item frequencies
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

        total_transactions = len(transactions)
        min_count = int(min_support * total_transactions)

        # Filter frequent items
        frequent_items = {item: count for item, count in item_counts.items() if count >= min_count}

        # Simple frequent itemset generation (simplified FPGrowth)
        frequent_itemsets = {}

        # Add single items
        for item, count in frequent_items.items():
            frequent_itemsets[frozenset([item])] = count / total_transactions

        # Add pairs and larger itemsets
        for size in range(2, 5):  # Up to 4-item bundles
            for transaction in transactions:
                # Get frequent items in this transaction
                frequent_in_transaction = [item for item in transaction if item in frequent_items]

                if len(frequent_in_transaction) >= size:
                    # Generate all combinations of this size
                    for combo in combinations(frequent_in_transaction, size):
                        itemset = frozenset(combo)
                        if itemset not in frequent_itemsets:
                            frequent_itemsets[itemset] = 0
                        frequent_itemsets[itemset] += 1

        # Convert counts to support
        result = {}
        for itemset, count in frequent_itemsets.items():
            if len(itemset) > 1:  # Only multi-item sets
                support = count / total_transactions
                if support >= min_support:
                    result[itemset] = support

        return result
    
    async def generate_item2vec_candidates(
        self,
        csv_upload_id: str,
        bundle_type: str,
        objective: str,
        embeddings: Optional[Dict[str, np.ndarray]] = None,
        sequences: Optional[List[List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate candidates using item2vec embeddings"""
        try:
            # Train or load item2vec embeddings
            if embeddings is None:
                embeddings = await self.get_or_train_embeddings(csv_upload_id, sequences=sequences)
            if not embeddings:
                logger.info("No embeddings available for item2vec candidates")
                return []
            
            candidates = []
            
            # Get anchor products based on objective
            anchor_products = await self.get_anchor_products_for_objective(csv_upload_id, objective)
            
            for anchor in anchor_products[:20]:  # Limit anchors
                if anchor in embeddings:
                    # Find similar products using cosine similarity
                    similar_products = self.find_similar_products(anchor, embeddings, top_k=10)
                    
                    for similar_product, similarity in similar_products:
                        if similar_product != anchor:
                            candidate = {
                                "products": [anchor, similar_product],
                                "support": 0.0,  # Will be computed from actual data
                                "confidence": float(similarity),  # Use similarity as confidence proxy
                                "lift": 1.0,
                                "bundle_type": bundle_type,
                                "generation_method": "item2vec",
                                "similarity_score": float(similarity),
                                "anchor_product": anchor
                            }
                            candidates.append(candidate)
            
            return candidates[:50]  # Limit item2vec candidates
            
        except Exception as e:
            logger.warning(f"Error in item2vec generation: {e}")
            return []
    
    async def get_or_train_embeddings(
        self,
        csv_upload_id: str,
        sequences: Optional[List[List[str]]] = None,
    ) -> Dict[str, np.ndarray]:
        """Get existing embeddings or train new ones"""
        try:
            # Check if embeddings exist for this upload
            existing_embeddings = await storage.get_variant_embeddings(csv_upload_id)
            if existing_embeddings:
                return self.deserialize_embeddings(existing_embeddings)
            
            # Train new embeddings
            embeddings = await self.train_item2vec_embeddings(csv_upload_id, sequences=sequences)
            
            # Store embeddings for future use
            if embeddings:
                await storage.store_variant_embeddings(csv_upload_id, self.serialize_embeddings(embeddings))
            
            return embeddings
            
        except Exception as e:
            logger.warning(f"Error getting/training embeddings: {e}")
            return {}
    
    async def train_item2vec_embeddings(
        self,
        csv_upload_id: str,
        sequences: Optional[List[List[str]]] = None,
    ) -> Dict[str, np.ndarray]:
        """Train item2vec embeddings using skip-gram approach"""
        try:
            # Get order sequences (baskets)
            if sequences is None:
                sequences = await self.get_purchase_sequences(csv_upload_id)
            if len(sequences) < 10:
                return {}
            
            # Build vocabulary
            vocab = self.build_vocabulary(sequences)
            if len(vocab) < 10:
                return {}
            
            # Simple skip-gram training (simplified implementation)
            embeddings = self.train_skipgram(sequences, vocab, self.embedding_dim)
            
            logger.info(f"Trained embeddings for {len(embeddings)} items")
            return embeddings
            
        except Exception as e:
            logger.warning(f"Error training embeddings: {e}")
            return {}
    
    def train_skipgram(self, sequences: List[List[str]], vocab: Dict[str, int], dim: int) -> Dict[str, np.ndarray]:
        """Simplified skip-gram training"""
        vocab_size = len(vocab)
        embeddings = {}
        
        # Initialize random embeddings
        np.random.seed(42)
        for item in vocab:
            embeddings[item] = np.random.normal(0, 0.1, dim)
        
        # Simple co-occurrence based training (simplified)
        learning_rate = 0.01
        window_size = 3
        
        for epoch in range(5):  # Small number of epochs for efficiency
            for sequence in sequences:
                for i, center_item in enumerate(sequence):
                    if center_item in vocab:
                        # Get context items
                        start = max(0, i - window_size)
                        end = min(len(sequence), i + window_size + 1)
                        
                        for j in range(start, end):
                            if j != i and j < len(sequence):
                                context_item = sequence[j]
                                if context_item in vocab:
                                    # Simple update rule (gradient approximation)
                                    center_emb = embeddings[center_item]
                                    context_emb = embeddings[context_item]
                                    
                                    # Positive sample update
                                    dot_product = np.dot(center_emb, context_emb)
                                    sigmoid = 1 / (1 + np.exp(-dot_product))
                                    
                                    gradient = learning_rate * (1 - sigmoid)
                                    embeddings[center_item] += gradient * context_emb
                                    embeddings[context_item] += gradient * center_emb
        
        # Normalize embeddings
        for item in embeddings:
            norm = np.linalg.norm(embeddings[item])
            if norm > 0:
                embeddings[item] = embeddings[item] / norm
        
        return embeddings
    
    def build_vocabulary(self, sequences: List[List[str]], order_count: Optional[int] = None) -> Dict[str, int]:
        """Build vocabulary from sequences with adaptive frequency threshold"""
        item_counts = defaultdict(int)
        for sequence in sequences:
            for item in sequence:
                item_counts[item] += 1

        # Use adaptive min_frequency based on store size
        if order_count is not None:
            tier = self._get_store_tier(order_count)
            min_freq = self.store_size_thresholds[tier]["min_frequency"]
        else:
            min_freq = self.min_frequency

        # Filter by minimum frequency
        vocab = {}
        idx = 0
        for item, count in item_counts.items():
            if count >= min_freq:
                vocab[item] = idx
                idx += 1

        return vocab
    
    def find_similar_products(self, anchor: str, embeddings: Dict[str, np.ndarray], top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar products using cosine similarity"""
        if anchor not in embeddings:
            return []
        
        anchor_emb = embeddings[anchor]
        similarities = []
        
        for item, item_emb in embeddings.items():
            if item != anchor:
                # Cosine similarity
                similarity = np.dot(anchor_emb, item_emb)
                similarities.append((item, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    async def get_transactions_for_mining(self, csv_upload_id: str) -> List[Set[str]]:
        """Get transaction data for pattern mining"""
        try:
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            orders = (
                await storage.get_orders_with_lines_by_run(run_id)
                if run_id else await storage.get_orders_with_lines(csv_upload_id)
            )
            transactions = []
            
            for order in orders:
                # Get all variant_ids (or SKUs) in this order
                order_items = set()
                for line in order.order_lines:
                    sku = getattr(line, 'sku', None)
                    variant_id = getattr(line, 'variant_id', None)

                    # ALWAYS use variant_id as primary identifier (uniform join key)
                    # variant_id is stable and always present from Shopify
                    # SKU is for display only and may be synthetic (no-sku-*)
                    if variant_id:
                        key = str(variant_id)
                    elif sku and not sku.startswith('no-sku-'):
                        # Fallback to real SKU only if variant_id is missing
                        key = sku
                    else:
                        continue

                    order_items.add(key)

                if len(order_items) >= 2:  # Only multi-item orders
                    transactions.append(order_items)
            
            return transactions
            
        except Exception as e:
            logger.warning(f"Error getting transactions: {e}")
            return []
    
    async def get_purchase_sequences(self, csv_upload_id: str) -> List[List[str]]:
        """Get purchase sequences for embedding training"""
        try:
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            orders = (
                await storage.get_orders_with_lines_by_run(run_id)
                if run_id else await storage.get_orders_with_lines(csv_upload_id)
            )
            sequences = []
            
            for order in orders:
                sequence = []
                for line in order.order_lines:
                    sku = getattr(line, 'sku', None)
                    variant_id = getattr(line, 'variant_id', None)

                    # ALWAYS use variant_id as primary identifier (uniform join key)
                    # variant_id is stable and always present from Shopify
                    # SKU is for display only and may be synthetic (no-sku-*)
                    if variant_id:
                        key = str(variant_id)
                    elif sku and not sku.startswith('no-sku-'):
                        # Fallback to real SKU only if variant_id is missing
                        key = sku
                    else:
                        continue

                    sequence.append(key)

                if len(sequence) >= 2:
                    sequences.append(sequence)
            
            return sequences
            
        except Exception as e:
            logger.warning(f"Error getting sequences: {e}")
            return []
    
    async def get_anchor_products_for_objective(self, csv_upload_id: str, objective: str) -> List[str]:
        """Get anchor products based on objective"""
        try:
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            catalog_items = (
                await storage.get_catalog_snapshots_by_run(run_id)
                if run_id else await storage.get_catalog_snapshots_by_upload(csv_upload_id)
            )
            anchors = []
            
            for item in catalog_items:
                include_as_anchor = False
                
                if objective == "clear_slow_movers":
                    # Use slow-moving items as anchors
                    if hasattr(item, 'objective_flags') and item.objective_flags:
                        flags = item.objective_flags if isinstance(item.objective_flags, dict) else {}
                        if flags.get("is_slow_mover", False):
                            include_as_anchor = True
                
                elif objective == "new_launch":
                    # Use new products as anchors
                    if hasattr(item, 'objective_flags') and item.objective_flags:
                        flags = item.objective_flags if isinstance(item.objective_flags, dict) else {}
                        if flags.get("is_new_launch", False):
                            include_as_anchor = True
                
                elif objective == "increase_aov":
                    # Use higher-priced items as anchors
                    if item.price > Decimal('25'):
                        include_as_anchor = True
                
                else:
                    # Default: use items with good availability
                    if item.available_total > 0:
                        include_as_anchor = True
                
                # Use variant_id as the uniform join key (not SKU)
                if include_as_anchor and item.variant_id:
                    anchors.append(item.variant_id)
            
            return anchors[:50]  # Limit number of anchors
            
        except Exception as e:
            logger.warning(f"Error getting anchor products: {e}")
            return []
    
    def convert_rules_to_candidates(self, association_rules: List, bundle_type: str) -> List[Dict[str, Any]]:
        """Convert association rules to candidate format"""
        candidates = []
        
        for rule in association_rules:
            try:
                antecedent = rule.antecedent if isinstance(rule.antecedent, list) else [rule.antecedent]
                consequent = rule.consequent if isinstance(rule.consequent, list) else [rule.consequent]
                
                candidate = {
                    "products": antecedent + consequent,
                    "support": float(rule.support),
                    "confidence": float(rule.confidence),
                    "lift": float(rule.lift),
                    "bundle_type": bundle_type,
                    "generation_method": "apriori",
                    "rule_id": rule.id
                }
                candidates.append(candidate)
                
            except Exception as e:
                logger.warning(f"Error converting rule to candidate: {e}")
                continue
        
        return candidates
    
    def combine_candidates(self, *candidate_lists) -> List[Dict[str, Any]]:
        """Combine and deduplicate candidates from multiple sources"""
        all_candidates = []
        seen_signatures = set()
        
        for candidate_list in candidate_lists:
            for candidate in candidate_list:
                # Create signature for deduplication
                products = sorted(candidate.get("products", []))
                bundle_type = candidate.get("bundle_type", "")
                signature = f"{bundle_type}:{':'.join(products)}"
                
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    all_candidates.append(candidate)
        
        return all_candidates
    
    def identify_sources(
        self,
        candidate: Dict[str, Any],
        apriori_candidates: List,
        fpgrowth_candidates: List,
        item2vec_candidates: List,
        top_pair_candidates: List
    ) -> List[str]:
        """Identify which generation methods produced this candidate"""
        sources = []

        candidate_products = sorted(candidate.get("products", []))
        candidate_type = candidate.get("bundle_type", "")

        # Check each source
        for source_name, source_candidates in [
            ("apriori", apriori_candidates),
            ("fpgrowth", fpgrowth_candidates), 
            ("item2vec", item2vec_candidates),
            ("top_pair_mining", top_pair_candidates)
        ]:
            for source_candidate in source_candidates:
                source_products = sorted(source_candidate.get("products", []))
                source_type = source_candidate.get("bundle_type", "")

                if candidate_products == source_products and candidate_type == source_type:
                    sources.append(source_name)
                    break

        return sources
    
    def serialize_embeddings(self, embeddings: Dict[str, np.ndarray]) -> str:
        """Serialize embeddings for storage"""
        serializable = {}
        for item, embedding in embeddings.items():
            serializable[item] = embedding.tolist()
        return json.dumps(serializable)
    
    def deserialize_embeddings(self, serialized: str) -> Dict[str, np.ndarray]:
        """Deserialize embeddings from storage"""
        try:
            data = json.loads(serialized)
            embeddings = {}
            for item, embedding_list in data.items():
                embeddings[item] = np.array(embedding_list)
            return embeddings
        except Exception as e:
            logger.warning(f"Error deserializing embeddings: {e}")
            return {}
