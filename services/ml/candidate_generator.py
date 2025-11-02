"""
ML Candidate Generator Service
Implements LLM embeddings (replacing item2vec) and FPGrowth algorithm for better candidate generation
"""
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import numpy as np
from decimal import Decimal
from collections import defaultdict
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
    run_id: Optional[str]
    valid_skus: Set[str]
    varid_to_sku: Dict[str, str]
    transactions: List[Set[str]]
    sequences: List[List[str]]
    embeddings: Dict[str, np.ndarray]
    llm_only: bool = False

class CandidateGenerator:
    """Advanced candidate generation using ML techniques"""
    
    def __init__(self):
        self.item_embeddings = {}
        self.embedding_dim = 64
        self.min_frequency = 2
        self.use_fpgrowth = True  # Feature flag for FPGrowth vs Apriori
        self.small_store_relax_threshold = 30  # Allow looser filtering when catalog is tiny
    
    async def prepare_context(self, csv_upload_id: str) -> CandidateGenerationContext:
        """Prefetch expensive data needed by candidate generation."""
        import time
        start_time = time.time()

        run_id = await storage.get_run_id_for_upload(csv_upload_id)
        valid_skus = await self.get_valid_skus_for_csv(csv_upload_id)
        varid_to_sku = await self.get_variantid_to_sku_map(csv_upload_id)
        transactions = await self.get_transactions_for_mining(csv_upload_id)
        sequences = await self.get_purchase_sequences(csv_upload_id)

        # NEW: Generate LLM embeddings (replaces item2vec)
        # Expected time: 2-3 seconds (vs 60-120s for item2vec)
        embeddings = {}
        try:
            # Get catalog products for embedding generation
            catalog = await storage.get_catalog_snapshots_by_run(run_id)
            catalog_list = [
                {
                    'sku': getattr(item, 'sku', ''),
                    'title': getattr(item, 'title', ''),
                    'product_category': getattr(item, 'product_category', ''),
                    'brand': getattr(item, 'brand', ''),
                    'vendor': getattr(item, 'vendor', ''),
                    'product_type': getattr(item, 'product_type', ''),
                    'description': getattr(item, 'description', ''),
                    'tags': getattr(item, 'tags', ''),
                }
                for item in catalog
            ]
            with_sku = sum(1 for item in catalog_list if item.get('sku'))
            logger.info(
                "[%s] Preparing LLM embeddings | catalog_items=%d with_sku=%d",
                csv_upload_id,
                len(catalog_list),
                with_sku,
            )

            logger.info(f"Generating LLM embeddings for {len(catalog_list)} products...")
            embeddings = await llm_embedding_engine.get_embeddings_batch(catalog_list, use_cache=True)
            logger.info(f"LLM embeddings generated in {time.time() - start_time:.2f}s")
            logger.info(
                "[%s] Embedding result | embeddings_available=%d",
                csv_upload_id,
                len(embeddings),
            )

        except Exception as e:
            logger.warning(f"Failed to generate LLM embeddings: {e}. Continuing without embeddings.")
            embeddings = {}

        return CandidateGenerationContext(
            run_id=run_id,
            valid_skus=valid_skus,
            varid_to_sku=varid_to_sku,
            transactions=transactions,
            sequences=sequences,
            embeddings=embeddings or {},
        )
    
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
            "invalid_sku_candidates_filtered": 0
        }
        
        try:
            run_id = context.run_id if context else await storage.get_run_id_for_upload(csv_upload_id)
            # CRITICAL FIX: Preload valid SKUs to prevent infinite loop
            logger.info(f"Preloading valid SKUs for scope: run_id={run_id} csv_upload_id={csv_upload_id}")
            valid_skus = context.valid_skus if context else await self.get_valid_skus_for_csv(csv_upload_id)
            # Map variant_id -> sku so rule products expressed as variant_ids can be validated against catalog
            varid_to_sku = context.varid_to_sku if context else await self.get_variantid_to_sku_map(csv_upload_id)
            logger.info(f"Found {len(valid_skus)} valid SKUs for prefiltering")
            
            llm_only_mode = bool(getattr(context, "llm_only", False))
            if llm_only_mode:
                metrics["generation_method"] = "llm_only"
                logger.info(f"[{csv_upload_id}] LLM-only candidate generation enabled (sparse dataset)")
            
            # Generate candidates from multiple sources
            apriori_candidates = []
            item2vec_candidates = []
            fpgrowth_candidates = []
            
            if not llm_only_mode:
                # 1. Traditional Apriori rules (existing)
                association_rules = (
                    await storage.get_association_rules_by_run(run_id)
                    if run_id else await storage.get_association_rules(csv_upload_id)
                )
                apriori_candidates = self.convert_rules_to_candidates(association_rules, bundle_type)
                metrics["apriori_candidates"] = len(apriori_candidates)
                
                # 2. FPGrowth algorithm (more efficient)
                if self.use_fpgrowth:
                    transactions = context.transactions if context else None
                    fpgrowth_candidates = await self.generate_fpgrowth_candidates(
                        csv_upload_id,
                        bundle_type,
                        transactions=transactions,
                    )
                    metrics["fpgrowth_candidates"] = len(fpgrowth_candidates)
            else:
                metrics["apriori_candidates"] = 0
                metrics["fpgrowth_candidates"] = 0
            
            # 3. LLM embeddings (semantic similarity) - REPLACES item2vec
            embeddings = context.embeddings if context else {}
            llm_candidates = []
            if embeddings:
                try:
                    # Get catalog for LLM candidate generation
                    catalog = await storage.get_catalog_snapshots_by_run(run_id)
                    catalog_list = [
                        {
                            'sku': getattr(item, 'sku', ''),
                            'title': getattr(item, 'title', ''),
                            'product_category': getattr(item, 'product_category', ''),
                        }
                        for item in catalog
                    ]

                    llm_candidates = await llm_embedding_engine.generate_candidates_by_similarity(
                        csv_upload_id=csv_upload_id,
                        bundle_type=bundle_type,
                        objective=objective,
                        catalog=catalog_list,
                        embeddings=embeddings,
                        num_candidates=20
                    )
                    logger.info(f"Generated {len(llm_candidates)} LLM-based candidates")
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
            
            # 4. Deterministic top-pair mining to guarantee strongest co-purchases make it through
            top_pair_candidates = []
            if not llm_only_mode:
                transactions_for_pairs = context.transactions if context else None
                top_pair_candidates = await self.generate_top_pair_candidates(
                    csv_upload_id,
                    bundle_type,
                    transactions=transactions_for_pairs,
                )
                metrics["top_pair_candidates"] = len(top_pair_candidates)
            else:
                metrics["top_pair_candidates"] = 0

            # 5. Combine and deduplicate candidates (using LLM + transactional)
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
            all_candidates = self.filter_candidates_by_valid_skus(
                all_candidates, valid_skus, varid_to_sku, csv_upload_id=csv_upload_id
            )
            invalid_filtered = original_count - len(all_candidates)
            metrics["invalid_sku_candidates_filtered"] = invalid_filtered

            if invalid_filtered > 0:
                logger.warning(f"Candidates: filtered_invalid={invalid_filtered} remaining={len(all_candidates)} valid_skus={len(valid_skus)} varid_map={len(varid_to_sku)}")

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

    async def get_variantid_to_sku_map(self, csv_upload_id: str) -> Dict[str, str]:
        """Build a mapping from variant_id to sku using catalog snapshots (run-aware)."""
        try:
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            if run_id:
                snapshots = await storage.get_catalog_snapshots_by_run(run_id)
            else:
                snapshots = await storage.get_catalog_snapshots_by_upload(csv_upload_id)
            mapping = {}
            for s in snapshots:
                vid = getattr(s, 'variant_id', None)
                sku = getattr(s, 'sku', None)
                if vid and sku:
                    mapping[str(vid)] = str(sku)
            return mapping
        except Exception as e:
            logger.warning(f"Error building variant_id→sku map: {e}")
            return {}
    
    async def get_valid_skus_for_csv(self, csv_upload_id: str) -> Set[str]:
        """Get all valid SKUs for the given CSV upload ID"""
        try:
            # Resolve run_id to aggregate across all related uploads
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            if run_id:
                catalog_entries = await storage.get_catalog_snapshots_by_run(run_id)
            else:
                catalog_entries = await storage.get_catalog_snapshots_by_upload(csv_upload_id)
            valid_skus = {entry.sku for entry in catalog_entries if entry.sku}
            
            # CRITICAL FIX: Remove placeholder and invalid SKUs including gid:// identifiers
            valid_skus.discard(None)
            filtered_skus = set()
            for sku in valid_skus:
                if (sku and 
                    not sku.startswith("no-sku-") and 
                    not sku.startswith("gid://") and
                    not sku.startswith("null") and
                    sku.strip() != ""):
                    filtered_skus.add(sku)
            
            logger.info(f"Filtered SKUs for {csv_upload_id}: {len(catalog_entries)} total -> {len(valid_skus)} non-null -> {len(filtered_skus)} valid")
            return filtered_skus
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

            order_sku_map: Dict[str, Set[str]] = defaultdict(set)
            for line in order_lines:
                order_id = getattr(line, "order_id", None)
                sku = getattr(line, "sku", None)
                if not order_id or not sku:
                    continue
                sku = str(sku).strip()
                if not sku or sku.startswith("gid://") or sku.startswith("no-sku-"):
                    continue
                order_sku_map[order_id].add(sku)

            transactions = [skus for skus in order_sku_map.values() if len(skus) >= 2]

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
    
    def filter_candidates_by_valid_skus(
        self,
        candidates: List[Dict[str, Any]],
        valid_skus: Set[str],
        varid_to_sku: Dict[str, str],
        csv_upload_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Filter out candidates containing invalid products.
        Accept either SKU directly or variant_id that can be mapped to a SKU via varid_to_sku.
        """
        if not valid_skus:
            logger.warning("No valid SKUs available for filtering")
            return candidates
        
        filtered_candidates = []
        scope = csv_upload_id or "unknown_upload"
        small_store = len(valid_skus) <= self.small_store_relax_threshold
        if small_store:
            logger.info(
                "[%s] Small catalog detected (valid_skus=%d). Relaxed candidate filtering enabled.",
                scope,
                len(valid_skus),
            )
        filtered_reason_counts: Dict[str, int] = defaultdict(int)
        relaxed_reason_counts: Dict[str, int] = defaultdict(int)
        relaxed_candidates = 0
        for candidate in candidates:
            products = candidate.get("products", [])
            failure_reason = None
            failure_product = None
            normalized: List[str] = []
            all_valid = True

            if not isinstance(products, list) or len(products) < 2:
                all_valid = False
                failure_reason = "insufficient products"
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

                    if p_str in valid_skus:
                        normalized.append(p_str)
                        continue

                    mapped = varid_to_sku.get(p_str)
                    if mapped and mapped in valid_skus:
                        normalized.append(mapped)
                        continue

                    all_valid = False
                    failure_product = p_str
                    if mapped and mapped not in valid_skus:
                        failure_reason = f"variant resolved to SKU {mapped} not present in catalog"
                    else:
                        failure_reason = "unmapped product (not SKU or variant)"
                    break

            if all_valid and len(normalized) >= 2:
                try:
                    candidate["products"] = normalized
                except Exception:
                    pass
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
                        if original_str in valid_skus:
                            fallback_products.append(original_str)
                        else:
                            mapped = varid_to_sku.get(original_str)
                            fallback_products.append(mapped if mapped else original_str or original)
                    try:
                        candidate["products"] = fallback_products
                    except Exception:
                        pass
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
            if len(transactions) < 10:
                logger.info("Insufficient transactions for FPGrowth, falling back to Apriori")
                return []
            
            # Build FP-tree and mine frequent itemsets
            frequent_itemsets = self.fpgrowth_mining(transactions, min_support=0.05)
            
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
        """Simple FPGrowth implementation for frequent itemset mining"""
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
                    from itertools import combinations
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
    
    def build_vocabulary(self, sequences: List[List[str]]) -> Dict[str, int]:
        """Build vocabulary from sequences"""
        item_counts = defaultdict(int)
        for sequence in sequences:
            for item in sequence:
                item_counts[item] += 1
        
        # Filter by minimum frequency
        vocab = {}
        idx = 0
        for item, count in item_counts.items():
            if count >= self.min_frequency:
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
                    key = None
                    if getattr(line, 'sku', None):
                        key = line.sku
                    elif getattr(line, 'variant_id', None):
                        key = line.variant_id
                    if key:
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
                    key = None
                    if getattr(line, 'sku', None):
                        key = line.sku
                    elif getattr(line, 'variant_id', None):
                        key = line.variant_id
                    if key:
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
                
                if include_as_anchor and item.sku:
                    anchors.append(item.sku)
            
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
