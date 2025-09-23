"""
ML Candidate Generator Service
Implements item2vec embeddings and FPGrowth algorithm for better candidate generation
"""
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import numpy as np
from decimal import Decimal
from collections import defaultdict
import hashlib
import json

from services.storage import storage

logger = logging.getLogger(__name__)

class CandidateGenerator:
    """Advanced candidate generation using ML techniques"""
    
    def __init__(self):
        self.item_embeddings = {}
        self.embedding_dim = 64
        self.min_frequency = 2
        self.use_fpgrowth = True  # Feature flag for FPGrowth vs Apriori
    
    async def generate_candidates(self, csv_upload_id: str, bundle_type: str, objective: str) -> Dict[str, Any]:
        """Generate bundle candidates using both Apriori and item2vec"""
        metrics = {
            "apriori_candidates": 0,
            "item2vec_candidates": 0,
            "fpgrowth_candidates": 0,
            "total_unique_candidates": 0,
            "generation_method": "hybrid",
            "invalid_sku_candidates_filtered": 0
        }
        
        try:
            run_id = await storage.get_run_id_for_upload(csv_upload_id)
            # CRITICAL FIX: Preload valid SKUs to prevent infinite loop
            logger.info(f"Preloading valid SKUs for scope: run_id={run_id} csv_upload_id={csv_upload_id}")
            valid_skus = await self.get_valid_skus_for_csv(csv_upload_id)
            # Map variant_id -> sku so rule products expressed as variant_ids can be validated against catalog
            varid_to_sku = await self.get_variantid_to_sku_map(csv_upload_id)
            logger.info(f"Found {len(valid_skus)} valid SKUs for prefiltering")
            
            # Generate candidates from multiple sources
            apriori_candidates = []
            item2vec_candidates = []
            fpgrowth_candidates = []
            
            # 1. Traditional Apriori rules (existing)
            association_rules = (
                await storage.get_association_rules_by_run(run_id)
                if run_id else await storage.get_association_rules(csv_upload_id)
            )
            apriori_candidates = self.convert_rules_to_candidates(association_rules, bundle_type)
            metrics["apriori_candidates"] = len(apriori_candidates)
            
            # 2. FPGrowth algorithm (more efficient)
            if self.use_fpgrowth:
                fpgrowth_candidates = await self.generate_fpgrowth_candidates(csv_upload_id, bundle_type)
                metrics["fpgrowth_candidates"] = len(fpgrowth_candidates)
            
            # 3. item2vec embeddings (semantic similarity)
            item2vec_candidates = await self.generate_item2vec_candidates(csv_upload_id, bundle_type, objective)
            metrics["item2vec_candidates"] = len(item2vec_candidates)
            
            # 4. Combine and deduplicate candidates
            all_candidates = self.combine_candidates(
                apriori_candidates, fpgrowth_candidates, item2vec_candidates
            )
            
            # CRITICAL FIX: Filter out candidates with invalid products (allow variant_id by mapping to sku)
            original_count = len(all_candidates)
            all_candidates = self.filter_candidates_by_valid_skus(all_candidates, valid_skus, varid_to_sku)
            invalid_filtered = original_count - len(all_candidates)
            metrics["invalid_sku_candidates_filtered"] = invalid_filtered
            metrics["total_unique_candidates"] = len(all_candidates)
            
            if invalid_filtered > 0:
                logger.warning(f"Filtered out {invalid_filtered} candidates with invalid SKUs")
            
            # 5. Add source information for explainability
            for candidate in all_candidates:
                candidate["generation_sources"] = self.identify_sources(
                    candidate, apriori_candidates, fpgrowth_candidates, item2vec_candidates
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
    
    def filter_candidates_by_valid_skus(self, candidates: List[Dict[str, Any]], valid_skus: Set[str], varid_to_sku: Dict[str, str]) -> List[Dict[str, Any]]:
        """Filter out candidates containing invalid products.
        Accept either SKU directly or variant_id that can be mapped to a SKU via varid_to_sku.
        """
        if not valid_skus:
            logger.warning("No valid SKUs available for filtering")
            return candidates
        
        filtered_candidates = []
        for candidate in candidates:
            products = candidate.get("products", [])
            if isinstance(products, list) and len(products) >= 2:
                # Normalize products to SKUs for validation
                normalized = []
                all_valid = True
                for p in products:
                    if p is None:
                        all_valid = False
                        break
                    p_str = str(p)
                    if p_str in valid_skus:
                        normalized.append(p_str)
                    elif p_str in varid_to_sku and varid_to_sku[p_str] in valid_skus:
                        normalized.append(varid_to_sku[p_str])
                    else:
                        all_valid = False
                        break
                if all_valid:
                    filtered_candidates.append(candidate)
                else:
                    logger.debug(f"Filtered candidate with invalid products: {products}")
        
        return filtered_candidates
    
    async def generate_fpgrowth_candidates(self, csv_upload_id: str, bundle_type: str) -> List[Dict[str, Any]]:
        """Generate candidates using FPGrowth algorithm"""
        try:
            # Get transaction data (order baskets)
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
    
    async def generate_item2vec_candidates(self, csv_upload_id: str, bundle_type: str, objective: str) -> List[Dict[str, Any]]:
        """Generate candidates using item2vec embeddings"""
        try:
            # Train or load item2vec embeddings
            embeddings = await self.get_or_train_embeddings(csv_upload_id)
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
    
    async def get_or_train_embeddings(self, csv_upload_id: str) -> Dict[str, np.ndarray]:
        """Get existing embeddings or train new ones"""
        try:
            # Check if embeddings exist for this upload
            existing_embeddings = await storage.get_variant_embeddings(csv_upload_id)
            if existing_embeddings:
                return self.deserialize_embeddings(existing_embeddings)
            
            # Train new embeddings
            embeddings = await self.train_item2vec_embeddings(csv_upload_id)
            
            # Store embeddings for future use
            if embeddings:
                await storage.store_variant_embeddings(csv_upload_id, self.serialize_embeddings(embeddings))
            
            return embeddings
            
        except Exception as e:
            logger.warning(f"Error getting/training embeddings: {e}")
            return {}
    
    async def train_item2vec_embeddings(self, csv_upload_id: str) -> Dict[str, np.ndarray]:
        """Train item2vec embeddings using skip-gram approach"""
        try:
            # Get order sequences (baskets)
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
    
    def identify_sources(self, candidate: Dict[str, Any], apriori_candidates: List, 
                        fpgrowth_candidates: List, item2vec_candidates: List) -> List[str]:
        """Identify which generation methods produced this candidate"""
        sources = []
        
        candidate_products = sorted(candidate.get("products", []))
        candidate_type = candidate.get("bundle_type", "")
        
        # Check each source
        for source_name, source_candidates in [
            ("apriori", apriori_candidates),
            ("fpgrowth", fpgrowth_candidates), 
            ("item2vec", item2vec_candidates)
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
