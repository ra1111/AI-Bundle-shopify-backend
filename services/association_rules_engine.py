"""
Association Rules Engine
Implements Apriori algorithm for market basket analysis
"""
from typing import List, Dict, Set, Tuple, Any, Optional
import logging
from collections import defaultdict
from itertools import combinations
from decimal import Decimal
import uuid

from services.storage import storage

logger = logging.getLogger(__name__)

class AssociationRulesEngine:
    """Association rules engine using Apriori algorithm"""
    
    def __init__(self):
        self.min_support = 0.02  # 2% minimum support (more friendly for small shops)
        self.min_confidence = 0.05  # 5% minimum confidence (lowered for small shops)
        self.max_itemset_size = 3  # Maximum size of itemsets
    
    async def generate_association_rules(self, csv_upload_id: str) -> None:
        """Generate association rules for transactions"""
        if not csv_upload_id:
            raise ValueError("csv_upload_id is required")
        try:
            logger.info(f"Generating association rules for upload: {csv_upload_id}")
            
            # Get transactions from order lines
            transactions = await self.get_transactions(csv_upload_id)
            if not transactions:
                logger.warning("No transactions found")
                return
            
            logger.info(f"Processing {len(transactions)} transactions")
            
            # Generate frequent itemsets
            frequent_itemsets = self.generate_frequent_itemsets(transactions)
            if not frequent_itemsets:
                logger.warning("No frequent itemsets found")
                return
            
            # Generate association rules
            rules = self.generate_rules_from_itemsets(frequent_itemsets, transactions, csv_upload_id)
            if not rules:
                logger.warning("No association rules generated")
                return
            
            # Clear existing rules for this upload
            await storage.clear_association_rules(csv_upload_id)
            
            # Save rules to database
            await storage.create_association_rules(rules)
            
            logger.info(f"Generated {len(rules)} association rules")
            
        except Exception as e:
            logger.error(f"Association rules generation error: {e}")
            raise
    
    async def get_transactions(self, csv_upload_id: Optional[str] = None) -> List[Set[str]]:
        """Get transactions (order lines grouped by order)"""
        if not csv_upload_id:
            return []
        order_lines = await storage.get_order_lines(csv_upload_id)
        
        # Group order lines by order_id
        orders_map: Dict[str, Set[str]] = defaultdict(set)
        
        for line in order_lines:
            # Use variant_id as canonical key, fallback to sku for compatibility
            variant_id = getattr(line, 'variant_id', None)
            sku = getattr(line, 'sku', None)
            order_id = getattr(line, 'order_id', None)
            
            # Prefer variant_id, but allow sku fallback
            product_key = variant_id or sku
            if product_key and order_id:
                orders_map[order_id].add(product_key)
        
        # Filter out single-item transactions (need at least 2 items for rules)
        transactions = [skus for skus in orders_map.values() if len(skus) >= 2]
        return transactions
    
    def generate_frequent_itemsets(self, transactions: List[Set[str]]) -> Dict[int, List[Tuple[frozenset, float]]]:
        """Generate frequent itemsets using Apriori algorithm"""
        total_transactions = len(transactions)
        frequent_itemsets: Dict[int, List[Tuple[frozenset, float]]] = {}
        
        # Generate 1-itemsets
        item_counts: Dict[str, int] = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # Filter by minimum support
        min_support_count = self.min_support * total_transactions
        frequent_1_itemsets = []
        
        for item, count in item_counts.items():
            if count >= min_support_count:
                support = count / total_transactions
                frequent_1_itemsets.append((frozenset([item]), support))
        
        if not frequent_1_itemsets:
            return frequent_itemsets
        
        frequent_itemsets[1] = frequent_1_itemsets
        logger.info(f"Generated {len(frequent_1_itemsets)} frequent 1-itemsets")
        
        # Generate k-itemsets (k > 1)
        for k in range(2, self.max_itemset_size + 1):
            candidate_itemsets = self.generate_candidate_itemsets(
                frequent_itemsets[k-1], k
            )
            
            if not candidate_itemsets:
                break
            
            # Count support for candidates
            itemset_counts: Dict[frozenset, int] = defaultdict(int)
            
            for transaction in transactions:
                for candidate in candidate_itemsets:
                    if candidate.issubset(transaction):
                        itemset_counts[candidate] += 1
            
            # Filter by minimum support
            frequent_k_itemsets = []
            for itemset, count in itemset_counts.items():
                if count >= min_support_count:
                    support = count / total_transactions
                    frequent_k_itemsets.append((itemset, support))
            
            if not frequent_k_itemsets:
                break
            
            frequent_itemsets[k] = frequent_k_itemsets
            logger.info(f"Generated {len(frequent_k_itemsets)} frequent {k}-itemsets")
        
        return frequent_itemsets
    
    def generate_candidate_itemsets(
        self, 
        frequent_itemsets: List[Tuple[frozenset, float]], 
        k: int
    ) -> List[frozenset]:
        """Generate candidate itemsets of size k"""
        candidates = []
        
        # Get just the itemsets (without support values)
        itemsets = [itemset for itemset, _ in frequent_itemsets]
        
        # Generate candidates by joining frequent (k-1)-itemsets
        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                # Join two itemsets if they differ by exactly one item
                union = itemsets[i] | itemsets[j]
                if len(union) == k:
                    # Check if all (k-1)-subsets are frequent
                    if self.has_frequent_subsets(union, itemsets):
                        candidates.append(union)
        
        return candidates
    
    def has_frequent_subsets(self, itemset: frozenset, frequent_itemsets: List[frozenset]) -> bool:
        """Check if all (k-1)-subsets of itemset are frequent"""
        for item in itemset:
            subset = itemset - {item}
            if subset not in frequent_itemsets:
                return False
        return True
    
    def generate_rules_from_itemsets(
        self, 
        frequent_itemsets: Dict[int, List[Tuple[frozenset, float]]],
        transactions: List[Set[str]],
        csv_upload_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate association rules from frequent itemsets"""
        rules = []
        
        # Process itemsets of size 2 or more
        for k in range(2, len(frequent_itemsets) + 1):
            for itemset, support in frequent_itemsets[k]:
                # Generate all possible rule combinations
                items = list(itemset)
                
                # Generate rules with different antecedent/consequent splits
                for r in range(1, len(items)):
                    for antecedent_items in combinations(items, r):
                        antecedent = frozenset(antecedent_items)
                        consequent = itemset - antecedent
                        
                        # Calculate confidence and lift
                        confidence = self.calculate_confidence(
                            antecedent, itemset, frequent_itemsets, transactions
                        )
                        
                        if confidence >= self.min_confidence:
                            lift = self.calculate_lift(
                                antecedent, consequent, support, frequent_itemsets
                            )
                            
                            rule = {
                                "id": str(uuid.uuid4()),
                                "csv_upload_id": csv_upload_id,
                                "antecedent": list(antecedent),
                                "consequent": list(consequent),
                                "support": support,
                                "confidence": confidence,
                                "lift": lift
                            }
                            rules.append(rule)
        
        # Sort by lift (descending)
        rules.sort(key=lambda r: float(r["lift"]), reverse=True)
        
        return rules
    
    def calculate_confidence(
        self, 
        antecedent: frozenset, 
        itemset: frozenset, 
        frequent_itemsets: Dict[int, List[Tuple[frozenset, float]]],
        transactions: List[Set[str]]
    ) -> float:
        """Calculate confidence for a rule"""
        # Find support of antecedent
        antecedent_support = self.find_itemset_support(antecedent, frequent_itemsets)
        
        if antecedent_support == 0:
            return 0.0
        
        # Find support of full itemset
        itemset_support = self.find_itemset_support(itemset, frequent_itemsets)
        
        # Confidence = support(itemset) / support(antecedent)
        return itemset_support / antecedent_support
    
    def calculate_lift(
        self, 
        antecedent: frozenset, 
        consequent: frozenset, 
        itemset_support: float,
        frequent_itemsets: Dict[int, List[Tuple[frozenset, float]]]
    ) -> float:
        """Calculate lift for a rule"""
        # Find support of antecedent and consequent
        antecedent_support = self.find_itemset_support(antecedent, frequent_itemsets)
        consequent_support = self.find_itemset_support(consequent, frequent_itemsets)
        
        if antecedent_support == 0 or consequent_support == 0:
            return 1.0
        
        # Lift = support(itemset) / (support(antecedent) * support(consequent))
        expected_support = antecedent_support * consequent_support
        return itemset_support / expected_support if expected_support > 0 else 1.0
    
    def find_itemset_support(
        self, 
        target_itemset: frozenset, 
        frequent_itemsets: Dict[int, List[Tuple[frozenset, float]]]
    ) -> float:
        """Find support value for a given itemset"""
        itemset_size = len(target_itemset)
        
        if itemset_size not in frequent_itemsets:
            return 0.0
        
        for itemset, support in frequent_itemsets[itemset_size]:
            if itemset == target_itemset:
                return support
        
        return 0.0