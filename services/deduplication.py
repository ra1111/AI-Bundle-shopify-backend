"""
Deduplication Service
Hash-based bundle uniqueness using sorted variant IDs + bundle type + objective
"""
from typing import List, Dict, Any, Optional, Set
import logging
import hashlib
import json

from services.storage import storage

logger = logging.getLogger(__name__)

class DeduplicationService:
    """Service for preventing duplicate bundle recommendations"""
    
    def __init__(self):
        self.seen_hashes = set()
        self.hash_to_candidate = {}
    
    def generate_bundle_hash(self, products: List[str], bundle_type: str, objective: str, 
                           additional_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate unique hash for bundle deduplication"""
        try:
            # Create canonical representation
            canonical_data = {
                "products": sorted(products),  # Sort for consistency
                "bundle_type": bundle_type,
                "objective": objective
            }
            
            # Add additional context if provided (e.g., pricing rules, constraints)
            if additional_context:
                # Only include relevant context that affects bundle uniqueness
                relevant_context = {}
                for key in ["min_quantity", "volume_threshold", "discount_structure"]:
                    if key in additional_context:
                        relevant_context[key] = additional_context[key]
                if relevant_context:
                    canonical_data["context"] = relevant_context
            
            # Generate hash
            canonical_string = json.dumps(canonical_data, sort_keys=True)
            hash_object = hashlib.sha256(canonical_string.encode())
            return hash_object.hexdigest()
            
        except Exception as e:
            logger.warning(f"Error generating bundle hash: {e}")
            # Fallback hash
            fallback_string = f"{bundle_type}:{objective}:{':'.join(sorted(products))}"
            return hashlib.md5(fallback_string.encode()).hexdigest()
    
    async def deduplicate_candidates(self, candidates: List[Dict[str, Any]], 
                                   csv_upload_id: str) -> Dict[str, Any]:
        """Deduplicate bundle candidates and handle conflicts"""
        metrics = {
            "total_candidates": len(candidates),
            "unique_candidates": 0,
            "duplicates_removed": 0,
            "duplicates_merged": 0,
            "hash_collisions": 0
        }
        
        try:
            # Check against existing database hashes first
            existing_hashes = await self.get_existing_bundle_hashes(csv_upload_id)
            
            unique_candidates = []
            candidate_hashes = {}
            duplicate_groups = {}
            
            for candidate in candidates:
                try:
                    # Generate hash for this candidate
                    bundle_hash = self.generate_bundle_hash(
                        candidate.get("products", []),
                        candidate.get("bundle_type", ""),
                        candidate.get("objective", ""),
                        candidate.get("additional_context")
                    )
                    
                    candidate["dedupe_hash"] = bundle_hash
                    
                    # Check if this is a completely new bundle
                    if bundle_hash not in existing_hashes and bundle_hash not in candidate_hashes:
                        # New unique candidate
                        candidate_hashes[bundle_hash] = candidate
                        unique_candidates.append(candidate)
                        
                    elif bundle_hash in candidate_hashes:
                        # Duplicate within current batch - merge information
                        existing_candidate = candidate_hashes[bundle_hash]
                        merged_candidate = self.merge_duplicate_candidates(existing_candidate, candidate)
                        
                        # Update the existing candidate in place
                        candidate_hashes[bundle_hash] = merged_candidate
                        # Find and update in unique_candidates list
                        for i, uc in enumerate(unique_candidates):
                            if uc.get("dedupe_hash") == bundle_hash:
                                unique_candidates[i] = merged_candidate
                                break
                        
                        metrics["duplicates_merged"] += 1
                        
                        # Track duplicate groups for analysis
                        if bundle_hash not in duplicate_groups:
                            duplicate_groups[bundle_hash] = []
                        duplicate_groups[bundle_hash].append(candidate)
                        
                    else:
                        # Duplicate with existing database entry
                        metrics["duplicates_removed"] += 1
                        logger.debug(f"Skipping duplicate bundle with hash: {bundle_hash}")
                
                except Exception as e:
                    logger.warning(f"Error processing candidate for deduplication: {e}")
                    # Include candidate anyway to avoid losing data
                    candidate["dedupe_hash"] = "error_" + str(hash(str(candidate)))
                    unique_candidates.append(candidate)
                    continue
            
            metrics["unique_candidates"] = len(unique_candidates)
            metrics["duplicates_removed"] = metrics["total_candidates"] - metrics["unique_candidates"] - metrics["duplicates_merged"]
            
            # Add deduplication metadata to candidates
            for candidate in unique_candidates:
                candidate["deduplication_info"] = {
                    "is_unique": True,
                    "hash": candidate.get("dedupe_hash"),
                    "duplicate_count": len(duplicate_groups.get(candidate.get("dedupe_hash"), [])),
                    "generation_sources": candidate.get("generation_sources", [])
                }
            
            logger.info(f"Deduplication completed: {metrics}")
            
            return {
                "unique_candidates": unique_candidates,
                "metrics": metrics,
                "duplicate_groups": duplicate_groups
            }
            
        except Exception as e:
            logger.error(f"Error in deduplication: {e}")
            # Return original candidates with error info
            for candidate in candidates:
                candidate["deduplication_info"] = {"error": str(e)}
            
            return {
                "unique_candidates": candidates,
                "metrics": metrics,
                "error": str(e)
            }
    
    def merge_duplicate_candidates(self, candidate1: Dict[str, Any], candidate2: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two duplicate candidates, keeping the best information"""
        try:
            # Start with the candidate that has higher confidence/score
            score1 = candidate1.get("ranking_score", candidate1.get("confidence", 0))
            score2 = candidate2.get("ranking_score", candidate2.get("confidence", 0))
            
            if score2 > score1:
                primary = candidate2.copy()
                secondary = candidate1
            else:
                primary = candidate1.copy()
                secondary = candidate2
            
            # Merge generation sources
            sources1 = set(candidate1.get("generation_sources", []))
            sources2 = set(candidate2.get("generation_sources", []))
            primary["generation_sources"] = list(sources1.union(sources2))
            
            # Keep the best statistical measures
            primary["confidence"] = max(
                candidate1.get("confidence", 0),
                candidate2.get("confidence", 0)
            )
            primary["lift"] = max(
                candidate1.get("lift", 1),
                candidate2.get("lift", 1)
            )
            primary["support"] = max(
                candidate1.get("support", 0),
                candidate2.get("support", 0)
            )
            
            # Merge objective fit scores (average)
            obj_fit1 = candidate1.get("objective_fit_raw", 0)
            obj_fit2 = candidate2.get("objective_fit_raw", 0)
            if obj_fit1 > 0 or obj_fit2 > 0:
                primary["objective_fit_raw"] = (obj_fit1 + obj_fit2) / 2
            
            # Keep better pricing if available
            pricing1 = candidate1.get("pricing")
            pricing2 = candidate2.get("pricing")
            if pricing2 and (not pricing1 or pricing2.get("bundle_price", 0) > 0):
                primary["pricing"] = pricing2
            
            # Merge metadata
            primary["merge_info"] = {
                "merged_from": [
                    candidate1.get("generation_method", "unknown"),
                    candidate2.get("generation_method", "unknown")
                ],
                "merge_reason": "duplicate_hash",
                "kept_primary": score2 > score1
            }
            
            return primary
            
        except Exception as e:
            logger.warning(f"Error merging duplicate candidates: {e}")
            return candidate1  # Return first candidate if merge fails
    
    async def get_existing_bundle_hashes(self, csv_upload_id: str) -> Set[str]:
        """Get hashes of existing bundle recommendations for this upload"""
        try:
            existing_recommendations = await storage.get_bundle_recommendations_hashes(csv_upload_id)
            return set(rec.dedupe_hash for rec in existing_recommendations if rec.dedupe_hash)
            
        except Exception as e:
            logger.warning(f"Error getting existing bundle hashes: {e}")
            return set()
    
    async def store_bundle_hashes(self, candidates: List[Dict[str, Any]], csv_upload_id: str) -> None:
        """Store bundle hashes for future deduplication"""
        try:
            hash_records = []
            for candidate in candidates:
                if candidate.get("dedupe_hash"):
                    hash_record = {
                        "csv_upload_id": csv_upload_id,
                        "dedupe_hash": candidate["dedupe_hash"],
                        "bundle_type": candidate.get("bundle_type"),
                        "objective": candidate.get("objective"),
                        "products": candidate.get("products", []),
                        "created_at": "now()"
                    }
                    hash_records.append(hash_record)
            
            if hash_records:
                await storage.store_bundle_hashes(hash_records)
                logger.info(f"Stored {len(hash_records)} bundle hashes")
                
        except Exception as e:
            logger.warning(f"Error storing bundle hashes: {e}")
    
    def detect_near_duplicates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect near-duplicate bundles (similar products, different types/objectives)"""
        try:
            near_duplicates = []
            
            for i, candidate1 in enumerate(candidates):
                for j, candidate2 in enumerate(candidates[i+1:], i+1):
                    similarity = self.compute_product_similarity(
                        candidate1.get("products", []),
                        candidate2.get("products", [])
                    )
                    
                    # If products are very similar but bundles are different
                    if (similarity > 0.8 and 
                        (candidate1.get("bundle_type") != candidate2.get("bundle_type") or
                         candidate1.get("objective") != candidate2.get("objective"))):
                        
                        near_duplicate = {
                            "candidate1_index": i,
                            "candidate2_index": j,
                            "product_similarity": similarity,
                            "type_difference": candidate1.get("bundle_type") != candidate2.get("bundle_type"),
                            "objective_difference": candidate1.get("objective") != candidate2.get("objective"),
                            "recommendation": self.get_near_duplicate_recommendation(candidate1, candidate2)
                        }
                        near_duplicates.append(near_duplicate)
            
            return near_duplicates
            
        except Exception as e:
            logger.warning(f"Error detecting near duplicates: {e}")
            return []
    
    def compute_product_similarity(self, products1: List[str], products2: List[str]) -> float:
        """Compute Jaccard similarity between two product lists"""
        try:
            if not products1 or not products2:
                return 0.0
            
            set1 = set(products1)
            set2 = set(products2)
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.warning(f"Error computing product similarity: {e}")
            return 0.0
    
    def get_near_duplicate_recommendation(self, candidate1: Dict[str, Any], 
                                        candidate2: Dict[str, Any]) -> str:
        """Get recommendation for handling near-duplicates"""
        try:
            score1 = candidate1.get("ranking_score", candidate1.get("confidence", 0))
            score2 = candidate2.get("ranking_score", candidate2.get("confidence", 0))
            
            if score1 > score2 * 1.2:
                return f"Keep candidate 1 (score: {score1:.3f} vs {score2:.3f})"
            elif score2 > score1 * 1.2:
                return f"Keep candidate 2 (score: {score2:.3f} vs {score1:.3f})"
            else:
                # Scores are similar, recommend based on other factors
                obj1 = candidate1.get("objective", "")
                obj2 = candidate2.get("objective", "")
                
                # Prefer certain objectives
                priority_objectives = ["clear_slow_movers", "margin_guard", "increase_aov"]
                
                if obj1 in priority_objectives and obj2 not in priority_objectives:
                    return f"Keep candidate 1 (priority objective: {obj1})"
                elif obj2 in priority_objectives and obj1 not in priority_objectives:
                    return f"Keep candidate 2 (priority objective: {obj2})"
                else:
                    return "Consider merging or manual review"
                    
        except Exception as e:
            logger.warning(f"Error getting near-duplicate recommendation: {e}")
            return "Manual review recommended"
    
    def cleanup_old_hashes(self, days_to_keep: int = 30) -> None:
        """Clean up old bundle hashes to prevent database bloat"""
        try:
            # This would be implemented to remove hashes older than specified days
            # For now, just log the intent
            logger.info(f"Cleanup requested for hashes older than {days_to_keep} days")
            # Implementation would call storage.cleanup_old_bundle_hashes(days_to_keep)
            
        except Exception as e:
            logger.warning(f"Error cleaning up old hashes: {e}")
    
    def get_deduplication_stats(self) -> Dict[str, Any]:
        """Get statistics about deduplication performance"""
        return {
            "total_hashes_seen": len(self.seen_hashes),
            "current_session_hashes": len(self.hash_to_candidate),
            "deduplication_active": True
        }