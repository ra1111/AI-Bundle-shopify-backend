"""
Feature Flags & Kill Switches System (PR-8)
Dynamic feature control with per-shop overrides and kill switches
"""
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime
from dataclasses import dataclass

from services.storage import storage

logger = logging.getLogger(__name__)

@dataclass
class FeatureFlag:
    """Feature flag configuration"""
    key: str
    value: bool
    description: str
    shop_id: Optional[str] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None

class FeatureFlagsManager:
    """Manages feature flags and kill switches for the bundle system"""
    
    def __init__(self):
        # Default global feature flags
        self.default_flags = {
            # Core bundling system
            "bundling.enabled": True,
            "bundling.v2_pipeline": True,
            
            # Pipeline phases
            "phase.csv_validation": True,
            "phase.data_mapping": True,
            "phase.objective_scoring": True,
            "phase.candidate_generation": True,
            "phase.ml_optimization": True,
            "phase.enterprise_optimization": True,
            "phase.bayesian_pricing": True,
            "phase.weighted_ranking": True,
            "phase.deduplication": True,
            "phase.explainability": True,

            # Advanced features (PR implementations)
            "advanced.pareto_optimization": True,
            "advanced.constraint_management": True,
            "advanced.performance_monitoring": True,
            "advanced.normalized_ranking": True,
            "advanced.cold_start_coverage": True,
            "advanced.metadata_similarity": True,

            # Data mapping controls
            "data_mapping.auto_reenrich_on_csv": True,
            "data_mapping.reset_cache_on_new_data": True,
            "data_mapping.unresolved_cache_ttl_seconds": 600,
            "data_mapping.enable_run_scope_fallback": True,
            "data_mapping.prefetch_enabled": True,
            "data_mapping.concurrent_mapping": True,
            "data_mapping.concurrent_map_limit": 25,
            "data_mapping.prefetch_batch_size": 100,
            "data_mapping.vectorized_normalization": True,
            "data_mapping.log_timings": True,

            # Bundling threshold + fallback tuning
            "bundling.relaxed_thresholds": True,
            "bundling.relaxed_min_support": 0.02,
            "bundling.relaxed_min_confidence": 0.18,
            "bundling.relaxed_min_lift": 1.1,
            "bundling.max_per_bundle_type": 15,
            "bundling.fallback_force_top_pairs": True,
            "bundling.fallback_force_pair_limit": 12,
            "bundling.llm_adaptive_similarity": True,

            # Analytics and insights
            "analytics.insights_engine": True,
            "analytics.predictive_models": True,
            "analytics.business_intelligence": True,
            "analytics.cohort_analysis": True,
            
            # API features
            "api.bundle_recommendations": True,
            "api.diagnostics": True,
            "api.admin_controls": True,
            "api.metrics_export": True,
            
            # Safety and performance
            "safety.max_bundle_size": 10,
            "safety.max_processing_time_ms": 12000,
            "safety.max_memory_usage_mb": 512,
            "safety.rate_limiting": True,
            
            # Monitoring and observability
            "monitoring.metrics_collection": True,
            "monitoring.performance_tracking": True,
            "monitoring.error_reporting": True,
            "monitoring.real_time_dashboard": True,
            
            # Experimental features
            "experimental.ai_copy_generation": True,
            "experimental.advanced_constraints": True,
            "experimental.dynamic_objectives": False,
            "experimental.multi_tenant_optimization": False
        }
        
        # Runtime flag cache
        self._flag_cache = {}
        self._shop_overrides = {}
        
        # Kill switch registry
        self.kill_switches = {
            "emergency.disable_all_bundling": False,
            "emergency.disable_ml_optimization": False,
            "emergency.disable_enterprise_features": False,
            "emergency.disable_analytics": False,
            "emergency.force_fallback_mode": False
        }
        
        # Flag change history
        self.change_history = []
    
    async def initialize_flags(self):
        """Initialize feature flags from storage"""
        try:
            # Load flags from database
            await self._load_flags_from_storage()
            
            # Apply any emergency kill switches
            await self._apply_kill_switches()
            
            logger.info("Feature flags initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing feature flags: {e}")
            # Fall back to default flags
            self._flag_cache = self.default_flags.copy()
    
    def get_flag(self, flag_key: str, shop_id: Optional[str] = None, default: bool = False) -> bool:
        """Get feature flag value with shop-specific overrides"""
        try:
            # Check for emergency kill switches first
            if self._is_killed_by_emergency_switch(flag_key):
                return False
            
            # Check shop-specific override
            if shop_id and shop_id in self._shop_overrides:
                shop_flags = self._shop_overrides[shop_id]
                if flag_key in shop_flags:
                    return shop_flags[flag_key]
            
            # Check global flag
            return self._flag_cache.get(flag_key, default)
            
        except Exception as e:
            logger.warning(f"Error getting flag {flag_key}: {e}")
            return default
    
    def get_all_flags(self, shop_id: Optional[str] = None) -> Dict[str, Any]:
        """Get all feature flags for a shop or globally"""
        try:
            # Start with global flags
            all_flags = self._flag_cache.copy()
            
            # Apply shop overrides
            if shop_id and shop_id in self._shop_overrides:
                all_flags.update(self._shop_overrides[shop_id])
            
            # Apply kill switches
            for flag_key in all_flags:
                if self._is_killed_by_emergency_switch(flag_key):
                    all_flags[flag_key] = False
            
            return all_flags
            
        except Exception as e:
            logger.error(f"Error getting all flags: {e}")
            return self.default_flags.copy()
    
    async def set_flag(self, flag_key: str, value: bool, shop_id: Optional[str] = None, 
                      updated_by: str = "system") -> bool:
        """Set feature flag value"""
        try:
            # Validate flag key
            if not self._is_valid_flag_key(flag_key):
                logger.warning(f"Invalid flag key: {flag_key}")
                return False
            
            # Update cache
            if shop_id:
                if shop_id not in self._shop_overrides:
                    self._shop_overrides[shop_id] = {}
                self._shop_overrides[shop_id][flag_key] = value
            else:
                self._flag_cache[flag_key] = value
            
            # Persist to storage
            await self._persist_flag(flag_key, value, shop_id, updated_by)
            
            # Record change
            self._record_flag_change(flag_key, value, shop_id, updated_by)
            
            logger.info(f"Set flag {flag_key}={value} for shop={shop_id} by {updated_by}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting flag {flag_key}: {e}")
            return False
    
    async def activate_kill_switch(self, kill_switch: str, activated_by: str = "system") -> bool:
        """Activate emergency kill switch"""
        try:
            if kill_switch not in self.kill_switches:
                logger.warning(f"Unknown kill switch: {kill_switch}")
                return False
            
            self.kill_switches[kill_switch] = True
            
            # Log critical event
            logger.critical(f"KILL SWITCH ACTIVATED: {kill_switch} by {activated_by}")
            
            # Record change
            self._record_flag_change(kill_switch, True, None, activated_by)
            
            # Apply kill switch effects immediately
            await self._apply_kill_switches()
            
            return True
            
        except Exception as e:
            logger.error(f"Error activating kill switch {kill_switch}: {e}")
            return False
    
    async def deactivate_kill_switch(self, kill_switch: str, deactivated_by: str = "system") -> bool:
        """Deactivate emergency kill switch"""
        try:
            if kill_switch not in self.kill_switches:
                logger.warning(f"Unknown kill switch: {kill_switch}")
                return False
            
            self.kill_switches[kill_switch] = False
            
            logger.warning(f"Kill switch deactivated: {kill_switch} by {deactivated_by}")
            
            # Record change
            self._record_flag_change(kill_switch, False, None, deactivated_by)
            
            # Reload normal flags
            await self.initialize_flags()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deactivating kill switch {kill_switch}: {e}")
            return False
    
    def get_flag_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics information about feature flags"""
        try:
            # Calculate flag statistics
            total_flags = len(self._flag_cache)
            enabled_flags = sum(1 for v in self._flag_cache.values() if v)
            disabled_flags = total_flags - enabled_flags
            
            # Shop override statistics
            shops_with_overrides = len(self._shop_overrides)
            total_overrides = sum(len(overrides) for overrides in self._shop_overrides.values())
            
            # Kill switch status
            active_kill_switches = [k for k, v in self.kill_switches.items() if v]
            
            return {
                "flag_summary": {
                    "total_flags": total_flags,
                    "enabled_flags": enabled_flags,
                    "disabled_flags": disabled_flags,
                    "flag_types": self._categorize_flags()
                },
                "shop_overrides": {
                    "shops_with_overrides": shops_with_overrides,
                    "total_overrides": total_overrides,
                    "override_distribution": self._get_override_distribution()
                },
                "kill_switches": {
                    "total_switches": len(self.kill_switches),
                    "active_switches": len(active_kill_switches),
                    "active_switch_names": active_kill_switches
                },
                "recent_changes": self.change_history[-10:] if self.change_history else [],
                "system_status": self._get_system_status()
            }
            
        except Exception as e:
            logger.error(f"Error getting flag diagnostics: {e}")
            return {"error": str(e)}
    
    def is_feature_enabled(self, feature_group: str, shop_id: Optional[str] = None) -> bool:
        """Check if a feature group is enabled"""
        # Feature group mappings
        feature_groups = {
            "bundling": ["bundling.enabled", "bundling.v2_pipeline"],
            "optimization": ["phase.ml_optimization", "phase.enterprise_optimization", "advanced.pareto_optimization"],
            "analytics": ["analytics.insights_engine", "analytics.predictive_models", "analytics.business_intelligence"],
            "advanced_features": ["advanced.normalized_ranking", "advanced.cold_start_coverage", "advanced.constraint_management"],
            "monitoring": ["monitoring.metrics_collection", "monitoring.performance_tracking"]
        }
        
        required_flags = feature_groups.get(feature_group, [])
        return all(self.get_flag(flag, shop_id) for flag in required_flags)
    
    # Private helper methods
    
    async def _load_flags_from_storage(self):
        """Load feature flags from database storage"""
        try:
            # This would load from database in a real implementation
            # For now, use defaults
            self._flag_cache = self.default_flags.copy()
            
            # Example of loading shop overrides
            # shop_flags = await storage.get_all_feature_flags()
            # for flag_record in shop_flags:
            #     if flag_record.shop_id:
            #         if flag_record.shop_id not in self._shop_overrides:
            #             self._shop_overrides[flag_record.shop_id] = {}
            #         self._shop_overrides[flag_record.shop_id][flag_record.key] = flag_record.value
            #     else:
            #         self._flag_cache[flag_record.key] = flag_record.value
            
        except Exception as e:
            logger.warning(f"Could not load flags from storage: {e}")
    
    async def _persist_flag(self, flag_key: str, value: bool, shop_id: Optional[str], updated_by: str):
        """Persist feature flag to storage"""
        try:
            # This would persist to database in a real implementation
            # await storage.upsert_feature_flag(flag_key, value, shop_id, updated_by)
            pass
            
        except Exception as e:
            logger.warning(f"Could not persist flag to storage: {e}")
    
    async def _apply_kill_switches(self):
        """Apply emergency kill switch effects"""
        try:
            if self.kill_switches.get("emergency.disable_all_bundling"):
                self._flag_cache["bundling.enabled"] = False
                logger.critical("All bundling disabled by emergency kill switch")
            
            if self.kill_switches.get("emergency.disable_ml_optimization"):
                self._flag_cache["phase.ml_optimization"] = False
                self._flag_cache["phase.enterprise_optimization"] = False
                self._flag_cache["advanced.pareto_optimization"] = False
                logger.critical("ML optimization disabled by emergency kill switch")
            
            if self.kill_switches.get("emergency.disable_enterprise_features"):
                for key in self._flag_cache:
                    if key.startswith("advanced.") or key.startswith("experimental."):
                        self._flag_cache[key] = False
                logger.critical("Enterprise features disabled by emergency kill switch")
            
            if self.kill_switches.get("emergency.disable_analytics"):
                for key in self._flag_cache:
                    if key.startswith("analytics."):
                        self._flag_cache[key] = False
                logger.critical("Analytics disabled by emergency kill switch")
            
            if self.kill_switches.get("emergency.force_fallback_mode"):
                # Enable only basic features
                fallback_flags = ["bundling.enabled", "phase.csv_validation", "phase.data_mapping", "api.bundle_recommendations"]
                for key in self._flag_cache:
                    self._flag_cache[key] = key in fallback_flags
                logger.critical("System in emergency fallback mode")
                
        except Exception as e:
            logger.error(f"Error applying kill switches: {e}")
    
    def _is_killed_by_emergency_switch(self, flag_key: str) -> bool:
        """Check if flag is disabled by emergency kill switch"""
        if self.kill_switches.get("emergency.disable_all_bundling") and flag_key.startswith("bundling."):
            return True
        if self.kill_switches.get("emergency.disable_ml_optimization") and "optimization" in flag_key:
            return True
        if self.kill_switches.get("emergency.disable_enterprise_features") and (flag_key.startswith("advanced.") or flag_key.startswith("experimental.")):
            return True
        if self.kill_switches.get("emergency.disable_analytics") and flag_key.startswith("analytics."):
            return True
        return False
    
    def _is_valid_flag_key(self, flag_key: str) -> bool:
        """Validate flag key format"""
        # Allow alphanumeric, dots, and underscores
        return all(c.isalnum() or c in '._' for c in flag_key)
    
    def _record_flag_change(self, flag_key: str, value: bool, shop_id: Optional[str], updated_by: str):
        """Record flag change in history"""
        change_record = {
            "timestamp": datetime.now().isoformat(),
            "flag_key": flag_key,
            "new_value": value,
            "shop_id": shop_id,
            "updated_by": updated_by
        }
        
        self.change_history.append(change_record)
        
        # Keep only last 100 changes
        if len(self.change_history) > 100:
            self.change_history = self.change_history[-100:]
    
    def _categorize_flags(self) -> Dict[str, int]:
        """Categorize flags by prefix"""
        categories = {}
        for flag_key in self._flag_cache:
            prefix = flag_key.split('.')[0]
            categories[prefix] = categories.get(prefix, 0) + 1
        return categories
    
    def _get_override_distribution(self) -> Dict[str, int]:
        """Get distribution of overrides per shop"""
        distribution = {}
        for shop_id, overrides in self._shop_overrides.items():
            count = len(overrides)
            range_key = f"{count//5*5}-{count//5*5+4}" if count < 20 else "20+"
            distribution[range_key] = distribution.get(range_key, 0) + 1
        return distribution
    
    def _get_system_status(self) -> str:
        """Get overall system status based on flags"""
        if any(self.kill_switches.values()):
            return "emergency_mode"
        elif not self.get_flag("bundling.enabled"):
            return "disabled"
        elif not self.get_flag("bundling.v2_pipeline"):
            return "legacy_mode"
        else:
            return "operational"

# Global feature flags manager instance
feature_flags = FeatureFlagsManager()
