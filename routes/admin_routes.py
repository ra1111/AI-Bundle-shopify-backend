"""
Admin API Routes for Feature Flags and Diagnostics (PR-8)
"""
from fastapi import APIRouter, HTTPException, Depends, Security, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
import os

from services.feature_flags import feature_flags
from services.obs.metrics import metrics_collector

logger = logging.getLogger(__name__)

# Security for admin endpoints
security = HTTPBearer()

class AdminAuth:
    """Simple admin authentication"""
    
    @staticmethod
    def verify_admin_key(credentials: HTTPAuthorizationCredentials = Security(security)):
        """Verify admin API key"""
        expected_key = os.getenv("ADMIN_API_KEY", "admin-dev-key-change-in-production")
        
        if not credentials or credentials.credentials != expected_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid admin API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return credentials.credentials

# Request/Response models
class FlagUpdateRequest(BaseModel):
    value: bool
    shop_id: Optional[str] = None
    updated_by: str = "api"

class BulkFlagUpdateRequest(BaseModel):
    flags: Dict[str, bool]
    shop_id: Optional[str] = None
    updated_by: str = "api"

class KillSwitchRequest(BaseModel):
    activated_by: str = "api"

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(AdminAuth.verify_admin_key)])

# Feature Flags Endpoints

@router.get("/flags")
async def get_all_feature_flags(shop_id: Optional[str] = None) -> Dict[str, Any]:
    """Get all feature flags, optionally for a specific shop"""
    try:
        flags = feature_flags.get_all_flags(shop_id)
        diagnostics = feature_flags.get_flag_diagnostics()
        
        return {
            "flags": flags,
            "diagnostics": diagnostics,
            "shop_id": shop_id
        }
    except Exception as e:
        logger.error(f"Error getting feature flags: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/flags/{flag_key}")
async def get_feature_flag(flag_key: str, shop_id: Optional[str] = None) -> Dict[str, Any]:
    """Get a specific feature flag value"""
    try:
        value = feature_flags.get_flag(flag_key, shop_id)
        
        return {
            "flag_key": flag_key,
            "value": value,
            "shop_id": shop_id
        }
    except Exception as e:
        logger.error(f"Error getting feature flag {flag_key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/flags/{flag_key}")
async def set_feature_flag(
    flag_key: str, 
    request: FlagUpdateRequest
) -> Dict[str, Any]:
    """Set a feature flag value"""
    try:
        value = request.value
        shop_id = request.shop_id
        updated_by = request.updated_by
        
        success = await feature_flags.set_flag(flag_key, value, shop_id, updated_by)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to set flag")
        
        return {
            "success": True,
            "flag_key": flag_key,
            "value": value,
            "shop_id": shop_id,
            "updated_by": updated_by
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting feature flag {flag_key}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/flags/bulk")
async def set_multiple_flags(request: BulkFlagUpdateRequest) -> Dict[str, Any]:
    """Set multiple feature flags at once"""
    try:
        flags_data = request.flags
        shop_id = request.shop_id
        updated_by = request.updated_by
        
        results = {}
        errors = []
        
        for flag_key, value in flags_data.items():
            try:
                success = await feature_flags.set_flag(flag_key, value, shop_id, updated_by)
                results[flag_key] = {"success": success, "value": value}
            except Exception as e:
                results[flag_key] = {"success": False, "error": str(e)}
                errors.append(f"{flag_key}: {e}")
        
        return {
            "results": results,
            "errors": errors,
            "shop_id": shop_id
        }
    except Exception as e:
        logger.error(f"Error setting multiple flags: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Kill Switch Endpoints

@router.post("/kill-switches/{switch_name}/activate")
async def activate_kill_switch(switch_name: str, request: KillSwitchRequest) -> Dict[str, Any]:
    """Activate an emergency kill switch"""
    try:
        activated_by = request.activated_by
        
        success = await feature_flags.activate_kill_switch(switch_name, activated_by)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to activate kill switch")
        
        return {
            "success": True,
            "kill_switch": switch_name,
            "activated_by": activated_by,
            "message": f"Kill switch {switch_name} activated"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error activating kill switch {switch_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/kill-switches/{switch_name}/deactivate")
async def deactivate_kill_switch(switch_name: str, request: KillSwitchRequest) -> Dict[str, Any]:
    """Deactivate an emergency kill switch"""
    try:
        deactivated_by = request.activated_by  # Using same field for consistency
        
        success = await feature_flags.deactivate_kill_switch(switch_name, deactivated_by)
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to deactivate kill switch")
        
        return {
            "success": True,
            "kill_switch": switch_name,
            "deactivated_by": deactivated_by,
            "message": f"Kill switch {switch_name} deactivated"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deactivating kill switch {switch_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Metrics and Monitoring Endpoints

@router.get("/metrics/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics and diagnostics"""
    try:
        performance_summary = metrics_collector.get_performance_summary()
        real_time_metrics = metrics_collector.get_real_time_metrics()
        
        return {
            "performance_summary": performance_summary,
            "real_time_metrics": real_time_metrics
        }
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/phases")
async def get_phase_diagnostics(phase_name: Optional[str] = None) -> Dict[str, Any]:
    """Get detailed phase performance diagnostics"""
    try:
        diagnostics = metrics_collector.get_phase_diagnostics(phase_name)
        
        return {
            "phase_diagnostics": diagnostics,
            "phase_filter": phase_name
        }
    except Exception as e:
        logger.error(f"Error getting phase diagnostics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/drops")
async def get_drop_analysis() -> Dict[str, Any]:
    """Get analysis of why candidates are dropped"""
    try:
        drop_analysis = metrics_collector.get_drop_analysis()
        
        return drop_analysis
    except Exception as e:
        logger.error(f"Error getting drop analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# System Health and Status

@router.get("/health")
async def get_system_health() -> Dict[str, Any]:
    """Get overall system health status"""
    try:
        flag_diagnostics = feature_flags.get_flag_diagnostics()
        performance_summary = metrics_collector.get_performance_summary()
        real_time_metrics = metrics_collector.get_real_time_metrics()
        
        # Determine overall health
        health_status = "healthy"
        issues = []
        
        # Check feature flag system
        if flag_diagnostics.get("system_status") == "emergency_mode":
            health_status = "degraded"
            issues.append("System in emergency mode - some features disabled")
        
        # Check performance alerts
        alerts = performance_summary.get("alerts", [])
        if alerts:
            critical_alerts = [a for a in alerts if a["severity"] == "critical"]
            if critical_alerts:
                health_status = "critical"
                issues.extend([a["message"] for a in critical_alerts])
            elif health_status == "healthy":
                health_status = "warning"
                issues.extend([a["message"] for a in alerts if a["severity"] == "warning"])
        
        # Check real-time status
        rt_health = real_time_metrics.get("health_status", "healthy")
        if rt_health == "critical" and health_status != "critical":
            health_status = "critical"
        elif rt_health == "warning" and health_status == "healthy":
            health_status = "warning"
        
        return {
            "status": health_status,
            "issues": issues,
            "timestamp": real_time_metrics.get("timestamp"),
            "components": {
                "feature_flags": flag_diagnostics.get("system_status", "unknown"),
                "metrics_collection": "operational" if performance_summary else "degraded",
                "bundle_generation": rt_health
            },
            "summary": {
                "total_runs": performance_summary.get("total_runs", 0),
                "success_rate": performance_summary.get("success_rate", 0),
                "active_runs": real_time_metrics.get("active_runs", 0)
            }
        }
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configuration and Settings

@router.get("/config")
async def get_system_configuration() -> Dict[str, Any]:
    """Get current system configuration"""
    try:
        flag_diagnostics = feature_flags.get_flag_diagnostics()
        
        return {
            "feature_flags": flag_diagnostics,
            "enabled_features": {
                "bundling": feature_flags.is_feature_enabled("bundling"),
                "optimization": feature_flags.is_feature_enabled("optimization"),
                "analytics": feature_flags.is_feature_enabled("analytics"),
                "advanced_features": feature_flags.is_feature_enabled("advanced_features"),
                "monitoring": feature_flags.is_feature_enabled("monitoring")
            },
            "performance_thresholds": metrics_collector.performance_thresholds
        }
    except Exception as e:
        logger.error(f"Error getting system configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))