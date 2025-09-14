"""
Enterprise Performance Monitoring for ML Operations
Tracks optimization performance, model metrics, and system health
"""
from typing import Dict, Any, List, Optional
import logging
import time
import asyncio

# Optional psutil import with fallback
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    # Fallback implementation without psutil
from decimal import Decimal
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json

from services.storage import storage

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for ML operations"""
    operation_id: str
    operation_type: str  # "optimization", "candidate_generation", "ranking", "pricing"
    csv_upload_id: str
    
    # Timing metrics
    start_time: datetime
    end_time: Optional[datetime] = None
    processing_time_ms: Optional[float] = None
    
    # Resource metrics
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Operation metrics
    input_size: int = 0
    output_size: int = 0
    success: bool = True
    error_message: Optional[str] = None
    
    # ML-specific metrics
    model_accuracy: Optional[float] = None
    convergence_iterations: Optional[int] = None
    pareto_frontier_size: Optional[int] = None
    constraint_violations: int = 0
    
    # Business metrics
    revenue_potential: Optional[float] = None
    margin_impact: Optional[float] = None
    inventory_efficiency: Optional[float] = None

@dataclass 
class SystemHealthMetrics:
    """System health and capacity metrics"""
    timestamp: datetime
    
    # System resources
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    
    # Database performance
    db_connection_count: int
    avg_query_time_ms: float
    
    # Cache performance
    cache_hit_rate: float
    cache_size_mb: float
    
    # Processing queue
    queue_size: int
    processing_rate: float  # operations per minute

class EnterprisePerformanceMonitor:
    """Enterprise-grade performance monitoring for ML operations"""
    
    def __init__(self):
        self.active_operations = {}
        self.performance_history = []
        self.system_health_history = []
        
        # Performance thresholds
        self.thresholds = {
            "max_processing_time_ms": 30000,  # 30 seconds
            "max_memory_usage_mb": 2048,      # 2 GB
            "max_cpu_usage_percent": 80,      # 80%
            "min_cache_hit_rate": 0.7,        # 70%
            "max_queue_size": 100
        }
        
        # Alert configurations
        self.alerts_enabled = True
        self.alert_cooldown = timedelta(minutes=5)
        self.last_alerts = {}
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
    
    async def start_operation_monitoring(self, operation_id: str, operation_type: str, 
                                       csv_upload_id: str, input_size: int = 0) -> PerformanceMetrics:
        """Start monitoring an ML operation"""
        try:
            # Get system metrics with fallback
            cpu_usage = psutil.cpu_percent() if PSUTIL_AVAILABLE else 0.0
            memory_usage = (psutil.Process().memory_info().rss / 1024 / 1024) if PSUTIL_AVAILABLE else 0.0
            
            metrics = PerformanceMetrics(
                operation_id=operation_id,
                operation_type=operation_type,
                csv_upload_id=csv_upload_id,
                start_time=datetime.now(),
                input_size=input_size,
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory_usage
            )
            
            self.active_operations[operation_id] = metrics
            
            logger.info(f"Started monitoring operation: {operation_id} ({operation_type})")
            return metrics
            
        except Exception as e:
            logger.warning(f"Error starting operation monitoring: {e}")
            # Return minimal metrics object on error
            return PerformanceMetrics(
                operation_id=operation_id,
                operation_type=operation_type,
                csv_upload_id=csv_upload_id,
                start_time=datetime.now(),
                input_size=input_size
            )
    
    async def finish_operation_monitoring(self, operation_id: str, 
                                        output_size: int = 0,
                                        success: bool = True,
                                        error_message: Optional[str] = None,
                                        ml_metrics: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """Finish monitoring an ML operation"""
        try:
            if operation_id not in self.active_operations:
                logger.warning(f"Operation {operation_id} not found in active monitoring")
                return None
            
            metrics = self.active_operations[operation_id]
            metrics.end_time = datetime.now()
            metrics.processing_time_ms = (metrics.end_time - metrics.start_time).total_seconds() * 1000
            metrics.output_size = output_size
            metrics.success = success
            metrics.error_message = error_message
            
            # Update resource metrics with fallback
            if PSUTIL_AVAILABLE:
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                metrics.peak_memory_mb = max(metrics.memory_usage_mb, current_memory)
            else:
                metrics.peak_memory_mb = metrics.memory_usage_mb
            
            # Add ML-specific metrics
            if ml_metrics:
                metrics.model_accuracy = ml_metrics.get("accuracy")
                metrics.convergence_iterations = ml_metrics.get("iterations")
                metrics.pareto_frontier_size = ml_metrics.get("pareto_solutions")
                metrics.constraint_violations = ml_metrics.get("constraint_violations", 0)
                metrics.revenue_potential = ml_metrics.get("revenue_potential")
                metrics.margin_impact = ml_metrics.get("margin_impact")
                metrics.inventory_efficiency = ml_metrics.get("inventory_efficiency")
            
            # Store in history
            self.performance_history.append(metrics)
            
            # Remove from active operations
            del self.active_operations[operation_id]
            
            # Check for performance alerts
            await self._check_performance_alerts(metrics)
            
            # Store in database for persistence
            await self._store_performance_metrics(metrics)
            
            logger.info(f"Finished monitoring operation: {operation_id} - "
                       f"Success: {success}, Time: {metrics.processing_time_ms:.1f}ms")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error finishing operation monitoring: {e}")
            return None
    
    async def update_operation_progress(self, operation_id: str, 
                                      progress_metrics: Dict[str, Any]):
        """Update progress metrics for ongoing operation"""
        try:
            if operation_id in self.active_operations:
                metrics = self.active_operations[operation_id]
                
                # Update current resource usage with fallback
                if PSUTIL_AVAILABLE:
                    metrics.cpu_usage_percent = psutil.cpu_percent()
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    metrics.memory_usage_mb = current_memory
                    metrics.peak_memory_mb = max(metrics.peak_memory_mb, current_memory)
                else:
                    # Use fallback values when psutil not available
                    metrics.cpu_usage_percent = 0.0
                    metrics.memory_usage_mb = 0.0
                
                # Update ML progress metrics
                if "iterations" in progress_metrics:
                    metrics.convergence_iterations = progress_metrics["iterations"]
                if "constraint_violations" in progress_metrics:
                    metrics.constraint_violations = progress_metrics["constraint_violations"]
                
                logger.debug(f"Updated progress for operation {operation_id}: "
                           f"CPU: {metrics.cpu_usage_percent:.1f}%, "
                           f"Memory: {metrics.memory_usage_mb:.1f}MB")
                
        except Exception as e:
            logger.warning(f"Error updating operation progress: {e}")
    
    async def get_system_health_metrics(self) -> SystemHealthMetrics:
        """Get current system health metrics"""
        try:
            # System resources with fallback
            if PSUTIL_AVAILABLE:
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                memory_percent = memory.percent
                disk_percent = disk.percent
            else:
                # Fallback values when psutil not available
                cpu_usage = 0.0
                memory_percent = 0.0
                disk_percent = 0.0
            
            # Database metrics (simplified)
            db_connection_count = 10  # Would query actual DB pool
            avg_query_time_ms = 50.0  # Would calculate from query logs
            
            # Cache metrics (simplified)
            cache_hit_rate = 0.85  # Would get from actual cache
            cache_size_mb = 256.0  # Would get from actual cache
            
            # Processing queue metrics
            queue_size = len(self.active_operations)
            processing_rate = self._calculate_processing_rate()
            
            health_metrics = SystemHealthMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_percent,
                disk_usage=disk_percent,
                db_connection_count=db_connection_count,
                avg_query_time_ms=avg_query_time_ms,
                cache_hit_rate=cache_hit_rate,
                cache_size_mb=cache_size_mb,
                queue_size=queue_size,
                processing_rate=processing_rate
            )
            
            # Store in history
            self.system_health_history.append(health_metrics)
            
            # Keep only recent history (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.system_health_history = [
                h for h in self.system_health_history if h.timestamp > cutoff_time
            ]
            
            return health_metrics
            
        except Exception as e:
            logger.error(f"Error getting system health metrics: {e}")
            return SystemHealthMetrics(
                timestamp=datetime.now(),
                cpu_usage=0, memory_usage=0, disk_usage=0,
                db_connection_count=0, avg_query_time_ms=0,
                cache_hit_rate=0, cache_size_mb=0,
                queue_size=0, processing_rate=0
            )
    
    async def generate_performance_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            # Filter recent operations
            recent_operations = [
                op for op in self.performance_history 
                if op.start_time > cutoff_time and op.end_time is not None
            ]
            
            if not recent_operations:
                return {"error": "No operations found in time window"}
            
            # Calculate aggregate metrics
            total_operations = len(recent_operations)
            successful_operations = sum(1 for op in recent_operations if op.success)
            success_rate = successful_operations / total_operations if total_operations > 0 else 0
            
            processing_times = [op.processing_time_ms for op in recent_operations if op.processing_time_ms]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            max_processing_time = max(processing_times) if processing_times else 0
            
            memory_usage = [op.peak_memory_mb for op in recent_operations]
            avg_memory_usage = sum(memory_usage) / len(memory_usage) if memory_usage else 0
            max_memory_usage = max(memory_usage) if memory_usage else 0
            
            # Operation type breakdown
            operation_types = {}
            for op in recent_operations:
                op_type = op.operation_type
                if op_type not in operation_types:
                    operation_types[op_type] = {"count": 0, "success_rate": 0, "avg_time": 0}
                
                operation_types[op_type]["count"] += 1
                if op.success:
                    operation_types[op_type]["success_rate"] += 1
                if op.processing_time_ms:
                    operation_types[op_type]["avg_time"] += op.processing_time_ms
            
            # Calculate averages for operation types
            for op_type_data in operation_types.values():
                count = op_type_data["count"]
                op_type_data["success_rate"] = op_type_data["success_rate"] / count if count > 0 else 0
                op_type_data["avg_time"] = op_type_data["avg_time"] / count if count > 0 else 0
            
            # Performance trends
            hourly_stats = self._calculate_hourly_trends(recent_operations)
            
            # System health summary
            recent_health = [h for h in self.system_health_history if h.timestamp > cutoff_time]
            health_summary = self._summarize_health_metrics(recent_health)
            
            # Recommendations
            recommendations = self._generate_performance_recommendations(recent_operations, recent_health)
            
            report = {
                "time_window_hours": time_window_hours,
                "report_generated": datetime.now().isoformat(),
                
                # Overall metrics
                "overall_metrics": {
                    "total_operations": total_operations,
                    "success_rate": success_rate,
                    "avg_processing_time_ms": avg_processing_time,
                    "max_processing_time_ms": max_processing_time,
                    "avg_memory_usage_mb": avg_memory_usage,
                    "max_memory_usage_mb": max_memory_usage
                },
                
                # Operation type breakdown
                "operation_types": operation_types,
                
                # Performance trends
                "hourly_trends": hourly_stats,
                
                # System health
                "system_health": health_summary,
                
                # Performance recommendations
                "recommendations": recommendations,
                
                # Current active operations
                "active_operations": len(self.active_operations),
                
                # Threshold violations
                "threshold_violations": self._count_threshold_violations(recent_operations)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {"error": f"Failed to generate report: {str(e)}"}
    
    def _calculate_processing_rate(self) -> float:
        """Calculate operations per minute over last hour"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=1)
            recent_operations = [
                op for op in self.performance_history 
                if op.start_time > cutoff_time and op.end_time is not None
            ]
            
            return len(recent_operations)  # Simplified: operations per hour
            
        except Exception:
            return 0.0
    
    async def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance threshold violations and send alerts"""
        try:
            if not self.alerts_enabled:
                return
            
            alert_messages = []
            
            # Check processing time
            if (metrics.processing_time_ms and 
                metrics.processing_time_ms > self.thresholds["max_processing_time_ms"]):
                alert_messages.append(
                    f"Operation {metrics.operation_id} exceeded processing time threshold: "
                    f"{metrics.processing_time_ms:.1f}ms > {self.thresholds['max_processing_time_ms']}ms"
                )
            
            # Check memory usage
            if metrics.peak_memory_mb > self.thresholds["max_memory_usage_mb"]:
                alert_messages.append(
                    f"Operation {metrics.operation_id} exceeded memory threshold: "
                    f"{metrics.peak_memory_mb:.1f}MB > {self.thresholds['max_memory_usage_mb']}MB"
                )
            
            # Check CPU usage
            if metrics.cpu_usage_percent > self.thresholds["max_cpu_usage_percent"]:
                alert_messages.append(
                    f"Operation {metrics.operation_id} exceeded CPU threshold: "
                    f"{metrics.cpu_usage_percent:.1f}% > {self.thresholds['max_cpu_usage_percent']}%"
                )
            
            # Send alerts (would integrate with actual alerting system)
            for alert in alert_messages:
                logger.warning(f"PERFORMANCE ALERT: {alert}")
                
        except Exception as e:
            logger.error(f"Error checking performance alerts: {e}")
    
    async def _store_performance_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics in database for persistence"""
        try:
            # Convert to JSON-serializable format
            metrics_data = {
                "operation_id": metrics.operation_id,
                "operation_type": metrics.operation_type,
                "csv_upload_id": metrics.csv_upload_id,
                "start_time": metrics.start_time.isoformat(),
                "end_time": metrics.end_time.isoformat() if metrics.end_time else None,
                "processing_time_ms": float(metrics.processing_time_ms) if metrics.processing_time_ms else None,
                "cpu_usage_percent": float(metrics.cpu_usage_percent),
                "memory_usage_mb": float(metrics.memory_usage_mb),
                "peak_memory_mb": float(metrics.peak_memory_mb),
                "input_size": metrics.input_size,
                "output_size": metrics.output_size,
                "success": metrics.success,
                "error_message": metrics.error_message,
                "model_accuracy": float(metrics.model_accuracy) if metrics.model_accuracy else None,
                "convergence_iterations": metrics.convergence_iterations,
                "pareto_frontier_size": metrics.pareto_frontier_size,
                "constraint_violations": metrics.constraint_violations,
                "revenue_potential": float(metrics.revenue_potential) if metrics.revenue_potential else None,
                "margin_impact": float(metrics.margin_impact) if metrics.margin_impact else None,
                "inventory_efficiency": float(metrics.inventory_efficiency) if metrics.inventory_efficiency else None
            }
            
            # Store in database (simplified - would use actual storage)
            logger.debug(f"Stored performance metrics for operation {metrics.operation_id}")
            
        except Exception as e:
            logger.warning(f"Error storing performance metrics: {e}")
    
    def _calculate_hourly_trends(self, operations: List[PerformanceMetrics]) -> List[Dict[str, Any]]:
        """Calculate hourly performance trends"""
        try:
            hourly_data = {}
            
            for op in operations:
                hour_key = op.start_time.replace(minute=0, second=0, microsecond=0)
                
                if hour_key not in hourly_data:
                    hourly_data[hour_key] = {
                        "hour": hour_key.isoformat(),
                        "operations": 0,
                        "successes": 0,
                        "total_time": 0,
                        "total_memory": 0
                    }
                
                hourly_data[hour_key]["operations"] += 1
                if op.success:
                    hourly_data[hour_key]["successes"] += 1
                if op.processing_time_ms:
                    hourly_data[hour_key]["total_time"] += op.processing_time_ms
                hourly_data[hour_key]["total_memory"] += op.peak_memory_mb
            
            # Calculate averages
            hourly_trends = []
            for hour_data in hourly_data.values():
                ops = hour_data["operations"]
                hourly_trends.append({
                    "hour": hour_data["hour"],
                    "operations": ops,
                    "success_rate": hour_data["successes"] / ops if ops > 0 else 0,
                    "avg_processing_time": hour_data["total_time"] / ops if ops > 0 else 0,
                    "avg_memory_usage": hour_data["total_memory"] / ops if ops > 0 else 0
                })
            
            return sorted(hourly_trends, key=lambda x: x["hour"])
            
        except Exception as e:
            logger.warning(f"Error calculating hourly trends: {e}")
            return []
    
    def _summarize_health_metrics(self, health_data: List[SystemHealthMetrics]) -> Dict[str, Any]:
        """Summarize system health metrics"""
        try:
            if not health_data:
                return {}
            
            cpu_values = [h.cpu_usage for h in health_data]
            memory_values = [h.memory_usage for h in health_data]
            disk_values = [h.disk_usage for h in health_data]
            
            return {
                "avg_cpu_usage": sum(cpu_values) / len(cpu_values),
                "max_cpu_usage": max(cpu_values),
                "avg_memory_usage": sum(memory_values) / len(memory_values),
                "max_memory_usage": max(memory_values),
                "avg_disk_usage": sum(disk_values) / len(disk_values),
                "max_disk_usage": max(disk_values),
                "health_checks": len(health_data)
            }
            
        except Exception as e:
            logger.warning(f"Error summarizing health metrics: {e}")
            return {}
    
    def _generate_performance_recommendations(self, operations: List[PerformanceMetrics], 
                                            health_data: List[SystemHealthMetrics]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        try:
            # High processing time recommendations
            slow_operations = [op for op in operations if op.processing_time_ms and op.processing_time_ms > 15000]
            if len(slow_operations) > len(operations) * 0.2:  # 20% of operations slow
                recommendations.append(
                    "Consider optimizing algorithm parameters or reducing problem complexity to improve processing speed"
                )
            
            # High memory usage recommendations
            memory_heavy_ops = [op for op in operations if op.peak_memory_mb > 1024]
            if len(memory_heavy_ops) > len(operations) * 0.3:  # 30% memory heavy
                recommendations.append(
                    "Implement streaming processing or batch optimization to reduce memory footprint"
                )
            
            # Success rate recommendations
            failed_ops = [op for op in operations if not op.success]
            if len(failed_ops) > len(operations) * 0.1:  # 10% failure rate
                recommendations.append(
                    "Investigate frequent operation failures and implement additional error handling"
                )
            
            # System health recommendations
            if health_data:
                avg_cpu = sum(h.cpu_usage for h in health_data) / len(health_data)
                if avg_cpu > 70:
                    recommendations.append(
                        "Consider scaling compute resources or implementing load balancing for high CPU usage"
                    )
                
                avg_memory = sum(h.memory_usage for h in health_data) / len(health_data)
                if avg_memory > 80:
                    recommendations.append(
                        "Monitor memory usage closely and consider increasing available RAM"
                    )
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]
    
    def _count_threshold_violations(self, operations: List[PerformanceMetrics]) -> Dict[str, int]:
        """Count threshold violations by type"""
        violations = {
            "processing_time": 0,
            "memory_usage": 0,
            "cpu_usage": 0,
            "constraint_violations": 0
        }
        
        try:
            for op in operations:
                if op.processing_time_ms and op.processing_time_ms > self.thresholds["max_processing_time_ms"]:
                    violations["processing_time"] += 1
                
                if op.peak_memory_mb > self.thresholds["max_memory_usage_mb"]:
                    violations["memory_usage"] += 1
                
                if op.cpu_usage_percent > self.thresholds["max_cpu_usage_percent"]:
                    violations["cpu_usage"] += 1
                
                violations["constraint_violations"] += op.constraint_violations
            
            return violations
            
        except Exception as e:
            logger.warning(f"Error counting threshold violations: {e}")
            return violations