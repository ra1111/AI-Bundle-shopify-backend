"""
Observability Metrics System (PR-8)
Phase timings, drop reasons, P50/P95 metrics, and performance counters
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import statistics
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

@dataclass
class PhaseMetric:
    """Metrics for a single pipeline phase"""
    phase_name: str
    start_time: float
    end_time: float
    duration_ms: float
    input_count: int
    output_count: int
    drop_count: int
    drop_reasons: Dict[str, int]
    success: bool
    error_message: Optional[str] = None
    memory_usage_mb: Optional[float] = None

@dataclass
class PipelineMetrics:
    """Complete pipeline metrics"""
    run_id: str
    csv_upload_id: str
    objective: str
    start_time: float
    end_time: float
    total_duration_ms: float
    phases: List[PhaseMetric]
    final_recommendations: int
    success: bool
    feature_flags: Dict[str, bool]

class MetricsCollector:
    """Collects and aggregates observability metrics"""
    
    def __init__(self):
        # Active phase tracking
        self.active_phases = {}  # run_id -> phase_name -> start_time
        self.pipeline_metrics = {}  # run_id -> PipelineMetrics
        
        # Historical metrics (rolling window)
        self.historical_metrics = deque(maxlen=1000)
        self.phase_timings = defaultdict(list)  # phase_name -> [duration_ms]
        self.drop_reasons = defaultdict(int)
        self.staged_publish_counters = {
            "runs": 0,
            "published": 0,
            "dropped": 0,
            "stages": 0,
        }
        self.last_staged_publish: Optional[Dict[str, Any]] = None
        
        # Real-time counters
        self.counters = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "total_recommendations": 0,
            "avg_pipeline_duration_ms": 0,
            "p95_pipeline_duration_ms": 0,
            "p99_pipeline_duration_ms": 0
        }
        
        # Thread-safe lock
        self._lock = threading.Lock()
        
        # Performance thresholds
        self.performance_thresholds = {
            "max_pipeline_duration_ms": 12000,  # 12 seconds
            "max_phase_duration_ms": {
                "csv_processing": 3000,
                "data_mapping": 2000,
                "candidate_generation": 4000,
                "optimization": 5000,
                "ranking": 2000,
                "deduplication": 1000
            },
            "min_success_rate": 0.95,
            "max_error_rate": 0.05
        }
    
    def start_pipeline(self, run_id: str, csv_upload_id: str, objective: str, 
                      feature_flags: Dict[str, bool]) -> None:
        """Start tracking a new pipeline run"""
        with self._lock:
            self.pipeline_metrics[run_id] = PipelineMetrics(
                run_id=run_id,
                csv_upload_id=csv_upload_id,
                objective=objective,
                start_time=time.time(),
                end_time=0,
                total_duration_ms=0,
                phases=[],
                final_recommendations=0,
                success=False,
                feature_flags=feature_flags.copy()
            )
            
            self.counters["total_runs"] += 1
            logger.info(f"Started pipeline tracking for run: {run_id}")
    
    @asynccontextmanager
    async def phase_timer(self, run_id: str, phase_name: str, input_count: int = 0):
        """Context manager for timing pipeline phases"""
        start_time = time.time()
        phase_metric = PhaseMetric(
            phase_name=phase_name,
            start_time=start_time,
            end_time=0,
            duration_ms=0,
            input_count=input_count,
            output_count=0,
            drop_count=0,
            drop_reasons={},
            success=False
        )
        
        try:
            yield phase_metric
            phase_metric.success = True
        except Exception as e:
            phase_metric.success = False
            phase_metric.error_message = str(e)
            logger.error(f"Phase {phase_name} failed for run {run_id}: {e}")
            raise
        finally:
            end_time = time.time()
            phase_metric.end_time = end_time
            phase_metric.duration_ms = (end_time - start_time) * 1000
            
            # Calculate drops
            if phase_metric.input_count > 0:
                phase_metric.drop_count = max(0, phase_metric.input_count - phase_metric.output_count)
            
            # Add to pipeline metrics
            with self._lock:
                if run_id in self.pipeline_metrics:
                    self.pipeline_metrics[run_id].phases.append(phase_metric)
                
                # Update historical phase timings
                self.phase_timings[phase_name].append(phase_metric.duration_ms)
                
                # Track drop reasons
                for reason, count in phase_metric.drop_reasons.items():
                    self.drop_reasons[f"{phase_name}:{reason}"] += count
                
                # Check performance thresholds
                self._check_phase_performance(phase_name, phase_metric.duration_ms)
    
    def finish_pipeline(self, run_id: str, final_recommendations: int, success: bool) -> Dict[str, Any]:
        """Finish tracking a pipeline run and return metrics"""
        with self._lock:
            if run_id not in self.pipeline_metrics:
                logger.warning(f"No pipeline metrics found for run: {run_id}")
                return {}
            
            pipeline = self.pipeline_metrics[run_id]
            pipeline.end_time = time.time()
            pipeline.total_duration_ms = (pipeline.end_time - pipeline.start_time) * 1000
            pipeline.final_recommendations = final_recommendations
            pipeline.success = success
            
            # Update counters
            if success:
                self.counters["successful_runs"] += 1
            else:
                self.counters["failed_runs"] += 1
            
            self.counters["total_recommendations"] += final_recommendations
            
            # Add to historical metrics
            self.historical_metrics.append(pipeline)
            
            # Update aggregate metrics
            self._update_aggregate_metrics()
            
            # Check performance thresholds
            self._check_pipeline_performance(pipeline.total_duration_ms)
            
            # Return metrics summary
            return self._generate_metrics_summary(pipeline)

    def record_phase_timings(self, timings: Dict[str, float]) -> None:
        """Record externally computed phase timings."""
        with self._lock:
            for phase_name, duration in timings.items():
                self.phase_timings[phase_name].append(duration)

    def record_drop_summary(self, reason_counts: Dict[str, int], namespace: Optional[str] = None) -> None:
        """Aggregate drop reasons coming from auxiliary tracks."""
        if not reason_counts:
            return
        with self._lock:
            for reason, count in reason_counts.items():
                key = f"{namespace}:{reason}" if namespace else reason
                self.drop_reasons[key] += count

    def record_staged_publish(self, summary: Dict[str, Any]) -> None:
        """Track staged publishing outcomes for observability dashboards."""
        if not summary:
            return
        with self._lock:
            self.staged_publish_counters["runs"] += 1
            self.staged_publish_counters["published"] += int(summary.get("published", 0) or 0)
            self.staged_publish_counters["dropped"] += int(summary.get("dropped", 0) or 0)
            self.staged_publish_counters["stages"] += len(summary.get("stages", []) or [])
            self.last_staged_publish = summary.copy()
    
    def get_phase_diagnostics(self, phase_name: Optional[str] = None) -> Dict[str, Any]:
        """Get detailed phase performance diagnostics"""
        diagnostics = {}
        
        with self._lock:
            if phase_name:
                # Specific phase diagnostics
                timings = self.phase_timings.get(phase_name, [])
                if timings:
                    diagnostics[phase_name] = self._calculate_phase_stats(phase_name, timings)
            else:
                # All phases diagnostics
                for phase, timings in self.phase_timings.items():
                    if timings:
                        diagnostics[phase] = self._calculate_phase_stats(phase, timings)
        
        return diagnostics
    
    def get_drop_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of why candidates are dropped"""
        with self._lock:
            # Aggregate drop reasons
            total_drops = sum(self.drop_reasons.values())
            
            if total_drops == 0:
                return {"total_drops": 0, "drop_reasons": {}}
            
            # Calculate drop percentages
            drop_percentages = {}
            for reason, count in self.drop_reasons.items():
                drop_percentages[reason] = {
                    "count": count,
                    "percentage": (count / total_drops) * 100
                }
            
            # Sort by frequency
            sorted_drops = dict(sorted(drop_percentages.items(), key=lambda x: x[1]["count"], reverse=True))
            
            return {
                "total_drops": total_drops,
                "drop_reasons": sorted_drops,
                "top_drop_reasons": list(sorted_drops.keys())[:5]
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary"""
        with self._lock:
            recent_metrics = list(self.historical_metrics)[-100:]  # Last 100 runs
            
            if not recent_metrics:
                return {"error": "No metrics available"}
            
            # Calculate success rate
            successful_runs = sum(1 for m in recent_metrics if m.success)
            success_rate = successful_runs / len(recent_metrics)
            
            # Calculate duration percentiles
            durations = [m.total_duration_ms for m in recent_metrics]
            
            summary = {
                "total_runs": self.counters["total_runs"],
                "success_rate": round(success_rate, 3),
                "recent_runs_analyzed": len(recent_metrics),
                "performance": {
                    "avg_duration_ms": round(statistics.mean(durations), 2),
                    "p50_duration_ms": round(statistics.median(durations), 2),
                    "p95_duration_ms": round(self._percentile(durations, 95), 2),
                    "p99_duration_ms": round(self._percentile(durations, 99), 2),
                    "max_duration_ms": round(max(durations), 2)
                },
                "recommendations": {
                    "total_generated": self.counters["total_recommendations"],
                    "avg_per_run": round(self.counters["total_recommendations"] / self.counters["total_runs"], 2) if self.counters["total_runs"] > 0 else 0
                },
                "alerts": self._check_performance_alerts()
            }
            
            return summary
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for monitoring dashboards"""
        with self._lock:
            # Recent activity (last 5 minutes)
            cutoff_time = time.time() - 300
            recent_metrics = [m for m in self.historical_metrics if m.start_time >= cutoff_time]
            
            # Active runs
            active_runs = len(self.pipeline_metrics)
            
            # Recent performance
            recent_durations = [m.total_duration_ms for m in recent_metrics if m.success]
            recent_failures = [m for m in recent_metrics if not m.success]
            
            return {
                "timestamp": datetime.now().isoformat(),
                "active_runs": active_runs,
                "recent_activity": {
                    "runs_last_5min": len(recent_metrics),
                    "successes_last_5min": len(recent_metrics) - len(recent_failures),
                    "failures_last_5min": len(recent_failures),
                    "avg_duration_last_5min": round(statistics.mean(recent_durations), 2) if recent_durations else 0
                },
                "current_load": {
                    "memory_usage_estimate": self._estimate_memory_usage(),
                    "processing_queue_length": active_runs
                },
                "health_status": self._get_health_status()
            }
    
    # Private helper methods
    
    def _calculate_phase_stats(self, phase_name: str, timings: List[float]) -> Dict[str, Any]:
        """Calculate statistics for a phase"""
        return {
            "count": len(timings),
            "avg_ms": round(statistics.mean(timings), 2),
            "median_ms": round(statistics.median(timings), 2),
            "p95_ms": round(self._percentile(timings, 95), 2),
            "p99_ms": round(self._percentile(timings, 99), 2),
            "min_ms": round(min(timings), 2),
            "max_ms": round(max(timings), 2),
            "std_dev_ms": round(statistics.stdev(timings) if len(timings) > 1 else 0, 2)
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def _update_aggregate_metrics(self):
        """Update aggregate performance metrics"""
        if not self.historical_metrics:
            return
        
        recent_metrics = list(self.historical_metrics)[-100:]
        durations = [m.total_duration_ms for m in recent_metrics if m.success]
        
        if durations:
            self.counters["avg_pipeline_duration_ms"] = statistics.mean(durations)
            self.counters["p95_pipeline_duration_ms"] = self._percentile(durations, 95)
            self.counters["p99_pipeline_duration_ms"] = self._percentile(durations, 99)
    
    def _check_phase_performance(self, phase_name: str, duration_ms: float):
        """Check if phase performance exceeds thresholds"""
        threshold = self.performance_thresholds["max_phase_duration_ms"].get(phase_name, 5000)
        if duration_ms > threshold:
            logger.warning(f"Phase {phase_name} exceeded threshold: {duration_ms:.2f}ms > {threshold}ms")
    
    def _check_pipeline_performance(self, duration_ms: float):
        """Check if pipeline performance exceeds thresholds"""
        threshold = self.performance_thresholds["max_pipeline_duration_ms"]
        if duration_ms > threshold:
            logger.warning(f"Pipeline exceeded threshold: {duration_ms:.2f}ms > {threshold}ms")
    
    def _check_performance_alerts(self) -> List[Dict[str, Any]]:
        """Check for performance alerts"""
        alerts = []
        
        # Check success rate
        if self.historical_metrics:
            recent_metrics = list(self.historical_metrics)[-50:]
            success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
            
            if success_rate < self.performance_thresholds["min_success_rate"]:
                alerts.append({
                    "type": "low_success_rate",
                    "severity": "critical",
                    "message": f"Success rate {success_rate:.2%} below threshold {self.performance_thresholds['min_success_rate']:.2%}",
                    "value": success_rate,
                    "threshold": self.performance_thresholds["min_success_rate"]
                })
        
        # Check average duration
        if self.counters["avg_pipeline_duration_ms"] > self.performance_thresholds["max_pipeline_duration_ms"]:
            alerts.append({
                "type": "high_duration",
                "severity": "warning",
                "message": f"Average duration {self.counters['avg_pipeline_duration_ms']:.0f}ms exceeds threshold",
                "value": self.counters["avg_pipeline_duration_ms"],
                "threshold": self.performance_thresholds["max_pipeline_duration_ms"]
            })
        
        return alerts
    
    def _generate_metrics_summary(self, pipeline: PipelineMetrics) -> Dict[str, Any]:
        """Generate metrics summary for a completed pipeline"""
        return {
            "run_id": pipeline.run_id,
            "success": pipeline.success,
            "total_duration_ms": round(pipeline.total_duration_ms, 2),
            "final_recommendations": pipeline.final_recommendations,
            "phases": [
                {
                    "name": phase.phase_name,
                    "duration_ms": round(phase.duration_ms, 2),
                    "input_count": phase.input_count,
                    "output_count": phase.output_count,
                    "drop_count": phase.drop_count,
                    "success": phase.success
                } for phase in pipeline.phases
            ],
            "feature_flags": pipeline.feature_flags
        }
    
    def _estimate_memory_usage(self) -> str:
        """Estimate current memory usage"""
        # Simplified memory estimation
        active_pipelines = len(self.pipeline_metrics)
        historical_size = len(self.historical_metrics)
        
        estimated_mb = (active_pipelines * 2) + (historical_size * 0.1)
        return f"{estimated_mb:.1f}MB"
    
    def _get_health_status(self) -> str:
        """Get overall system health status"""
        alerts = self._check_performance_alerts()
        
        critical_alerts = [a for a in alerts if a["severity"] == "critical"]
        warning_alerts = [a for a in alerts if a["severity"] == "warning"]
        
        if critical_alerts:
            return "critical"
        elif warning_alerts:
            return "warning"
        else:
            return "healthy"

# Global metrics collector instance
metrics_collector = MetricsCollector()
