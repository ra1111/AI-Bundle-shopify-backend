"""
Feature Normalizers for Enhanced Ranking (PR-5)
Min-max and robust scaling with frozen parameters for consistency
"""
from typing import Dict, List, Any, Tuple, Optional
import logging
import numpy as np
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class NormalizationParams:
    """Frozen normalization parameters for consistency across runs"""
    min_val: float
    max_val: float
    median: float
    iqr: float
    mean: float
    std: float
    method: str  # "min_max", "robust", "z_score"

class FeatureNormalizer:
    """Feature normalization with frozen parameters for ranking consistency"""
    
    def __init__(self):
        self.normalization_cache = {}  # csv_upload_id -> feature -> params
        
    def compute_normalization_params(self, values: List[float], method: str = "robust") -> NormalizationParams:
        """Compute normalization parameters for a feature"""
        if not values:
            return NormalizationParams(0, 1, 0, 1, 0, 1, method)
        
        values_array = np.array(values)
        
        # Remove outliers for robust statistics
        q25, q75 = np.percentile(values_array, [25, 75])
        iqr = q75 - q25
        
        # Filter extreme outliers (beyond 3 IQR)
        lower_bound = q25 - 3 * iqr
        upper_bound = q75 + 3 * iqr
        filtered_values = values_array[(values_array >= lower_bound) & (values_array <= upper_bound)]
        
        if len(filtered_values) == 0:
            filtered_values = values_array
        
        params = NormalizationParams(
            min_val=float(np.min(filtered_values)),
            max_val=float(np.max(filtered_values)),
            median=float(np.median(filtered_values)),
            iqr=float(iqr) if iqr > 0 else 1.0,
            mean=float(np.mean(filtered_values)),
            std=float(np.std(filtered_values)) if np.std(filtered_values) > 0 else 1.0,
            method=method
        )
        
        return params
    
    def normalize_feature(self, value: float, params: NormalizationParams) -> float:
        """Normalize a single feature value using frozen parameters"""
        try:
            if params.method == "min_max":
                # Min-max scaling to [0, 1]
                if params.max_val == params.min_val:
                    return 0.5  # Neutral value when no variance
                return max(0, min(1, (value - params.min_val) / (params.max_val - params.min_val)))
            
            elif params.method == "robust":
                # Robust scaling using median and IQR
                return max(0, min(1, 0.5 + (value - params.median) / (2 * params.iqr)))
            
            elif params.method == "z_score":
                # Z-score normalization, then sigmoid to [0, 1]
                z_score = (value - params.mean) / params.std
                return 1 / (1 + np.exp(-z_score))  # Sigmoid
            
            else:
                return max(0, min(1, value))  # Clamp to [0, 1]
                
        except Exception as e:
            logger.warning(f"Error normalizing feature value {value}: {e}")
            return 0.5  # Safe fallback
    
    def normalize_features_batch(self, candidates: List[Dict[str, Any]], 
                                csv_upload_id: str, 
                                feature_config: Dict[str, str]) -> List[Dict[str, Any]]:
        """Normalize all ranking features for a batch of candidates"""
        try:
            # Extract all feature values for parameter computation
            feature_values = {}
            for feature_name in feature_config.keys():
                feature_values[feature_name] = []
                
                for candidate in candidates:
                    value = candidate.get(feature_name, 0)
                    if isinstance(value, (int, float)) and not np.isnan(value):
                        feature_values[feature_name].append(float(value))
            
            # Compute normalization parameters
            normalization_params = {}
            for feature_name, values in feature_values.items():
                method = feature_config.get(feature_name, "robust")
                normalization_params[feature_name] = self.compute_normalization_params(values, method)
            
            # Store parameters for this run
            self.normalization_cache[csv_upload_id] = normalization_params
            
            # Normalize all candidates
            normalized_candidates = []
            for candidate in candidates:
                normalized_candidate = candidate.copy()
                
                # Add normalized features
                normalized_candidate["normalized_features"] = {}
                for feature_name, params in normalization_params.items():
                    raw_value = candidate.get(feature_name, 0)
                    if isinstance(raw_value, (int, float)):
                        normalized_value = self.normalize_feature(float(raw_value), params)
                        normalized_candidate["normalized_features"][feature_name] = normalized_value
                    else:
                        normalized_candidate["normalized_features"][feature_name] = 0.5
                
                normalized_candidates.append(normalized_candidate)
            
            logger.info(f"Normalized {len(feature_config)} features for {len(candidates)} candidates")
            return normalized_candidates
            
        except Exception as e:
            logger.error(f"Error normalizing features batch: {e}")
            return candidates  # Return original candidates on error
    
    def get_normalization_diagnostics(self, csv_upload_id: str) -> Dict[str, Any]:
        """Get normalization diagnostics for analysis"""
        try:
            if csv_upload_id not in self.normalization_cache:
                return {"error": "No normalization data found"}
            
            params = self.normalization_cache[csv_upload_id]
            diagnostics = {}
            
            for feature_name, param in params.items():
                diagnostics[feature_name] = {
                    "method": param.method,
                    "range": [param.min_val, param.max_val],
                    "median": param.median,
                    "iqr": param.iqr,
                    "mean": param.mean,
                    "std": param.std,
                    "dynamic_range": param.max_val - param.min_val,
                    "coefficient_of_variation": param.std / param.mean if param.mean != 0 else 0
                }
            
            return {
                "normalization_diagnostics": diagnostics,
                "features_normalized": len(diagnostics),
                "csv_upload_id": csv_upload_id
            }
            
        except Exception as e:
            logger.error(f"Error getting normalization diagnostics: {e}")
            return {"error": str(e)}