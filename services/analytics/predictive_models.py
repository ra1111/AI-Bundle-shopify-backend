"""
Predictive Models for Advanced Analytics (PR-5)
Implements machine learning models for forecasting and trend prediction
"""
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Represents a model prediction with confidence intervals"""
    predicted_value: float
    confidence_lower: float
    confidence_upper: float
    prediction_date: datetime
    model_accuracy: float

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    trend_direction: str  # "upward", "downward", "stable", "volatile"
    trend_strength: float  # 0-1 scale
    seasonal_pattern: bool
    anomalies_detected: List[Dict[str, Any]]
    key_factors: List[str]

class PredictiveModelsEngine:
    """Advanced predictive models for business forecasting"""
    
    def __init__(self):
        # Model configuration
        self.models = {
            "linear_regression": {"enabled": True, "weight": 0.3},
            "moving_average": {"enabled": True, "weight": 0.2},
            "exponential_smoothing": {"enabled": True, "weight": 0.3},
            "seasonal_decomposition": {"enabled": True, "weight": 0.2}
        }
        
        # Forecasting parameters
        self.forecast_horizon = 30  # days
        self.confidence_level = 0.95
        self.min_data_points = 7
        
        # Seasonality detection
        self.seasonal_periods = [7, 14, 30, 90]  # Weekly, bi-weekly, monthly, quarterly
        
    async def predict_bundle_performance(self, historical_data: List[Dict[str, Any]], 
                                       bundle_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Predict bundle performance based on historical data and characteristics"""
        try:
            if len(historical_data) < self.min_data_points:
                return {"error": "Insufficient historical data for prediction"}
            
            # Extract features
            features = self._extract_bundle_features(historical_data, bundle_characteristics)
            
            # Generate predictions using ensemble approach
            predictions = {
                "conversion_rate": await self._predict_conversion_rate(features),
                "revenue_impact": await self._predict_revenue_impact(features),
                "adoption_rate": await self._predict_adoption_rate(features),
                "customer_satisfaction": await self._predict_customer_satisfaction(features),
                "competitive_resilience": await self._assess_competitive_resilience(features)
            }
            
            # Calculate overall performance score
            performance_score = self._calculate_performance_score(predictions)
            
            # Generate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(predictions)
            
            return {
                "predictions": predictions,
                "performance_score": performance_score,
                "confidence_intervals": confidence_intervals,
                "model_accuracy": self._estimate_model_accuracy(historical_data),
                "key_factors": self._identify_key_factors(features),
                "risk_assessment": self._assess_prediction_risks(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error predicting bundle performance: {e}")
            return {"error": str(e)}
    
    async def forecast_market_trends(self, market_data: List[Dict[str, Any]], 
                                   forecast_period: int = 30) -> Dict[str, Any]:
        """Forecast market trends and opportunities"""
        try:
            # Time series analysis
            time_series = self._prepare_time_series(market_data)
            
            # Trend decomposition
            trend_analysis = self._decompose_trends(time_series)
            
            # Generate forecasts
            forecasts = {}
            for metric, series in time_series.items():
                if len(series) >= self.min_data_points:
                    forecast = await self._generate_forecast(series, forecast_period, metric)
                    forecasts[metric] = forecast
            
            # Market opportunity analysis
            opportunities = self._identify_market_opportunities(forecasts, trend_analysis)
            
            # Risk factors
            risk_factors = self._assess_market_risks(forecasts, trend_analysis)
            
            return {
                "forecasts": forecasts,
                "trend_analysis": trend_analysis,
                "market_opportunities": opportunities,
                "risk_factors": risk_factors,
                "forecast_accuracy": self._calculate_forecast_accuracy(time_series),
                "recommended_strategies": self._recommend_market_strategies(forecasts, opportunities)
            }
            
        except Exception as e:
            logger.error(f"Error forecasting market trends: {e}")
            return {"error": str(e)}
    
    async def predict_customer_behavior(self, customer_data: List[Dict[str, Any]], 
                                      customer_segments: Dict[str, Any]) -> Dict[str, Any]:
        """Predict customer behavior patterns and lifetime value"""
        try:
            # Customer segmentation analysis
            segment_predictions = {}
            
            for segment_name, segment_data in customer_segments.items():
                segment_predictions[segment_name] = {
                    "retention_probability": self._predict_retention(segment_data),
                    "lifetime_value": self._predict_lifetime_value(segment_data),
                    "purchase_frequency": self._predict_purchase_frequency(segment_data),
                    "bundle_affinity": self._predict_bundle_affinity(segment_data),
                    "price_sensitivity": self._analyze_price_sensitivity(segment_data)
                }
            
            # Churn prediction
            churn_analysis = self._predict_customer_churn(customer_data)
            
            # Cross-sell opportunities
            cross_sell_predictions = self._predict_cross_sell_opportunities(customer_data)
            
            # Seasonal behavior patterns
            seasonal_patterns = self._analyze_seasonal_behavior(customer_data)
            
            return {
                "segment_predictions": segment_predictions,
                "churn_analysis": churn_analysis,
                "cross_sell_opportunities": cross_sell_predictions,
                "seasonal_patterns": seasonal_patterns,
                "personalization_insights": self._generate_personalization_insights(segment_predictions),
                "retention_strategies": self._recommend_retention_strategies(churn_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error predicting customer behavior: {e}")
            return {"error": str(e)}
    
    async def analyze_competitive_landscape(self, market_data: Dict[str, Any], 
                                          competitor_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze competitive landscape and predict market position"""
        try:
            # Market share analysis
            market_position = self._analyze_market_position(market_data)
            
            # Competitive threats assessment
            competitive_threats = self._assess_competitive_threats(market_data, competitor_data)
            
            # Market opportunity scoring
            opportunity_scores = self._score_market_opportunities(market_data)
            
            # Competitive advantages
            competitive_advantages = self._identify_competitive_advantages(market_data)
            
            # Strategic recommendations
            strategic_recommendations = self._generate_strategic_recommendations(
                market_position, competitive_threats, opportunity_scores
            )
            
            return {
                "market_position": market_position,
                "competitive_threats": competitive_threats,
                "opportunity_scores": opportunity_scores,
                "competitive_advantages": competitive_advantages,
                "strategic_recommendations": strategic_recommendations,
                "market_outlook": self._generate_market_outlook(market_data)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing competitive landscape: {e}")
            return {"error": str(e)}
    
    # Helper methods for feature extraction and prediction
    
    def _extract_bundle_features(self, historical_data: List[Dict[str, Any]], 
                                characteristics: Dict[str, Any]) -> Dict[str, float]:
        """Extract relevant features for bundle prediction"""
        features = {}
        
        # Historical performance features
        if historical_data:
            conversions = [d.get("conversion_rate", 0) for d in historical_data]
            revenues = [d.get("revenue", 0) for d in historical_data]
            
            features.update({
                "avg_conversion_rate": np.mean(conversions) if conversions else 0,
                "conversion_trend": self._calculate_simple_trend(conversions),
                "revenue_volatility": np.std(revenues) if revenues else 0,
                "performance_consistency": 1 - (np.std(conversions) / np.mean(conversions)) if conversions and np.mean(conversions) > 0 else 0
            })
        
        # Bundle characteristics features
        if characteristics:
            features.update({
                "bundle_size": len(characteristics.get("products", [])),
                "price_point": characteristics.get("total_price", 0),
                "discount_percentage": characteristics.get("discount_pct", 0),
                "category_diversity": len(set(characteristics.get("categories", []))),
                "brand_diversity": len(set(characteristics.get("brands", [])))
            })
        
        return features
    
    async def _predict_conversion_rate(self, features: Dict[str, float]) -> float:
        """Predict bundle conversion rate"""
        # Simplified model - in production, would use trained ML model
        base_rate = 0.12  # Industry baseline
        
        # Adjust based on features
        adjustments = 0
        if features.get("avg_conversion_rate", 0) > 0:
            adjustments += (features["avg_conversion_rate"] - base_rate) * 0.5
        
        if features.get("bundle_size", 0) > 0:
            # Optimal bundle size is 2-3 items
            optimal_size = 2.5
            size_factor = 1 - abs(features["bundle_size"] - optimal_size) * 0.05
            adjustments += (size_factor - 1) * base_rate
        
        if features.get("discount_percentage", 0) > 0:
            # Moderate discounts improve conversion
            discount_factor = min(1.5, 1 + features["discount_percentage"] * 0.01)
            adjustments += (discount_factor - 1) * base_rate * 0.3
        
        predicted_rate = max(0.01, min(0.5, base_rate + adjustments))
        return round(predicted_rate, 4)
    
    async def _predict_revenue_impact(self, features: Dict[str, float]) -> float:
        """Predict revenue impact of bundle"""
        base_impact = features.get("price_point", 0)
        
        # Adjust for conversion probability
        conversion_rate = await self._predict_conversion_rate(features)
        expected_revenue = base_impact * conversion_rate
        
        # Account for volume effects
        if features.get("bundle_size", 0) > 1:
            volume_multiplier = 1 + (features["bundle_size"] - 1) * 0.15
            expected_revenue *= volume_multiplier
        
        return round(expected_revenue, 2)
    
    async def _predict_adoption_rate(self, features: Dict[str, float]) -> float:
        """Predict customer adoption rate for bundle"""
        base_adoption = 0.25  # 25% baseline adoption
        
        # Adjust based on historical performance
        if features.get("performance_consistency", 0) > 0.7:
            base_adoption *= 1.2
        
        # Adjust for price sensitivity
        if features.get("discount_percentage", 0) > 15:
            base_adoption *= 1.15
        
        # Adjust for complexity
        if features.get("bundle_size", 0) > 4:
            base_adoption *= 0.9  # Complex bundles have lower adoption
        
        return round(min(0.8, base_adoption), 3)
    
    async def _predict_customer_satisfaction(self, features: Dict[str, float]) -> float:
        """Predict customer satisfaction score"""
        base_satisfaction = 7.5  # Out of 10
        
        # Higher category diversity often means better satisfaction
        if features.get("category_diversity", 0) > 1:
            base_satisfaction += 0.5
        
        # Reasonable pricing improves satisfaction
        if features.get("discount_percentage", 0) > 5:
            base_satisfaction += 0.3
        
        # Consistency in performance
        if features.get("performance_consistency", 0) > 0.8:
            base_satisfaction += 0.4
        
        return round(min(10.0, base_satisfaction), 1)
    
    async def _assess_competitive_resilience(self, features: Dict[str, float]) -> float:
        """Assess how resilient the bundle is to competitive pressure"""
        base_resilience = 0.6  # 60% baseline
        
        # Unique combinations are more resilient
        if features.get("category_diversity", 0) > 2:
            base_resilience += 0.2
        
        # Strong historical performance indicates resilience
        if features.get("avg_conversion_rate", 0) > 0.15:
            base_resilience += 0.15
        
        # Price competitiveness
        if features.get("discount_percentage", 0) > 10:
            base_resilience += 0.1
        
        return round(min(1.0, base_resilience), 3)
    
    async def _generate_forecast(self, series: List[float], periods: int, metric_name: str) -> Dict[str, Any]:
        """Generate forecast for a time series"""
        if len(series) < 3:
            return {"error": "Insufficient data"}
        
        # Simple ensemble forecast
        forecasts = []
        
        # Linear trend
        x = np.arange(len(series))
        coeffs = np.polyfit(x, series, 1)
        future_x = np.arange(len(series), len(series) + periods)
        linear_forecast = np.polyval(coeffs, future_x)
        forecasts.append(linear_forecast)
        
        # Moving average
        window = min(7, len(series) // 2)
        ma_base = np.mean(series[-window:])
        ma_forecast = [ma_base] * periods
        forecasts.append(ma_forecast)
        
        # Exponential smoothing
        alpha = 0.3
        es_forecast = []
        last_value = series[-1]
        for _ in range(periods):
            es_forecast.append(last_value)
            last_value = alpha * last_value + (1 - alpha) * last_value
        forecasts.append(es_forecast)
        
        # Ensemble average
        ensemble_forecast = np.mean(forecasts, axis=0)
        
        # Calculate confidence intervals
        forecast_std = np.std(forecasts, axis=0)
        confidence_lower = ensemble_forecast - 1.96 * forecast_std
        confidence_upper = ensemble_forecast + 1.96 * forecast_std
        
        return {
            "metric": metric_name,
            "forecast_values": [round(val, 2) for val in ensemble_forecast],
            "confidence_lower": [round(val, 2) for val in confidence_lower],
            "confidence_upper": [round(val, 2) for val in confidence_upper],
            "forecast_accuracy": self._estimate_forecast_accuracy(series),
            "trend_direction": "increasing" if coeffs[0] > 0 else "decreasing"
        }
    
    def _calculate_simple_trend(self, values: List[float]) -> float:
        """Calculate simple trend coefficient"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope / np.mean(values) if np.mean(values) != 0 else 0.0
    
    def _estimate_forecast_accuracy(self, series: List[float]) -> float:
        """Estimate forecast accuracy based on historical validation"""
        if len(series) < 6:
            return 0.7  # Default moderate accuracy
        
        # Simple cross-validation approach
        split_point = len(series) // 2
        train_data = series[:split_point]
        test_data = series[split_point:]
        
        # Generate forecast for test period
        x = np.arange(len(train_data))
        coeffs = np.polyfit(x, train_data, 1)
        test_x = np.arange(len(train_data), len(series))
        predicted = np.polyval(coeffs, test_x)
        
        # Calculate accuracy
        mape = np.mean(np.abs((test_data - predicted) / test_data)) if all(val != 0 for val in test_data) else 0.3
        accuracy = max(0.1, 1 - mape)
        
        return round(min(0.95, accuracy), 3)