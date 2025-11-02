"""
AI Copy Generator Service
Generates marketing copy using OpenAI
"""
from typing import List, Dict, Any, Optional
import logging
import json

from services.ml.llm_utils import (
    get_async_client,
    load_settings,
    run_with_common_errors,
    should_use_llm,
)

logger = logging.getLogger(__name__)


class AICopyGenerator:
    """AI copy generator using OpenAI"""
    
    def __init__(self):
        self._settings = load_settings()
        self.client = get_async_client()
        self.model = self._settings.completion_model
        self.max_tokens = self._settings.completion_max_tokens
        self.temperature = self._settings.completion_temperature
        
    async def generate_bundle_copy(
        self, 
        products: List[Dict[str, Any]], 
        bundle_type: str, 
        context: str = ""
    ) -> Dict[str, str]:
        """Generate marketing copy for a product bundle"""
        if not should_use_llm():
            logger.warning("OpenAI API key not found, using fallback copy")
            return self.generate_fallback_copy(products, bundle_type)

        # Create prompt for OpenAI
        prompt = self.create_bundle_prompt(products, bundle_type, context)

        async def _call():
            return await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert e-commerce copywriter specializing in product bundles. Generate compelling, conversion-optimized marketing copy that highlights value and encourages purchase.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

        response = await run_with_common_errors(
            "bundle copy generation",
            _call,
        )

        if response is None or not response.choices:
            return self.generate_fallback_copy(products, bundle_type)

        content = response.choices[0].message.content
        if not content:
            logger.warning("Empty response from OpenAI; using fallback copy")
            return self.generate_fallback_copy(products, bundle_type)

        # Try to parse as JSON, fallback if needed
        try:
            ai_copy = json.loads(content)
            # Validate required fields
            required_fields = ["title", "description", "valueProposition", "explanation"]
            if all(field in ai_copy for field in required_fields):
                return ai_copy
            else:
                logger.warning("AI response missing required fields")
                return self.parse_text_response(content, bundle_type)

        except json.JSONDecodeError:
            # Parse as structured text
            return self.parse_text_response(content, bundle_type)
    
    def create_bundle_prompt(
        self, 
        products: List[Dict[str, Any]], 
        bundle_type: str, 
        context: str
    ) -> str:
        """Create prompt for OpenAI bundle copy generation"""
        
        # Extract product information
        product_names = [p.get("name", "Product") for p in products]
        product_categories = list(set(p.get("category", "General") for p in products))
        product_brands = list(set(p.get("brand", "Various") for p in products))
        
        # Bundle type descriptions
        bundle_descriptions = {
            "FBT": "frequently bought together items that complement each other perfectly",
            "VOLUME_DISCOUNT": "volume discount bundle offering better value when buying more",
            "MIX_MATCH": "mix and match bundle allowing customers to choose from different categories",
            "BXGY": "buy X get Y promotional bundle with extra value",
            "FIXED": "curated bundle at an attractive fixed price"
        }
        
        bundle_desc = bundle_descriptions.get(bundle_type, "product bundle")
        
        prompt = f"""
Generate compelling marketing copy for a {bundle_desc} containing these products:

Products: {', '.join(product_names[:5])}  # Limit to first 5 for readability
Categories: {', '.join(product_categories[:3])}
Brands: {', '.join(product_brands[:3])}
Bundle Type: {bundle_type}
Context: {context}

Please provide the response as JSON with exactly these fields:
{{
    "title": "Catchy bundle title (max 60 characters)",
    "description": "Detailed description highlighting benefits and value (max 200 characters)", 
    "valueProposition": "Clear value statement explaining why customers should buy this bundle (max 150 characters)",
    "explanation": "Clear explanation of why this bundle is recommended based on data and business objectives (max 180 characters)",
    "features": ["Key feature 1", "Key feature 2", "Key feature 3"],
    "benefits": ["Customer benefit 1", "Customer benefit 2"]
}}

Focus on:
- Value and savings
- Product synergy and complementarity  
- Convenience and bundling benefits
- Clear, actionable language
- Urgency or scarcity if appropriate

Make it compelling and conversion-focused while staying truthful.
"""
        return prompt
    
    def parse_text_response(self, content: str, bundle_type: str) -> Dict[str, str]:
        """Parse non-JSON AI response into structured format"""
        try:
            lines = content.strip().split('\n')
            result = {
                "title": "Great Bundle Deal",
                "description": "Perfect combination of products at an unbeatable price.",
                "valueProposition": "Save money while getting everything you need.",
                "explanation": "Recommended based on customer purchase patterns and business objectives.",
                "features": ["High quality products", "Great value", "Perfect combination"],
                "benefits": ["Save money", "Convenience"]
            }
            
            # Try to extract structured information
            current_field = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Look for field indicators
                if "title:" in line.lower() or "title" in line.lower():
                    title = line.split(":", 1)[-1].strip().strip('"\'')
                    if title:
                        result["title"] = title[:60]
                        
                elif "description:" in line.lower():
                    desc = line.split(":", 1)[-1].strip().strip('"\'')
                    if desc:
                        result["description"] = desc[:200]
                        
                elif "value" in line.lower() and ("proposition" in line.lower() or "statement" in line.lower()):
                    value_prop = line.split(":", 1)[-1].strip().strip('"\'')
                    if value_prop:
                        result["valueProposition"] = value_prop[:150]
            
            return result
            
        except Exception as e:
            logger.warning(f"Error parsing AI response: {e}")
            return self.generate_fallback_copy([], bundle_type)
    
    def generate_fallback_copy(self, products: List[Dict[str, Any]], bundle_type: str) -> Dict[str, str]:
        """Generate fallback copy when AI is unavailable"""
        
        product_count = len(products)
        
        bundle_templates = {
            "FBT": {
                "title": f"Perfect Pair Bundle - {product_count} Items",
                "description": f"These {product_count} products are frequently bought together for good reason. Save when you get them all!",
                "valueProposition": "Get everything you need in one convenient bundle with instant savings.",
                "explanation": "Recommended based on strong customer purchase patterns showing these items are frequently bought together.",
                "features": ["Frequently bought together", "Perfect combination", "Instant savings"],
                "benefits": ["Save time shopping", "Save money", "Get complete solution"]
            },
            "VOLUME_DISCOUNT": {
                "title": f"Volume Saver Bundle - {product_count} Products",
                "description": f"Buy more, save more! Get {product_count} quality products at an unbeatable volume discount.",
                "valueProposition": "The more you buy, the more you save with this volume discount bundle.",
                "explanation": "Recommended to drive higher order values while providing customer savings through volume pricing.",
                "features": ["Volume discount", "Multiple quantities", "Better unit pricing"],
                "benefits": ["Maximum savings", "Stock up and save", "Better value"]
            },
            "MIX_MATCH": {
                "title": f"Mix & Match Bundle - {product_count} Choices",
                "description": f"Choose from {product_count} great products and create your perfect combination with mix & match savings.",
                "valueProposition": "Create your ideal combination and save with flexible mix & match pricing.",
                "explanation": "Recommended based on cross-category purchase patterns to increase basket size and customer satisfaction.",
                "features": ["Flexible choice", "Multiple options", "Mix and match savings"],
                "benefits": ["Customize your order", "Save on combinations", "Perfect variety"]
            },
            "BXGY": {
                "title": f"Buy & Get Bundle - Special Offer",
                "description": f"Buy select items and get additional products at a huge discount in this limited-time bundle.",
                "valueProposition": "Get more for less with this exclusive buy X get Y bundle offer.",
                "explanation": "Recommended to move slow-moving inventory while rewarding customers with high-value primary purchases.",
                "features": ["Buy and get deal", "Extra products", "Limited time offer"],
                "benefits": ["Get more value", "Try new products", "Exclusive savings"]
            },
            "FIXED": {
                "title": f"Curated Bundle - {product_count} Products",
                "description": f"Expertly curated {product_count}-product bundle at one attractive fixed price. No complicated pricing!",
                "valueProposition": "Expertly selected products at one simple, attractive price.",
                "explanation": "Recommended based on complementary product analysis and optimized for high-margin objectives.",
                "features": ["Curated selection", "Fixed price", "No complicated pricing"],
                "benefits": ["Expert selection", "Simple pricing", "Great value"]
            }
        }
        
        template = bundle_templates.get(bundle_type, bundle_templates["FIXED"])
        
        # Customize with product information if available
        if products:
            product_names = [p.get("name", "") for p in products if p.get("name")]
            if product_names:
                first_product = product_names[0]
                template["title"] = template["title"].replace("Bundle", f"{first_product} Bundle")
        
        return template
