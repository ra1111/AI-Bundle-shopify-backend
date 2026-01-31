"""
AI Copy Generator Service
Generates marketing copy using OpenAI
"""
from typing import List, Dict, Any, Optional
import logging
import json
import time
import traceback

from services.ml.llm_utils import (
    get_async_client,
    load_settings,
    run_with_common_errors,
    should_use_llm,
)

logger = logging.getLogger(__name__)


def _generate_request_id() -> str:
    """Generate a short request ID for log correlation."""
    import uuid
    return str(uuid.uuid4())[:8]


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
        context: str = "",
        request_id: str = None
    ) -> Dict[str, str]:
        """Generate marketing copy for a product bundle"""
        req_id = request_id or _generate_request_id()
        start_time = time.time()
        
        logger.info(
            f"[{req_id}] ✨ AI Copy generation STARTED\n"
            f"  Bundle type: {bundle_type}\n"
            f"  Products: {len(products)}\n"
            f"  Model: {self.model}"
        )
        
        try:
            # Step 1: Check if LLM is available
            logger.info(f"[{req_id}] 🔑 Checking OpenAI API key...")
            if not should_use_llm():
                logger.warning(f"[{req_id}] ⚠️ OpenAI API key not found, using fallback copy")
                fallback = self.generate_fallback_copy(products, bundle_type)
                duration = (time.time() - start_time) * 1000
                logger.info(f"[{req_id}] ✅ Fallback copy generated in {duration:.0f}ms")
                return fallback

            logger.info(f"[{req_id}] ✅ OpenAI API key verified")

            # Step 2: Create prompt
            logger.info(f"[{req_id}] 📝 Creating prompt for {len(products)} products...")
            prompt_start = time.time()
            prompt = self.create_bundle_prompt(products, bundle_type, context)
            prompt_duration = (time.time() - prompt_start) * 1000
            logger.info(f"[{req_id}] ✅ Prompt created in {prompt_duration:.0f}ms (length: {len(prompt)} chars)")

            # Step 3: Call OpenAI API
            logger.info(
                f"[{req_id}] 🤖 Calling OpenAI API...\n"
                f"  Model: {self.model}\n"
                f"  Max tokens: {self.max_tokens}\n"
                f"  Temperature: {self.temperature}"
            )
            api_start = time.time()

            async def _call():
                return await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a world-class e-commerce copywriter who has written for major brands like "
                                "Apple, Nike, and Amazon. Your copy drives conversions because you understand "
                                "customer psychology. You write punchy, benefit-focused copy that creates desire. "
                                "You NEVER write generic copy like 'Get both X and Y together' - every line you "
                                "write is crafted to persuade and convert. Return ONLY valid JSON, no markdown."
                            ),
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
            api_duration = (time.time() - api_start) * 1000

            # Step 4: Check response
            if response is None:
                logger.error(f"[{req_id}] ❌ OpenAI API returned None after {api_duration:.0f}ms")
                fallback = self.generate_fallback_copy(products, bundle_type)
                total_duration = (time.time() - start_time) * 1000
                logger.info(f"[{req_id}] ✅ Fallback copy generated after API failure, total: {total_duration:.0f}ms")
                return fallback
                
            if not response.choices:
                logger.error(f"[{req_id}] ❌ OpenAI API returned empty choices after {api_duration:.0f}ms")
                fallback = self.generate_fallback_copy(products, bundle_type)
                total_duration = (time.time() - start_time) * 1000
                logger.info(f"[{req_id}] ✅ Fallback copy generated after empty response, total: {total_duration:.0f}ms")
                return fallback

            logger.info(
                f"[{req_id}] ✅ OpenAI API responded in {api_duration:.0f}ms\n"
                f"  Choices: {len(response.choices)}\n"
                f"  Usage: {response.usage.total_tokens if response.usage else 'N/A'} tokens"
            )

            content = response.choices[0].message.content
            if not content:
                logger.warning(f"[{req_id}] ⚠️ Empty content from OpenAI; using fallback copy")
                fallback = self.generate_fallback_copy(products, bundle_type)
                total_duration = (time.time() - start_time) * 1000
                logger.info(f"[{req_id}] ✅ Fallback copy generated, total: {total_duration:.0f}ms")
                return fallback

            logger.info(f"[{req_id}] 📦 Received content ({len(content)} chars), parsing...")

            # Step 5: Parse response
            parse_start = time.time()
            try:
                ai_copy = json.loads(content)
                # Validate required fields
                required_fields = ["title", "description", "valueProposition", "explanation"]
                if all(field in ai_copy for field in required_fields):
                    parse_duration = (time.time() - parse_start) * 1000
                    total_duration = (time.time() - start_time) * 1000
                    logger.info(
                        f"[{req_id}] ✅ AI Copy generation COMPLETE in {total_duration:.0f}ms\n"
                        f"  Title: {ai_copy.get('title', 'N/A')[:50]}...\n"
                        f"  Parse time: {parse_duration:.0f}ms"
                    )
                    return ai_copy
                else:
                    missing = [f for f in required_fields if f not in ai_copy]
                    logger.warning(f"[{req_id}] ⚠️ AI response missing fields: {missing}")
                    result = self.parse_text_response(content, bundle_type)
                    total_duration = (time.time() - start_time) * 1000
                    logger.info(f"[{req_id}] ✅ Text response parsed, total: {total_duration:.0f}ms")
                    return result

            except json.JSONDecodeError as je:
                logger.warning(f"[{req_id}] ⚠️ JSON parse failed: {je}, trying text parse...")
                result = self.parse_text_response(content, bundle_type)
                total_duration = (time.time() - start_time) * 1000
                logger.info(f"[{req_id}] ✅ Text response parsed after JSON error, total: {total_duration:.0f}ms")
                return result
                
        except Exception as e:
            total_duration = (time.time() - start_time) * 1000
            logger.error(
                f"[{req_id}] ❌ AI Copy generation FAILED after {total_duration:.0f}ms!\n"
                f"  Error type: {type(e).__name__}\n"
                f"  Error message: {str(e)}\n"
                f"  Traceback:\n{traceback.format_exc()}"
            )
            # Return fallback instead of crashing
            logger.info(f"[{req_id}] 🔄 Generating fallback copy after exception...")
            return self.generate_fallback_copy(products, bundle_type)
    
    def create_bundle_prompt(
        self,
        products: List[Dict[str, Any]],
        bundle_type: str,
        context: str
    ) -> str:
        """Create prompt for OpenAI bundle copy generation"""

        # Extract and clean product information
        def clean_product_name(name: str) -> str:
            """Convert slug-style names to human-readable format."""
            if not name or name == "Product":
                return "Product"
            # Replace hyphens/underscores with spaces, title case
            cleaned = name.replace("-", " ").replace("_", " ")
            # Title case but preserve known acronyms
            return " ".join(
                word.upper() if word.upper() in ["USB", "LED", "HD", "4K", "XL", "XXL"]
                else word.capitalize()
                for word in cleaned.split()
            )

        product_names = [clean_product_name(p.get("name") or p.get("title", "Product")) for p in products]
        product_categories = list(set(p.get("category", "") for p in products if p.get("category")))
        product_brands = list(set(p.get("brand", "") for p in products if p.get("brand")))

        # Extract pricing info if available
        prices = [float(p.get("price", 0)) for p in products if p.get("price")]
        total_price = sum(prices) if prices else None

        # Extract descriptions and tags for richer context
        product_descriptions = [p.get("description", "") for p in products if p.get("description")]
        product_tags = []
        for p in products:
            tags = p.get("tags", "")
            if tags:
                # Tags might be comma-separated string or list
                if isinstance(tags, str):
                    product_tags.extend([t.strip() for t in tags.split(",") if t.strip()])
                elif isinstance(tags, list):
                    product_tags.extend(tags)
        product_tags = list(set(product_tags))[:10]  # Dedupe and limit

        # Bundle type descriptions with copywriting angles
        bundle_angles = {
            "FBT": {
                "desc": "frequently bought together bundle",
                "angle": "These products are often purchased together by customers who know what they need. Create copy that emphasizes the natural pairing and convenience.",
            },
            "VOLUME_DISCOUNT": {
                "desc": "volume discount bundle",
                "angle": "Customers get better value buying in bulk. Emphasize per-unit savings and the smart shopping decision.",
            },
            "BXGY": {
                "desc": "buy one get one promotional bundle",
                "angle": "This is a promotional deal - create excitement and urgency. Focus on the 'free' or heavily discounted item.",
            },
            "MIX_MATCH": {
                "desc": "mix and match bundle",
                "angle": "Customers can customize their selection. Highlight variety, choice, and personalization.",
            },
            "FIXED": {
                "desc": "curated bundle",
                "angle": "This is an expertly curated selection. Position it as a complete solution hand-picked for the customer.",
            },
        }

        bundle_info = bundle_angles.get(bundle_type, {"desc": "product bundle", "angle": "Create compelling copy."})

        # Build rich product details string
        product_details_lines = []
        for i, p in enumerate(products[:5]):
            name = clean_product_name(p.get("name") or p.get("title", "Product"))
            price = float(p.get("price", 0)) if p.get("price") else 0
            desc = p.get("description", "")

            line = f"  - {name}"
            if price > 0:
                line += f" (${price:.2f})"
            if desc:
                # Truncate long descriptions
                short_desc = desc[:100] + "..." if len(desc) > 100 else desc
                line += f"\n    Description: {short_desc}"
            product_details_lines.append(line)

        product_details = "\n".join(product_details_lines)

        # Build optional context sections
        tags_section = f"**PRODUCT TAGS:** {', '.join(product_tags)}" if product_tags else ""
        descriptions_section = ""
        if product_descriptions:
            desc_text = " | ".join(d[:80] for d in product_descriptions[:2])
            descriptions_section = f"**PRODUCT CONTEXT:** {desc_text}"

        prompt = f"""You are a top-tier e-commerce copywriter. Write compelling marketing copy for this {bundle_info['desc']}.

**PRODUCTS IN BUNDLE:**
{product_details}

**CATEGORIES:** {', '.join(product_categories[:3]) if product_categories else 'Mixed'}
**BRANDS:** {', '.join(product_brands[:3]) if product_brands else 'Various'}
{f"**APPROXIMATE VALUE:** ${total_price:.2f}" if total_price else ""}
{tags_section}
{descriptions_section}

**COPYWRITING ANGLE:** {bundle_info['angle']}
**BUSINESS CONTEXT:** {context}

**REQUIREMENTS:**
Write persuasive, conversion-focused copy that makes customers WANT to buy this bundle. Avoid generic phrases like "Get both X and Y together" - be creative!

Return JSON with these exact fields:
{{
    "badge": "Short widget label, 2-4 words ONLY, ALL CAPS (e.g. BUNDLE & SAVE, BOUGHT TOGETHER, COMBO DEAL). This is NOT the title - it's a tiny category tag.",
    "title": "Concise bundle name (max 40 chars, under 6 words). Should read like a product heading, NOT ad copy. Good: 'Complete Snowboard Kit', 'Winter Essentials Pack'. Bad: 'Unleash Your Winter Adventure Duo', 'Elevate Your Ride'.",
    "description": "One concrete sentence about what's in the bundle and why it's a good deal (max 120 chars). Be specific about the products - no vague marketing fluff like 'elevate your experience'. Good: 'Two top-rated boards at 6% off. Perfect for sharing the slopes.' Bad: 'Designed for thrill-seekers who crave the ultimate ride.' NEVER use em dashes.",
    "valueProposition": "Clear, specific reason to buy NOW (max 120 chars)",
    "explanation": "Data-driven reasoning for why this bundle makes sense (max 150 chars)",
    "features": ["Specific feature 1", "Specific feature 2", "Specific feature 3"],
    "benefits": ["Tangible benefit 1", "Tangible benefit 2"]
}}

**COPYWRITING TIPS:**
- Title should sound like a product label, NOT an ad headline. Keep it short and functional.
- Description should mention the actual products and the concrete savings, not generic hype.
- Be specific about value (not just "save money" - say how much)
- NEVER use em dashes (—) anywhere in the output. Use periods or commas instead.
- Sound human, not robotic
- AVOID vague marketing phrases like "unleash", "elevate", "ultimate", "thrill-seekers", "adventure duo"
- NEVER use the exact product slugs/handles in the title"""
        return prompt
    
    def parse_text_response(self, content: str, bundle_type: str) -> Dict[str, str]:
        """Parse non-JSON AI response into structured format"""
        try:
            lines = content.strip().split('\n')
            result = {
                "badge": "BUNDLE & SAVE",
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

        product_count = len(products) if products else 2

        # Extract first product name for personalization (cleaned)
        first_product = "Premium"
        if products and len(products) > 0:
            raw_name = products[0].get("name") or products[0].get("title", "")
            if raw_name and raw_name != "Product":
                # Clean slug-style names
                cleaned = raw_name.replace("-", " ").replace("_", " ")
                words = cleaned.split()[:3]  # Take first 3 words max
                first_product = " ".join(w.capitalize() for w in words)

        bundle_templates = {
            "FBT": {
                "badge": "BOUGHT TOGETHER",
                "title": f"Complete {first_product} Kit",
                "description": f"Everything you need in one smart bundle. Customers who buy these together love the results.",
                "valueProposition": "One click for the complete solution - no guesswork needed.",
                "explanation": "Popular pairing based on real customer behavior - these items work better together.",
                "features": ["Proven combination", "Complete solution", "Bundle savings"],
                "benefits": ["Skip the research", "Guaranteed compatibility", "Instant value"]
            },
            "VOLUME_DISCOUNT": {
                "badge": "BUY MORE SAVE MORE",
                "title": f"Smart Stock-Up: {first_product}",
                "description": f"Lock in savings now with this volume bundle. Perfect for regular users who know quality when they see it.",
                "valueProposition": "Smart shoppers stock up and save - simple math, better value.",
                "explanation": "Volume pricing optimized for value-conscious customers seeking the best per-unit cost.",
                "features": ["Bulk pricing unlocked", "Per-unit savings", "Premium quality"],
                "benefits": ["Lower cost per item", "Never run out", "Long-term value"]
            },
            "MIX_MATCH": {
                "badge": "MIX & MATCH",
                "title": f"Your Choice: {first_product} Collection",
                "description": f"Pick your favorites and watch the savings stack up. Your bundle, your way.",
                "valueProposition": "Mix freely, save automatically - the more you add, the more you keep.",
                "explanation": "Flexible bundle designed to reward customers who explore the collection.",
                "features": ["Full flexibility", "Stackable savings", "No compromises"],
                "benefits": ["Total control", "Discover new favorites", "Save on variety"]
            },
            "BXGY": {
                "badge": "BONUS DEAL",
                "title": f"Today's Deal: {first_product} Bonus",
                "description": f"This is the deal you've been waiting for. Buy your favorites and unlock bonus items at an incredible price.",
                "valueProposition": "Your purchase unlocks exclusive bonus value - don't miss this limited offer.",
                "explanation": "Strategic promotion designed to reward loyal customers while introducing them to new products.",
                "features": ["Bonus unlocked", "Limited availability", "Premium selection"],
                "benefits": ["More for your money", "Try before you commit", "VIP treatment"]
            },
            "FIXED": {
                "badge": "BUNDLE & SAVE",
                "title": f"The Essential {first_product} Bundle",
                "description": f"Hand-picked essentials at one unbeatable price. No math required - just pure value.",
                "valueProposition": "One price, zero hassle - everything you need in a single click.",
                "explanation": "Curated by experts based on what works best together - optimized for maximum satisfaction.",
                "features": ["Expert curation", "One simple price", "Complete package"],
                "benefits": ["Decision made easy", "Guaranteed fit", "Premium without premium pricing"]
            }
        }

        template = bundle_templates.get(bundle_type, bundle_templates["FIXED"])

        # Return a copy to avoid mutation issues
        return {**template}
