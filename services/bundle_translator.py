"""
Bundle Translator Layer
Converts quick-start bundle recommendations into canonical bundle_def objects
for the unified bundle engine.

This translator ensures that:
- Quick-start FBT bundles → FBT bundle_def
- Quick-start BOGO bundles → BXGY bundle_def
- Quick-start VOLUME bundles → VOLUME bundle_def

All bundle_defs are compatible with:
- Shopify Function discount logic
- PDP widget rendering
- Cart/Drawer UI
- Unified pricing engine
"""

from typing import Dict, Any, List, Optional
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


def translate_fbt(rec: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate FBT (Frequently Bought Together) bundle to canonical bundle_def.

    Input (quick-start format):
    {
        "bundle_type": "FBT",
        "products": [
            {"sku": "A", "variant_id": "...", "price": 29.99},
            {"sku": "B", "variant_id": "...", "price": 39.99}
        ],
        "pricing": {
            "original_total": 69.98,
            "bundle_price": 62.98,
            "discount_amount": 7.00,
            "discount_pct": "10.0%"
        }
    }

    Output (bundle_def format):
    {
        "bundle_type": "FBT",
        "items": [
            {"variant_id": "...", "quantity": 1},
            {"variant_id": "...", "quantity": 1}
        ],
        "pricing": {
            "discount_type": "PERCENTAGE",
            "discount_value": 10.0
        },
        "rules": {
            "min_items_required": 2,
            "max_items_allowed": 2,
            "allow_substitution": false
        }
    }
    """
    products = rec.get("products", [])
    pricing = rec.get("pricing", {})

    # Extract discount percentage
    discount_pct_str = pricing.get("discount_pct", "10.0%")
    discount_value = float(discount_pct_str.strip('%'))

    # Build items list
    items = []
    for product in products:
        variant_id = product.get("variant_id")
        if not variant_id:
            # Fallback: lookup from catalog by SKU
            sku = product.get("sku")
            if sku and sku in catalog:
                variant_id = catalog[sku].variant_id

        if variant_id:
            items.append({
                "variant_id": str(variant_id),
                "quantity": 1
            })

    return {
        "bundle_type": "FBT",
        "items": items,
        "pricing": {
            "discount_type": "PERCENTAGE",
            "discount_value": discount_value
        },
        "rules": {
            "min_items_required": len(items),
            "max_items_allowed": len(items),
            "allow_substitution": False
        }
    }


def translate_bogo(rec: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate BOGO (Buy X Get Y) bundle to canonical BXGY bundle_def.

    Input (quick-start format):
    {
        "bundle_type": "BOGO",
        "products": [
            {
                "sku": "SLOW-SKU",
                "variant_id": "...",
                "price": 20.00,
                "min_quantity": 2,
                "reward_quantity": 1
            }
        ],
        "pricing": {
            "bogo_config": {
                "buy_qty": 2,
                "get_qty": 1,
                "discount_percent": 33.3
            }
        }
    }

    Output (BXGY bundle_def format):
    {
        "bundle_type": "BXGY",
        "qualifiers": [
            {"variant_id": "...", "min_qty": 2}
        ],
        "rewards": [
            {
                "variant_id": "...",
                "discount_type": "PERCENTAGE",
                "discount_value": 100.0  // 100% off the reward item (free)
            }
        ],
        "pricing": {},
        "rules": {
            "auto_apply": true
        }
    }
    """
    products = rec.get("products", [])
    pricing = rec.get("pricing", {})
    bogo_config = pricing.get("bogo_config", {})

    if not products:
        raise ValueError("BOGO bundle must have at least one product")

    product = products[0]
    variant_id = product.get("variant_id")

    # Fallback: lookup from catalog by SKU
    if not variant_id:
        sku = product.get("sku")
        if sku and sku in catalog:
            variant_id = catalog[sku].variant_id

    if not variant_id:
        raise ValueError(f"Cannot translate BOGO bundle: missing variant_id for SKU {product.get('sku')}")

    buy_qty = bogo_config.get("buy_qty", 2)
    get_qty = bogo_config.get("get_qty", 1)

    # BXGY format: qualifiers (what you must buy) + rewards (what you get discounted)
    return {
        "bundle_type": "BXGY",
        "qualifiers": [
            {
                "variant_id": str(variant_id),
                "min_qty": buy_qty
            }
        ],
        "rewards": [
            {
                "variant_id": str(variant_id),
                "quantity": get_qty,
                "discount_type": "PERCENTAGE",
                "discount_value": 100.0  # 100% off = free
            }
        ],
        "pricing": {},
        "rules": {
            "auto_apply": True
        }
    }


def translate_volume(rec: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate VOLUME (tiered pricing) bundle to canonical bundle_def.

    Input (quick-start format):
    {
        "bundle_type": "VOLUME",
        "products": [
            {
                "sku": "POPULAR",
                "variant_id": "...",
                "price": 25.00
            }
        ],
        "pricing": {
            "volume_tiers": [
                {"min_qty": 1, "discount_type": "NONE", "discount_value": 0},
                {"min_qty": 2, "discount_type": "PERCENTAGE", "discount_value": 5},
                {"min_qty": 3, "discount_type": "PERCENTAGE", "discount_value": 10},
                {"min_qty": 5, "discount_type": "PERCENTAGE", "discount_value": 15}
            ]
        }
    }

    Output (bundle_def format):
    {
        "bundle_type": "VOLUME",
        "items": [
            {"variant_id": "...", "quantity": 1}
        ],
        "pricing": {
            "volume_tiers": [
                {"min_qty": 1, "discount_type": "NONE", "discount_value": 0},
                {"min_qty": 2, "discount_type": "PERCENTAGE", "discount_value": 5},
                {"min_qty": 3, "discount_type": "PERCENTAGE", "discount_value": 10},
                {"min_qty": 5, "discount_type": "PERCENTAGE", "discount_value": 15}
            ]
        },
        "rules": {
            "auto_apply": true,
            "min_qty": 1
        }
    }
    """
    products = rec.get("products", [])
    pricing = rec.get("pricing", {})

    if not products:
        raise ValueError("VOLUME bundle must have at least one product")

    product = products[0]
    variant_id = product.get("variant_id")

    # Fallback: lookup from catalog by SKU
    if not variant_id:
        sku = product.get("sku")
        if sku and sku in catalog:
            variant_id = catalog[sku].variant_id

    if not variant_id:
        raise ValueError(f"Cannot translate VOLUME bundle: missing variant_id for SKU {product.get('sku')}")

    # Extract volume tiers
    volume_tiers = pricing.get("volume_tiers", [])

    return {
        "bundle_type": "VOLUME",
        "items": [
            {
                "variant_id": str(variant_id),
                "quantity": 1
            }
        ],
        "pricing": {
            "volume_tiers": volume_tiers
        },
        "rules": {
            "auto_apply": True,
            "min_qty": 1
        }
    }


def translate_bundle_rec(rec: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main translator router function.

    Converts a bundle recommendation from database format to canonical bundle_def format.

    Args:
        rec: Bundle recommendation dict (from BundleRecommendation model)
        catalog: Dictionary mapping SKU -> CatalogSnapshot

    Returns:
        Canonical bundle_def dict for unified engine

    Raises:
        ValueError: If bundle_type is unknown or translation fails
    """
    bundle_type = rec.get("bundle_type", "").upper()

    if not bundle_type:
        raise ValueError("Bundle recommendation missing bundle_type field")

    logger.debug(f"Translating bundle {rec.get('id', 'unknown')} of type {bundle_type}")

    # Route to appropriate translator
    if bundle_type == "FBT":
        return translate_fbt(rec, catalog)

    elif bundle_type == "BOGO":
        return translate_bogo(rec, catalog)

    elif bundle_type == "VOLUME" or bundle_type == "VOLUME_DISCOUNT":
        return translate_volume(rec, catalog)

    else:
        # For other bundle types (MIX_MATCH, FIXED, BXGY), assume they're already in bundle_def format
        # This handles v2 bundles that don't need translation
        logger.warning(
            f"No translator for bundle_type '{bundle_type}'. "
            f"Assuming bundle is already in canonical format."
        )
        return rec


def generate_ai_copy_for_bundle(
    bundle_type: str,
    products: List[Dict[str, Any]],
    pricing: Dict[str, Any]
) -> Dict[str, str]:
    """
    Generate simple AI copy for quick-start bundles.

    This is a lightweight alternative to GPT-4 generation for quick-start mode.
    Full v2 pipeline will replace these with proper GPT-4 generated copy.

    Args:
        bundle_type: Type of bundle (FBT, BOGO, VOLUME)
        products: List of product dicts with name, price, sku
        pricing: Pricing dict with discount info

    Returns:
        Dict with title, description, value_proposition
    """
    bundle_type = bundle_type.upper()

    if bundle_type == "FBT":
        # Frequently Bought Together
        if len(products) >= 2:
            p1_name = products[0].get("name", "Product 1")
            p2_name = products[1].get("name", "Product 2")
            discount_pct = pricing.get("discount_pct", "10%")

            return {
                "title": f"{p1_name} + {p2_name}",
                "description": f"Customers who bought {p1_name} also bought {p2_name}. Get both together and save {discount_pct}!",
                "value_proposition": f"Save {discount_pct} when you bundle"
            }
        else:
            return {
                "title": "Bundle Deal",
                "description": "Frequently bought together",
                "value_proposition": "Save when you bundle"
            }

    elif bundle_type == "BOGO":
        # Buy X Get Y
        if products:
            product_name = products[0].get("name", "Product")
            bogo_config = pricing.get("bogo_config", {})
            buy_qty = bogo_config.get("buy_qty", 2)
            get_qty = bogo_config.get("get_qty", 1)

            return {
                "title": f"Buy {buy_qty}, Get {get_qty} Free - {product_name}",
                "description": f"Stock up on {product_name}! Buy {buy_qty} and get {get_qty} free.",
                "value_proposition": f"Get {get_qty} free when you buy {buy_qty}"
            }
        else:
            return {
                "title": "Buy More, Get More",
                "description": "Special offer on this product",
                "value_proposition": "Get free items with purchase"
            }

    elif bundle_type == "VOLUME" or bundle_type == "VOLUME_DISCOUNT":
        # Volume discount
        if products:
            product_name = products[0].get("name", "Product")
            volume_tiers = pricing.get("volume_tiers", [])

            # Find best discount
            max_discount = 0
            for tier in volume_tiers:
                if tier.get("discount_type") == "PERCENTAGE":
                    max_discount = max(max_discount, tier.get("discount_value", 0))

            return {
                "title": f"Volume Discount - {product_name}",
                "description": f"Buy more {product_name} and save up to {max_discount}%! The more you buy, the more you save.",
                "value_proposition": f"Save up to {max_discount}% on bulk orders"
            }
        else:
            return {
                "title": "Volume Discount",
                "description": "Buy more, save more",
                "value_proposition": "Bulk pricing available"
            }

    else:
        # Default fallback
        return {
            "title": "Bundle Deal",
            "description": "Special bundle offer",
            "value_proposition": "Save with this bundle"
        }
