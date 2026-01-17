"""
Standardized Bundle Schemas
===========================

This module defines the canonical data structures for all bundle types.
All bundle generation code MUST output data conforming to these schemas.

CRITICAL FIELDS (Required for Frontend):
----------------------------------------
1. products[].product_gid  - Required for Shopify API enrichment
2. products[].name         - Required for display
3. products[].price        - Required for pricing calculations
4. confidence              - Required for sorting/priority (0.0-1.0)
5. ranking_score           - Required for bundle ordering
6. ai_copy.title           - Required for display

BUNDLE TYPES:
-------------
- FBT (Frequently Bought Together): Cross-sell bundles shown on product pages
- VOLUME: "Buy more, save more" - quantity-based tiered discounts
- BOGO (Buy X Get Y): Promotional bundles like "Buy 2 Get 1 Free"

SHOP SIZE TIERS:
----------------
- Very Small: <50 orders (uses FallbackLadder Tier 7-5)
- Small: 50-200 orders (uses FallbackLadder Tier 5-3)
- Medium: 200-500 orders (uses standard ML pipeline)
- Large: 500+ orders (uses strict ML pipeline)

All tiers MUST produce bundles conforming to these schemas.
"""

from typing import List, Dict, Any, Optional, Union, TypedDict, Literal
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from datetime import datetime
import logging
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE DEFINITIONS (TypedDict for runtime validation)
# =============================================================================

class ProductDataDict(TypedDict, total=False):
    """Product data structure - REQUIRED fields for frontend.

    NOTE: variant_id is the PRIMARY identifier used for all internal lookups.
    SKU is optional and only used for display purposes.
    """
    # === REQUIRED ===
    product_gid: str      # "gid://shopify/Product/123" - CRITICAL for Shopify API
    name: str             # Product display name
    price: float          # Product price (number, not string)
    variant_id: str       # Shopify variant ID - PRIMARY KEY for all lookups

    # === RECOMMENDED ===
    variant_gid: str      # "gid://shopify/ProductVariant/456"
    sku: str              # Product SKU - DISPLAY ONLY (may be empty)
    title: str            # Same as name (for backward compatibility)
    image_url: Optional[str]  # Product image URL
    product_id: str       # Shopify product ID (numeric string)


class VolumeTierDict(TypedDict, total=False):
    """Volume tier configuration."""
    min_qty: int           # Minimum quantity for this tier (e.g., 2, 3, 5)
    discount_type: str     # "PERCENTAGE" | "FIXED" | "NONE"
    discount_value: float  # Discount amount (percentage or fixed)
    label: Optional[str]   # Display label (e.g., "Best Value")
    type: str              # "percentage" | "fixed" (lowercase)
    value: float           # Same as discount_value (for compatibility)


class BOGOConfigDict(TypedDict, total=False):
    """BOGO configuration."""
    buy_qty: int           # How many to buy (e.g., 2)
    get_qty: int           # How many free/discounted (e.g., 1)
    discount_type: str     # "free" | "percent" | "fixed"
    discount_percent: float  # 100 = FREE, 50 = half off
    same_product: bool     # Is it same product BOGO?
    mode: str              # "free_same_variant" | "percent_off" | "fixed_off"


class FBTPricingDict(TypedDict, total=False):
    """FBT bundle pricing structure."""
    original_total: float     # Sum of individual prices
    bundle_price: float       # Final price after discount
    discount_amount: float    # Savings amount (original - bundle)
    discount_pct: str         # "10%" formatted string
    discount_percentage: float  # 10.0 numeric value
    discount_type: str        # "percentage" | "fixed_amount"
    product_details: List[Dict[str, Any]]  # Per-product breakdown


class VolumePricingDict(TypedDict, total=False):
    """Volume bundle pricing structure."""
    original_total: float     # Single unit price
    bundle_price: float       # Same as original (tiers define discounts)
    discount_amount: float    # 0 (tiers define discounts)
    discount_type: str        # "tiered"
    volume_tiers: List[VolumeTierDict]


class BOGOPricingDict(TypedDict, total=False):
    """BOGO bundle pricing structure."""
    buy_total: float          # Total for "buy" items
    get_value: float          # Value of free/discounted items
    final_price: float        # What customer pays
    savings: float            # Amount saved
    discount_type: str        # "bogo"
    bogo_config: BOGOConfigDict


class AICopyDict(TypedDict, total=False):
    """AI-generated marketing copy."""
    title: str                # Bundle display title
    description: str          # Bundle description
    tagline: Optional[str]    # Short tagline
    cta_text: Optional[str]   # Call-to-action text
    savings_message: Optional[str]  # "Save $X!"
    show_on: List[str]        # ["product", "cart"]
    is_active: bool           # Bundle active status
    features: Dict[str, Any]  # ML features for explainability


class FBTProductsDict(TypedDict, total=False):
    """FBT products structure - supports both flat and enhanced formats."""
    items: List[ProductDataDict]           # Flat array of products
    trigger_product: ProductDataDict       # Main product (first in pair)
    addon_products: List[ProductDataDict]  # Recommended addons


class VolumeProductsDict(TypedDict, total=False):
    """Volume products structure."""
    items: List[ProductDataDict]           # Single product (for volume)
    volume_tiers: List[VolumeTierDict]     # Tiers also stored here


class BOGOProductsDict(TypedDict, total=False):
    """BOGO products structure."""
    items: List[ProductDataDict]
    qualifiers: List[Dict[str, Any]]  # "Buy" products with quantity
    rewards: List[Dict[str, Any]]     # "Get" products with discount


class FBTBundleDict(TypedDict, total=False):
    """Complete FBT bundle structure."""
    id: str
    csv_upload_id: str
    shop_id: Optional[str]
    bundle_type: Literal["FBT"]
    objective: str
    products: Union[FBTProductsDict, List[ProductDataDict]]
    pricing: FBTPricingDict
    ai_copy: AICopyDict
    confidence: float
    predicted_lift: float
    ranking_score: float
    support: Optional[float]
    lift: Optional[float]
    is_approved: bool
    is_used: bool
    rank_position: Optional[int]
    discount_reference: Optional[str]
    created_at: datetime


class VolumeBundleDict(TypedDict, total=False):
    """Complete VOLUME bundle structure."""
    id: str
    csv_upload_id: str
    shop_id: Optional[str]
    bundle_type: Literal["VOLUME"]
    objective: str
    products: VolumeProductsDict
    pricing: VolumePricingDict
    ai_copy: AICopyDict
    volume_tiers: List[VolumeTierDict]  # Top-level for frontend
    confidence: float
    predicted_lift: float
    ranking_score: float
    is_approved: bool
    is_used: bool
    rank_position: Optional[int]
    discount_reference: Optional[str]
    created_at: datetime


class BOGOBundleDict(TypedDict, total=False):
    """Complete BOGO bundle structure."""
    id: str
    csv_upload_id: str
    shop_id: Optional[str]
    bundle_type: Literal["BOGO"]
    objective: str
    products: BOGOProductsDict
    pricing: BOGOPricingDict
    ai_copy: AICopyDict
    bogo_config: BOGOConfigDict  # Top-level for frontend
    qualifiers: List[Dict[str, Any]]  # Top-level for frontend
    rewards: List[Dict[str, Any]]  # Top-level for frontend
    confidence: float
    predicted_lift: float
    ranking_score: float
    is_approved: bool
    is_used: bool
    rank_position: Optional[int]
    discount_reference: Optional[str]
    created_at: datetime


# =============================================================================
# DATACLASS DEFINITIONS (for type safety in code)
# =============================================================================

@dataclass
class ProductData:
    """Product data - REQUIRED fields for frontend.

    NOTE: variant_id is the PRIMARY identifier used for all internal lookups.
    SKU is optional and only used for display purposes.
    """
    product_gid: str  # "gid://shopify/Product/123"
    name: str
    price: float
    variant_id: str  # PRIMARY KEY - required for all lookups
    variant_gid: str = ""
    sku: str = ""  # Display only - may be empty
    title: str = ""
    image_url: Optional[str] = None
    product_id: str = ""

    def __post_init__(self):
        if not self.title:
            self.title = self.name
        if not self.variant_gid and self.variant_id:
            self.variant_gid = f"gid://shopify/ProductVariant/{self.variant_id}"

    def to_dict(self) -> ProductDataDict:
        return {
            "product_gid": self.product_gid,
            "name": self.name,
            "title": self.title or self.name,
            "price": self.price,
            "variant_id": self.variant_id,
            "variant_gid": self.variant_gid,
            "sku": self.sku,
            "image_url": self.image_url,
            "product_id": self.product_id,
        }


@dataclass
class VolumeTier:
    """Volume tier configuration."""
    min_qty: int
    discount_type: str  # "PERCENTAGE" | "FIXED" | "NONE"
    discount_value: float
    label: Optional[str] = None

    def to_dict(self) -> VolumeTierDict:
        return {
            "min_qty": self.min_qty,
            "discount_type": self.discount_type,
            "discount_value": self.discount_value,
            "label": self.label,
            "type": self.discount_type.lower() if self.discount_type != "NONE" else "percentage",
            "value": self.discount_value,
        }


@dataclass
class BOGOConfig:
    """BOGO configuration."""
    buy_qty: int
    get_qty: int
    discount_type: str  # "free" | "percent" | "fixed"
    discount_percent: float = 100.0
    same_product: bool = True
    mode: str = "free_same_variant"

    def to_dict(self) -> BOGOConfigDict:
        return {
            "buy_qty": self.buy_qty,
            "get_qty": self.get_qty,
            "discount_type": self.discount_type,
            "discount_percent": self.discount_percent,
            "same_product": self.same_product,
            "mode": self.mode,
        }


@dataclass
class FBTPricing:
    """FBT pricing structure."""
    original_total: float
    bundle_price: float
    discount_amount: float
    discount_percentage: float = 0.0
    discount_type: str = "percentage"

    def __post_init__(self):
        if self.discount_amount == 0 and self.original_total > 0:
            self.discount_amount = self.original_total - self.bundle_price
        if self.discount_percentage == 0 and self.original_total > 0:
            self.discount_percentage = (self.discount_amount / self.original_total) * 100

    def to_dict(self) -> FBTPricingDict:
        return {
            "original_total": self.original_total,
            "bundle_price": self.bundle_price,
            "discount_amount": self.discount_amount,
            "discount_pct": f"{self.discount_percentage:.0f}%",
            "discount_percentage": self.discount_percentage,
            "discount_type": self.discount_type,
        }


@dataclass
class VolumePricing:
    """Volume pricing structure."""
    original_total: float  # Single unit price
    volume_tiers: List[VolumeTier] = field(default_factory=list)

    def to_dict(self) -> VolumePricingDict:
        return {
            "original_total": self.original_total,
            "bundle_price": self.original_total,
            "discount_amount": 0,
            "discount_type": "tiered",
            "volume_tiers": [t.to_dict() for t in self.volume_tiers],
        }


@dataclass
class BOGOPricing:
    """BOGO pricing structure."""
    buy_total: float
    get_value: float
    final_price: float
    bogo_config: BOGOConfig

    @property
    def savings(self) -> float:
        return self.get_value * (self.bogo_config.discount_percent / 100)

    def to_dict(self) -> BOGOPricingDict:
        return {
            "buy_total": self.buy_total,
            "get_value": self.get_value,
            "final_price": self.final_price,
            "savings": self.savings,
            "discount_type": "bogo",
            "bogo_config": self.bogo_config.to_dict(),
        }


@dataclass
class AICopy:
    """AI-generated marketing copy."""
    title: str
    description: str
    tagline: Optional[str] = None
    cta_text: Optional[str] = None
    savings_message: Optional[str] = None
    show_on: List[str] = field(default_factory=lambda: ["product", "cart"])
    is_active: bool = True
    features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> AICopyDict:
        return {
            "title": self.title,
            "description": self.description,
            "tagline": self.tagline,
            "cta_text": self.cta_text,
            "savings_message": self.savings_message,
            "show_on": self.show_on,
            "is_active": self.is_active,
            "features": self.features,
        }


@dataclass
class FBTBundle:
    """Complete FBT bundle."""
    id: str
    csv_upload_id: str
    products: List[ProductData]
    pricing: FBTPricing
    ai_copy: AICopy
    confidence: float
    ranking_score: float
    objective: str = "increase_aov"
    shop_id: Optional[str] = None
    predicted_lift: float = 1.0
    support: Optional[float] = None
    lift: Optional[float] = None
    is_approved: bool = False
    is_used: bool = False
    rank_position: Optional[int] = None
    discount_reference: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def bundle_type(self) -> str:
        return "FBT"

    def to_dict(self) -> FBTBundleDict:
        product_dicts = [p.to_dict() for p in self.products]
        return {
            "id": self.id,
            "csv_upload_id": self.csv_upload_id,
            "shop_id": self.shop_id,
            "bundle_type": "FBT",
            "objective": self.objective,
            "products": {
                "items": product_dicts,
                "trigger_product": product_dicts[0] if product_dicts else {},
                "addon_products": product_dicts[1:] if len(product_dicts) > 1 else [],
            },
            "pricing": self.pricing.to_dict(),
            "ai_copy": self.ai_copy.to_dict(),
            "confidence": self.confidence,
            "predicted_lift": self.predicted_lift,
            "ranking_score": self.ranking_score,
            "support": self.support,
            "lift": self.lift,
            "is_approved": self.is_approved,
            "is_used": self.is_used,
            "rank_position": self.rank_position,
            "discount_reference": self.discount_reference,
            "created_at": self.created_at,
        }


@dataclass
class VolumeBundle:
    """Complete VOLUME bundle."""
    id: str
    csv_upload_id: str
    product: ProductData  # Single product for volume
    volume_tiers: List[VolumeTier]
    ai_copy: AICopy
    confidence: float
    ranking_score: float
    objective: str = "increase_aov"
    shop_id: Optional[str] = None
    predicted_lift: float = 1.0
    is_approved: bool = False
    is_used: bool = False
    rank_position: Optional[int] = None
    discount_reference: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def bundle_type(self) -> str:
        return "VOLUME"

    @property
    def pricing(self) -> VolumePricing:
        return VolumePricing(
            original_total=self.product.price,
            volume_tiers=self.volume_tiers,
        )

    def to_dict(self) -> VolumeBundleDict:
        product_dict = self.product.to_dict()
        tier_dicts = [t.to_dict() for t in self.volume_tiers]
        return {
            "id": self.id,
            "csv_upload_id": self.csv_upload_id,
            "shop_id": self.shop_id,
            "bundle_type": "VOLUME",
            "objective": self.objective,
            "products": {
                "items": [product_dict],
                "volume_tiers": tier_dicts,
            },
            "pricing": {
                "original_total": self.product.price,
                "bundle_price": self.product.price,
                "discount_amount": 0,
                "discount_type": "tiered",
                "volume_tiers": tier_dicts,
            },
            "volume_tiers": tier_dicts,  # Top-level for frontend
            "ai_copy": self.ai_copy.to_dict(),
            "confidence": self.confidence,
            "predicted_lift": self.predicted_lift,
            "ranking_score": self.ranking_score,
            "is_approved": self.is_approved,
            "is_used": self.is_used,
            "rank_position": self.rank_position,
            "discount_reference": self.discount_reference,
            "created_at": self.created_at,
        }


@dataclass
class BOGOBundle:
    """Complete BOGO bundle."""
    id: str
    csv_upload_id: str
    product: ProductData  # BOGO is typically same product
    bogo_config: BOGOConfig
    ai_copy: AICopy
    confidence: float
    ranking_score: float
    objective: str = "clear_slow_movers"
    shop_id: Optional[str] = None
    predicted_lift: float = 1.0
    is_approved: bool = False
    is_used: bool = False
    rank_position: Optional[int] = None
    discount_reference: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def bundle_type(self) -> str:
        return "BOGO"

    def to_dict(self) -> BOGOBundleDict:
        product_dict = self.product.to_dict()
        bogo_dict = self.bogo_config.to_dict()

        qualifiers = [{
            "quantity": self.bogo_config.buy_qty,
            **product_dict,
        }]
        rewards = [{
            "quantity": self.bogo_config.get_qty,
            "discount_type": self.bogo_config.discount_type,
            "discount_percent": self.bogo_config.discount_percent,
            **product_dict,
        }]

        return {
            "id": self.id,
            "csv_upload_id": self.csv_upload_id,
            "shop_id": self.shop_id,
            "bundle_type": "BOGO",
            "objective": self.objective,
            "products": {
                "items": [product_dict],
                "qualifiers": qualifiers,
                "rewards": rewards,
            },
            "pricing": {
                "buy_total": self.product.price * self.bogo_config.buy_qty,
                "get_value": self.product.price * self.bogo_config.get_qty,
                "final_price": self.product.price * self.bogo_config.buy_qty,
                "savings": self.product.price * self.bogo_config.get_qty * (self.bogo_config.discount_percent / 100),
                "discount_type": "bogo",
                "bogo_config": bogo_dict,
            },
            "bogo_config": bogo_dict,  # Top-level for frontend
            "qualifiers": qualifiers,   # Top-level for frontend
            "rewards": rewards,         # Top-level for frontend
            "ai_copy": self.ai_copy.to_dict(),
            "confidence": self.confidence,
            "predicted_lift": self.predicted_lift,
            "ranking_score": self.ranking_score,
            "is_approved": self.is_approved,
            "is_used": self.is_used,
            "rank_position": self.rank_position,
            "discount_reference": self.discount_reference,
            "created_at": self.created_at,
        }


# =============================================================================
# DEFAULT VALUES
# =============================================================================

DEFAULT_VOLUME_TIERS = [
    VolumeTier(min_qty=1, discount_type="NONE", discount_value=0, label=None),
    VolumeTier(min_qty=2, discount_type="PERCENTAGE", discount_value=5, label="Starter Pack"),
    VolumeTier(min_qty=3, discount_type="PERCENTAGE", discount_value=10, label="Popular"),
    VolumeTier(min_qty=5, discount_type="PERCENTAGE", discount_value=15, label="Best Value"),
]

DEFAULT_BOGO_CONFIG = BOGOConfig(
    buy_qty=2,
    get_qty=1,
    discount_type="free",
    discount_percent=100,
    same_product=True,
    mode="free_same_variant",
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_product_data(
    product: Union[str, Dict[str, Any]],
    catalog: Optional[Dict[str, Any]] = None,
) -> ProductDataDict:
    """
    Normalize a product entry to the standard ProductDataDict format.

    Handles:
    - String (variant_id - primary identifier)
    - Dict with partial data
    - Dict with full data

    Args:
        product: Product data in any format (variant_id string or dict)
        catalog: Optional catalog mapping variant_id -> CatalogSnapshot

    Returns:
        Normalized ProductDataDict with all required fields

    NOTE: variant_id is the PRIMARY identifier. SKU is only for display.
    """
    if isinstance(product, str):
        # Assume it's a variant_id (primary identifier)
        variant_id = product
        if catalog and variant_id in catalog:
            snap = catalog[variant_id]
            product_id = getattr(snap, 'product_id', '')
            return {
                "product_gid": f"gid://shopify/Product/{product_id}",
                "name": getattr(snap, 'product_title', 'Product'),
                "title": getattr(snap, 'product_title', 'Product'),
                "price": float(getattr(snap, 'price', 0) or 0),
                "variant_id": variant_id,
                "variant_gid": f"gid://shopify/ProductVariant/{variant_id}",
                "sku": getattr(snap, 'sku', '') or "",  # Display only
                "image_url": getattr(snap, 'image_url', None),
                "product_id": product_id,
            }
        else:
            # No catalog data - return minimal structure
            return {
                "product_gid": "",
                "name": "Unknown Product",
                "title": "Unknown Product",
                "price": 0.0,
                "variant_id": variant_id,
                "variant_gid": f"gid://shopify/ProductVariant/{variant_id}",
                "sku": "",  # Don't use variant_id as SKU fallback
                "image_url": None,
                "product_id": "",
            }

    elif isinstance(product, dict):
        # Already a dict - normalize it
        # variant_id is PRIMARY, do not fall back to SKU
        variant_id = product.get("variant_id", "")
        product_id = product.get("product_id", "")

        # Try to get product_gid from various sources
        product_gid = product.get("product_gid", "")
        if not product_gid and product_id:
            product_gid = f"gid://shopify/Product/{product_id}"

        # Try to get name from various sources
        name = product.get("name") or product.get("title") or product.get("product_title", "Unknown Product")

        return {
            "product_gid": product_gid,
            "name": name,
            "title": name,
            "price": float(product.get("price", 0) or 0),
            "variant_id": variant_id,
            "variant_gid": product.get("variant_gid", f"gid://shopify/ProductVariant/{variant_id}") if variant_id else "",
            "sku": product.get("sku", "") or "",  # Display only
            "image_url": product.get("image_url"),
            "product_id": product_id,
        }

    else:
        logger.warning(f"Unknown product format: {type(product)}")
        return {
            "product_gid": "",
            "name": "Unknown Product",
            "title": "Unknown Product",
            "price": 0.0,
            "variant_id": "",
            "variant_gid": "",
            "sku": "",
            "image_url": None,
            "product_id": "",
        }


def enrich_products_from_catalog(
    products: Union[List[str], List[Dict[str, Any]]],
    catalog: Dict[str, Any],
) -> List[ProductDataDict]:
    """
    Enrich a list of products with catalog data.

    Args:
        products: List of product identifiers (strings or partial dicts)
        catalog: Mapping of variant_id -> CatalogSnapshot

    Returns:
        List of fully enriched ProductDataDict objects
    """
    enriched = []
    for product in products:
        normalized = normalize_product_data(product, catalog)
        # Only include if we have valid product_gid
        if normalized.get("product_gid") or normalized.get("price", 0) > 0:
            enriched.append(normalized)
    return enriched


def normalize_fbt_bundle(
    bundle: Dict[str, Any],
    catalog: Optional[Dict[str, Any]] = None,
) -> FBTBundleDict:
    """
    Normalize an FBT bundle to the standard format.

    Args:
        bundle: Bundle dict in any format
        catalog: Optional catalog for product enrichment

    Returns:
        Normalized FBTBundleDict
    """
    products = bundle.get("products", [])

    # Handle different products formats
    if isinstance(products, dict):
        items = products.get("items", [])
    elif isinstance(products, list):
        items = products
    else:
        items = []

    # Enrich products
    enriched_items = enrich_products_from_catalog(items, catalog or {})

    # Build pricing
    pricing = bundle.get("pricing", {})
    if not pricing.get("original_total"):
        pricing["original_total"] = sum(p.get("price", 0) for p in enriched_items)
    if not pricing.get("bundle_price"):
        discount_pct = float(pricing.get("discount_percentage", 10))
        pricing["bundle_price"] = pricing["original_total"] * (1 - discount_pct / 100)
    if not pricing.get("discount_amount"):
        pricing["discount_amount"] = pricing["original_total"] - pricing["bundle_price"]

    # Build ai_copy
    ai_copy = bundle.get("ai_copy", {})
    if not ai_copy.get("title"):
        names = [p.get("name", "Product") for p in enriched_items[:2]]
        ai_copy["title"] = " + ".join(names) if names else "FBT Bundle"
    if not ai_copy.get("description"):
        discount = pricing.get("discount_percentage", 10)
        ai_copy["description"] = f"Save {discount:.0f}% when bought together!"

    return {
        "id": bundle.get("id", str(uuid.uuid4())),
        "csv_upload_id": bundle.get("csv_upload_id", ""),
        "shop_id": bundle.get("shop_id"),
        "bundle_type": "FBT",
        "objective": bundle.get("objective", "increase_aov"),
        "products": {
            "items": enriched_items,
            "trigger_product": enriched_items[0] if enriched_items else {},
            "addon_products": enriched_items[1:] if len(enriched_items) > 1 else [],
        },
        "pricing": pricing,
        "ai_copy": ai_copy,
        "confidence": float(bundle.get("confidence", 0.5)),
        "predicted_lift": float(bundle.get("predicted_lift", 1.0)),
        "ranking_score": float(bundle.get("ranking_score", 0.5)),
        "support": bundle.get("support"),
        "lift": bundle.get("lift"),
        "is_approved": bundle.get("is_approved", False),
        "is_used": bundle.get("is_used", False),
        "rank_position": bundle.get("rank_position"),
        "discount_reference": bundle.get("discount_reference"),
        "created_at": bundle.get("created_at", datetime.utcnow()),
    }


def normalize_volume_bundle(
    bundle: Dict[str, Any],
    catalog: Optional[Dict[str, Any]] = None,
) -> VolumeBundleDict:
    """
    Normalize a VOLUME bundle to the standard format.

    Args:
        bundle: Bundle dict in any format
        catalog: Optional catalog for product enrichment

    Returns:
        Normalized VolumeBundleDict
    """
    products = bundle.get("products", [])

    # Handle different products formats
    if isinstance(products, dict):
        items = products.get("items", [])
    elif isinstance(products, list):
        items = products
    else:
        items = []

    # Volume bundles should have 1 product
    enriched_items = enrich_products_from_catalog(items[:1], catalog or {})

    if not enriched_items:
        enriched_items = [{
            "product_gid": "",
            "name": "Unknown Product",
            "title": "Unknown Product",
            "price": 0.0,
            "variant_id": "",
            "variant_gid": "",
            "sku": "",
            "image_url": None,
            "product_id": "",
        }]

    product = enriched_items[0]

    # Build volume tiers
    volume_tiers = bundle.get("volume_tiers") or bundle.get("pricing", {}).get("volume_tiers") or [
        t.to_dict() for t in DEFAULT_VOLUME_TIERS
    ]

    # Build pricing
    pricing = {
        "original_total": product.get("price", 0),
        "bundle_price": product.get("price", 0),
        "discount_amount": 0,
        "discount_type": "tiered",
        "volume_tiers": volume_tiers,
    }

    # Build ai_copy
    ai_copy = bundle.get("ai_copy", {})
    if not ai_copy.get("title"):
        ai_copy["title"] = f"Buy More, Save More - {product.get('name', 'Product')}"
    if not ai_copy.get("description"):
        ai_copy["description"] = f"The more {product.get('name', 'you')} buy, the more you save!"

    return {
        "id": bundle.get("id", str(uuid.uuid4())),
        "csv_upload_id": bundle.get("csv_upload_id", ""),
        "shop_id": bundle.get("shop_id"),
        "bundle_type": "VOLUME",
        "objective": bundle.get("objective", "increase_aov"),
        "products": {
            "items": enriched_items,
            "volume_tiers": volume_tiers,
        },
        "pricing": pricing,
        "volume_tiers": volume_tiers,  # Top-level for frontend
        "ai_copy": ai_copy,
        "confidence": float(bundle.get("confidence", 0.5)),
        "predicted_lift": float(bundle.get("predicted_lift", 1.0)),
        "ranking_score": float(bundle.get("ranking_score", 0.5)),
        "is_approved": bundle.get("is_approved", False),
        "is_used": bundle.get("is_used", False),
        "rank_position": bundle.get("rank_position"),
        "discount_reference": bundle.get("discount_reference"),
        "created_at": bundle.get("created_at", datetime.utcnow()),
    }


def normalize_bogo_bundle(
    bundle: Dict[str, Any],
    catalog: Optional[Dict[str, Any]] = None,
) -> BOGOBundleDict:
    """
    Normalize a BOGO bundle to the standard format.

    Args:
        bundle: Bundle dict in any format
        catalog: Optional catalog for product enrichment

    Returns:
        Normalized BOGOBundleDict
    """
    products = bundle.get("products", [])

    # Handle different products formats
    if isinstance(products, dict):
        items = products.get("items", [])
    elif isinstance(products, list):
        items = products
    else:
        items = []

    # BOGO bundles should have 1 product
    enriched_items = enrich_products_from_catalog(items[:1], catalog or {})

    if not enriched_items:
        enriched_items = [{
            "product_gid": "",
            "name": "Unknown Product",
            "title": "Unknown Product",
            "price": 0.0,
            "variant_id": "",
            "variant_gid": "",
            "sku": "",
            "image_url": None,
            "product_id": "",
        }]

    product = enriched_items[0]

    # Build BOGO config
    bogo_config = bundle.get("bogo_config") or bundle.get("pricing", {}).get("bogo_config") or DEFAULT_BOGO_CONFIG.to_dict()

    # Build qualifiers and rewards
    qualifiers = [{
        "quantity": bogo_config.get("buy_qty", 2),
        **product,
    }]
    rewards = [{
        "quantity": bogo_config.get("get_qty", 1),
        "discount_type": bogo_config.get("discount_type", "free"),
        "discount_percent": bogo_config.get("discount_percent", 100),
        **product,
    }]

    # Build pricing
    price = product.get("price", 0)
    buy_qty = bogo_config.get("buy_qty", 2)
    get_qty = bogo_config.get("get_qty", 1)
    discount_pct = bogo_config.get("discount_percent", 100)

    pricing = {
        "buy_total": price * buy_qty,
        "get_value": price * get_qty,
        "final_price": price * buy_qty,
        "savings": price * get_qty * (discount_pct / 100),
        "discount_type": "bogo",
        "bogo_config": bogo_config,
    }

    # Build ai_copy
    ai_copy = bundle.get("ai_copy", {})
    if not ai_copy.get("title"):
        if discount_pct >= 100:
            ai_copy["title"] = f"Buy {buy_qty}, Get {get_qty} FREE"
        else:
            ai_copy["title"] = f"Buy {buy_qty}, Get {get_qty} at {discount_pct}% Off"
    if not ai_copy.get("description"):
        ai_copy["description"] = f"Stock up on {product.get('name', 'this product')} and save!"

    return {
        "id": bundle.get("id", str(uuid.uuid4())),
        "csv_upload_id": bundle.get("csv_upload_id", ""),
        "shop_id": bundle.get("shop_id"),
        "bundle_type": "BOGO",
        "objective": bundle.get("objective", "clear_slow_movers"),
        "products": {
            "items": enriched_items,
            "qualifiers": qualifiers,
            "rewards": rewards,
        },
        "pricing": pricing,
        "bogo_config": bogo_config,  # Top-level for frontend
        "qualifiers": qualifiers,    # Top-level for frontend
        "rewards": rewards,          # Top-level for frontend
        "ai_copy": ai_copy,
        "confidence": float(bundle.get("confidence", 0.5)),
        "predicted_lift": float(bundle.get("predicted_lift", 1.0)),
        "ranking_score": float(bundle.get("ranking_score", 0.5)),
        "is_approved": bundle.get("is_approved", False),
        "is_used": bundle.get("is_used", False),
        "rank_position": bundle.get("rank_position"),
        "discount_reference": bundle.get("discount_reference"),
        "created_at": bundle.get("created_at", datetime.utcnow()),
    }


def validate_bundle(bundle: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate a bundle against the schema requirements.

    Args:
        bundle: Bundle dict to validate

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required top-level fields
    required_fields = ["id", "bundle_type", "products", "pricing", "ai_copy"]
    for field in required_fields:
        if not bundle.get(field):
            errors.append(f"Missing required field: {field}")

    # Check bundle type
    bundle_type = bundle.get("bundle_type")
    if bundle_type not in ("FBT", "VOLUME", "BOGO"):
        errors.append(f"Invalid bundle_type: {bundle_type}")

    # Check products structure
    products = bundle.get("products", {})
    if isinstance(products, dict):
        items = products.get("items", [])
    elif isinstance(products, list):
        items = products
    else:
        items = []
        errors.append("products must be a list or dict with 'items' key")

    # Check each product has required fields
    for i, product in enumerate(items):
        if isinstance(product, dict):
            if not product.get("product_gid"):
                errors.append(f"Product {i}: missing product_gid")
            if not product.get("name") and not product.get("title"):
                errors.append(f"Product {i}: missing name/title")
            if product.get("price", 0) <= 0:
                errors.append(f"Product {i}: invalid price")
        elif isinstance(product, str):
            errors.append(f"Product {i}: is a string ('{product}'), should be a dict with product_gid, name, price")

    # Check confidence and ranking_score
    confidence = bundle.get("confidence", 0)
    if confidence == 0:
        errors.append("confidence is 0 - ML scoring may not be working")

    ranking_score = bundle.get("ranking_score", 0)
    if ranking_score == 0:
        errors.append("ranking_score is 0 - ranking may not be working")

    # Check ai_copy
    ai_copy = bundle.get("ai_copy", {})
    if not ai_copy.get("title"):
        errors.append("ai_copy.title is missing")

    # Bundle-type specific checks
    if bundle_type == "VOLUME":
        volume_tiers = bundle.get("volume_tiers") or products.get("volume_tiers")
        if not volume_tiers:
            errors.append("VOLUME bundle missing volume_tiers")

    elif bundle_type == "BOGO":
        bogo_config = bundle.get("bogo_config") or bundle.get("pricing", {}).get("bogo_config")
        if not bogo_config:
            errors.append("BOGO bundle missing bogo_config")

    return (len(errors) == 0, errors)


# =============================================================================
# SHOP SIZE TIER MAPPING
# =============================================================================

SHOP_SIZE_TIERS = {
    "very_small": {
        "max_orders": 50,
        "fallback_tiers": [7, 6, 5],  # Cold-start, popularity, heuristics
        "min_support": 0.01,
        "description": "Very small shops with minimal order history"
    },
    "small": {
        "max_orders": 200,
        "fallback_tiers": [5, 4, 3],  # Heuristics, similarity, smoothed
        "min_support": 0.015,
        "description": "Small shops with limited order history"
    },
    "medium": {
        "max_orders": 500,
        "fallback_tiers": [3, 2, 1],  # Smoothed, adaptive, strict rules
        "min_support": 0.02,
        "description": "Medium shops with moderate order history"
    },
    "large": {
        "max_orders": float("inf"),
        "fallback_tiers": [1],  # Strict association rules only
        "min_support": 0.05,
        "description": "Large shops with extensive order history"
    },
}


def get_shop_tier(order_count: int) -> str:
    """Determine shop tier based on order count."""
    if order_count < SHOP_SIZE_TIERS["very_small"]["max_orders"]:
        return "very_small"
    elif order_count < SHOP_SIZE_TIERS["small"]["max_orders"]:
        return "small"
    elif order_count < SHOP_SIZE_TIERS["medium"]["max_orders"]:
        return "medium"
    else:
        return "large"
