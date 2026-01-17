"""
Bundle Schemas Package
Provides standardized data structures for all bundle types.
"""

from .bundle_schemas import (
    # Product schemas
    ProductData,
    ProductDataDict,

    # Bundle type schemas
    FBTBundle,
    FBTBundleDict,
    VolumeBundle,
    VolumeBundleDict,
    BOGOBundle,
    BOGOBundleDict,

    # Pricing schemas
    FBTPricing,
    FBTPricingDict,
    VolumePricing,
    VolumePricingDict,
    BOGOPricing,
    BOGOPricingDict,
    VolumeTier,
    VolumeTierDict,
    BOGOConfig,
    BOGOConfigDict,

    # AI Copy schemas
    AICopy,
    AICopyDict,

    # Helper functions
    normalize_product_data,
    normalize_fbt_bundle,
    normalize_volume_bundle,
    normalize_bogo_bundle,
    validate_bundle,
    enrich_products_from_catalog,
)
