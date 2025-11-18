"""
Tests for Bundle Translator Layer

Validates that quick-start bundles are correctly translated into canonical bundle_def format.
"""

import pytest
from decimal import Decimal
from services.bundle_translator import (
    translate_fbt,
    translate_bogo,
    translate_volume,
    translate_bundle_rec,
    generate_ai_copy_for_bundle
)


class MockCatalogSnapshot:
    """Mock CatalogSnapshot for testing"""
    def __init__(self, sku, variant_id, product_id, product_title, price):
        self.sku = sku
        self.variant_id = variant_id
        self.product_id = product_id
        self.product_title = product_title
        self.price = Decimal(str(price))


# Sample catalog for testing
SAMPLE_CATALOG = {
    "SKU-A": MockCatalogSnapshot("SKU-A", "gid://shopify/ProductVariant/001", "gid://shopify/Product/101", "Premium Shirt", 29.99),
    "SKU-B": MockCatalogSnapshot("SKU-B", "gid://shopify/ProductVariant/002", "gid://shopify/Product/102", "Designer Pants", 49.99),
    "SLOW-SKU": MockCatalogSnapshot("SLOW-SKU", "gid://shopify/ProductVariant/003", "gid://shopify/Product/103", "Clearance Hat", 19.99),
    "POPULAR": MockCatalogSnapshot("POPULAR", "gid://shopify/ProductVariant/004", "gid://shopify/Product/104", "Best Seller", 25.00),
}


class TestTranslateFBT:
    """Test FBT (Frequently Bought Together) translation"""

    def test_translate_fbt_basic(self):
        """Test basic FBT bundle translation"""
        rec = {
            "bundle_type": "FBT",
            "products": [
                {
                    "sku": "SKU-A",
                    "variant_id": "gid://shopify/ProductVariant/001",
                    "price": 29.99
                },
                {
                    "sku": "SKU-B",
                    "variant_id": "gid://shopify/ProductVariant/002",
                    "price": 49.99
                }
            ],
            "pricing": {
                "original_total": 79.98,
                "bundle_price": 71.98,
                "discount_amount": 8.00,
                "discount_pct": "10.0%"
            }
        }

        result = translate_fbt(rec, SAMPLE_CATALOG)

        assert result["bundle_type"] == "FBT"
        assert len(result["items"]) == 2
        assert result["items"][0]["variant_id"] == "gid://shopify/ProductVariant/001"
        assert result["items"][0]["quantity"] == 1
        assert result["items"][1]["variant_id"] == "gid://shopify/ProductVariant/002"
        assert result["items"][1]["quantity"] == 1

        assert result["pricing"]["discount_type"] == "PERCENTAGE"
        assert result["pricing"]["discount_value"] == 10.0

        assert result["rules"]["min_items_required"] == 2
        assert result["rules"]["max_items_allowed"] == 2
        assert result["rules"]["allow_substitution"] is False

    def test_translate_fbt_missing_variant_id(self):
        """Test FBT translation with variant_id lookup from catalog"""
        rec = {
            "bundle_type": "FBT",
            "products": [
                {"sku": "SKU-A", "price": 29.99},  # Missing variant_id
                {"sku": "SKU-B", "price": 49.99}   # Missing variant_id
            ],
            "pricing": {
                "discount_pct": "15.0%"
            }
        }

        result = translate_fbt(rec, SAMPLE_CATALOG)

        # Should lookup variant_ids from catalog
        assert result["items"][0]["variant_id"] == "gid://shopify/ProductVariant/001"
        assert result["items"][1]["variant_id"] == "gid://shopify/ProductVariant/002"
        assert result["pricing"]["discount_value"] == 15.0

    def test_translate_fbt_different_discount(self):
        """Test FBT with different discount percentage"""
        rec = {
            "bundle_type": "FBT",
            "products": [
                {"sku": "SKU-A", "variant_id": "gid://shopify/ProductVariant/001"},
                {"sku": "SKU-B", "variant_id": "gid://shopify/ProductVariant/002"}
            ],
            "pricing": {
                "discount_pct": "20.0%"
            }
        }

        result = translate_fbt(rec, SAMPLE_CATALOG)
        assert result["pricing"]["discount_value"] == 20.0


class TestTranslateBOGO:
    """Test BOGO (Buy X Get Y) translation to BXGY"""

    def test_translate_bogo_basic(self):
        """Test basic BOGO bundle translation"""
        rec = {
            "bundle_type": "BOGO",
            "products": [
                {
                    "sku": "SLOW-SKU",
                    "variant_id": "gid://shopify/ProductVariant/003",
                    "price": 19.99,
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

        result = translate_bogo(rec, SAMPLE_CATALOG)

        assert result["bundle_type"] == "BXGY"

        # Check qualifiers (what you must buy)
        assert len(result["qualifiers"]) == 1
        assert result["qualifiers"][0]["variant_id"] == "gid://shopify/ProductVariant/003"
        assert result["qualifiers"][0]["min_qty"] == 2

        # Check rewards (what you get discounted)
        assert len(result["rewards"]) == 1
        assert result["rewards"][0]["variant_id"] == "gid://shopify/ProductVariant/003"
        assert result["rewards"][0]["quantity"] == 1
        assert result["rewards"][0]["discount_type"] == "PERCENTAGE"
        assert result["rewards"][0]["discount_value"] == 100.0  # 100% off = free

        assert result["rules"]["auto_apply"] is True

    def test_translate_bogo_missing_variant_id(self):
        """Test BOGO translation with variant_id lookup from catalog"""
        rec = {
            "bundle_type": "BOGO",
            "products": [
                {"sku": "SLOW-SKU", "price": 19.99}  # Missing variant_id
            ],
            "pricing": {
                "bogo_config": {
                    "buy_qty": 3,
                    "get_qty": 1
                }
            }
        }

        result = translate_bogo(rec, SAMPLE_CATALOG)

        # Should lookup variant_id from catalog
        assert result["qualifiers"][0]["variant_id"] == "gid://shopify/ProductVariant/003"
        assert result["qualifiers"][0]["min_qty"] == 3
        assert result["rewards"][0]["quantity"] == 1

    def test_translate_bogo_buy_3_get_2(self):
        """Test BOGO with different quantities"""
        rec = {
            "bundle_type": "BOGO",
            "products": [
                {"sku": "SLOW-SKU", "variant_id": "gid://shopify/ProductVariant/003"}
            ],
            "pricing": {
                "bogo_config": {
                    "buy_qty": 3,
                    "get_qty": 2
                }
            }
        }

        result = translate_bogo(rec, SAMPLE_CATALOG)

        assert result["qualifiers"][0]["min_qty"] == 3
        assert result["rewards"][0]["quantity"] == 2

    def test_translate_bogo_missing_product(self):
        """Test BOGO translation with missing product raises error"""
        rec = {
            "bundle_type": "BOGO",
            "products": [],
            "pricing": {"bogo_config": {}}
        }

        with pytest.raises(ValueError, match="BOGO bundle must have at least one product"):
            translate_bogo(rec, SAMPLE_CATALOG)


class TestTranslateVolume:
    """Test VOLUME (tiered pricing) translation"""

    def test_translate_volume_basic(self):
        """Test basic volume bundle translation"""
        rec = {
            "bundle_type": "VOLUME",
            "products": [
                {
                    "sku": "POPULAR",
                    "variant_id": "gid://shopify/ProductVariant/004",
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

        result = translate_volume(rec, SAMPLE_CATALOG)

        assert result["bundle_type"] == "VOLUME"

        # Check items
        assert len(result["items"]) == 1
        assert result["items"][0]["variant_id"] == "gid://shopify/ProductVariant/004"
        assert result["items"][0]["quantity"] == 1

        # Check volume tiers
        assert len(result["pricing"]["volume_tiers"]) == 4
        assert result["pricing"]["volume_tiers"][0]["min_qty"] == 1
        assert result["pricing"]["volume_tiers"][0]["discount_value"] == 0
        assert result["pricing"]["volume_tiers"][3]["min_qty"] == 5
        assert result["pricing"]["volume_tiers"][3]["discount_value"] == 15

        # Check rules
        assert result["rules"]["auto_apply"] is True
        assert result["rules"]["min_qty"] == 1

    def test_translate_volume_missing_variant_id(self):
        """Test volume translation with variant_id lookup from catalog"""
        rec = {
            "bundle_type": "VOLUME",
            "products": [
                {"sku": "POPULAR", "price": 25.00}  # Missing variant_id
            ],
            "pricing": {
                "volume_tiers": [
                    {"min_qty": 1, "discount_type": "NONE", "discount_value": 0},
                    {"min_qty": 2, "discount_type": "PERCENTAGE", "discount_value": 10}
                ]
            }
        }

        result = translate_volume(rec, SAMPLE_CATALOG)

        # Should lookup variant_id from catalog
        assert result["items"][0]["variant_id"] == "gid://shopify/ProductVariant/004"
        assert len(result["pricing"]["volume_tiers"]) == 2

    def test_translate_volume_missing_product(self):
        """Test volume translation with missing product raises error"""
        rec = {
            "bundle_type": "VOLUME",
            "products": [],
            "pricing": {"volume_tiers": []}
        }

        with pytest.raises(ValueError, match="VOLUME bundle must have at least one product"):
            translate_volume(rec, SAMPLE_CATALOG)


class TestTranslateBundleRec:
    """Test main router function"""

    def test_translate_bundle_rec_fbt(self):
        """Test router correctly delegates to translate_fbt"""
        rec = {
            "id": "bundle-001",
            "bundle_type": "FBT",
            "products": [
                {"sku": "SKU-A", "variant_id": "gid://shopify/ProductVariant/001"},
                {"sku": "SKU-B", "variant_id": "gid://shopify/ProductVariant/002"}
            ],
            "pricing": {"discount_pct": "10.0%"}
        }

        result = translate_bundle_rec(rec, SAMPLE_CATALOG)

        assert result["bundle_type"] == "FBT"
        assert len(result["items"]) == 2

    def test_translate_bundle_rec_bogo(self):
        """Test router correctly delegates to translate_bogo"""
        rec = {
            "id": "bundle-002",
            "bundle_type": "BOGO",
            "products": [
                {"sku": "SLOW-SKU", "variant_id": "gid://shopify/ProductVariant/003"}
            ],
            "pricing": {
                "bogo_config": {
                    "buy_qty": 2,
                    "get_qty": 1
                }
            }
        }

        result = translate_bundle_rec(rec, SAMPLE_CATALOG)

        assert result["bundle_type"] == "BXGY"
        assert len(result["qualifiers"]) == 1

    def test_translate_bundle_rec_volume(self):
        """Test router correctly delegates to translate_volume"""
        rec = {
            "id": "bundle-003",
            "bundle_type": "VOLUME",
            "products": [
                {"sku": "POPULAR", "variant_id": "gid://shopify/ProductVariant/004"}
            ],
            "pricing": {
                "volume_tiers": [
                    {"min_qty": 1, "discount_type": "NONE", "discount_value": 0}
                ]
            }
        }

        result = translate_bundle_rec(rec, SAMPLE_CATALOG)

        assert result["bundle_type"] == "VOLUME"
        assert len(result["items"]) == 1

    def test_translate_bundle_rec_unknown_type(self):
        """Test router handles unknown bundle types gracefully"""
        rec = {
            "id": "bundle-004",
            "bundle_type": "MIX_MATCH",
            "items": [{"variant_id": "gid://shopify/ProductVariant/005"}]
        }

        # Should return rec as-is with warning
        result = translate_bundle_rec(rec, SAMPLE_CATALOG)

        assert result["bundle_type"] == "MIX_MATCH"
        assert result == rec

    def test_translate_bundle_rec_missing_type(self):
        """Test router raises error for missing bundle_type"""
        rec = {
            "id": "bundle-005",
            "products": []
        }

        with pytest.raises(ValueError, match="Bundle recommendation missing bundle_type field"):
            translate_bundle_rec(rec, SAMPLE_CATALOG)

    def test_translate_bundle_rec_case_insensitive(self):
        """Test router handles lowercase bundle types"""
        rec = {
            "bundle_type": "fbt",  # lowercase
            "products": [
                {"sku": "SKU-A", "variant_id": "gid://shopify/ProductVariant/001"},
                {"sku": "SKU-B", "variant_id": "gid://shopify/ProductVariant/002"}
            ],
            "pricing": {"discount_pct": "10.0%"}
        }

        result = translate_bundle_rec(rec, SAMPLE_CATALOG)

        assert result["bundle_type"] == "FBT"


class TestGenerateAICopy:
    """Test AI copy generation for quick-start bundles"""

    def test_generate_fbt_copy(self):
        """Test AI copy generation for FBT bundles"""
        products = [
            {"name": "Premium Shirt", "price": 29.99, "sku": "SKU-A"},
            {"name": "Designer Pants", "price": 49.99, "sku": "SKU-B"}
        ]
        pricing = {"discount_pct": "10%"}

        result = generate_ai_copy_for_bundle("FBT", products, pricing)

        assert "Premium Shirt" in result["title"]
        assert "Designer Pants" in result["title"]
        assert "10%" in result["description"]
        assert "value_proposition" in result

    def test_generate_bogo_copy(self):
        """Test AI copy generation for BOGO bundles"""
        products = [
            {"name": "Clearance Hat", "price": 19.99, "sku": "SLOW-SKU"}
        ]
        pricing = {
            "bogo_config": {
                "buy_qty": 2,
                "get_qty": 1
            }
        }

        result = generate_ai_copy_for_bundle("BOGO", products, pricing)

        assert "Buy 2" in result["title"]
        assert "Get 1 Free" in result["title"]
        assert "Clearance Hat" in result["title"]
        assert "free" in result["description"].lower()

    def test_generate_volume_copy(self):
        """Test AI copy generation for VOLUME bundles"""
        products = [
            {"name": "Best Seller", "price": 25.00, "sku": "POPULAR"}
        ]
        pricing = {
            "volume_tiers": [
                {"min_qty": 1, "discount_type": "NONE", "discount_value": 0},
                {"min_qty": 2, "discount_type": "PERCENTAGE", "discount_value": 5},
                {"min_qty": 5, "discount_type": "PERCENTAGE", "discount_value": 15}
            ]
        }

        result = generate_ai_copy_for_bundle("VOLUME", products, pricing)

        assert "Volume Discount" in result["title"]
        assert "Best Seller" in result["title"]
        assert "15%" in result["description"]  # Max discount

    def test_generate_copy_unknown_type(self):
        """Test AI copy generation for unknown bundle types"""
        products = [{"name": "Product", "price": 10.00}]
        pricing = {}

        result = generate_ai_copy_for_bundle("UNKNOWN", products, pricing)

        assert result["title"] == "Bundle Deal"
        assert result["description"] == "Special bundle offer"


class TestEndToEndTranslation:
    """End-to-end translation tests"""

    def test_full_quick_start_fbt_bundle(self):
        """Test complete translation of quick-start FBT bundle"""
        # This simulates a bundle from generate_quick_start_bundles()
        rec = {
            "id": "quick-001",
            "csv_upload_id": "upload-123",
            "bundle_type": "FBT",
            "objective": "increase_aov",
            "products": [
                {
                    "sku": "SKU-A",
                    "name": "Premium Shirt",
                    "price": 29.99,
                    "variant_id": "gid://shopify/ProductVariant/001",
                    "product_id": "gid://shopify/Product/101"
                },
                {
                    "sku": "SKU-B",
                    "name": "Designer Pants",
                    "price": 49.99,
                    "variant_id": "gid://shopify/ProductVariant/002",
                    "product_id": "gid://shopify/Product/102"
                }
            ],
            "pricing": {
                "original_total": 79.98,
                "bundle_price": 71.98,
                "discount_amount": 8.00,
                "discount_pct": "10.0%"
            },
            "confidence": Decimal("0.85"),
            "ranking_score": Decimal("1.3"),
            "discount_reference": "__quick_start_upload-123__"
        }

        bundle_def = translate_bundle_rec(rec, SAMPLE_CATALOG)

        # Verify bundle_def structure
        assert bundle_def["bundle_type"] == "FBT"
        assert len(bundle_def["items"]) == 2
        assert bundle_def["pricing"]["discount_type"] == "PERCENTAGE"
        assert bundle_def["pricing"]["discount_value"] == 10.0
        assert bundle_def["rules"]["min_items_required"] == 2

        # This bundle_def can now be used by:
        # - Shopify Function for discounts
        # - PDP widget for rendering
        # - Cart/Drawer for UI
        # - Unified pricing engine


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
