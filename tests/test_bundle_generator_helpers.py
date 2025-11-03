import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("asyncpg", reason="BundleGenerator module requires asyncpg dependency")

from services.bundle_generator import BundleGenerator


def test_extract_sku_list_handles_mixed_payload():
    recommendation = {
        "products": [
            "SKU-001",
            {"sku": "SKU-002", "name": "Widget"},
            {"variant_id": "gid://shopify/Variant/123", "title": "Variant 123"},
            {"id": "fallback-id"},
            "",
            None,
        ]
    }

    skus = BundleGenerator._extract_sku_list(None, recommendation)

    assert skus == [
        "SKU-001",
        "SKU-002",
        "gid://shopify/Variant/123",
        "fallback-id",
    ]


def test_extract_sku_list_empty_products():
    recommendation = {"products": []}
    skus = BundleGenerator._extract_sku_list(None, recommendation)
    assert skus == []
