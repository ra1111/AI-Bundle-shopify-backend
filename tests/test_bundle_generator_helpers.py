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


def test_normalize_drop_reason_standardization():
    gen = object.__new__(BundleGenerator)
    assert gen._normalize_drop_reason("out_of_stock", "inventory") == "OUT_OF_STOCK"
    assert gen._normalize_drop_reason("policy_violation", "compliance") == "POLICY_BLOCK"
    assert gen._normalize_drop_reason("pricing_error", "pricing") == "PRICE_ANOMALY"
    assert gen._normalize_drop_reason("unknown", "inventory") == "FINALIZE_TX_FAIL"


def test_build_staged_progress_payload_shape():
    gen = object.__new__(BundleGenerator)
    staged_state = {
        "waves": [
            {
                "index": 0,
                "target": 3,
                "published": 3,
                "drops": {"OUT_OF_STOCK": 1},
                "duration_ms": 1200,
                "finalize_tx": "abc",
            }
        ],
        "totals": {"published": 3, "dropped": 1},
        "cursor": {"stage_idx": 1, "published": 3, "last_bundle_id": "b3"},
        "backpressure": {"active": True, "reason": "drop_rate_high"},
    }

    payload = gen._build_staged_progress_payload(
        "run_123", staged_state, next_eta_seconds=25
    )

    assert payload["run_id"] == "run_123"
    assert payload["staged"] is True
    assert payload["waves"][0]["index"] == 0
    assert payload["waves"][0]["target"] == 3
    assert payload["totals"]["published"] == 3
    assert payload["cursor"]["stage_idx"] == 1
    assert payload["next_wave_eta_sec"] == 25
