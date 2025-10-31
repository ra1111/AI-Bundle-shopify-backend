"""
Centralized configuration helpers for shop scoping.
"""
from __future__ import annotations

import os
from typing import Iterable, Mapping, Optional, Any

DEFAULT_SHOP_ID: str = os.getenv("DEFAULT_SHOP_ID") or "demo-shop"

# Column names we recognize for shop identification inside CSV payloads.
SHOP_ID_FIELD_CANDIDATES: tuple[str, ...] = (
    "shop_id",
    "shopid",
    "shop",
    "shop_domain",
    "shopdomain",
    "store_domain",
    "storedomain",
    "merchant_domain",
    "merchantdomain",
    "domain",
)


def sanitize_shop_id(value: Optional[Any]) -> Optional[str]:
    """Normalize raw IDs (strip whitespace, lower-case domains)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        value = str(value)
    text = str(value).strip()
    if not text:
        return None
    return text.lower()


def resolve_shop_id(*candidates: Optional[Any]) -> str:
    """
    Pick the first usable shop identifier from candidates, otherwise fall back to DEFAULT_SHOP_ID.
    """
    for candidate in candidates:
        normalized = sanitize_shop_id(candidate)
        if normalized:
            return normalized
    return DEFAULT_SHOP_ID


def infer_shop_id_from_rows(rows: Iterable[Mapping[str, Any]]) -> Optional[str]:
    """
    Scan parsed CSV rows for a known shop identifier column.
    Returns a normalized value or None if not present.
    """
    for row in rows:
        if not row:
            continue
        # Direct matches (exact key)
        for key in SHOP_ID_FIELD_CANDIDATES:
            if key in row:
                value = sanitize_shop_id(row.get(key))
                if value:
                    return value
        # Also check canonicalized keys (snake_case via underscores)
        for key, value in row.items():
            if not key:
                continue
            canonical = (
                key.strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
            if canonical in SHOP_ID_FIELD_CANDIDATES:
                normalized = sanitize_shop_id(value)
                if normalized:
                    return normalized
    return None
