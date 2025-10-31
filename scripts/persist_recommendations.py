#!/usr/bin/env python3
"""Persist bundle recommendations from a JSON payload into the database.

The remote bundle API can return recommendations without storing them. This
utility bridges that gap by reading the generated JSON (from bundle_run.sh) and
inserting rows into the bundle_recommendations table using the existing
SQLAlchemy models.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

# Ensure DATABASE_URL is available before importing the SQLAlchemy engine.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
DEFAULT_DB_URL_FILE = PROJECT_ROOT / "dburl.txt"

RAW_DB_URL = os.getenv("DATABASE_URL")
if not RAW_DB_URL and DEFAULT_DB_URL_FILE.exists():
    candidate = DEFAULT_DB_URL_FILE.read_text(encoding="utf-8").strip()
    if candidate:
        RAW_DB_URL = candidate
        os.environ["DATABASE_URL"] = candidate

print(f"PYTHON: {sys.executable}")
print(f"DB_URL(env raw): {RAW_DB_URL}")
try:
    print(f"HOSTNAME: {socket.gethostname()}")
except Exception as exc:  # pragma: no cover - defensive logging
    print(f"HOSTNAME_ERR: {exc}")
sys.stdout.flush()

if RAW_DB_URL and RAW_DB_URL.startswith("postgresql://") and "+asyncpg" not in RAW_DB_URL:
    normalized_url = RAW_DB_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    os.environ["DATABASE_URL"] = normalized_url
    print(f"DB_URL(normalized driver): {normalized_url}")
    sys.stdout.flush()

if os.getenv("DATABASE_URL"):
    print(f"DB_URL(final): {os.getenv('DATABASE_URL')}")
    sys.stdout.flush()

from database import init_db
from services.storage import StorageService
from settings import resolve_shop_id


def _to_decimal(value: Any, fallback: str = "0") -> Decimal:
    """Best-effort conversion to Decimal with a fallback."""
    if value in (None, "", "null", "NULL"):
        return Decimal(fallback)
    try:
        return Decimal(str(value))
    except Exception:
        return Decimal(fallback)


def _normalise_recommendation(
    rec: Dict[str, Any],
    csv_upload_id: str,
    default_rank: int,
    shop_id: str,
) -> Dict[str, Any]:
    """Map API recommendation fields onto the BundleRecommendation schema."""

    bundle_id = rec.get("id") or str(uuid.uuid4())
    rank_position = rec.get("rank_position") or default_rank

    confidence = _to_decimal(rec.get("confidence"), "0")
    predicted_lift = rec.get("predicted_lift")
    lift = rec.get("lift")
    support = rec.get("support")

    # Default ranking score to confidence if none supplied.
    ranking_score = rec.get("ranking_score")
    ranking_score = _to_decimal(
        ranking_score if ranking_score is not None else rec.get("confidence"), "0"
    )

    now = datetime.now(timezone.utc)

    raw_created = rec.get("created_at")
    created_at = now
    if raw_created:
        try:
            created_at = datetime.fromisoformat(str(raw_created).replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            created_at = now

    normalised = {
        "id": bundle_id,
        "csv_upload_id": csv_upload_id,
        "shop_id": shop_id,
        "bundle_type": rec.get("bundle_type") or "unknown",
        "objective": rec.get("objective") or "",
        "products": rec.get("products") or [],
        "pricing": rec.get("pricing") or {},
        "ai_copy": rec.get("ai_copy") or {},
        "confidence": confidence,
        "predicted_lift": _to_decimal(predicted_lift if predicted_lift is not None else lift, "0"),
        "support": (_to_decimal(support, "0") if support is not None else None),
        "lift": (_to_decimal(lift, "0") if lift is not None else None),
        "ranking_score": ranking_score,
        "discount_reference": rec.get("discount_reference"),
        "is_approved": bool(rec.get("is_approved", False)),
        "is_used": bool(rec.get("is_used", False)),
        "rank_position": rank_position,
        "created_at": created_at.replace(tzinfo=None),
    }

    return normalised


async def _persist(
    json_path: Path,
    csv_upload_id: str,
    *,
    clear_existing: bool = False,
    run_id: Optional[str] = None,
    shop_id: Optional[str] = None,
) -> int:
    """Load recommendations from disk and persist them."""

    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    recommendations = (
        payload.get("result", {}).get("recommendations")
        if isinstance(payload, dict)
        else None
    )

    if not recommendations:
        return 0

    await init_db()
    storage = StorageService()

    upload_shop_id = await storage.get_shop_id_for_upload(csv_upload_id)
    resolved_shop_id = resolve_shop_id(shop_id, upload_shop_id)

    if clear_existing:
        await storage.clear_bundle_recommendations(csv_upload_id, resolved_shop_id)

    normalised: List[Dict[str, Any]] = []
    for idx, rec in enumerate(recommendations, start=1):
        try:
            normalised.append(_normalise_recommendation(rec, csv_upload_id, idx, resolved_shop_id))
        except Exception as exc:  # Defensive: skip malformed items, keep going
            print(f"⚠️  Skipped recommendation {idx} due to error: {exc}", file=sys.stderr)

    if not normalised:
        return 0

    print(f"Persisting {len(normalised)} recommendations for upload {csv_upload_id} (run_id={run_id or '-'}).")
    sys.stdout.flush()

    await storage.create_bundle_recommendations(normalised)
    return len(normalised)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", nargs="?", type=Path, help="Path to bundles_<RUN>.json produced by bundle_run.sh")
    parser.add_argument("csv_upload_id", nargs="?", help="CSV upload identifier to associate with the recommendations")
    parser.add_argument("--input", dest="json_path_opt", type=Path, help="Alias for the JSON payload path.")
    parser.add_argument("--csv-upload-id", dest="csv_upload_id_opt", help="Alias for the CSV upload ID.")
    parser.add_argument("--run-id", dest="run_id", default=None, help="Optional run identifier for logging.")
    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Delete existing recommendations for this upload before inserting",
    )
    parser.add_argument(
        "--shop-id",
        dest="shop_id",
        default=None,
        help="Optional explicit shop identifier (defaults to environment or CSV upload metadata)",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    json_path = args.json_path_opt or args.json_path
    if not json_path:
        raise SystemExit("✗ JSON input path is required")

    csv_upload_id = args.csv_upload_id_opt or args.csv_upload_id or os.getenv("CSV_UPLOAD_ID")
    if not csv_upload_id:
        raise SystemExit("✗ csv_upload_id is required")

    try:
        count = asyncio.run(
            _persist(
                json_path,
                csv_upload_id,
                clear_existing=args.clear_existing,
                run_id=args.run_id or os.getenv("RUN_ID"),
                shop_id=args.shop_id,
            )
        )
    except Exception as exc:
        print(f"✗ Failed to persist recommendations: {exc}", file=sys.stderr)
        return 1

    if count == 0:
        print("⚠️  No recommendations were persisted (payload empty or filtered).")
    else:
        print(f"✓ Persisted {count} recommendations for upload {csv_upload_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
