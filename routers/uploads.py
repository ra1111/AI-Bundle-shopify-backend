"""
CSV Upload Router
Handles file uploads and processing
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Form
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, desc
from typing import List, Optional, Tuple
import logging
import uuid
from datetime import datetime

from database import get_db, CsvUpload
from services.csv_processor import CSVProcessor
from services.association_rules_engine import AssociationRulesEngine
from services.bundle_generator import BundleGenerator
from services.data_mapper import DataMapper
from services.storage import storage
from settings import resolve_shop_id

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload-csv")
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    csvType: Optional[str] = Form(None),
    runId: Optional[str] = Form(None),
    shopId: Optional[str] = Form(None),
    shopDomain: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    request_id = str(uuid.uuid4())
    try:
        logger.info(f"[{request_id}] Upload attempt filename={file.filename!r} csvType={csvType!r} content_type={file.content_type!r}")

        if not file.filename or not file.filename.lower().endswith(".csv"):
            logger.warning(f"[{request_id}] Reject non-CSV filename={file.filename!r}")
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")

        content = await file.read()
        size = len(content or b"")
        logger.info(f"[{request_id}] Received payload size={size} bytes")

        if size == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        if size > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 50MB")

        # Light peek at the first line for diagnostics
        try:
            head_preview = content[:200].decode("utf-8", errors="replace")
            first_line = head_preview.splitlines()[0] if head_preview else ""
            logger.info(f"[{request_id}] CSV preview firstLine={first_line!r}")
        except Exception:
            logger.exception(f"[{request_id}] Could not decode CSV preview")


        # AFTER (add catalog_joined; keep products→products_variants alias)
        valid_csv_types = ['orders', 'products', 'inventory_levels', 'variants', 'catalog_joined']

        alias_map = {
            # Keep both so older clients continue working
            'products': 'catalog_joined',
            'products_variants': 'variants',
            'variants': 'variants',
            'catalog': 'catalog_joined',       # optional helper
            'catalog_joined': 'catalog_joined',
            'orders': 'orders',
            'inventory_levels': 'inventory_levels',
            None: 'auto',
        }
 
        normalized_type = alias_map.get(csvType, csvType)
        if csvType and normalized_type not in valid_csv_types and normalized_type != "auto":
            logger.warning(f"[{request_id}] Invalid csvType={csvType!r} (normalized={normalized_type!r})")
            raise HTTPException(status_code=400, detail=f"Invalid CSV type. Must be one of: {', '.join(valid_csv_types)}")

        # Quick DB smoke test (fast, safe)
        try:
            await db.execute(text("SELECT 1"))
            logger.info(f"[{request_id}] DB session OK")
        except Exception:
            logger.exception(f"[{request_id}] DB session not OK")
            raise HTTPException(status_code=500, detail="Database unavailable")

        upload_id = str(uuid.uuid4())
        # Use provided runId to correlate multi-file ingests; generate if absent
        effective_run_id = (runId or str(uuid.uuid4())).strip()
        resolved_shop_id = resolve_shop_id(shopId, shopDomain)

        if runId:
            logger.info(f"[{request_id}] Using provided runId={effective_run_id}")
        else:
            logger.warning(f"[{request_id}] ⚠️ No runId provided - generated new runId={effective_run_id} (CSVs won't be grouped!)")

        csv_upload = CsvUpload(
            id=upload_id,
            filename=file.filename,
            csv_type=normalized_type,
            run_id=effective_run_id,
            shop_id=resolved_shop_id,
            total_rows=0,
            processed_rows=0,
            status="processing",
            error_message=None
        )
        db.add(csv_upload)
        await db.commit()
        logger.info(f"[{request_id}] Created CsvUpload id={upload_id} run_id={effective_run_id} shop_id={resolved_shop_id} csv_type={normalized_type}")

        # hand off to background (use the normalized type exactly once)
        csv_content = content.decode('utf-8', errors="replace")
        background_tasks.add_task(process_csv_background, csv_content, upload_id, normalized_type)
        logger.info(f"[{request_id}] Background task scheduled for id={upload_id}")

        return {
            "uploadId": upload_id,
            "runId": effective_run_id,
            "status": "processing",
            "shopId": resolved_shop_id,
            "requestId": request_id
        }

    except HTTPException as he:
        logger.error(f"[{request_id}] HTTP {he.status_code} during upload: {he.detail}")
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Upload error")
        # Return the message to caller during debugging; change to generic later
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/upload-status/{upload_id}")
async def get_upload_status(upload_id: str, db: AsyncSession = Depends(get_db)):
    """
    ⚠️ DEPRECATED: Use /api/shopify/status/{upload_id} instead

    This endpoint is maintained for backward compatibility but will be removed in a future version.
    Please migrate to /api/shopify/status/{upload_id} which provides:
    - Simplified status values (processing/completed/failed)
    - bundle_count field
    - Better error handling
    """
    try:
        logger.warning(
            f"⚠️ DEPRECATED ENDPOINT CALLED: /api/upload-status/{upload_id} "
            f"- Please migrate to /api/shopify/status/{upload_id}"
        )

        result = await db.get(CsvUpload, upload_id)
        if not result:
            raise HTTPException(status_code=404, detail="Upload not found")

        # Map internal status to simplified status
        frontend_status = result.status
        if result.status in {"bundle_generation_completed"}:
            frontend_status = "completed"
        elif result.status in {"bundle_generation_in_progress", "bundle_generation_queued", "bundle_generation_async"}:
            frontend_status = "processing"
        elif result.status in {"bundle_generation_failed", "bundle_generation_timed_out", "bundle_generation_cancelled"}:
            frontend_status = "failed"

        return {
            "id": result.id,
            "filename": result.filename,
            "csvType": result.csv_type,
            "runId": result.run_id,
            "status": frontend_status,  # Simplified status
            "totalRows": result.total_rows,
            "processedRows": result.processed_rows,
            "errorMessage": result.error_message,
            "_deprecated": True,
            "_useInstead": f"/api/shopify/status/{upload_id}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get upload status")

@router.get("/uploads")
async def get_uploads(db: AsyncSession = Depends(get_db)):
    """Get list of recent uploads"""
    try:
        from sqlalchemy import select, desc
        
        query = select(CsvUpload).order_by(desc(CsvUpload.created_at)).limit(10)
        result = await db.execute(query)
        uploads = result.scalars().all()
        
        return [
            {
                "id": upload.id,
                "filename": upload.filename,
                "csvType": upload.csv_type,
                "status": upload.status,
                "totalRows": upload.total_rows,
                "processedRows": upload.processed_rows,
                "createdAt": upload.created_at.isoformat() if upload.created_at is not None else None
            }
            for upload in uploads
        ]
        
    except Exception as e:
        logger.error(f"Get uploads error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get uploads")

async def process_csv_background(csv_content: str, upload_id: str, csv_type: str = "auto"):
    """Background task to process CSV content"""
    try:
        processor = CSVProcessor()
        await processor.process_csv(csv_content, upload_id, csv_type)
        
    except Exception as e:
        logger.error(f"Background processing error: {e}")
        # Update upload status to failed
        from database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            upload = await db.get(CsvUpload, upload_id)
            if upload:
                # Update upload status to failed
                from sqlalchemy import update
                await db.execute(
                    update(CsvUpload)
                    .where(CsvUpload.id == upload_id)
                    .values(status="failed", error_message=str(e))
                )
                await db.commit()


async def _resolve_orders_upload(csv_upload_id: str, db: AsyncSession) -> Tuple[str, Optional[str], CsvUpload]:
    upload = await db.get(CsvUpload, csv_upload_id)
    if not upload:
        raise HTTPException(status_code=404, detail="Upload not found")

    if upload.csv_type == 'orders':
        return upload.id, upload.run_id, upload

    if not upload.run_id:
        raise HTTPException(status_code=400, detail="Run ID required to resolve associated orders upload")

    query = (
        select(CsvUpload)
        .where(CsvUpload.run_id == upload.run_id, CsvUpload.csv_type == 'orders')
        .order_by(desc(CsvUpload.created_at))
    )
    result = await db.execute(query)
    orders_upload = result.scalars().first()
    if not orders_upload:
        raise HTTPException(status_code=400, detail="No orders upload found for the provided run")

    return orders_upload.id, upload.run_id, upload


async def _get_run_upload(db: AsyncSession, run_id: Optional[str], csv_type: str) -> Optional[CsvUpload]:
    if not run_id:
        return None
    query = (
        select(CsvUpload)
        .where(CsvUpload.run_id == run_id, CsvUpload.csv_type == csv_type)
        .order_by(desc(CsvUpload.created_at))
    )
    result = await db.execute(query)
    return result.scalars().first()


async def _ensure_data_ready(orders_upload_id: str, run_id: Optional[str], db: AsyncSession) -> None:
    mapper = DataMapper()
    enrichment = await mapper.enrich_order_lines_with_variants(orders_upload_id)
    metrics = enrichment.get("metrics", {})

    if not metrics.get("resolved_variants"):
        raise HTTPException(status_code=400, detail="No order lines resolved; verify dataset completeness before continuing")

    if not run_id:
        return

    variant_upload = await _get_run_upload(db, run_id, 'variants')
    catalog_upload = await _get_run_upload(db, run_id, 'catalog_joined')

    if not variant_upload or variant_upload.status != 'completed':
        raise HTTPException(status_code=400, detail="Variants upload missing or incomplete for this run")
    if not catalog_upload or catalog_upload.status != 'completed':
        raise HTTPException(status_code=400, detail="Catalog upload missing or incomplete for this run")

    order_lines = await storage.get_order_lines_by_run(run_id)
    order_skus = {str(getattr(line, 'sku', '')).strip() for line in order_lines if getattr(line, 'sku', None)}

    variant_rows = await storage.get_variants(variant_upload.id)
    variant_skus = {getattr(v, 'sku') for v in variant_rows if getattr(v, 'sku', None)}

    catalog_map = await storage.get_catalog_snapshots_map(catalog_upload.id)
    catalog_skus = set(catalog_map.keys())

    missing_variant_skus = sorted(sku for sku in order_skus if sku not in variant_skus)
    missing_catalog_skus = sorted(sku for sku in order_skus if sku not in catalog_skus)

    if missing_variant_skus:
        sample = ", ".join(missing_variant_skus[:5])
        raise HTTPException(
            status_code=400,
            detail=f"Variants missing for SKUs: {sample} (total {len(missing_variant_skus)})"
        )

    if missing_catalog_skus:
        sample = ", ".join(missing_catalog_skus[:5])
        raise HTTPException(
            status_code=400,
            detail=f"Catalog entries missing for SKUs: {sample} (total {len(missing_catalog_skus)})"
        )

@router.post("/generate-rules")
async def generate_rules(request: dict, db: AsyncSession = Depends(get_db)):
    """Generate association rules from uploaded data"""
    try:
        csv_upload_id = request.get("csvUploadId")
        if not csv_upload_id:
            raise HTTPException(status_code=400, detail="csvUploadId is required")
        orders_upload_id, run_id, _ = await _resolve_orders_upload(csv_upload_id, db)
        await _ensure_data_ready(orders_upload_id, run_id, db)

        # Initialize services
        association_engine = AssociationRulesEngine()
        # Generate association rules
        logger.info(f"Starting association rules generation for upload: {orders_upload_id}")
        await association_engine.generate_association_rules(orders_upload_id)
        
        logger.info("Association rules generation completed")
        return {
            "success": True,
            "message": "Association rules generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Association rules generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate association rules: {str(e)}")

@router.post("/generate-bundles")
async def generate_bundles(request: dict, db: AsyncSession = Depends(get_db)):
    """Generate bundle recommendations (automatically runs association rules first)"""
    try:
        csv_upload_id = request.get("csvUploadId")
        if not csv_upload_id:
            raise HTTPException(status_code=400, detail="csvUploadId is required")
        orders_upload_id, run_id, source_upload = await _resolve_orders_upload(csv_upload_id, db)
        await _ensure_data_ready(orders_upload_id, run_id, db)

        # Initialize services
        association_engine = AssociationRulesEngine()
        bundle_generator = BundleGenerator()
        
        # Step 1: Generate association rules first
        logger.info(f"Starting association rules generation for upload: {orders_upload_id}")
        await association_engine.generate_association_rules(orders_upload_id)
        logger.info("Association rules generation completed")
        
        # Step 2: Generate bundle recommendations
        target_upload_id = orders_upload_id if source_upload.csv_type != 'orders' else csv_upload_id
        logger.info(f"Starting bundle recommendations generation for upload: {target_upload_id}")
        result = await bundle_generator.generate_bundle_recommendations(target_upload_id)
        logger.info("Bundle recommendations generation completed")
        
        return {
            "success": True,
            "message": "Bundle recommendations generated successfully",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Bundle generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate bundles: {str(e)}")
