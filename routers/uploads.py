"""
CSV Upload Router
Handles file uploads and processing
"""
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends, Form
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import List, Optional
import logging
import uuid
from datetime import datetime

from database import get_db, CsvUpload
from services.csv_processor import CSVProcessor
from services.association_rules_engine import AssociationRulesEngine
from services.bundle_generator import BundleGenerator

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/upload-csv")
async def upload_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    csvType: Optional[str] = Form(None),
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
        csv_upload = CsvUpload(
            id=upload_id,
            filename=file.filename,
            csv_type=normalized_type,
            total_rows=0,
            processed_rows=0,
            status="processing",
            error_message=None
        )
        db.add(csv_upload)
        await db.commit()
        logger.info(f"[{request_id}] Created CsvUpload id={upload_id}")

        # hand off to background (use the normalized type exactly once)
        csv_content = content.decode('utf-8', errors="replace")
        background_tasks.add_task(process_csv_background, csv_content, upload_id, normalized_type)
        logger.info(f"[{request_id}] Background task scheduled for id={upload_id}")

        return {"uploadId": upload_id, "status": "processing", "requestId": request_id}

    except HTTPException as he:
        logger.error(f"[{request_id}] HTTP {he.status_code} during upload: {he.detail}")
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Upload error")
        # Return the message to caller during debugging; change to generic later
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/upload-status/{upload_id}")
async def get_upload_status(upload_id: str, db: AsyncSession = Depends(get_db)):
    """Get upload processing status"""
    try:
        result = await db.get(CsvUpload, upload_id)
        if not result:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        return {
            "id": result.id,
            "filename": result.filename,
            "csvType": result.csv_type,
            "status": result.status,
            "totalRows": result.total_rows,
            "processedRows": result.processed_rows,
            "errorMessage": result.error_message
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

@router.post("/generate-rules")
async def generate_rules(request: dict, db: AsyncSession = Depends(get_db)):
    """Generate association rules from uploaded data"""
    try:
        csv_upload_id = request.get("csvUploadId")
        if not csv_upload_id:
            raise HTTPException(status_code=400, detail="csvUploadId is required")
            
        # Initialize services
        association_engine = AssociationRulesEngine()
        
        # Generate association rules
        logger.info(f"Starting association rules generation for upload: {csv_upload_id}")
        await association_engine.generate_association_rules(csv_upload_id)
        
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
            
        # Initialize services
        association_engine = AssociationRulesEngine()
        bundle_generator = BundleGenerator()
        
        # Step 1: Generate association rules first
        logger.info(f"Starting association rules generation for upload: {csv_upload_id}")
        await association_engine.generate_association_rules(csv_upload_id)
        logger.info("Association rules generation completed")
        
        # Step 2: Generate bundle recommendations
        logger.info(f"Starting bundle recommendations generation for upload: {csv_upload_id}")
        result = await bundle_generator.generate_bundle_recommendations(csv_upload_id)
        logger.info("Bundle recommendations generation completed")
        
        return {
            "success": True,
            "message": "Bundle recommendations generated successfully",
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Bundle generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate bundles: {str(e)}")
