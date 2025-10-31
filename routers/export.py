"""
Export Router
Handles data export functionality
"""
from fastapi import APIRouter, HTTPException, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional
import logging
import json
import csv
import io
from datetime import datetime

from database import get_db, Bundle, BundleRecommendation
from settings import resolve_shop_id

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/export/bundles")
async def export_bundles(db: AsyncSession = Depends(get_db)):
    """Export bundles as JSON"""
    try:
        from sqlalchemy import select
        
        query = select(Bundle).where(Bundle.is_active == True)
        result = await db.execute(query)
        bundles = result.scalars().all()
        
        export_data = {
            "exportDate": datetime.now().isoformat(),
            "bundleCount": len(bundles),
            "bundles": [
                {
                    "id": bundle.id,
                    "name": bundle.name,
                    "description": bundle.description,
                    "bundleType": bundle.bundle_type,
                    "products": bundle.products,
                    "pricing": bundle.pricing,
                    "isActive": bundle.is_active,
                    "createdAt": bundle.created_at.isoformat() if bundle.created_at is not None else None,
                    "updatedAt": bundle.updated_at.isoformat() if bundle.updated_at is not None else None
                }
                for bundle in bundles
            ]
        }
        
        json_content = json.dumps(export_data, indent=2)
        
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=bundles.json"}
        )
        
    except Exception as e:
        logger.error(f"Export bundles error: {e}")
        raise HTTPException(status_code=500, detail="Failed to export bundles")

@router.get("/export/recommendations")
async def export_recommendations(
    shopId: Optional[str] = None,
    uploadId: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Export recommendations as JSON"""
    try:
        shop_id = resolve_shop_id(shopId)
        from sqlalchemy import select
        
        query = select(BundleRecommendation).where(BundleRecommendation.shop_id == shop_id)
        if uploadId:
            query = query.where(BundleRecommendation.csv_upload_id == uploadId)
        
        result = await db.execute(query)
        recommendations = result.scalars().all()
        
        export_data = {
            "exportDate": datetime.now().isoformat(),
            "uploadId": uploadId,
            "shopId": shop_id,
            "recommendationCount": len(recommendations),
            "recommendations": [
                {
                    "id": rec.id,
                    "csvUploadId": rec.csv_upload_id,
                    "bundleType": rec.bundle_type,
                    "products": rec.products,
                    "pricing": rec.pricing,
                    "aiCopy": rec.ai_copy,
                    "confidence": str(rec.confidence),
                    "predictedLift": str(rec.predicted_lift),
                    "support": str(rec.support) if rec.support is not None else None,
                    "lift": str(rec.lift) if rec.lift is not None else None,
                    "isApproved": getattr(rec, 'is_approved', False),
                    "createdAt": rec.created_at.isoformat() if rec.created_at is not None else None
                }
                for rec in recommendations
            ]
        }
        
        json_content = json.dumps(export_data, indent=2)
        filename = f"bundle-recommendations{'-' + uploadId if uploadId else ''}.json"
        
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Export recommendations error: {e}")
        raise HTTPException(status_code=500, detail="Failed to export recommendations")

@router.get("/recommendations/{upload_id}/export")
async def export_csv_recommendations(
    upload_id: str,
    shopId: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Export CSV-specific recommendations as CSV format"""
    try:
        shop_id = resolve_shop_id(shopId)
        from sqlalchemy import select
        
        query = select(BundleRecommendation).where(
            BundleRecommendation.csv_upload_id == upload_id,
        )
        query = query.where(BundleRecommendation.shop_id == shop_id)
        result = await db.execute(query)
        recommendations = result.scalars().all()
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found for this upload")
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # CSV headers
        writer.writerow([
            "ID", "Bundle Type", "Confidence", "Predicted Lift", 
            "Products", "AI Title", "Is Approved", "Created At"
        ])
        
        # CSV data rows
        for rec in recommendations:
            ai_copy = getattr(rec, 'ai_copy', None) or {}
            products_str = ""
            
            if isinstance(rec.products, list):
                product_names = []
                for product in rec.products:
                    if isinstance(product, dict) and 'name' in product:
                        product_names.append(product['name'])
                products_str = "; ".join(product_names)
            
            writer.writerow([
                rec.id,
                rec.bundle_type,
                str(rec.confidence),
                str(rec.predicted_lift),
                products_str,
                ai_copy.get("title", ""),
                "Yes" if getattr(rec, 'is_approved', False) else "No",
                rec.created_at.isoformat() if rec.created_at is not None else ""
            ])
        
        csv_content = output.getvalue()
        output.close()
        
        filename = f"recommendations-{upload_id}-{datetime.now().strftime('%Y%m%d')}.csv"
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export CSV recommendations error: {e}")
        raise HTTPException(status_code=500, detail="Failed to export CSV recommendations")
