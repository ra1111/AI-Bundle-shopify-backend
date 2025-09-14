"""
Bundles Router
Handles active bundle management
"""
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
import uuid

from database import get_db, Bundle, BundleRecommendation

logger = logging.getLogger(__name__)
router = APIRouter()

class CreateBundleRequest(BaseModel):
    recommendationId: str

@router.post("/bundles")
async def create_bundle(
    request: CreateBundleRequest,
    db: AsyncSession = Depends(get_db)
):
    """Create bundle from recommendation"""
    try:
        # Get recommendation
        recommendation = await db.get(BundleRecommendation, request.recommendationId)
        if not recommendation:
            raise HTTPException(status_code=404, detail="Recommendation not found")
        
        # Create bundle from recommendation
        ai_copy = recommendation.ai_copy if recommendation.ai_copy is not None else {}
        bundle = Bundle(
            id=str(uuid.uuid4()),
            name=ai_copy.get("title", f"Bundle {recommendation.bundle_type}"),
            description=ai_copy.get("description", "Generated bundle"),
            bundle_type=recommendation.bundle_type,
            products=recommendation.products,
            pricing=recommendation.pricing,
            is_active=True
        )
        
        db.add(bundle)
        
        # Mark recommendation as approved
        from sqlalchemy import update
        await db.execute(
            update(BundleRecommendation)
            .where(BundleRecommendation.id == request.recommendationId)
            .values(is_approved=True)
        )
        
        await db.commit()
        
        return {
            "success": True,
            "message": "Bundle created successfully",
            "bundleId": bundle.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create bundle error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create bundle")

@router.get("/bundles")
async def get_bundles(db: AsyncSession = Depends(get_db)):
    """Get list of active bundles"""
    try:
        from sqlalchemy import select
        
        query = select(Bundle).order_by(Bundle.created_at.desc())
        result = await db.execute(query)
        bundles = result.scalars().all()
        
        return [
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
        
    except Exception as e:
        logger.error(f"Get bundles error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get bundles")