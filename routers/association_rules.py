"""
Association Rules Router
Handles association rules generation and retrieval
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional
import logging

from database import get_db
from services.association_rules_engine import AssociationRulesEngine

logger = logging.getLogger(__name__)
router = APIRouter()

class GenerateRulesRequest(BaseModel):
    csvUploadId: Optional[str] = None

@router.post("/generate-rules")
async def generate_rules(
    request: GenerateRulesRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Generate association rules"""
    try:
        csv_upload_id = request.csvUploadId
        
        # Start background task to generate rules
        background_tasks.add_task(generate_rules_background, csv_upload_id)
        
        scope = f"for CSV upload {csv_upload_id}" if csv_upload_id else "overall"
        return {"success": True, "message": f"Association rules generation started {scope}"}
        
    except Exception as e:
        logger.error(f"Rules generation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate association rules")

@router.get("/association-rules")
async def get_association_rules(
    uploadId: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get association rules with optional CSV filtering"""
    try:
        from sqlalchemy import select
        from database import AssociationRule
        
        query = select(AssociationRule)
        if uploadId:
            query = query.where(AssociationRule.csv_upload_id == uploadId)
        
        query = query.order_by(AssociationRule.lift.desc()).limit(100)
        result = await db.execute(query)
        rules = result.scalars().all()
        
        return [
            {
                "id": rule.id,
                "csvUploadId": rule.csv_upload_id,
                "antecedent": rule.antecedent,
                "consequent": rule.consequent,
                "support": str(rule.support),
                "confidence": str(rule.confidence),
                "lift": str(rule.lift),
                "createdAt": rule.created_at.isoformat() if rule.created_at is not None else None
            }
            for rule in rules
        ]
        
    except Exception as e:
        logger.error(f"Get association rules error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get association rules")

async def generate_rules_background(csv_upload_id: Optional[str]):
    """Background task to generate association rules"""
    try:
        engine = AssociationRulesEngine()
        await engine.generate_association_rules(csv_upload_id)
        
    except Exception as e:
        logger.error(f"Background rules generation error: {e}")
        # Could update status or send notification here