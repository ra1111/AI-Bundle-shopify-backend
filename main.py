"""
FastAPI Application Entry Point
AI Bundle Creator - Python Backend
"""
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import JSONResponse
import os
from dotenv import load_dotenv
import logging

from routers import (
    uploads,
    association_rules, 
    bundle_recommendations,
    bundles,
    analytics,
    export
)
# Import admin routes from routes directory
from routes.admin_routes import router as admin_router
from database import init_db

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI Bundle Creator API",
    description="AI-powered bundle creation system for e-commerce",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add middleware
app.add_middleware(
    SessionMiddleware,
    secret_key=os.getenv("SESSION_SECRET") or os.urandom(32).hex(),
    max_age=86400 * 7,  # 7 days
    same_site="lax",
    https_only=os.getenv("NODE_ENV") == "production"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "ai-bundle-creator"}

# Validation error handler for 422 errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    logger.error(f"Request body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "error": "Validation failed"}
    )

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

# Include routers
app.include_router(uploads.router, prefix="/api", tags=["uploads"])
app.include_router(association_rules.router, prefix="/api", tags=["association-rules"])
app.include_router(bundle_recommendations.router, prefix="/api", tags=["bundle-recommendations"])
app.include_router(bundles.router, prefix="/api", tags=["bundles"])
app.include_router(analytics.router, prefix="/api", tags=["analytics"])
app.include_router(export.router, prefix="/api", tags=["export"])
app.include_router(admin_router, prefix="/api", tags=["admin"])

# Initialize database on startup
@app.on_event("startup")
async def startup():
    logger.info("Starting AI Bundle Creator API...")
    await init_db()
    logger.info("Database initialized successfully")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down AI Bundle Creator API...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("NODE_ENV") != "production"
    )