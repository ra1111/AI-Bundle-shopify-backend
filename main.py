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
import asyncio
from dotenv import load_dotenv
from routes.admin_routes import router as admin_router
from starlette.middleware.base import BaseHTTPMiddleware
import logging
import json
import sys
import time
import uuid as _uuid
from typing import Callable, Dict
from routers import (
    uploads,
    association_rules,
    bundle_recommendations,
    bundles,
    analytics,
    export,
    shopify_upload,
    generation_progress,
    bundle_status,
)

from database import init_db, check_db_health, get_pool_status
from collections import defaultdict
from asyncio import Lock

# Load environment variables
load_dotenv()


# ---- Logging setup (JSON; good for Cloud Run) ----
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "severity": record.levelname,
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include traceback if present
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())

root = logging.getLogger()
root.handlers = [handler]
root.setLevel(LOG_LEVEL)

# Optional: crank down noisy libs
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)  # bump to INFO to see SQL


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
    max_age=86400 * 7,
    same_site="lax",
    https_only=os.getenv("NODE_ENV") == "production"
)

cors_origins_raw = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5000")
cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
if not cors_origins:
    cors_origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---- Request/Response logging middleware ----
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-Id") or str(_uuid.uuid4())
        start = time.time()

        # Attach request_id so handlers can use it
        request.state.request_id = request_id

        # Log request (avoid reading full body for large uploads)
        logger.info(
            f"REQ {request.method} {request.url.path} "
            f"qs={request.url.query!s} ip={request.client.host if request.client else '-'} "
            f"rid={request_id} ua={request.headers.get('user-agent','-')}"
        )
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(f"Uncaught exception in request pipeline rid={request_id}")
            raise

        dur_ms = int((time.time() - start) * 1000)
        logger.info(
            f"RES {request.method} {request.url.path} "
            f"status={response.status_code} durMs={dur_ms} rid={request_id}"
        )
        # Make request id visible to clients
        response.headers["X-Request-Id"] = request_id
        return response

app.add_middleware(RequestIDMiddleware)


# ---- Rate Limiting Middleware ----
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple in-memory rate limiter.
    Limits requests per IP per time window.
    """
    def __init__(self, app, requests_per_minute: int = 60, burst_size: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.window_seconds = 60
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = Lock()
        # Endpoints exempt from rate limiting
        self._exempt_paths = {"/healthz", "/api/health", "/", "/api/docs", "/api/redoc", "/api/openapi.json"}

    async def dispatch(self, request: Request, call_next: Callable):
        # Skip rate limiting for health checks and docs
        if request.url.path in self._exempt_paths:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        async with self._lock:
            # Clean old requests outside window
            self._requests[client_ip] = [
                t for t in self._requests[client_ip]
                if current_time - t < self.window_seconds
            ]

            # Check if over limit
            if len(self._requests[client_ip]) >= self.requests_per_minute:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Rate limit exceeded",
                        "retry_after_seconds": self.window_seconds,
                    },
                    headers={"Retry-After": str(self.window_seconds)},
                )

            # Check burst limit (too many in very short time)
            recent_requests = [t for t in self._requests[client_ip] if current_time - t < 1]
            if len(recent_requests) >= self.burst_size // 10:  # 10 requests per second max
                logger.warning(f"Burst limit exceeded for IP: {client_ip}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Too many requests in short time",
                        "retry_after_seconds": 1,
                    },
                    headers={"Retry-After": "1"},
                )

            # Record this request
            self._requests[client_ip].append(current_time)

        response = await call_next(request)

        # Add rate limit headers
        remaining = self.requests_per_minute - len(self._requests.get(client_ip, []))
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(int(current_time + self.window_seconds))

        return response


# Add rate limiting (configurable via env)
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "120"))  # 120 requests per minute default

if RATE_LIMIT_ENABLED:
    app.add_middleware(RateLimitMiddleware, requests_per_minute=RATE_LIMIT_RPM)
    logger.info(f"Rate limiting enabled: {RATE_LIMIT_RPM} requests/minute")


@app.get("/")
async def root():
    return {"ok": True, "service": "ai-bundle-creator"}

@app.get("/healthz")
async def healthz():
    """Basic health check for load balancers."""
    return {"ok": True}


@app.get("/api/health")
async def api_health():
    """Comprehensive health check including database status."""
    db_health = await check_db_health()
    overall_status = "healthy" if db_health["status"] == "healthy" else "degraded"
    return {
        "status": overall_status,
        "database": db_health,
        "timestamp": time.time(),
    }


@app.get("/api/health/pool")
async def api_health_pool():
    """Get database connection pool status."""
    pool_status = await get_pool_status()
    return {
        "pool": pool_status,
        "timestamp": time.time(),
    }



# --- Error handlers ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    logger.error(f"Request body: {await request.body()}")
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "error": "Validation failed"})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

# --- Routers ---
app.include_router(uploads.router, prefix="/api", tags=["uploads"])
app.include_router(association_rules.router, prefix="/api", tags=["association-rules"])
app.include_router(bundle_recommendations.router, prefix="/api", tags=["bundle-recommendations"])
app.include_router(bundles.router, prefix="/api", tags=["bundles"])
app.include_router(analytics.router, prefix="/api", tags=["analytics"])
app.include_router(export.router, prefix="/api", tags=["export"])
app.include_router(generation_progress.router, prefix="/api", tags=["generation-progress"])
app.include_router(bundle_status.router)
app.include_router(shopify_upload.router)
app.include_router(admin_router, prefix="/api", tags=["admin"])

# --- Startup/shutdown ---
@app.on_event("startup")
async def startup():
    logger.info("Starting AI Bundle Creator API...")
    # Don't initialize DB on Cloud Run startup - do it separately
    if os.getenv("INIT_DB_ON_STARTUP", "false").lower() == "true":
        try:
            logger.info("Initializing database tables...")
            await asyncio.wait_for(init_db(), timeout=120)  # 2 minutes for slow connections
            logger.info("✅ Database initialized successfully")
        except asyncio.TimeoutError:
            logger.error("❌ DB init timed out after 120s, continuing without init")
        except Exception as e:
            logger.error(f"❌ DB init failed (continuing to serve): {e}", exc_info=True)
    else:
        logger.info("Skipping DB init on startup")
@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down AI Bundle Creator API...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8080)),
        reload=os.getenv("NODE_ENV") != "production"
    )
