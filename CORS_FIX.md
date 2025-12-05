# CORS Configuration Fix

## Problem
Frontend polling requests from Cloudflare Tunnel were returning **400 Bad Request** on OPTIONS preflight:
```
OPTIONS /api/shopify/status/{upload_id} → 400
OPTIONS /api/generation-progress/{upload_id} → 400
```

**Root Cause:** The Cloudflare tunnel URL (`https://pest-template-nov-not.trycloudflare.com`) was not in the `CORS_ORIGINS` environment variable.

## Solution
Updated Cloud Run environment variable to include the frontend origin:

```bash
gcloud run services update bundle-api --region=us-central1 \
  --update-env-vars "^@^CORS_ORIGINS=http://localhost:3000,http://localhost:5000,https://pest-template-nov-not.trycloudflare.com,https://admin.shopify.com"
```

## Current CORS Configuration
```
http://localhost:3000
http://localhost:5000
https://pest-template-nov-not.trycloudflare.com
https://admin.shopify.com
```

## Testing
Run the test script to verify CORS is working:
```bash
./test_cors.sh
```

Expected output:
```
HTTP/2 200
access-control-allow-origin: https://pest-template-nov-not.trycloudflare.com
access-control-allow-methods: DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT
access-control-allow-credentials: true
```

## If Cloudflare Tunnel URL Changes

Cloudflare tunnels generate random URLs like `https://xyz-abc-123.trycloudflare.com`. If your tunnel URL changes:

### Option 1: Update CORS_ORIGINS (Quick Fix)
```bash
# Replace NEW_TUNNEL_URL with your new Cloudflare tunnel URL
gcloud run services update bundle-api --region=us-central1 \
  --update-env-vars "^@^CORS_ORIGINS=http://localhost:3000,http://localhost:5000,https://NEW_TUNNEL_URL,https://admin.shopify.com"
```

### Option 2: Add Wildcard CORS Support (Better Solution)

FastAPI's built-in `CORSMiddleware` doesn't support wildcard patterns like `https://*.trycloudflare.com`.

To support dynamic tunnel URLs, you need to implement custom CORS logic. Add this to [main.py](main.py):

```python
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import re

class CustomCORSMiddleware(BaseHTTPMiddleware):
    """Allow dynamic CORS for Cloudflare tunnels"""

    ALLOWED_PATTERNS = [
        r"^http://localhost:\d+$",
        r"^https://.*\.trycloudflare\.com$",
        r"^https://admin\.shopify\.com$",
    ]

    async def dispatch(self, request, call_next):
        origin = request.headers.get("origin")

        # Check if origin matches any allowed pattern
        is_allowed = False
        if origin:
            for pattern in self.ALLOWED_PATTERNS:
                if re.match(pattern, origin):
                    is_allowed = True
                    break

        # Handle OPTIONS (preflight)
        if request.method == "OPTIONS" and is_allowed:
            return Response(
                content="OK",
                headers={
                    "Access-Control-Allow-Origin": origin,
                    "Access-Control-Allow-Methods": "DELETE, GET, HEAD, OPTIONS, PATCH, POST, PUT",
                    "Access-Control-Allow-Headers": "*",
                    "Access-Control-Allow-Credentials": "true",
                    "Access-Control-Max-Age": "600",
                }
            )

        # Process request
        response = await call_next(request)

        # Add CORS headers to response
        if is_allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Vary"] = "Origin"

        return response

# In main.py, replace the CORSMiddleware with:
# app.add_middleware(CustomCORSMiddleware)
```

**Note:** The current fixed CORS_ORIGINS approach is sufficient for now. Only implement custom CORS if you frequently change tunnel URLs.

## Verification Checklist

After any CORS changes:

- [ ] OPTIONS requests return 200 (not 400)
- [ ] Response includes `access-control-allow-origin` header
- [ ] Frontend can poll `/api/shopify/status/{uploadId}`
- [ ] Frontend can poll `/api/generation-progress/{uploadId}`
- [ ] No CORS errors in browser console

## Related Files
- Backend CORS config: [main.py:93-103](main.py#L93-L103)
- Status endpoints:
  - [routers/shopify_upload.py:161](routers/shopify_upload.py#L161) - `/api/shopify/status/{upload_id}`
  - [routers/generation_progress.py:20](routers/generation_progress.py#L20) - `/api/generation-progress/{upload_id}`

## Deployment
The fix was deployed to Cloud Run revision:
```
bundle-api-00142-qtp
Service URL: https://bundle-api-250755735924.us-central1.run.app
```
