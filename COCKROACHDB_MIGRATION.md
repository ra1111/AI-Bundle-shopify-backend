# CockroachDB Migration Guide

## Overview
This document details the migration from Google Cloud SQL (PostgreSQL) to CockroachDB for the AI Bundle Creator Python server.

## What Changed

### 1. Database Connection ([database.py](database.py))
- **SSL Handling**: Modified to handle CockroachDB's SSL requirements via `connect_args`
- **JSON Codec Patch**: Added monkey-patch to handle CockroachDB's lack of `json` type (only supports `jsonb`)
- **Version String Patch**: Added patch to parse CockroachDB version strings (format differs from PostgreSQL)
- **Python 3.9 Compatibility**: Changed type annotations from `Type | None` to `Optional[Type]`

### 2. Dependencies ([requirements.txt](requirements.txt))
- Made `pandas` and `numpy` versions more flexible to support Python 3.9

### 3. Environment Configuration ([.env](.env))
- Updated `DATABASE_URL` to point to CockroachDB cluster

## CockroachDB Connection Details

```bash
Host: lively-lobster-17738.j77.aws-ap-south-1.cockroachlabs.cloud
Port: 26257
Database: defaultdb
User: rahula
Region: AWS ap-south-1 (Mumbai)
```

**Connection String:**
```
postgresql://rahula:QChHu3o5tJn7zOZXhFmUhA@lively-lobster-17738.j77.aws-ap-south-1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full
```

## Key Technical Changes

### Compatibility Patches

**1. JSON Codec Issue**
- **Problem**: CockroachDB doesn't have the PostgreSQL `json` type, only `jsonb`
- **Solution**: Monkey-patched `PGDialect_asyncpg.setup_asyncpg_json_codec()` to only set up JSONB codec

**2. Version Detection Issue**
- **Problem**: CockroachDB version string format: `CockroachDB CCL v25.2.6 ...` doesn't match PostgreSQL regex
- **Solution**: Monkey-patched `PGDialect._get_server_version_info()` to detect and parse CockroachDB versions

**3. SSL Configuration**
- **Problem**: CockroachDB requires SSL, asyncpg needs `ssl='require'` parameter
- **Solution**: Added `ssl: "require"` to `connect_args` in engine configuration

### Database Models
- All existing SQLAlchemy models are compatible with CockroachDB
- CockroachDB supports all PostgreSQL data types we use: `VARCHAR`, `TEXT`, `INTEGER`, `NUMERIC`, `TIMESTAMP`, `BOOLEAN`, `JSONB`
- Indexes and constraints work as expected

## Testing

### Local Testing
Run the connection test:
```bash
source venv/bin/activate
python test_cockroach_connection.py
```

**Expected Output:**
```
✓ Successfully connected to CockroachDB!
✓ Created 12 tables: association_rules, bundle_recommendations, bundles, catalog_snapshot, csv_uploads, inventory_levels, order_lines, orders, products, shop_sync_status, users, variants
✓ Database initialization completed successfully!
```

### Verify Tables
All 12 tables were successfully created:
1. users
2. csv_uploads
3. orders
4. order_lines
5. products
6. variants
7. inventory_levels
8. catalog_snapshot
9. association_rules
10. bundles
11. bundle_recommendations
12. shop_sync_status

## Deployment Instructions

### Option 1: Deploy to Cloud Run (Google Cloud)

1. **Set Environment Variables in Cloud Run:**
   ```bash
   gcloud run services update YOUR_SERVICE_NAME \
     --set-env-vars="DATABASE_URL=postgresql://rahula:QChHu3o5tJn7zOZXhFmUhA@lively-lobster-17738.j77.aws-ap-south-1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full" \
     --region=YOUR_REGION
   ```

2. **Build and Deploy:**
   ```bash
   # Build Docker image
   docker build -t gcr.io/YOUR_PROJECT_ID/ai-bundle-creator:latest .

   # Push to Google Container Registry
   docker push gcr.io/YOUR_PROJECT_ID/ai-bundle-creator:latest

   # Deploy to Cloud Run
   gcloud run deploy ai-bundle-creator \
     --image gcr.io/YOUR_PROJECT_ID/ai-bundle-creator:latest \
     --platform managed \
     --region YOUR_REGION \
     --allow-unauthenticated
   ```

### Option 2: Deploy with Docker Compose

1. **Create docker-compose.yml:**
   ```yaml
   version: '3.8'
   services:
     api:
       build: .
       ports:
         - "8080:8080"
       environment:
         - DATABASE_URL=postgresql://rahula:QChHu3o5tJn7zOZXhFmUhA@lively-lobster-17738.j77.aws-ap-south-1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full
         - NODE_ENV=production
         - PORT=8080
       restart: unless-stopped
   ```

2. **Start the service:**
   ```bash
   docker-compose up -d
   ```

### Option 3: Manual Deployment

1. **Set up Python environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export DATABASE_URL="postgresql://rahula:QChHu3o5tJn7zOZXhFmUhA@lively-lobster-17738.j77.aws-ap-south-1.cockroachlabs.cloud:26257/defaultdb?sslmode=verify-full"
   export NODE_ENV="production"
   export PORT=8080
   ```

3. **Run the application:**
   ```bash
   gunicorn main:app \
     -w 4 \
     -k uvicorn.workers.UvicornWorker \
     --bind 0.0.0.0:8080 \
     --timeout 120 \
     --access-logfile - \
     --error-logfile -
   ```

## Environment Variables Reference

Required:
- `DATABASE_URL` - CockroachDB connection string
- `OPENAI_API_KEY` - OpenAI API key for AI copy generation
- `SESSION_SECRET` - Secret for session management
- `ADMIN_API_KEY` - Admin API authentication key

Optional:
- `NODE_ENV` - Environment (development/production)
- `PORT` - Server port (default: 8080)
- `CORS_ORIGINS` - Comma-separated allowed origins
- `INIT_DB_ON_STARTUP` - Auto-create tables on startup (default: false)

## Verification Checklist

After deployment, verify:
- [ ] Application starts successfully
- [ ] Health endpoint responds: `GET /api/health`
- [ ] Database tables exist (12 tables)
- [ ] CSV upload works: `POST /api/upload-csv`
- [ ] Bundle generation works: `POST /api/generate-bundles`
- [ ] Recommendations retrieval works: `GET /api/bundle-recommendations`

## Performance Notes

CockroachDB Performance Characteristics:
- **Distributed**: Data is automatically replicated across multiple regions
- **ACID Compliance**: Full transactional guarantees
- **Horizontal Scaling**: Scales across multiple nodes
- **Connection Pooling**: Using 10 base connections, 20 overflow (as configured)
- **Latency**: Cross-region queries may have higher latency than single-region GCP SQL

## Rollback Plan

If issues arise, you can quickly roll back to Google Cloud SQL:

1. Update `.env` or environment variables with old `DATABASE_URL`
2. Restart the application
3. The code still supports standard PostgreSQL

## Monitoring

Monitor these metrics in production:
- Database connection pool utilization
- Query latency (especially cross-region queries)
- Error rates (watch for SSL/connection errors)
- CockroachDB Console: https://cockroachlabs.cloud/

## Support

For CockroachDB-specific issues:
- CockroachDB Docs: https://www.cockroachlabs.com/docs/
- Support: Through CockroachDB Cloud console

## Cost Optimization

CockroachDB pricing is based on:
- Storage used
- Request units (RUs)
- Data transfer

To optimize costs:
- Use appropriate indexes
- Batch operations where possible
- Monitor RU consumption in CockroachDB console
- Consider using connection pooling (already configured)

## Security Notes

**Current Configuration:**
- SSL/TLS enabled (required by CockroachDB)
- Connection string contains credentials - keep `.env` file secure
- For production, consider using secret management (GCP Secret Manager, AWS Secrets Manager, etc.)

**Recommendations:**
- Rotate database password regularly
- Use IAM-based authentication if available
- Restrict database access by IP if possible
- Monitor access logs in CockroachDB console
