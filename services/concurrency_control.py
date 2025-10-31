"""
Production-Ready Concurrency Control Service
Provides per-shop mutex locking and atomic status updates to prevent race conditions
"""
import asyncio
import hashlib
import time
import logging
import random
from typing import Optional, Dict, Any, Callable, Awaitable, Union, List
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession, AsyncConnection
from sqlalchemy import text, update, select
from database import AsyncSessionLocal, CsvUpload, engine
from settings import resolve_shop_id

logger = logging.getLogger(__name__)

class ConcurrencyController:
    """
    Production-ready concurrency control for shop operations including:
    - True per-shop mutex locking using shop_id (extracted from CSV uploads)
    - Locks held during entire generation phase to prevent race conditions
    - Dedicated connection handling to prevent lock leaks
    - True exponential backoff with jitter for lock acquisition
    - Status precondition checks with compare-and-set semantics
    - Proper error handling and automatic cleanup
    """
    
    def __init__(self):
        self.lock_timeout_seconds = 900  # 15 minutes for bundle generation
        self.max_retries = 12  # More retries with exponential backoff
        self.base_retry_delay = 0.25  # Starting retry delay
        self.max_retry_delay = 120.0  # Maximum retry delay (2 minutes)
        self.jitter_factor = 0.3  # Add randomization to prevent thundering herd
        self._active_locks: Dict[str, AsyncConnection] = {}  # Track active lock connections
    
    async def get_shop_id_from_csv_upload(self, csv_upload_id: str) -> Optional[str]:
        """
        Extract shop_id from CSV upload record for proper per-shop locking
        Returns None if CSV upload not found
        """
        if not csv_upload_id:
            return None
            
        db = AsyncSessionLocal()
        try:
            result = await db.execute(
                select(CsvUpload.shop_id).where(CsvUpload.id == csv_upload_id)
            )
            shop_id = result.scalar()
            return resolve_shop_id(shop_id)
        except Exception as e:
            logger.error(f"Failed to get shop_id for CSV upload {csv_upload_id}: {e}")
            return None
        finally:
            await db.close()
    
    def _generate_lock_key(self, shop_id: str, operation: str = "bundle_generation") -> int:
        """
        Generate a unique integer lock key from shop_id and operation
        PostgreSQL advisory locks require integer keys
        """
        # Create a deterministic hash from shop_id + operation
        hash_input = f"shop:{shop_id}:operation:{operation}"
        hash_digest = hashlib.sha256(hash_input.encode()).hexdigest()
        # Convert to 32-bit signed integer (PostgreSQL advisory lock requirement)
        lock_key = int(hash_digest[:8], 16)
        # Ensure positive 32-bit signed integer
        if lock_key > 2147483647:
            lock_key = lock_key - 4294967296
        return abs(lock_key)
    
    async def _acquire_advisory_lock_with_backoff(self, conn: AsyncConnection, lock_key: int, shop_id: str, operation: str) -> bool:
        """
        Acquire PostgreSQL advisory lock with true exponential backoff and jitter
        Returns True if lock acquired, False if timeout
        """
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                # Try to acquire lock (non-blocking)
                result = await conn.execute(
                    text("SELECT pg_try_advisory_lock(:lock_key)"),
                    {"lock_key": lock_key}
                )
                acquired = result.scalar()
                
                if acquired:
                    logger.info(f"Acquired advisory lock {lock_key} for shop {shop_id} operation {operation} on attempt {attempt + 1}")
                    return True
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed >= self.lock_timeout_seconds:
                    logger.warning(f"Lock acquisition timed out after {elapsed:.1f}s for shop {shop_id}")
                    return False
                
                # Calculate exponential backoff delay with jitter
                base_delay = self.base_retry_delay * (2 ** attempt)
                jitter = base_delay * self.jitter_factor * random.random()
                delay = min(base_delay + jitter, self.max_retry_delay)
                
                logger.info(f"Lock {lock_key} busy for shop {shop_id}, retry {attempt + 1}/{self.max_retries} in {delay:.2f}s")
                await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Error acquiring advisory lock {lock_key} for shop {shop_id}: {e}")
                if attempt == self.max_retries - 1:
                    return False
                await asyncio.sleep(self.base_retry_delay)
        
        logger.warning(f"Failed to acquire advisory lock {lock_key} for shop {shop_id} after {self.max_retries} attempts")
        return False
    
    async def _release_advisory_lock(self, conn: AsyncConnection, lock_key: int, shop_id: str) -> bool:
        """
        Release PostgreSQL advisory lock with proper error handling
        Returns True if successfully released
        """
        try:
            result = await conn.execute(
                text("SELECT pg_advisory_unlock(:lock_key)"),
                {"lock_key": lock_key}
            )
            released = result.scalar()
            
            success = bool(released) if released is not None else False
            
            if success:
                logger.info(f"Released advisory lock {lock_key} for shop {shop_id}")
            else:
                logger.warning(f"Failed to release advisory lock {lock_key} for shop {shop_id} (may not have been held)")
            
            return success
            
        except Exception as e:
            logger.error(f"Error releasing advisory lock {lock_key} for shop {shop_id}: {e}")
            return False
    
    @asynccontextmanager
    async def acquire_shop_lock_for_csv_upload(
        self, 
        csv_upload_id: str, 
        operation: str = "bundle_generation"
    ):
        """
        Context manager for acquiring per-shop lock based on CSV upload's shop_id
        Lock is held for the entire duration of the context (entire generation phase)
        
        Usage:
            async with concurrency_controller.acquire_shop_lock_for_csv_upload(csv_upload_id):
                # Bundle generation happens here with lock held
                await generate_bundles()
        """
        if not csv_upload_id:
            raise ValueError("CSV upload ID cannot be empty")
        
        # First, get the shop_id from the CSV upload record
        shop_id = await self.get_shop_id_from_csv_upload(csv_upload_id)
        if not shop_id:
            raise ValueError(f"Cannot find shop_id for CSV upload {csv_upload_id}")
        
        lock_key = self._generate_lock_key(shop_id, operation)
        lock_identifier = f"{shop_id}:{operation}"
        
        # Create dedicated raw connection for advisory lock
        logger.info(f"Creating database connection for {operation} lock (shop: {shop_id}, csv_upload: {csv_upload_id})")
        conn = await engine.connect()
        logger.info(f"Database connection established, attempting to acquire {operation} lock (lock_key: {lock_key})")

        try:
            # Acquire lock with exponential backoff
            acquired = await self._acquire_advisory_lock_with_backoff(conn, lock_key, shop_id, operation)
            
            if not acquired:
                raise TimeoutError(f"Could not acquire {operation} lock for shop {shop_id} (CSV upload {csv_upload_id})")
            
            # Track active lock connection
            self._active_locks[lock_identifier] = conn
            
            logger.info(f"Acquired {operation} lock for shop {shop_id} (CSV upload {csv_upload_id})")
            
            # Yield connection and metadata to the caller
            yield {"conn": conn, "shop_id": shop_id, "csv_upload_id": csv_upload_id}
            
        finally:
            # Always attempt to release lock and close connection
            try:
                await self._release_advisory_lock(conn, lock_key, shop_id)
            except Exception as e:
                logger.error(f"Error during lock release for shop {shop_id}: {e}")
            finally:
                # Remove from active locks tracking
                self._active_locks.pop(lock_identifier, None)
                
                # Close database connection
                try:
                    await conn.close()
                except Exception as e:
                    logger.error(f"Error closing database connection for shop {shop_id}: {e}")
                
                logger.info(f"Released {operation} lock for shop {shop_id} (CSV upload {csv_upload_id})")
    
    async def atomic_status_update_with_precondition(
        self, 
        csv_upload_id: str, 
        new_status: str, 
        expected_current_status: Optional[str] = None,
        additional_fields: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Atomically update CSV upload status with compare-and-set semantics
        Uses short-lived session to prevent nested transaction issues
        
        Args:
            csv_upload_id: CSV upload ID to update
            new_status: New status to set
            expected_current_status: Expected current status (for compare-and-set), None to skip check
            additional_fields: Additional fields to update
            
        Returns:
            Dict with success status, current_status, and details
        """
        # Always create fresh short-lived session to avoid nested transaction issues
        async with AsyncSessionLocal() as db:
            try:
                # Start transaction
                async with db.begin():
                    # First, get current record with FOR UPDATE lock
                    result = await db.execute(
                        select(CsvUpload).where(CsvUpload.id == csv_upload_id)
                        .with_for_update()
                    )
                    csv_upload = result.scalar_one_or_none()
                    
                    if not csv_upload:
                        return {
                            "success": False,
                            "error": f"CSV upload {csv_upload_id} not found",
                            "current_status": None
                        }
                    
                    current_status = csv_upload.status
                    
                    # Check precondition if specified
                    if expected_current_status is not None and current_status != expected_current_status:
                        return {
                            "success": False,
                            "error": f"Status precondition failed: expected '{expected_current_status}', found '{current_status}'",
                            "current_status": current_status
                        }
                    
                    # Prepare update values
                    update_values = {"status": new_status}
                    if additional_fields:
                        update_values.update(additional_fields)
                    
                    # Execute atomic update
                    await db.execute(
                        update(CsvUpload)
                        .where(CsvUpload.id == csv_upload_id)
                        .values(**update_values)
                    )
                    
                    logger.info(f"Atomically updated CSV upload {csv_upload_id} status: '{current_status}' -> '{new_status}'")
                    
                    if additional_fields:
                        field_names = ", ".join(additional_fields.keys())
                        logger.info(f"Also updated fields: {field_names}")
                    
                    return {
                        "success": True,
                        "previous_status": current_status,
                        "new_status": new_status,
                        "csv_upload_id": csv_upload_id
                    }
                    
            except Exception as e:
                logger.error(f"Failed to update CSV upload {csv_upload_id} status: {e}")
                return {
                    "success": False,
                    "error": f"Update failed: {str(e)}",
                    "current_status": None
                }
    
    async def exponential_backoff_retry(
        self,
        operation: Callable[[], Awaitable[Any]],
        operation_name: str,
        max_retries: Optional[int] = None
    ) -> Any:
        """
        Execute operation with true exponential backoff retry logic and jitter
        """
        max_retries = max_retries or self.max_retries
        
        for attempt in range(max_retries + 1):
            try:
                return await operation()
                
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Operation {operation_name} failed after {max_retries} retries: {e}")
                    raise
                
                # Calculate delay with exponential backoff and jitter
                base_delay = self.base_retry_delay * (2 ** attempt)
                jitter = base_delay * self.jitter_factor * random.random()
                delay = min(base_delay + jitter, self.max_retry_delay)
                
                logger.warning(f"Operation {operation_name} failed (attempt {attempt + 1}), retrying in {delay:.2f}s: {e}")
                await asyncio.sleep(delay)
        
        # Should never reach here, but just in case
        raise RuntimeError(f"Unexpected error in retry logic for {operation_name}")
    
    async def get_active_locks(self) -> List[str]:
        """Get list of currently active locks for monitoring"""
        return list(self._active_locks.keys())
    
    async def force_release_lock(self, shop_id: str, operation: str = "bundle_generation") -> bool:
        """Force release a lock (emergency use only)"""
        lock_identifier = f"{shop_id}:{operation}"
        conn = self._active_locks.get(lock_identifier)
        
        if not conn:
            logger.warning(f"No active lock found for {lock_identifier}")
            return False
        
        lock_key = self._generate_lock_key(shop_id, operation)
        success = await self._release_advisory_lock(conn, lock_key, shop_id)
        
        if success:
            self._active_locks.pop(lock_identifier, None)
            await conn.close()
            logger.info(f"Force released lock for {lock_identifier}")
        
        return success

# Global instance for application use
concurrency_controller = ConcurrencyController()
