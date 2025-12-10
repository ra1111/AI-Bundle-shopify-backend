"""
Utility functions for the AI Bundle Creator backend.
Includes retry logic, decorators, and helper functions.
"""
import asyncio
import functools
import logging
import random
from typing import Callable, Type, Tuple, Optional, Any
from sqlalchemy.exc import (
    OperationalError,
    InterfaceError,
    TimeoutError as SQLAlchemyTimeoutError,
    DisconnectionError,
)

logger = logging.getLogger(__name__)

# Transient database errors that should be retried
TRANSIENT_DB_ERRORS: Tuple[Type[Exception], ...] = (
    OperationalError,
    InterfaceError,
    SQLAlchemyTimeoutError,
    DisconnectionError,
    ConnectionError,
    TimeoutError,
)


def is_transient_error(exc: Exception) -> bool:
    """Check if an exception is a transient error that should be retried."""
    # Check direct instance
    if isinstance(exc, TRANSIENT_DB_ERRORS):
        return True

    # Check for common transient error messages
    error_msg = str(exc).lower()
    transient_patterns = [
        "connection refused",
        "connection reset",
        "connection timed out",
        "timeout",
        "too many connections",
        "server closed the connection",
        "connection pool exhausted",
        "could not connect",
        "temporarily unavailable",
        "retry transaction",  # CockroachDB specific
        "restart transaction",  # CockroachDB specific
        "40001",  # Serialization failure (PostgreSQL/CockroachDB)
    ]
    return any(pattern in error_msg for pattern in transient_patterns)


def retry_async(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    exponential_backoff: bool = True,
    jitter: bool = True,
    retry_on: Optional[Tuple[Type[Exception], ...]] = None,
):
    """
    Decorator for async functions that retries on transient failures.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff
        jitter: Whether to add random jitter to delays
        retry_on: Tuple of exception types to retry on (defaults to transient DB errors)
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Check if we should retry
                    should_retry = False
                    if retry_on:
                        should_retry = isinstance(e, retry_on)
                    else:
                        should_retry = is_transient_error(e)

                    if not should_retry or attempt >= max_retries:
                        raise

                    # Calculate delay
                    if exponential_backoff:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay

                    if jitter:
                        delay = delay * (0.5 + random.random())  # 50-150% of delay

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay:.2f}s due to: {type(e).__name__}: {str(e)[:100]}"
                    )
                    await asyncio.sleep(delay)

            # This should never be reached, but just in case
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


def retry_sync(
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 10.0,
    exponential_backoff: bool = True,
    jitter: bool = True,
    retry_on: Optional[Tuple[Type[Exception], ...]] = None,
):
    """
    Decorator for sync functions that retries on transient failures.
    Same as retry_async but for synchronous functions.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import time
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    should_retry = False
                    if retry_on:
                        should_retry = isinstance(e, retry_on)
                    else:
                        should_retry = is_transient_error(e)

                    if not should_retry or attempt >= max_retries:
                        raise

                    if exponential_backoff:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                    else:
                        delay = base_delay

                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                        f"after {delay:.2f}s due to: {type(e).__name__}: {str(e)[:100]}"
                    )
                    time.sleep(delay)

            if last_exception:
                raise last_exception

        return wrapper
    return decorator


async def with_retry(
    coro_func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 0.5,
    **kwargs,
) -> Any:
    """
    Execute an async function with retry logic.

    Example:
        result = await with_retry(some_async_function, arg1, arg2, max_retries=5)
    """
    last_exception = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if not is_transient_error(e) or attempt >= max_retries:
                raise

            delay = min(base_delay * (2 ** attempt), 10.0)
            delay = delay * (0.5 + random.random())
            logger.warning(
                f"Retry {attempt + 1}/{max_retries} after {delay:.2f}s: {type(e).__name__}"
            )
            await asyncio.sleep(delay)

    if last_exception:
        raise last_exception


class CircuitBreaker:
    """
    Simple circuit breaker to prevent cascading failures.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Circuit is tripped, requests fail immediately
    - HALF_OPEN: Testing if service recovered
    """
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function through the circuit breaker."""
        import time

        async with self._lock:
            current_time = time.time()

            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == self.OPEN:
                if current_time - self._last_failure_time >= self.recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_calls = 0
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN. Retry in "
                        f"{self.recovery_timeout - (current_time - self._last_failure_time):.1f}s"
                    )

            # In HALF_OPEN, limit the number of test calls
            if self._state == self.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError("Circuit breaker is in HALF_OPEN, max test calls reached")
                self._half_open_calls += 1

        try:
            result = await func(*args, **kwargs)

            async with self._lock:
                # Success - reset or close circuit
                if self._state == self.HALF_OPEN:
                    self._state = self.CLOSED
                    logger.info("Circuit breaker transitioning to CLOSED")
                self._failure_count = 0

            return result

        except Exception as e:
            async with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.time()

                if self._failure_count >= self.failure_threshold:
                    self._state = self.OPEN
                    logger.warning(
                        f"Circuit breaker transitioning to OPEN after {self._failure_count} failures"
                    )

            raise


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and blocking calls."""
    pass


def sanitize_string(value: Optional[str], max_length: int = 1000, default: str = "") -> str:
    """Sanitize a string value for safe storage."""
    if value is None:
        return default
    # Remove null bytes and other problematic characters
    cleaned = str(value).replace("\x00", "").strip()
    # Truncate if too long
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length]
    return cleaned


def sanitize_dict(data: dict, max_string_length: int = 1000) -> dict:
    """Recursively sanitize all string values in a dictionary."""
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = sanitize_string(value, max_string_length)
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value, max_string_length)
        elif isinstance(value, list):
            result[key] = [
                sanitize_dict(item, max_string_length) if isinstance(item, dict)
                else sanitize_string(item, max_string_length) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            result[key] = value
    return result
