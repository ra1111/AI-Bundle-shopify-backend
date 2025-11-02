"""
Shared helpers for working with OpenAI-powered LLM flows.

Environment variables:
    OPENAI_API_KEY                 → required for live calls
    EMBED_MODEL / EMBED_DIM        → embedding model override + expected dimension
    EMBED_BATCH                    → embedding batch size
    EMBED_RETRY_MAX                → retry budget for embeddings + other calls
    EMBED_RETRY_BACKOFF_BASE_S     → base backoff seconds
    LLM_COMPLETION_MODEL           → chat/completions model for copy/explainability
    LLM_COMPLETION_MAX_TOKENS      → token limit for completion calls
    LLM_COMPLETION_TEMPERATURE     → sampling temperature for copy/explainability

Centralises configuration, client creation and logging so that every part of
the pipeline (embeddings, copy generation, explainability, etc.) behaves
consistently and can be tuned from environment variables.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Coroutine, Optional, TypeVar

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMSettings:
    """Resolved configuration for all LLM touchpoints."""

    api_key: str
    embedding_model: str
    embedding_dim: int
    embedding_batch_size: int
    completion_model: str
    completion_max_tokens: int
    completion_temperature: float
    retry_max: int
    retry_backoff_base_s: float


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid int for %s=%s; falling back to %s", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid float for %s=%s; falling back to %s", name, value, default)
        return default


@lru_cache(maxsize=1)
def load_settings() -> LLMSettings:
    """Load and cache LLM configuration from environment variables."""

    raw_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not raw_key:
        logger.warning("OPENAI_API_KEY not configured; LLM features will fallback")

    embedding_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    embedding_dim = _env_int("EMBED_DIM", 1536)
    embedding_batch_size = _env_int("EMBED_BATCH", 100)

    completion_model = os.getenv("LLM_COMPLETION_MODEL", "gpt-4o-mini")
    completion_max_tokens = _env_int("LLM_COMPLETION_MAX_TOKENS", 600)
    completion_temperature = _env_float("LLM_COMPLETION_TEMPERATURE", 0.7)

    retry_max = _env_int("EMBED_RETRY_MAX", 3)
    retry_backoff_base_s = _env_float("EMBED_RETRY_BACKOFF_BASE_S", 0.5)

    return LLMSettings(
        api_key=raw_key,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedding_batch_size=embedding_batch_size,
        completion_model=completion_model,
        completion_max_tokens=completion_max_tokens,
        completion_temperature=completion_temperature,
        retry_max=retry_max,
        retry_backoff_base_s=retry_backoff_base_s,
    )


@lru_cache(maxsize=1)
def get_async_client() -> AsyncOpenAI:
    """Return a singleton AsyncOpenAI client shared across services."""

    settings = load_settings()
    return AsyncOpenAI(api_key=settings.api_key or None)


def should_use_llm() -> bool:
    """Quick check to see if we have an API key configured."""

    return bool(load_settings().api_key)


T = TypeVar("T")


async def run_with_common_errors(
    operation: str,
    coro_factory: Callable[[], Coroutine[None, None, T]],
    *,
    on_error: Optional[Callable[[Exception], Optional[T]]] = None,
) -> Optional[T]:
    """
    Await an OpenAI coroutine and capture/log failures consistently.

    Args:
        operation: High-level description (e.g. "embedding batch").
        coro_factory: Callable returning the coroutine to await.
        on_error: Optional callback producing a fallback result.
    """

    try:
        return await coro_factory()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("LLM %s failed: %s", operation, exc, exc_info=True)
        if on_error:
            try:
                return on_error(exc)
            except Exception as fallback_exc:  # pragma: no cover
                logger.error("LLM fallback for %s failed: %s", operation, fallback_exc, exc_info=True)
        return None
