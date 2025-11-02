"""
LLM-based Product Embeddings Engine (v2)

- Replaces item2vec with OpenAI embeddings
- Zero training time, cold-start friendly
- Persistent cache (memory + storage)
- L2-normalized vectors for stable cosine
- Safe batching with retries & dedup
- Configurable similarity thresholds
"""

from __future__ import annotations
import os
import json
import math
import time
import hashlib
import logging
from typing import Any, Dict, List, Tuple, Optional, Iterable

import numpy as np
from openai import AsyncOpenAI

from services.storage import storage  # must expose async get(key)->str|None and set(key, value, ttl=None)

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


class LLMEmbeddingEngine:
    """
    Generate and cache product embeddings using OpenAI's embedding models.

    Public API:
      - get_embedding(product)
      - get_embeddings_batch(products)
      - find_similar_products(target_sku, catalog, embeddings, top_k, min_similarity)
      - compute_bundle_similarity(products, embeddings)
      - generate_candidates_by_similarity(csv_upload_id, bundle_type, objective, catalog, embeddings, num_candidates)
    """

    # ------- defaults / config -------
    DEFAULT_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")  # 1536 dims
    DEFAULT_DIM = int(os.getenv("EMBED_DIM", "1536"))
    BATCH_SIZE = int(os.getenv("EMBED_BATCH", "100"))  # OpenAI supports large inputs; keep modest for stability
    CACHE_NS = os.getenv("EMBED_CACHE_NS", "embeddings:v1")  # bump if _create_product_text changes materially
    CACHE_TTL_S = int(os.getenv("EMBED_CACHE_TTL_S", str(60 * 60 * 24 * 30)))  # 30 days

    # thresholds for candidate builders (env overridable)
    FBT_MIN = float(os.getenv("SIM_FBT_MIN", "0.40"))        # complementary floor
    TOO_SIM = float(os.getenv("SIM_TOO_SIMILAR", "0.70"))    # skip near-duplicates
    VOLUME_MIN = float(os.getenv("SIM_VOLUME_MIN", "0.70"))  # volume (very similar)

    # retry config
    RETRY_MAX = int(os.getenv("EMBED_RETRY_MAX", "3"))
    RETRY_BACKOFF_BASE_S = float(os.getenv("EMBED_RETRY_BACKOFF_BASE_S", "0.5"))

    def __init__(self) -> None:
        raw_key = os.getenv("OPENAI_API_KEY", "")
        api_key = raw_key.strip()  # remove accidental newlines/spaces
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = self.DEFAULT_MODEL
        self.embedding_dim = self.DEFAULT_DIM
        self.batch_size = self.BATCH_SIZE
        self.normalize = True
        self._memory_cache: Dict[str, np.ndarray] = {}

    # --------- text synthesis & cache keys ---------
    @staticmethod
    def _hash(s: str) -> str:
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    def _storage_key(self, cache_key: str) -> str:
        return f"{self.CACHE_NS}:{cache_key}"

    def _get_field(self, p: Dict[str, Any], *keys: str, default: Optional[str] = None) -> Optional[str]:
        for k in keys:
            v = p.get(k)
            if v:
                return str(v)
        return default

    def _create_product_text(self, product: Dict[str, Any]) -> str:
        parts: List[str] = []
        title = self._get_field(product, "title", "name")
        if title:
            parts.append(title)

        category = self._get_field(product, "product_type", "product_category", "category")
        if category:
            parts.append(f"Category: {category}")

        brand = self._get_field(product, "brand")
        if brand:
            parts.append(f"Brand: {brand}")

        vendor = self._get_field(product, "vendor")
        if vendor:
            parts.append(f"Vendor: {vendor}")

        ptype = self._get_field(product, "type")
        if ptype:
            parts.append(f"Type: {ptype}")

        desc = self._get_field(product, "description", "body_html", "body")
        if desc:
            parts.append(desc[:200])

        tags = product.get("tags")
        if tags:
            parts.append(f"Tags: {tags if isinstance(tags, str) else ', '.join(tags)}")

        return " | ".join(parts)

    # --------- math helpers ---------
    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v if n == 0 else (v / n)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        return 0.0 if denom == 0 else float(np.dot(a, b) / denom)

    # --------- storage-backed cache ---------
    async def _get_cached_embedding(self, cache_key: str) -> Optional[np.ndarray]:
        # memory first
        v = self._memory_cache.get(cache_key)
        if v is not None:
            return v
        # persistent
        try:
            skey = self._storage_key(cache_key)
            blob = await storage.get(skey)
            if not blob:
                return None
            vec = np.array(json.loads(blob), dtype=np.float32)
            if self.normalize:
                vec = self._l2(vec)
            self._memory_cache[cache_key] = vec
            return vec
        except Exception as e:
            logger.warning(f"EMBED_CACHE: read failed for key={cache_key}: {e}")
            return None

    async def _set_cached_embedding(self, cache_key: str, vec: np.ndarray) -> None:
        if self.normalize:
            vec = self._l2(vec)
        self._memory_cache[cache_key] = vec
        try:
            skey = self._storage_key(cache_key)
            await storage.set(skey, json.dumps(vec.tolist()), ttl=self.CACHE_TTL_S)
        except Exception as e:
            logger.warning(f"EMBED_CACHE: write failed for key={cache_key}: {e}")

    # --------- single embedding ---------
    async def get_embedding(self, product: Dict[str, Any], use_cache: bool = True) -> np.ndarray:
        text = self._create_product_text(product)
        cache_key = self._hash(text)

        if use_cache:
            cached = await self._get_cached_embedding(cache_key)
            if cached is not None:
                return cached

        # call API with retries
        inputs = [text]
        vecs = await self._embed_texts(inputs)
        vec = vecs[0] if vecs else np.zeros(self.embedding_dim, dtype=np.float32)

        if use_cache and vec is not None:
            await self._set_cached_embedding(cache_key, vec)

        return vec

    # --------- batched embedding (dedup + retries) ---------
    async def get_embeddings_batch(self, products: List[Dict[str, Any]], use_cache: bool = True) -> Dict[str, np.ndarray]:
        """
        Returns {sku: embedding}. Skips products with no sku.
        Deduplicates identical texts to save tokens.
        """
        logger.info("LLM_EMBEDDINGS: request count=%d use_cache=%s", len(products) if products else 0, use_cache)

        out: Dict[str, np.ndarray] = {}
        if not products:
            return out

        # prepare texts & dedup by text hash
        pending: List[Tuple[str, str, str]] = []  # (sku, text, cache_key)
        text_to_skus: Dict[str, List[str]] = {}
        cache_hits = 0

        for p in products:
            sku = p.get("sku")
            if not sku:
                continue
            text = self._create_product_text(p)
            cache_key = self._hash(text)

            if use_cache:
                cached = await self._get_cached_embedding(cache_key)
                if cached is not None:
                    out[sku] = cached
                    cache_hits += 1
                    continue

            pending.append((sku, text, cache_key))
            text_to_skus.setdefault(cache_key, []).append(sku)

        if not pending:
            logger.info("LLM_EMBEDDINGS: served fully from cache | cache_hits=%d", cache_hits)
            return out

        # deduplicate identical texts (by cache_key)
        unique_items: List[Tuple[str, str]] = []  # (cache_key, text)
        seen = set()
        for _, text, ck in pending:
            if ck in seen:
                continue
            seen.add(ck)
            unique_items.append((ck, text))

        logger.info("LLM_EMBEDDINGS: to_generate_unique=%d (from %d pending) batches≈%d",
                    len(unique_items), len(pending), (len(unique_items)-1)//self.batch_size + 1)

        # embed in batches
        generated: Dict[str, np.ndarray] = {}
        for i in range(0, len(unique_items), self.batch_size):
            batch = unique_items[i:i + self.batch_size]
            batch_keys = [ck for ck, _ in batch]
            batch_texts = [tx for _, tx in batch]

            # call API with retries
            vecs = await self._embed_texts(batch_texts)
            if not vecs or len(vecs) != len(batch_texts):
                # length mismatch or failure; fill zeros
                logger.error("LLM_EMBEDDINGS: batch mismatch/failure, got=%s expected=%s",
                             len(vecs) if vecs else 0, len(batch_texts))
                for ck in batch_keys:
                    generated[ck] = np.zeros(self.embedding_dim, dtype=np.float32)
                continue

            # write to cache maps
            for ck, v in zip(batch_keys, vecs):
                if v is None or not isinstance(v, np.ndarray):
                    v = np.zeros(self.embedding_dim, dtype=np.float32)
                await self._set_cached_embedding(ck, v)
                generated[ck] = v

        # fan out to SKUs
        for ck, skus in text_to_skus.items():
            v = generated.get(ck)
            if v is None:
                # might have been filled earlier by cache in the same run
                v = self._memory_cache.get(ck, np.zeros(self.embedding_dim, dtype=np.float32))
            for sku in skus:
                out[sku] = v

        logger.info("LLM_EMBEDDINGS: done | total=%d cache_hits=%d newly_embedded_unique=%d",
                    len(out), cache_hits, len(generated))
        return out

    # --------- low-level embed with retries ---------
    async def _embed_texts(self, texts: List[str]) -> List[np.ndarray]:
        """
        Returns list of np.ndarray, preserving order; zero-vector on failure.
        Retries 429/5xx with simple exponential backoff.
        """
        if not texts:
            return []

        attempt = 0
        while True:
            try:
                resp = await self.client.embeddings.create(model=self.model, input=texts)
                data = resp.data or []
                out: List[np.ndarray] = []
                for i in range(len(texts)):
                    if i >= len(data) or not getattr(data[i], "embedding", None):
                        out.append(np.zeros(self.embedding_dim, dtype=np.float32))
                    else:
                        vec = np.array(data[i].embedding, dtype=np.float32)
                        out.append(self._l2(vec) if self.normalize else vec)
                return out
            except Exception as e:
                attempt += 1
                msg = str(e)
                # non-retryable (e.g., bad key/headers)
                if "Illegal header value" in msg or "authentication" in msg.lower():
                    logger.error("EMBED: non-retryable error: %s", msg)
                    return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]
                if attempt > self.RETRY_MAX:
                    logger.error("EMBED: retries exhausted (%d). last error: %s", attempt, msg)
                    return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]
                backoff = self.RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1)) * (1.0 + 0.1 * (attempt))
                logger.warning("EMBED: transient error (attempt %d/%d): %s | sleeping %.2fs",
                               attempt, self.RETRY_MAX, msg, backoff)
                await self._sleep(backoff)

    @staticmethod
    async def _sleep(seconds: float) -> None:
        # tiny awaitable sleep without importing asyncio at top-level for tests
        import asyncio as _asyncio
        await _asyncio.sleep(seconds)

    # --------- similarity & candidates ---------
    async def find_similar_products(
        self,
        target_sku: str,
        catalog: List[Dict[str, Any]],
        embeddings: Dict[str, np.ndarray],
        top_k: int = 20,
        min_similarity: float = 0.5,
    ) -> List[Tuple[str, float]]:
        if target_sku not in embeddings:
            logger.warning("SIM: target SKU %s missing embedding", target_sku)
            return []

        t = embeddings[target_sku]
        sims: List[Tuple[str, float]] = []
        for p in catalog:
            sku = p.get("sku")
            if not sku or sku == target_sku:
                continue
            v = embeddings.get(sku)
            if v is None:
                continue
            s = self._cosine(t, v)
            if s >= min_similarity:
                sims.append((sku, float(s)))

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    def compute_bundle_similarity(self, products: List[str], embeddings: Dict[str, np.ndarray]) -> float:
        if len(products) < 2:
            return 0.0
        vals: List[float] = []
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                a = embeddings.get(products[i])
                b = embeddings.get(products[j])
                if a is None or b is None:
                    continue
                vals.append(self._cosine(a, b))
        return float(np.mean(vals)) if vals else 0.0

    async def generate_candidates_by_similarity(
        self,
        csv_upload_id: str,
        bundle_type: str,
        objective: str,
        catalog: List[Dict[str, Any]],
        embeddings: Dict[str, np.ndarray],
        num_candidates: int = 20,
    ) -> List[Dict[str, Any]]:
        if not catalog or not embeddings:
            return []
        bt = (bundle_type or "").upper()
        if bt == "FBT":
            return await self._gen_fbt(catalog, embeddings, num_candidates)
        if bt == "VOLUME_DISCOUNT":
            return await self._gen_volume(catalog, embeddings, num_candidates)
        if bt == "MIX_MATCH":
            return await self._gen_mixmatch(catalog, embeddings, num_candidates)
        if bt == "BXGY":
            return await self._gen_bxgy(catalog, embeddings, num_candidates)
        if bt == "FIXED":
            return await self._gen_fixed(catalog, embeddings, objective, num_candidates)
        return []

    # ---- internal generators ----
    async def _gen_fbt(self, catalog: List[Dict[str, Any]], embeddings: Dict[str, np.ndarray], n: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        head = catalog[: min(50, len(catalog))]
        for p in head:
            sku = p.get("sku")
            if not sku or sku not in embeddings:
                continue
            neigh = await self.find_similar_products(
                target_sku=sku,
                catalog=catalog,
                embeddings=embeddings,
                top_k=10,
                min_similarity=self.FBT_MIN,
            )
            for s_sku, sim in neigh[:5]:
                if sim > self.TOO_SIM:
                    continue  # skip near-duplicates
                out.append({"products": [sku, s_sku], "llm_similarity": sim, "source": "llm_fbt"})
                if len(out) >= n:
                    return out
        return out[:n]

    async def _gen_volume(self, catalog: List[Dict[str, Any]], embeddings: Dict[str, np.ndarray], n: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        head = catalog[: min(30, len(catalog))]
        for p in head:
            sku = p.get("sku")
            if not sku or sku not in embeddings:
                continue
            neigh = await self.find_similar_products(
                target_sku=sku,
                catalog=catalog,
                embeddings=embeddings,
                top_k=5,
                min_similarity=self.VOLUME_MIN,
            )
            if len(neigh) >= 2:
                skus = [s for s, _ in neigh[:3]]
                avg_sim = float(np.mean([x for _, x in neigh[:3]]))
                out.append({"products": [sku] + skus, "llm_similarity": avg_sim, "source": "llm_volume"})
                if len(out) >= n:
                    return out
        return out[:n]

    async def _gen_mixmatch(self, catalog: List[Dict[str, Any]], embeddings: Dict[str, np.ndarray], n: int) -> List[Dict[str, Any]]:
        # Variety within category — start with volume-like, but you could add category gating later
        return await self._gen_volume(catalog, embeddings, n)

    async def _gen_bxgy(self, catalog: List[Dict[str, Any]], embeddings: Dict[str, np.ndarray], n: int) -> List[Dict[str, Any]]:
        # Complementary pairing similar to FBT
        return await self._gen_fbt(catalog, embeddings, n)

    async def _gen_fixed(
        self,
        catalog: List[Dict[str, Any]],
        embeddings: Dict[str, np.ndarray],
        objective: str,
        n: int,
    ) -> List[Dict[str, Any]]:
        # Placeholder: curate 2-item complementary sets; add objective-specific logic later
        return await self._gen_fbt(catalog, embeddings, n)


# Global instance
llm_embedding_engine = LLMEmbeddingEngine()