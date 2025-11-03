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
import hashlib
import logging
from typing import Any, Dict, List, Tuple, Optional
import contextlib

import numpy as np
import inspect
from services.storage import storage  # must expose async get(key)->str|None and set(key, value, ttl=None)
from services.ml.llm_utils import get_async_client, load_settings
from services.feature_flags import feature_flags

try:
    from opentelemetry import trace
    from opentelemetry.trace import SpanKind
except ImportError:  # pragma: no cover - optional dependency
    trace = None
    SpanKind = None

logger = logging.getLogger(__name__)


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
    FBT_MIN = float(os.getenv("SIM_FBT_MIN", "0.30"))        # complementary floor (softened for sparse catalogs)
    TOO_SIM = float(os.getenv("SIM_TOO_SIMILAR", "0.90"))    # skip near-duplicates only
    VOLUME_MIN = float(os.getenv("SIM_VOLUME_MIN", "0.70"))  # volume (very similar)
    RELAX_FILTER_ORDERS = int(os.getenv("SIM_RELAX_FILTER_ORDERS", "12"))  # relax similarity filters below this order count

    # retry config
    RETRY_MAX = int(os.getenv("EMBED_RETRY_MAX", "3"))
    RETRY_BACKOFF_BASE_S = float(os.getenv("EMBED_RETRY_BACKOFF_BASE_S", "0.5"))

    PREFILTER_PRICE_DELTA = float(os.getenv("SIM_PREFILTER_PRICE_DELTA", "0.25"))
    PREFILTER_REQUIRE_VENDOR = os.getenv("SIM_PREFILTER_VENDOR_MATCH", "true").lower() == "true"
    PREFILTER_REQUIRE_CATEGORY = os.getenv("SIM_PREFILTER_CATEGORY_MATCH", "false").lower() == "true"

    def __init__(self) -> None:
        self._settings = load_settings()
        self.client = get_async_client()
        self.model = self._settings.embedding_model or self.DEFAULT_MODEL
        self.embedding_dim = self._settings.embedding_dim or self.DEFAULT_DIM
        self.batch_size = self._settings.embedding_batch_size or self.BATCH_SIZE
        self.normalize = True
        self._memory_cache: Dict[str, np.ndarray] = {}
        self._persist_ok = False
        self._read_method = None
        self._write_method = None
        self.retry_max = self._settings.retry_max or self.RETRY_MAX
        self.retry_backoff_base_s = self._settings.retry_backoff_base_s or self.RETRY_BACKOFF_BASE_S
        self._tracer = trace.get_tracer(__name__) if trace else None

        read_candidates = ("get", "get_text", "read", "read_text", "load")
        write_candidates = ("set", "put", "write", "write_text", "save")

        for obj in (storage, getattr(storage, "client", None)):
            if not obj:
                continue
            if self._read_method is None:
                for name in read_candidates:
                    fn = getattr(obj, name, None)
                    if fn:
                        self._read_method = fn
                        break
            if self._write_method is None:
                for name in write_candidates:
                    fn = getattr(obj, name, None)
                    if fn:
                        self._write_method = fn
                        break
            if self._read_method and self._write_method:
                self._persist_ok = True
                break

        if not self._persist_ok:
            logger.info("EMBED_CACHE: persistent cache disabled (no compatible storage methods found)")

    @contextlib.contextmanager
    def _start_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        if not self._tracer or not SpanKind:
            yield None
            return
        with self._tracer.start_as_current_span(name, kind=SpanKind.INTERNAL) as span:
            if attributes:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
            yield span

    def _resolve_similarity_thresholds(self, catalog_size: int) -> Tuple[float, float, float]:
        """
        Determine similarity thresholds, optionally adapting based on catalog size.
        Returns (fbt_min, too_similar, volume_min).
        """
        default = (self.FBT_MIN, self.TOO_SIM, self.VOLUME_MIN)

        if not feature_flags.get_flag("bundling.llm_adaptive_similarity", False):
            return default

        if catalog_size <= 0:
            return default

        if catalog_size < 100:
            thresholds = (0.25, 0.95, 0.45)
        elif catalog_size < 1000:
            thresholds = (0.30, 0.85, 0.55)
        else:
            thresholds = (0.35, 0.80, 0.65)

        if thresholds != default:
            logger.info(
                "LLM_SIM adaptive thresholds applied | catalog_size=%d fbt_min=%.2f too_sim=%.2f volume_min=%.2f",
                catalog_size,
                thresholds[0],
                thresholds[1],
                thresholds[2],
            )

        return thresholds

    # --------- storage adapters ---------
    async def _storage_get(self, key: str) -> Optional[str]:
        if not self._persist_ok or self._read_method is None:
            return None
        result = self._read_method(key)
        result = await result if inspect.isawaitable(result) else result
        if isinstance(result, (bytes, bytearray)):
            result = result.decode("utf-8", errors="ignore")
        return result

    async def _storage_set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        if not self._persist_ok or self._write_method is None:
            return
        try:
            result = None
            try:
                result = self._write_method(key, value, ttl=ttl)
            except TypeError:
                result = self._write_method(key, value)
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            logger.warning("EMBED_CACHE: write failed for key=%s: %s", key, e)

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
            blob = await self._storage_get(skey)
            if not blob:
                return None
            if isinstance(blob, (bytes, bytearray)):
                blob = blob.decode("utf-8", errors="ignore")
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
            payload = json.dumps(vec.tolist())
            await self._storage_set(skey, payload, ttl=self.CACHE_TTL_S)
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
        missing_sku = 0

        for p in products:
            sku = p.get("sku") or p.get("Variant SKU") or p.get("variant_sku")
            if not sku:
                missing_sku += 1
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

        if missing_sku:
            logger.info("LLM_EMBEDDINGS: skipped %d catalog entries with no SKU/variant identifier", missing_sku)

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
                resp = await self.client.embeddings.create(
                    model=self.model,
                    input=texts,
                    timeout=30,
                )
                data = resp.data or []
                if data and getattr(data[0], "embedding", None) is not None:
                    self.embedding_dim = len(data[0].embedding)
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
                nonretry_signals = ("Illegal header value", "authentication", "invalid_api_key", "401")
                if any(sig.lower() in msg.lower() for sig in nonretry_signals):
                    logger.error("EMBED: non-retryable error: %s", msg)
                    return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]
                if attempt > self.retry_max:
                    logger.error("EMBED: retries exhausted (%d). last error: %s", attempt, msg)
                    return [np.zeros(self.embedding_dim, dtype=np.float32) for _ in texts]
                backoff = self.retry_backoff_base_s * (2 ** (attempt - 1)) * (1.0 + 0.1 * (attempt))
                logger.warning("EMBED: transient error (attempt %d/%d): %s | sleeping %.2fs",
                               attempt, self.retry_max, msg, backoff)
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
        csv_upload_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        span_attrs = {
            "target_sku": target_sku,
            "catalog_size": len(catalog),
            "threshold": min_similarity,
        }
        with self._start_span("llm_embeddings.similarity_search", span_attrs):
            if target_sku not in embeddings:
                logger.warning("SIM: target SKU %s missing embedding", target_sku)
                return []

            t = embeddings[target_sku]
            sims: List[Tuple[str, float]] = []
            highest_miss: Optional[float] = None
            scope = csv_upload_id or "unknown_upload"
            for p in catalog:
                sku = p.get("sku") or p.get("Variant SKU") or p.get("variant_sku")
                if not sku or sku == target_sku:
                    continue
                v = embeddings.get(sku)
                if v is None:
                    continue
                s = self._cosine(t, v)
                if s >= min_similarity:
                    sims.append((sku, float(s)))
                else:
                    if highest_miss is None or s > highest_miss:
                        highest_miss = float(s)

            sims.sort(key=lambda x: x[1], reverse=True)
            if highest_miss is not None and logger.isEnabledFor(logging.DEBUG):
                delta = max(min_similarity - highest_miss, 0.0)
                logger.debug(
                    "[%s] SIM miss | target=%s threshold=%.3f below=%.3f delta=%.3f",
                    scope,
                    target_sku,
                    min_similarity,
                    highest_miss,
                    delta,
                )
            if not sims:
                logger.info(
                    "[%s] SIM search produced no matches | target=%s threshold=%.3f catalog_checked=%d",
                    scope,
                    target_sku,
                    min_similarity,
                    len(catalog),
                )
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
        orders_count: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not catalog or not embeddings:
            return []
        bt = (bundle_type or "").upper()
        span_attrs = {
            "csv_upload_id": csv_upload_id,
            "bundle_type": bt,
            "objective": objective,
            "catalog_size": len(catalog),
        }
        with self._start_span("llm_embeddings.generate_candidates", span_attrs):
            fbt_min, too_similar, volume_min = self._resolve_similarity_thresholds(len(catalog))
            if bt == "FBT":
                results = await self._gen_fbt(
                    catalog,
                    embeddings,
                    num_candidates,
                    csv_upload_id=csv_upload_id,
                    orders_count=orders_count,
                    fbt_min=fbt_min,
                    too_similar=too_similar,
                )
                logger.info(
                    "LLM candidates generated | upload=%s type=%s objective=%s count=%d",
                    csv_upload_id,
                    bt,
                    objective,
                    len(results),
                )
                return results
            if bt == "VOLUME_DISCOUNT":
                results = await self._gen_volume(
                    catalog,
                    embeddings,
                    num_candidates,
                    csv_upload_id=csv_upload_id,
                    orders_count=orders_count,
                    volume_min=volume_min,
                )
                logger.info(
                    "LLM candidates generated | upload=%s type=%s objective=%s count=%d",
                    csv_upload_id,
                    bt,
                    objective,
                    len(results),
                )
                return results
            if bt == "MIX_MATCH":
                results = await self._gen_mixmatch(
                    catalog,
                    embeddings,
                    num_candidates,
                    csv_upload_id=csv_upload_id,
                    orders_count=orders_count,
                    volume_min=volume_min,
                )
                logger.info(
                    "LLM candidates generated | upload=%s type=%s objective=%s count=%d",
                    csv_upload_id,
                    bt,
                    objective,
                    len(results),
                )
                return results
            if bt == "BXGY":
                results = await self._gen_bxgy(
                    catalog,
                    embeddings,
                    num_candidates,
                    csv_upload_id=csv_upload_id,
                    orders_count=orders_count,
                    fbt_min=fbt_min,
                    too_similar=too_similar,
                )
                logger.info(
                    "LLM candidates generated | upload=%s type=%s objective=%s count=%d",
                    csv_upload_id,
                    bt,
                    objective,
                    len(results),
                )
                return results
            if bt == "FIXED":
                results = await self._gen_fixed(
                    catalog,
                    embeddings,
                    objective,
                    num_candidates,
                    csv_upload_id=csv_upload_id,
                    orders_count=orders_count,
                    fbt_min=fbt_min,
                    too_similar=too_similar,
                )
                logger.info(
                    "LLM candidates generated | upload=%s type=%s objective=%s count=%d",
                    csv_upload_id,
                    bt,
                    objective,
                    len(results),
                )
                return results
            logger.warning("LLM candidate generation skipped: unsupported bundle type %s", bt)
            return []

    # ---- internal generators ----
    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        if value in (None, "", "null", "NULL"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _passes_prefilter(self, anchor: Optional[Dict[str, Any]], candidate: Optional[Dict[str, Any]]) -> bool:
        if not anchor or not candidate:
            return True
        price_a = self._safe_float(anchor.get("price"))
        price_b = self._safe_float(candidate.get("price"))
        if price_a is not None and price_b is not None:
            if price_a == 0:
                logger.debug("LLM prefilter rejected due to zero price | anchor=%s candidate=%s", anchor.get("sku"), candidate.get("sku"))
                return False
            delta = abs(price_a - price_b) / max(price_a, 1.0)
            if delta > self.PREFILTER_PRICE_DELTA:
                logger.debug(
                    "LLM prefilter price delta too high | anchor_price=%.2f candidate_price=%.2f delta=%.2f limit=%.2f",
                    price_a,
                    price_b,
                    delta,
                    self.PREFILTER_PRICE_DELTA,
                )
                return False
        if self.PREFILTER_REQUIRE_VENDOR:
            vendor_a = anchor.get("vendor")
            vendor_b = candidate.get("vendor")
            if vendor_a and vendor_b and vendor_a != vendor_b:
                logger.debug(
                    "LLM prefilter vendor mismatch | anchor_vendor=%s candidate_vendor=%s",
                    vendor_a,
                    vendor_b,
                )
                return False
        if self.PREFILTER_REQUIRE_CATEGORY:
            category_a = anchor.get("product_type") or anchor.get("category")
            category_b = candidate.get("product_type") or candidate.get("category")
            if category_a and category_b and category_a != category_b:
                logger.debug(
                    "LLM prefilter category mismatch | anchor_category=%s candidate_category=%s",
                    category_a,
                    category_b,
                )
                return False
        return True

    async def _gen_fbt(
        self,
        catalog: List[Dict[str, Any]],
        embeddings: Dict[str, np.ndarray],
        n: int,
        csv_upload_id: Optional[str] = None,
        orders_count: Optional[int] = None,
        fbt_min: Optional[float] = None,
        too_similar: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        head = catalog[: min(50, len(catalog))]
        scope = csv_upload_id or "unknown_upload"
        catalog_lookup = {item.get("sku"): item for item in catalog if item.get("sku")}
        anchors_total = len(head)
        anchors_missing_sku = 0
        anchors_missing_embedding = 0
        neighbors_total = 0
        neighbors_added = 0
        neighbors_too_similar = 0
        relax_filters = (
            orders_count is not None and orders_count < self.RELAX_FILTER_ORDERS
        )
        fbt_floor = self.FBT_MIN if fbt_min is None else fbt_min
        too_sim_threshold = self.TOO_SIM if too_similar is None else too_similar
        if relax_filters:
            logger.info(
                "[%s] LLM FBT: relaxing similarity filters (orders=%s < threshold=%s)",
                scope,
                orders_count,
                self.RELAX_FILTER_ORDERS,
            )
        for p in head:
            sku = p.get("sku")
            if not sku or sku not in embeddings:
                if not sku:
                    anchors_missing_sku += 1
                else:
                    anchors_missing_embedding += 1
                continue
            neigh = await self.find_similar_products(
                target_sku=sku,
                catalog=catalog,
                embeddings=embeddings,
                top_k=10,
                min_similarity=0.0 if relax_filters else fbt_floor,
                csv_upload_id=csv_upload_id,
            )
            for s_sku, sim in neigh[:5]:
                logger.debug("LLM_CANDIDATE_SIM sku=%s candidate=%s sim=%.3f", sku, s_sku, sim)
                neighbors_total += 1
                if not relax_filters and sim > too_sim_threshold:
                    if logger.isEnabledFor(logging.DEBUG):
                        delta = sim - too_sim_threshold
                        logger.debug(
                            "[%s] SIM near-duplicate filtered | target=%s candidate=%s sim=%.3f limit=%.3f delta=%.3f",
                            scope,
                            sku,
                            s_sku,
                            sim,
                            too_sim_threshold,
                            delta,
                        )
                    neighbors_too_similar += 1
                    continue  # skip near-duplicates
                anchor_meta = catalog_lookup.get(sku)
                candidate_meta = catalog_lookup.get(s_sku)
                if not self._passes_prefilter(anchor_meta, candidate_meta):
                    continue
                out.append({"products": [sku, s_sku], "llm_similarity": sim, "source": "llm_fbt"})
                neighbors_added += 1
                if len(out) >= n:
                    logger.info(
                        "[%s] LLM FBT generation summary | anchors=%d missing_sku=%d missing_embedding=%d neighbors=%d added=%d too_similar=%d truncated=%d",
                        scope,
                        anchors_total,
                        anchors_missing_sku,
                        anchors_missing_embedding,
                        neighbors_total,
                        neighbors_added,
                        neighbors_too_similar,
                        n,
                    )
                    return out
        logger.info(
            "[%s] LLM FBT generation summary | anchors=%d missing_sku=%d missing_embedding=%d neighbors=%d added=%d too_similar=%d truncated=%d",
            scope,
            anchors_total,
            anchors_missing_sku,
            anchors_missing_embedding,
            neighbors_total,
            neighbors_added,
            neighbors_too_similar,
            min(n, len(out)),
        )
        return out[:n]

    async def _gen_volume(
        self,
        catalog: List[Dict[str, Any]],
        embeddings: Dict[str, np.ndarray],
        n: int,
        csv_upload_id: Optional[str] = None,
        orders_count: Optional[int] = None,
        volume_min: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        head = catalog[: min(30, len(catalog))]
        scope = csv_upload_id or "unknown_upload"
        catalog_lookup = {item.get("sku"): item for item in catalog if item.get("sku")}
        anchors_total = len(head)
        anchors_missing = 0
        neighbors_total = 0
        neighbors_added = 0
        relax_filters = (
            orders_count is not None and orders_count < self.RELAX_FILTER_ORDERS
        )
        volume_floor = self.VOLUME_MIN if volume_min is None else volume_min
        if relax_filters:
            logger.info(
                "[%s] LLM volume: relaxing similarity filters (orders=%s < threshold=%s)",
                scope,
                orders_count,
                self.RELAX_FILTER_ORDERS,
            )
            for p in head:
                sku = p.get("sku")
                if not sku or sku not in embeddings:
                    anchors_missing += 1
                    continue
                neigh = await self.find_similar_products(
                    target_sku=sku,
                    catalog=catalog,
                    embeddings=embeddings,
                    top_k=5,
                    min_similarity=0.0 if relax_filters else volume_floor,
                    csv_upload_id=csv_upload_id,
                )
                if len(neigh) >= 2:
                    skus = [s for s, _ in neigh[:3]]
                    neighbors_total += len(neigh[:3])
                    anchor_meta = catalog_lookup.get(sku)
                    filtered_skus = []
                    sims_subset = []
                    for candidate_sku, sim in neigh[:3]:
                        if self._passes_prefilter(anchor_meta, catalog_lookup.get(candidate_sku)):
                            filtered_skus.append(candidate_sku)
                            sims_subset.append(sim)
                    if not filtered_skus:
                        continue
                    avg_sim = float(np.mean(sims_subset))
                    out.append({"products": [sku] + filtered_skus, "llm_similarity": avg_sim, "source": "llm_volume"})
                    neighbors_added += 1
                    if len(out) >= n:
                        logger.info(
                            "[%s] LLM volume generation summary | anchors=%d missing=%d neighbor_sets=%d",
                            scope,
                        anchors_total,
                        anchors_missing,
                        neighbors_added,
                    )
                    return out
        logger.info(
            "[%s] LLM volume generation summary | anchors=%d missing=%d neighbor_sets=%d neighbors_considered=%d",
            scope,
            anchors_total,
            anchors_missing,
            neighbors_added,
            neighbors_total,
        )
        return out[:n]

    async def _gen_mixmatch(
        self,
        catalog: List[Dict[str, Any]],
        embeddings: Dict[str, np.ndarray],
        n: int,
        csv_upload_id: Optional[str] = None,
        orders_count: Optional[int] = None,
        volume_min: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        # Variety within category — start with volume-like, but you could add category gating later
        return await self._gen_volume(
            catalog,
            embeddings,
            n,
            csv_upload_id=csv_upload_id,
            orders_count=orders_count,
            volume_min=volume_min,
        )

    async def _gen_bxgy(
        self,
        catalog: List[Dict[str, Any]],
        embeddings: Dict[str, np.ndarray],
        n: int,
        csv_upload_id: Optional[str] = None,
        orders_count: Optional[int] = None,
        fbt_min: Optional[float] = None,
        too_similar: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        # Complementary pairing similar to FBT
        return await self._gen_fbt(
            catalog,
            embeddings,
            n,
            csv_upload_id=csv_upload_id,
            orders_count=orders_count,
            fbt_min=fbt_min,
            too_similar=too_similar,
        )

    async def _gen_fixed(
        self,
        catalog: List[Dict[str, Any]],
        embeddings: Dict[str, np.ndarray],
        objective: str,
        n: int,
        csv_upload_id: Optional[str] = None,
        orders_count: Optional[int] = None,
        fbt_min: Optional[float] = None,
        too_similar: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        # Placeholder: curate 2-item complementary sets; add objective-specific logic later
        return await self._gen_fbt(
            catalog,
            embeddings,
            n,
            csv_upload_id=csv_upload_id,
            orders_count=orders_count,
            fbt_min=fbt_min,
            too_similar=too_similar,
        )


# Global instance
llm_embedding_engine = LLMEmbeddingEngine()
