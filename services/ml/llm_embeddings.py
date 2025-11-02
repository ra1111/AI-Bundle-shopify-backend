"""
LLM-based Product Embeddings Engine

Replaces item2vec with OpenAI embeddings for:
- Zero training time (vs 60-120s for item2vec)
- Cold start support (works for new products)
- Semantic understanding (not just behavioral)
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import asyncio
import hashlib
import json
from openai import AsyncOpenAI
import os

from services.storage import storage

logger = logging.getLogger(__name__)

class LLMEmbeddingEngine:
    """Generate and cache product embeddings using OpenAI's embedding models"""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-3-small"  # 1536 dims, $0.02/1M tokens
        self.embedding_dim = 1536
        self.batch_size = 100  # OpenAI allows up to 2048 inputs per request
        self._memory_cache = {}  # In-memory cache for this request

    def _create_product_text(self, product: Dict) -> str:
        """Create rich text representation of product for embedding"""
        parts = []

        # Title (most important)
        if product.get('title'):
            parts.append(product['title'])

        # Category
        if product.get('product_category'):
            parts.append(f"Category: {product['product_category']}")

        # Brand
        if product.get('brand'):
            parts.append(f"Brand: {product['brand']}")

        # Vendor
        if product.get('vendor'):
            parts.append(f"Vendor: {product['vendor']}")

        # Product type
        if product.get('product_type'):
            parts.append(f"Type: {product['product_type']}")

        # Description (truncated)
        if product.get('description'):
            desc = str(product['description'])[:200]  # Limit to 200 chars
            parts.append(desc)

        # Tags
        if product.get('tags'):
            tags = product['tags'] if isinstance(product['tags'], str) else ', '.join(product['tags'])
            parts.append(f"Tags: {tags}")

        return ' | '.join(parts)

    def _get_embedding_cache_key(self, text: str) -> str:
        """Generate cache key for embedding"""
        return hashlib.md5(text.encode()).hexdigest()

    async def get_embedding(self, product: Dict, use_cache: bool = True) -> np.ndarray:
        """Get embedding for a single product"""
        text = self._create_product_text(product)
        cache_key = self._get_embedding_cache_key(text)

        # Check cache if enabled
        if use_cache:
            cached = await self._get_cached_embedding(cache_key)
            if cached is not None:
                return cached

        # Generate new embedding
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Cache the embedding
            if use_cache:
                await self._cache_embedding(cache_key, embedding, text)

            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding for product {product.get('sku', 'unknown')}: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_dim, dtype=np.float32)

    async def get_embeddings_batch(self, products: List[Dict], use_cache: bool = True) -> Dict[str, np.ndarray]:
        """Get embeddings for multiple products efficiently (batched API calls)"""
        try:
            logger.info(
                f"LLM_EMBEDDINGS: Batch embedding request | "
                f"products={len(products) if products else 0}, use_cache={use_cache}"
            )

            embeddings = {}
            products_to_embed = []
            product_texts = {}

            # Prepare texts and check cache
            for product in products if products else []:
                sku = product.get('sku')
                if not sku:
                    continue

                text = self._create_product_text(product)
                cache_key = self._get_embedding_cache_key(text)
                product_texts[sku] = text

                # Check cache
                if use_cache:
                    cached = await self._get_cached_embedding(cache_key)
                    if cached is not None:
                        embeddings[sku] = cached
                        continue

                products_to_embed.append((sku, text, cache_key))

            if not products_to_embed:
                logger.info(
                    f"LLM_EMBEDDINGS: All embeddings found in cache | "
                    f"cached_count={len(embeddings)}"
                )
                return embeddings

            logger.info(
                f"LLM_EMBEDDINGS: Generating new embeddings | "
                f"cached={len(embeddings)}, to_generate={len(products_to_embed)}, "
                f"batches={(len(products_to_embed)-1)//self.batch_size + 1}"
            )

            # Batch the API calls
            for i in range(0, len(products_to_embed), self.batch_size):
                batch = products_to_embed[i:i + self.batch_size]
                batch_texts = [text for _, text, _ in batch]
                batch_num = i//self.batch_size + 1
                total_batches = (len(products_to_embed)-1)//self.batch_size + 1

                try:
                    logger.info(
                        f"LLM_EMBEDDINGS: Calling OpenAI API | "
                        f"batch={batch_num}/{total_batches}, batch_size={len(batch)}, "
                        f"model={self.model}"
                    )

                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=batch_texts
                    )

                    # Process response
                    for j, (sku, text, cache_key) in enumerate(batch):
                        embedding = np.array(response.data[j].embedding, dtype=np.float32)
                        embeddings[sku] = embedding

                        # Cache the embedding
                        if use_cache:
                            await self._cache_embedding(cache_key, embedding, text)

                    logger.info(
                        f"LLM_EMBEDDINGS: Batch complete | "
                        f"batch={batch_num}/{total_batches}, embeddings_generated={len(batch)}"
                    )

                except Exception as e:
                    logger.error(
                        f"LLM_EMBEDDINGS: Error generating embeddings for batch {batch_num}: {e} | "
                        f"Using zero vectors as fallback for {len(batch)} products",
                        exc_info=True
                    )
                    # Use zero vectors as fallback for failed batch
                    for sku, _, _ in batch:
                        embeddings[sku] = np.zeros(self.embedding_dim, dtype=np.float32)

            logger.info(
                f"LLM_EMBEDDINGS: Batch embedding complete | "
                f"total_embeddings={len(embeddings)}, "
                f"cached={len(embeddings) - len(products_to_embed)}, "
                f"generated={len(products_to_embed)}"
            )

            return embeddings

        except Exception as e:
            logger.error(
                f"LLM_EMBEDDINGS: Fatal error in batch embedding: {e} | "
                f"Returning partial results ({len(embeddings)} embeddings)",
                exc_info=True
            )
            return embeddings  # Return whatever we got

    async def find_similar_products(
        self,
        target_sku: str,
        catalog: List[Dict],
        embeddings: Dict[str, np.ndarray],
        top_k: int = 20,
        min_similarity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """Find products similar to target using cosine similarity"""
        if target_sku not in embeddings:
            logger.warning(f"Target SKU {target_sku} not in embeddings")
            return []

        target_emb = embeddings[target_sku]
        similarities = []

        for product in catalog:
            sku = product.get('sku')
            if not sku or sku == target_sku or sku not in embeddings:
                continue

            product_emb = embeddings[sku]
            similarity = self._cosine_similarity(target_emb, product_emb)

            if similarity >= min_similarity:
                similarities.append((sku, float(similarity)))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def compute_bundle_similarity(self, products: List[str], embeddings: Dict[str, np.ndarray]) -> float:
        """Compute average pairwise similarity for products in a bundle"""
        if len(products) < 2:
            return 0.0

        similarities = []
        for i in range(len(products)):
            for j in range(i + 1, len(products)):
                sku_i, sku_j = products[i], products[j]

                if sku_i in embeddings and sku_j in embeddings:
                    sim = self._cosine_similarity(embeddings[sku_i], embeddings[sku_j])
                    similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.0

    async def _get_cached_embedding(self, cache_key: str) -> Optional[np.ndarray]:
        """Retrieve embedding from in-memory cache"""
        return self._memory_cache.get(cache_key)

    async def _cache_embedding(self, cache_key: str, embedding: np.ndarray, text: str):
        """Store embedding in in-memory cache"""
        self._memory_cache[cache_key] = embedding

    async def generate_candidates_by_similarity(
        self,
        csv_upload_id: str,
        bundle_type: str,
        objective: str,
        catalog: List[Dict],
        embeddings: Dict[str, np.ndarray],
        num_candidates: int = 20
    ) -> List[Dict]:
        """Generate bundle candidates using LLM semantic similarity"""
        candidates = []

        # Strategy depends on bundle type
        if bundle_type == "FBT":
            # Frequently Bought Together: Find complementary products
            candidates = await self._generate_fbt_candidates(catalog, embeddings, num_candidates)

        elif bundle_type == "VOLUME_DISCOUNT":
            # Volume discount: Find similar products that make sense to buy multiple of
            candidates = await self._generate_volume_candidates(catalog, embeddings, num_candidates)

        elif bundle_type == "MIX_MATCH":
            # Mix & Match: Find products in same category with variety
            candidates = await self._generate_mix_match_candidates(catalog, embeddings, num_candidates)

        elif bundle_type == "BXGY":
            # Buy X Get Y: Find complementary pairs
            candidates = await self._generate_bxgy_candidates(catalog, embeddings, num_candidates)

        elif bundle_type == "FIXED":
            # Fixed bundle: Curated sets
            candidates = await self._generate_fixed_candidates(catalog, embeddings, objective, num_candidates)

        return candidates

    async def _generate_fbt_candidates(
        self,
        catalog: List[Dict],
        embeddings: Dict[str, np.ndarray],
        num_candidates: int
    ) -> List[Dict]:
        """Generate FBT candidates using complementary similarity"""
        candidates = []

        # For each product, find moderately similar products (0.4-0.7 similarity)
        # Not too similar (different products) but related (complementary)
        for product in catalog[:50]:  # Limit to top 50 products to avoid timeout
            sku = product.get('sku')
            if not sku or sku not in embeddings:
                continue

            similar = await self.find_similar_products(
                target_sku=sku,
                catalog=catalog,
                embeddings=embeddings,
                top_k=10,
                min_similarity=0.4  # Complementary, not identical
            )

            # Create bundles with 2-3 products
            for similar_sku, sim in similar[:5]:
                if sim > 0.7:  # Too similar, skip
                    continue

                candidates.append({
                    'products': [sku, similar_sku],
                    'llm_similarity': sim,
                    'source': 'llm_fbt'
                })

        return candidates[:num_candidates]

    async def _generate_volume_candidates(
        self,
        catalog: List[Dict],
        embeddings: Dict[str, np.ndarray],
        num_candidates: int
    ) -> List[Dict]:
        """Generate volume discount candidates (same/similar products)"""
        candidates = []

        # Find products with high similarity (same type, different variants)
        for product in catalog[:30]:
            sku = product.get('sku')
            if not sku or sku not in embeddings:
                continue

            similar = await self.find_similar_products(
                target_sku=sku,
                catalog=catalog,
                embeddings=embeddings,
                top_k=5,
                min_similarity=0.7  # Very similar products
            )

            if len(similar) >= 2:
                similar_skus = [s[0] for s in similar[:3]]
                candidates.append({
                    'products': [sku] + similar_skus,
                    'llm_similarity': np.mean([s[1] for s in similar]),
                    'source': 'llm_volume'
                })

        return candidates[:num_candidates]

    async def _generate_mix_match_candidates(
        self,
        catalog: List[Dict],
        embeddings: Dict[str, np.ndarray],
        num_candidates: int
    ) -> List[Dict]:
        """Generate mix & match candidates (variety within category)"""
        # Similar to volume but with more variety
        return await self._generate_volume_candidates(catalog, embeddings, num_candidates)

    async def _generate_bxgy_candidates(
        self,
        catalog: List[Dict],
        embeddings: Dict[str, np.ndarray],
        num_candidates: int
    ) -> List[Dict]:
        """Generate BXGY candidates (buy X, get complementary Y free)"""
        # Similar to FBT
        return await self._generate_fbt_candidates(catalog, embeddings, num_candidates)

    async def _generate_fixed_candidates(
        self,
        catalog: List[Dict],
        embeddings: Dict[str, np.ndarray],
        objective: str,
        num_candidates: int
    ) -> List[Dict]:
        """Generate fixed bundle candidates based on objective"""
        candidates = []

        # Use objective to guide bundle creation
        # For now, use FBT-style complementary matching
        # TODO: Add objective-specific logic (e.g., gift boxes, seasonal)

        return await self._generate_fbt_candidates(catalog, embeddings, num_candidates)


# Global instance
llm_embedding_engine = LLMEmbeddingEngine()
