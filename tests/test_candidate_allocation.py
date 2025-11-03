from collections import Counter
import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.ml.candidate_generator import CandidateGenerator


def test_choose_embedding_targets_prefers_frequent_skus():
    generator = CandidateGenerator()
    catalog_map = {
        f"SKU{i}": {"sku": f"SKU{i}", "is_slow_mover": i % 2 == 0}
        for i in range(30)
    }
    frequency = Counter({f"SKU{i}": 30 - i for i in range(30)})
    targets = generator._choose_embedding_targets(
        valid_skus=set(catalog_map.keys()),
        catalog_map=catalog_map,
        sku_frequency=frequency,
    )
    assert len(targets) <= generator.max_embedding_targets
    assert targets[0] == "SKU0"
    assert targets[1] == "SKU1"


def test_allocation_includes_flagged_items_when_low_frequency():
    generator = CandidateGenerator()
    catalog_map = {
        "A": {"sku": "A", "is_slow_mover": True},
        "B": {"sku": "B", "is_new_launch": True},
        "C": {"sku": "C"},
    }
    frequency = Counter({"C": 1})
    targets = generator._choose_embedding_targets(
        valid_skus={"A", "B", "C"},
        catalog_map=catalog_map,
        sku_frequency=frequency,
    )
    assert "A" in targets and "B" in targets
