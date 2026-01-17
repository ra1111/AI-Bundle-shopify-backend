from collections import Counter
import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.ml.candidate_generator import CandidateGenerator


def test_choose_embedding_targets_prefers_frequent_variant_ids():
    generator = CandidateGenerator()
    catalog_map = {
        f"VID{i}": {"variant_id": f"VID{i}", "is_slow_mover": i % 2 == 0}
        for i in range(30)
    }
    frequency = Counter({f"VID{i}": 30 - i for i in range(30)})
    targets = generator._choose_embedding_targets(
        valid_variant_ids=set(catalog_map.keys()),
        catalog_map=catalog_map,
        variant_id_frequency=frequency,
    )
    assert len(targets) <= generator.max_embedding_targets
    assert targets[0] == "VID0"
    assert targets[1] == "VID1"


def test_allocation_includes_flagged_items_when_low_frequency():
    generator = CandidateGenerator()
    catalog_map = {
        "A": {"variant_id": "A", "is_slow_mover": True},
        "B": {"variant_id": "B", "is_new_launch": True},
        "C": {"variant_id": "C"},
    }
    frequency = Counter({"C": 1})
    targets = generator._choose_embedding_targets(
        valid_variant_ids={"A", "B", "C"},
        catalog_map=catalog_map,
        variant_id_frequency=frequency,
    )
    assert "A" in targets and "B" in targets
