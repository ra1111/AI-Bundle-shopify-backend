import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.obs.metrics import MetricsCollector


def test_record_drop_summary_namespaced():
    collector = MetricsCollector()

    collector.record_drop_summary({"inventory": 3, "compliance": 1}, namespace="stage_1")
    collector.record_drop_summary({"inventory": 2}, namespace="stage_2")
    collector.record_drop_summary({"global": 4})

    assert collector.drop_reasons["stage_1:inventory"] == 3
    assert collector.drop_reasons["stage_1:compliance"] == 1
    assert collector.drop_reasons["stage_2:inventory"] == 2
    assert collector.drop_reasons["global"] == 4


def test_record_staged_publish_counters():
    collector = MetricsCollector()

    summary = {
        "published": 7,
        "dropped": 3,
        "stages": [{"stage_index": 1}, {"stage_index": 2}],
        "tracks": {"copy_ms": 120, "pricing_ms": 80},
        "drop_reasons": {"inventory:out_of_stock": 2},
    }

    collector.record_staged_publish(summary)

    assert collector.staged_publish_counters["runs"] == 1
    assert collector.staged_publish_counters["published"] == 7
    assert collector.staged_publish_counters["dropped"] == 3
    assert collector.staged_publish_counters["stages"] == 2
    assert collector.last_staged_publish == summary


def test_record_staged_publish_with_totals_section():
    collector = MetricsCollector()
    state_summary = {
        "totals": {"published": 5, "dropped": 2},
        "waves": [{"index": 0}],
    }
    collector.record_staged_publish(state_summary)
    assert collector.staged_publish_counters["published"] == 5
    assert collector.staged_publish_counters["dropped"] == 2
