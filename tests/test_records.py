import json
from dataclasses import replace
from datetime import timedelta

import pytest

from sfi.forecasts.record import ArchiveConflict, ForecastArchive, QualityGate, make_record

from .conftest import T0, make_game


def _rec(p=0.62, data_kind="synthetic"):
    return make_record(
        game=make_game(), as_of=T0 - timedelta(hours=1), p_home_win=p, model_name="baseline_elo",
        model_version="0.1.0", feature_spec_version="elo_v1", source_snapshot_id="snap",
        inputs={"home_elo": 1510.0}, gate=QualityGate("publish"), data_kind=data_kind,
    )


def test_record_hash_is_deterministic_and_content_sensitive():
    assert _rec().record_hash == _rec().record_hash
    assert _rec(0.62).record_hash != _rec(0.63).record_hash


def test_synthetic_records_are_flagged_simulation_only():
    assert _rec().simulation_only is True
    assert _rec(data_kind="real").simulation_only is False
    with pytest.raises(ValueError):
        _rec(data_kind="made_up")


def test_archive_is_append_only_and_idempotent(tmp_path):
    path = tmp_path / "a.jsonl"
    arc = ForecastArchive(path)
    assert arc.append([_rec()]) == (1, 0)
    assert ForecastArchive(path).append([_rec()]) == (0, 1)  # identical re-run is fine
    with pytest.raises(ArchiveConflict):
        ForecastArchive(path).append([replace(_rec(0.70), forecast_id=_rec().forecast_id)])
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["distribution"]["p_home_win"] == 0.62


def test_probabilities_must_be_strictly_inside_unit_interval():
    for bad in (0.0, 1.0, 1.2):
        with pytest.raises(ValueError):
            _rec(bad)
