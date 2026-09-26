"""Schema contracts: observations, imagery, samples, and the dataset registry file."""

from __future__ import annotations

import importlib
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from wave_analysis.registry import validate_registry
from wave_analysis.schemas import ImageRecord, MultimodalSample, Observation
from wave_analysis.schemas.observation import (
    SchemaError,
    coerce_observation_frame,
    empty_observation_frame,
    make_observation_id,
    validate_observation_frame,
)

ROOT = Path(__file__).resolve().parents[2]


def _obs(**kw):
    base = {
        "observation_id": "ndbc:41010:stdmet:wave_significant_height:2024-01-01T00:00:00Z",
        "dataset_id": "ndbc",
        "station_id": "41010",
        "time_utc": datetime(2024, 1, 1, tzinfo=UTC),
        "variable": "wave_significant_height",
        "value": 1.2,
        "units": "m",
        "source_file": "41010h2024.txt.gz",
        "processing_version": "0.1.0",
    }
    base.update(kw)
    return base


def test_observation_model():
    Observation(**_obs())
    with pytest.raises(ValidationError, match="unknown variable"):
        Observation(**_obs(variable="Hs"))
    with pytest.raises(ValidationError, match="UTC"):
        Observation(**_obs(time_utc=datetime(2024, 1, 1, tzinfo=timezone(timedelta(hours=-5)))))
    with pytest.raises(ValidationError):
        Observation(**_obs(time_utc=datetime(2024, 1, 1)))  # naive


def test_observation_frame_validation():
    df = coerce_observation_frame(pd.DataFrame([_obs()]))
    validate_observation_frame(df)
    bad = df.copy()
    bad.loc[0, "units"] = "ft"
    with pytest.raises(SchemaError, match="canonical"):
        validate_observation_frame(bad)
    dup = pd.concat([df, df])
    with pytest.raises(SchemaError, match="duplicate"):
        validate_observation_frame(dup)
    assert list(empty_observation_frame().columns) == list(df.columns)


def test_observation_id_is_deterministic():
    t = pd.Timestamp("2024-01-01", tz="UTC")
    assert (
        make_observation_id("ndbc", "41010", "stdmet", "x", t)
        == "ndbc:41010:stdmet:x:2024-01-01T00:00:00Z"
    )


def test_image_record_window():
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    ImageRecord(
        image_id="i",
        camera_id="c",
        media_type="video",
        capture_start_utc=t0,
        capture_end_utc=t0 + timedelta(minutes=3),
    )
    with pytest.raises(ValidationError):
        ImageRecord(
            image_id="i",
            camera_id="c",
            media_type="video",
            capture_start_utc=t0,
            capture_end_utc=t0 - timedelta(seconds=1),
        )


def test_multimodal_sample_example():
    s = MultimodalSample.model_validate(
        {
            "sample_id": "x",
            "group_keys": {"station": "41010", "day": "2024-01-01"},
            "time": {
                "start_utc": "2024-01-01T00:10:00Z",
                "reference_utc": "2024-01-01T00:40:00Z",
                "offset_s": -1800,
            },
            "wave_reference": {
                "reference_station_id": "41010",
                "Hs_m": 1.2,
                "hs_estimator": "spectral_hm0",
            },
            "provenance": {"sources": ["ndbc"], "pipeline_version": "0.1.0"},
        }
    )
    assert s.quality.synchronization == 2


def test_repository_registry_is_valid():
    data = yaml.safe_load((ROOT / "data" / "registry" / "datasets.yaml").read_text())
    assert validate_registry(data) == []
    ids = {d["dataset_id"] for d in data["datasets"]}
    assert {"ndbc", "cdip", "usgs_coastcam"} <= ids


def test_registry_adapters_resolve():
    """Every ``adapter`` in the registry names an importable module or attribute."""
    reg = yaml.safe_load((ROOT / "data" / "registry" / "datasets.yaml").read_text())
    adapters = [d["adapter"] for d in reg["datasets"] if d.get("adapter")]
    assert adapters
    for path in adapters:
        module, _, attr = path.rpartition(".")
        try:
            importlib.import_module(path)
        except ModuleNotFoundError:
            assert hasattr(importlib.import_module(module), attr), path


def test_registry_validation_catches_problems():
    probs = validate_registry({"datasets": [{"dataset_id": "Bad-Id"}]})
    assert probs and any("dataset_id" in p for p in probs)
    assert validate_registry({}) == ["top-level key 'datasets' must be a list"]
