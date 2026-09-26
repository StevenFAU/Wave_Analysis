"""Live WebCOOS checks (``pytest -m network``); skipped when no API token is configured."""

from __future__ import annotations

import pandas as pd
import pytest

from wave_analysis.sources.webcoos import (
    YIN2025_CAMERAS,
    TokenError,
    WebCOOSClient,
    camera_table,
    element_table,
    load_token,
)

pytestmark = pytest.mark.network


@pytest.fixture(scope="module")
def client():
    try:
        token = load_token()
    except TokenError as exc:
        pytest.skip(str(exc))
    with WebCOOSClient(token) as c:
        yield c


def test_yin_cameras_listed_with_stills(client):
    cams = camera_table(client.assets()).set_index("camera")
    assert set(YIN2025_CAMERAS) <= set(cams.index)
    assert cams.loc[list(YIN2025_CAMERAS), "stills_service"].notna().all()


def test_elements_have_times_names_and_sizes(client):
    end = pd.Timestamp.now(tz="UTC").floor("D")
    els = element_table(
        client.elements(
            "currituck_hampton_inn-one-minute-stills-s3", end - pd.Timedelta(days=3), end
        ),
        "currituck_hampton_inn",
    )
    assert len(els) > 0
    assert els["size_bytes"].notna().all() and els["url"].str.startswith("https://").all()
