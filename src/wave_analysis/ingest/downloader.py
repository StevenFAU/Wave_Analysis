"""Polite, byte-preserving HTTP downloader with manifest recording.

Guarantees
----------
* **Bytes are stored exactly as served** (no decompression, no re-encoding).
  ``.txt.gz`` files stay gzipped; parsing happens in a later stage.
* **Atomic writes** - data land in a temporary file and are renamed only after
  the checksum is computed, so an interrupted run never leaves a truncated
  file that looks valid.
* **Every attempt is recorded** in the manifest, including HTTP 404s, so that
  *absence* of a product for a station/period is documented evidence rather
  than an unrecorded assumption.
* **Rate limiting and retries** - NDBC asks users to "limit your retrievals to
  a minimal level" (https://www.ndbc.noaa.gov/faq/rt_data_access.shtml). A
  minimum inter-request interval (default 1 s) and exponential back-off on
  transient errors are applied to every provider.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from wave_analysis import __version__
from wave_analysis.ingest.manifest import EntryStatus, ManifestEntry, utcnow

USER_AGENT = f"wave-analysis/{__version__} (research data pipeline; +https://github.com/StevenFAU/Wave_Analysis)"
_RETRY_STATUS = {429, 500, 502, 503, 504}
_CHUNK_BYTES = 1 << 20


@dataclass
class Downloader:
    """HTTP client wrapper enforcing the project's acquisition rules.

    Parameters
    ----------
    min_interval_s
        Minimum wall-clock seconds between successive requests.
    max_retries
        Retries for connection errors and HTTP 429/5xx.
    timeout_s
        Per-request timeout.
    transport
        Optional ``httpx`` transport (used by tests to avoid the network).
    """

    min_interval_s: float = 1.0
    max_retries: int = 4
    backoff_s: float = 2.0
    timeout_s: float = 60.0
    transport: httpx.BaseTransport | None = None
    _client: httpx.Client = field(init=False, repr=False)
    _last_request: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=self.timeout_s,
            follow_redirects=True,
            transport=self.transport,
        )

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        self._client.close()

    def __enter__(self) -> Downloader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def get(self, url: str) -> httpx.Response:
        """GET with throttling and retries. Returns the final response (any status)."""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                resp = self._client.get(url)
            except httpx.TransportError as exc:
                last_exc = exc
            else:
                if resp.status_code not in _RETRY_STATUS:
                    return resp
                last_exc = httpx.HTTPStatusError(
                    f"HTTP {resp.status_code}", request=resp.request, response=resp
                )
            if attempt < self.max_retries:
                time.sleep(self.backoff_s * 2**attempt)
        assert last_exc is not None
        raise last_exc

    def _stream_to_temp(
        self, url: str, directory: Path, *, compress: bool
    ) -> tuple[httpx.Response, Path | None, str | None, int]:
        """GET ``url`` like :meth:`get`, streaming a 200 body into a temporary file.

        Returns ``(response, temp_path, sha256, size)``; ``temp_path`` is
        ``None`` for any other status. The body is hashed as it arrives, so
        files larger than memory can be fetched. A connection that drops
        mid-body is retried from the start like any other transport error.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            tmp: Path | None = None
            try:
                with self._client.stream("GET", url) as resp:
                    if resp.status_code == 200:
                        directory.mkdir(parents=True, exist_ok=True)
                        fd, name = tempfile.mkstemp(dir=directory, prefix=".partial-")
                        tmp = Path(name)
                        digest = hashlib.sha256()
                        size = 0
                        with os.fdopen(fd, "wb") as fh:
                            gz = (
                                gzip.GzipFile(filename="", fileobj=fh, mode="wb", mtime=0)
                                if compress
                                else None
                            )
                            write = fh.write if gz is None else gz.write
                            for chunk in resp.iter_bytes(_CHUNK_BYTES):
                                digest.update(chunk)
                                size += len(chunk)
                                write(chunk)
                            if gz is not None:
                                gz.close()
                        return resp, tmp, digest.hexdigest(), size
                    if resp.status_code not in _RETRY_STATUS:
                        return resp, None, None, 0
                    last_exc = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}", request=resp.request, response=resp
                    )
            except httpx.TransportError as exc:
                last_exc = exc
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
            except BaseException:
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
                raise
            if attempt < self.max_retries:
                time.sleep(self.backoff_s * 2**attempt)
        assert last_exc is not None
        raise last_exc

    def fetch(
        self,
        url: str,
        dest: str | Path,
        *,
        source_id: str,
        product: str,
        station_id: str | None = None,
        period: str | None = None,
        previous_sha256: str | None = None,
        overwrite: bool = True,
        compress: bool = False,
    ) -> ManifestEntry:
        """Download ``url`` to ``dest`` and return the manifest entry describing it.

        If ``previous_sha256`` is given (from an earlier manifest), the entry's
        status is ``UNCHANGED`` or ``CHANGED`` accordingly; upstream revisions
        of "historical" files do happen and must be detectable.

        With ``compress=True`` the payload is stored gzip-compressed (use a
        ``.gz`` suffix on ``dest``). ``sha256`` and ``size_bytes`` always
        describe the payload *as served*, so checksums stay comparable across
        storage choices; verify with ``sha256_file(path, gunzip=True)``.
        """
        dest = Path(dest)
        base = {
            "source_id": source_id,
            "product": product,
            "station_id": station_id,
            "period": period,
            "url": url,
            "software_version": __version__,
        }
        if dest.exists() and not overwrite:
            raise FileExistsError(dest)
        try:
            resp, tmp, digest, size = self._stream_to_temp(url, dest.parent, compress=compress)
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            return ManifestEntry(
                **base, retrieved_at=utcnow(), status=EntryStatus.FAILED, note=str(exc)[:500]
            )
        if resp.status_code == 404:
            return ManifestEntry(
                **base, retrieved_at=utcnow(), status=EntryStatus.NOT_FOUND, http_status=404
            )
        if tmp is None:
            return ManifestEntry(
                **base,
                retrieved_at=utcnow(),
                status=EntryStatus.FAILED,
                http_status=resp.status_code,
            )
        try:
            os.replace(tmp, dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        if previous_sha256 is None:
            status = EntryStatus.VERIFIED
        elif previous_sha256 == digest:
            status = EntryStatus.UNCHANGED
        else:
            status = EntryStatus.CHANGED
        return ManifestEntry(
            **base,
            local_path=str(dest),
            sha256=digest,
            size_bytes=size,
            http_status=200,
            http_last_modified=resp.headers.get("last-modified"),
            http_etag=resp.headers.get("etag"),
            retrieved_at=utcnow(),
            status=status,
            note="stored gzip-compressed; sha256/size_bytes are of the served payload"
            if compress
            else None,
        )
