from __future__ import annotations

import hashlib
import time
from pathlib import Path

import requests

from src.acquisition.wayback import wayback_raw_url


def safe_name(value: str) -> str:
    """Return a deterministic filename-safe identifier for a URL."""
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def fetch_wayback_html(
    timestamp: str,
    original_url: str,
    out_dir: Path,
    sleep_seconds: float = 2.0,
    max_attempts: int = 5,
    backoff_seconds: float = 5.0,
) -> tuple[Path | None, str | None]:
    """
    Download an archived HTML document from Wayback Machine.

    Transient HTTP errors and connection-related exceptions are retried
    using exponential backoff. Permanent HTTP errors are returned
    immediately.

    Returns
    -------
    tuple[Path | None, str | None]
        (html_path, None) on success or (None, error_message) on failure.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be >= 0")

    if backoff_seconds < 0:
        raise ValueError("backoff_seconds must be >= 0")

    out_dir.mkdir(parents=True, exist_ok=True)

    url = wayback_raw_url(timestamp, original_url)

    filename = f"{timestamp}_{safe_name(original_url)}.html"
    path = out_dir / filename

    # Reuse an HTML file that has already been downloaded.
    if path.exists():
        return path, None

    transient_statuses = {
        429,
        500,
        502,
        503,
        504,
    }

    last_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        # Important: do not reuse a response from a previous attempt.
        response: requests.Response | None = None

        try:
            response = requests.get(
                url,
                timeout=(20, 60),
                headers={
                    "User-Agent": "TFM-Violencia-Machista/0.1",
                },
            )

            status = response.status_code

            if status == 200:
                path.write_text(
                    response.text,
                    encoding="utf-8",
                    errors="ignore",
                )

                time.sleep(sleep_seconds)

                return path, None

            last_error = f"HTTP {status}"

            # Permanent HTTP error: retrying is not expected
            # to change the result.
            if status not in transient_statuses:
                time.sleep(sleep_seconds)
                return None, last_error

        except (
            requests.exceptions.ConnectTimeout,
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
        ) as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        except requests.exceptions.RequestException as exc:
            return None, f"{type(exc).__name__}: {exc}"

        # No additional delay is necessary after the final attempt.
        if attempt == max_attempts:
            break

        # Respect Retry-After when the current HTTP response provides it.
        retry_after: float | None = None

        if response is not None:
            value = response.headers.get("Retry-After")

            if value is not None:
                try:
                    retry_after = float(value)
                except (TypeError, ValueError):
                    retry_after = None

        if retry_after is not None:
            delay = retry_after
        else:
            delay = backoff_seconds * (2 ** (attempt - 1))

        time.sleep(delay)

    return None, last_error