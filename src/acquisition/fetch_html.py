from __future__ import annotations

import time
import hashlib
import requests
from pathlib import Path

from src.acquisition.wayback import wayback_raw_url


def safe_name(value: str) -> str:
    return hashlib.md5(value.encode("utf-8")).hexdigest()


def fetch_wayback_html(
    timestamp: str,
    original_url: str,
    out_dir: Path,
    sleep_seconds: float = 1.0,
) -> tuple[Path | None, str | None]:
    out_dir.mkdir(parents=True, exist_ok=True)

    url = wayback_raw_url(timestamp, original_url)
    filename = f"{timestamp}_{safe_name(original_url)}.html"
    path = out_dir / filename

    if path.exists():
        return path, None

    try:
        response = requests.get(
            url,
            timeout=60,
            headers={"User-Agent": "TFM-Violencia-Machista/0.1"},
        )

        time.sleep(sleep_seconds)

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"

        path.write_text(response.text, encoding="utf-8", errors="ignore")
        return path, None

    except Exception as e:
        return None, repr(e)