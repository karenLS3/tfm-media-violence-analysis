from __future__ import annotations

import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CDX_URL = "https://web.archive.org/cdx/search/cdx"


def build_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "TFM-Violencia-Machista/0.1"})

    return session


SESSION = build_session()


def query_cdx_simple(
    url_pattern: str,
    from_date: str,
    to_date: str,
    limit: int = 100,
) -> pd.DataFrame:
    params = {
        "url": url_pattern,
        "from": from_date,
        "to": to_date,
        "output": "json",
        "fl": "timestamp,original,statuscode,mimetype,digest",
        "limit": str(limit),
    }

    response = SESSION.get(CDX_URL, params=params, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(
            f"CDX status={response.status_code}. Preview={response.text[:500]!r}"
        )

    text = response.text.strip()

    if not text:
        return pd.DataFrame(
            columns=["timestamp", "original", "statuscode", "mimetype", "digest"]
        )

    data = response.json()

    if len(data) <= 1:
        return pd.DataFrame(
            columns=["timestamp", "original", "statuscode", "mimetype", "digest"]
        )

    return pd.DataFrame(data[1:], columns=data[0])