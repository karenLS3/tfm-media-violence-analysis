import pytest

from src.acquisition import cdx_client


class FakeResponse:
    """Respuesta HTTP simulada para probar CDX sin acceder a Internet."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        text: str = "",
        json_data=None,
    ):
        self.status_code = status_code
        self.text = text
        self._json_data = json_data

    def json(self):
        return self._json_data


def test_query_cdx_simple_builds_request_and_dataframe(monkeypatch):
    captured = {}

    payload = [
        [
            "timestamp",
            "original",
            "statuscode",
            "mimetype",
            "digest",
        ],
        [
            "20150101000000",
            "https://www.clarin.com/sociedad/ejemplo.html",
            "200",
            "text/html",
            "ABC123",
        ],
    ]

    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout

        return FakeResponse(
            status_code=200,
            text="non-empty response",
            json_data=payload,
        )

    monkeypatch.setattr(
        cdx_client.SESSION,
        "get",
        fake_get,
    )

    result = cdx_client.query_cdx_simple(
        url_pattern="clarin.com/*",
        from_date="20150101000000",
        to_date="20151231235959",
        limit=3,
    )

    assert captured["url"] == cdx_client.CDX_URL
    assert captured["timeout"] == 120

    assert captured["params"] == {
        "url": "clarin.com/*",
        "from": "20150101000000",
        "to": "20151231235959",
        "output": "json",
        "fl": "timestamp,original,statuscode,mimetype,digest",
        "limit": "3",
    }

    assert len(result) == 1

    assert (
        result.iloc[0]["timestamp"]
        == "20150101000000"
    )

    assert result.iloc[0]["original"] == (
        "https://www.clarin.com/sociedad/ejemplo.html"
    )

    assert result.iloc[0]["statuscode"] == "200"
    assert result.iloc[0]["mimetype"] == "text/html"
    assert result.iloc[0]["digest"] == "ABC123"


def test_query_cdx_simple_returns_empty_dataframe_for_empty_response(
    monkeypatch,
):
    def fake_get(url, params, timeout):
        return FakeResponse(
            status_code=200,
            text="",
            json_data=None,
        )

    monkeypatch.setattr(
        cdx_client.SESSION,
        "get",
        fake_get,
    )

    result = cdx_client.query_cdx_simple(
        url_pattern="clarin.com/*",
        from_date="20150101",
        to_date="20150131",
    )

    assert result.empty

    assert result.columns.tolist() == [
        "timestamp",
        "original",
        "statuscode",
        "mimetype",
        "digest",
    ]


def test_query_cdx_simple_raises_on_http_error(
    monkeypatch,
):
    def fake_get(url, params, timeout):
        return FakeResponse(
            status_code=503,
            text="Service unavailable",
            json_data=None,
        )

    monkeypatch.setattr(
        cdx_client.SESSION,
        "get",
        fake_get,
    )

    with pytest.raises(
        RuntimeError,
        match="CDX status=503",
    ):
        cdx_client.query_cdx_simple(
            url_pattern="clarin.com/*",
            from_date="20150101",
            to_date="20150131",
        )
        