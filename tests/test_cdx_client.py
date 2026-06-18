from src.acquisition.cdx_client import build_cdx_params, CDXRecord


def test_build_cdx_params_basic():
    params = build_cdx_params(
        url_pattern="clarin.com/*",
        from_timestamp="20150101000000",
        to_timestamp="20151231235959",
    )

    assert params["url"] == "clarin.com/*"
    assert params["from"] == "20150101000000"
    assert params["to"] == "20151231235959"
    assert params["output"] == "json"
    assert "statuscode:200" in params["filter"]
    assert "mimetype:text/html" in params["filter"]


def test_cdx_record_archive_url():
    record = CDXRecord(
        timestamp="20150101000000",
        original="https://www.clarin.com/sociedad/ejemplo.html",
    )

    assert record.archive_url == (
        "https://web.archive.org/web/20150101000000/"
        "https://www.clarin.com/sociedad/ejemplo.html"
    )


def test_cdx_record_to_dict():
    record = CDXRecord(
        timestamp="20150101000000",
        original="https://example.com/article",
        statuscode="200",
        mimetype="text/html",
        digest="ABC123",
        length="12345",
    )

    data = record.to_dict()

    assert data["timestamp"] == "20150101000000"
    assert data["original"] == "https://example.com/article"
    assert data["statuscode"] == "200"
    assert data["mimetype"] == "text/html"
    assert data["digest"] == "ABC123"
    assert data["length"] == "12345"
    assert data["archive_url"].startswith("https://web.archive.org/web/")