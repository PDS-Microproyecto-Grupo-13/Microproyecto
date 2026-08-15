import responses

from src.ingestion.client import FoorillaClient
from src.ingestion.config import FoorillaConfig


def _config() -> FoorillaConfig:
    return FoorillaConfig(
        api_key="test-key",
        base_url="https://foorilla.com/api/v1",
        requests_per_second=1000,  # avoid throttling slowing tests down
        page_size=2,
    )


@responses.activate
def test_iter_paginated_walks_all_pages():
    responses.add(
        responses.GET,
        "https://foorilla.com/api/v1/hiring/job/",
        json={"results": [{"id": "1"}, {"id": "2"}], "count": 3, "pages": 2, "page": 1, "page_size": 2},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://foorilla.com/api/v1/hiring/job/",
        json={"results": [{"id": "3"}], "count": 3, "pages": 2, "page": 2, "page_size": 2},
        status=200,
    )

    client = FoorillaClient(_config())
    records = list(client.iter_paginated("hiring/job/"))

    assert [r["id"] for r in records] == ["1", "2", "3"]


@responses.activate
def test_iter_paginated_stops_on_empty_results():
    responses.add(
        responses.GET,
        "https://foorilla.com/api/v1/hiring/job/",
        json={"results": [], "count": 0, "pages": 0, "page": 1, "page_size": 2},
        status=200,
    )

    client = FoorillaClient(_config())
    records = list(client.iter_paginated("hiring/job/"))

    assert records == []


@responses.activate
def test_auth_header_uses_api_key_not_bearer():
    responses.add(
        responses.GET,
        "https://foorilla.com/api/v1/hiring/job/",
        json={"results": [], "count": 0, "pages": 0, "page": 1, "page_size": 2},
        status=200,
    )

    client = FoorillaClient(_config())
    list(client.iter_paginated("hiring/job/"))

    sent_headers = responses.calls[0].request.headers
    assert sent_headers["Api-Key"] == "test-key"
    assert "Authorization" not in sent_headers


@responses.activate
def test_iter_jobs_applies_filters():
    responses.add(
        responses.GET,
        "https://foorilla.com/api/v1/hiring/job/",
        json={"results": [{"id": "1"}], "count": 1, "pages": 1, "page": 1, "page_size": 2},
        status=200,
    )

    client = FoorillaClient(_config())
    list(client.iter_jobs(title="ml engineer", location="remote"))

    from urllib.parse import urlparse, parse_qs
    query = parse_qs(urlparse(responses.calls[0].request.url).query)
    assert query["title"] == ["ml engineer"]
    assert query["location"] == ["remote"]
