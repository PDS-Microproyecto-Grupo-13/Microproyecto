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
def test_iter_paginated_stops_at_max_pages():
    responses.add(
        responses.GET,
        "https://foorilla.com/api/v1/hiring/job/",
        json={"results": [{"id": "1"}], "count": 300, "pages": 300, "page": 1, "page_size": 1},
        status=200,
    )
    responses.add(
        responses.GET,
        "https://foorilla.com/api/v1/hiring/job/",
        json={"results": [{"id": "2"}], "count": 300, "pages": 300, "page": 2, "page_size": 1},
        status=200,
    )

    client = FoorillaClient(_config())
    records = list(client.iter_paginated("hiring/job/", max_pages=2))

    # Should stop after page 2 even though 300 pages exist.
    assert [r["id"] for r in records] == ["1", "2"]
    assert len(responses.calls) == 2


@responses.activate
def test_iter_jobs_serializes_topic_as_repeated_query_param():
    responses.add(
        responses.GET,
        "https://foorilla.com/api/v1/hiring/job/",
        json={"results": [{"id": "1"}], "count": 1, "pages": 1, "page": 1, "page_size": 2},
        status=200,
    )

    client = FoorillaClient(_config())
    list(client.iter_jobs(topic=[17126, 42]))

    from urllib.parse import urlparse, parse_qs
    query = parse_qs(urlparse(responses.calls[0].request.url).query)
    assert query["topic"] == ["17126", "42"]


@responses.activate
def test_iter_jobs_topic_accepts_single_int():
    responses.add(
        responses.GET,
        "https://foorilla.com/api/v1/hiring/job/",
        json={"results": [{"id": "1"}], "count": 1, "pages": 1, "page": 1, "page_size": 2},
        status=200,
    )

    client = FoorillaClient(_config())
    list(client.iter_jobs(topic=17126))

    from urllib.parse import urlparse, parse_qs
    query = parse_qs(urlparse(responses.calls[0].request.url).query)
    assert query["topic"] == ["17126"]
    responses.add(
        responses.GET,
        "https://foorilla.com/api/v1/hiring/job/",
        json={"results": [{"id": "1"}], "count": 1, "pages": 1, "page": 1, "page_size": 2},
        status=200,
    )

    client = FoorillaClient(_config())
    list(client.iter_jobs(extra_params={"topic": "17126"}))

    from urllib.parse import urlparse, parse_qs
    query = parse_qs(urlparse(responses.calls[0].request.url).query)
    assert query["topic"] == ["17126"]


@responses.activate
def test_iter_jobs_published_date_filters():
    responses.add(
        responses.GET,
        "https://foorilla.com/api/v1/hiring/job/",
        json={"results": [{"id": "1"}], "count": 1, "pages": 1, "page": 1, "page_size": 2},
        status=200,
    )

    client = FoorillaClient(_config())
    list(client.iter_jobs(published_after="2026-08-01", published_before="2026-08-15"))

    from urllib.parse import urlparse, parse_qs
    query = parse_qs(urlparse(responses.calls[0].request.url).query)
    assert query["published_after"] == ["2026-08-01"]
    assert query["published_before"] == ["2026-08-15"]


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
