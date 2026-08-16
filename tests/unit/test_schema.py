"""Tests for schema validation and CSV-row flattening — in particular the
`company` field, which is a nested object in the live API but a flat string
in the dashboard CSV export. See schema.py's CompanyRef for context."""
from src.ingestion.fetch import _job_row
from src.ingestion.schema import JobRaw


def test_company_as_nested_api_object_is_flattened():
    record = {
        "title": "Data Architect (AWS, Azure, GCP)",
        "company": {"id": 54, "created": "2026-01-01", "name": "Acme Corp", "tags": []},
    }
    validated = JobRaw.from_api_record(record)

    assert validated.company_name == "Acme Corp"
    assert validated.company_id == 54

    row = _job_row(validated)
    assert row["company"] == "Acme Corp"
    assert row["company_id"] == 54


def test_company_as_plain_string_from_csv_export_still_works():
    record = {"title": "ML Engineer", "company": "Acme Corp"}
    validated = JobRaw.from_api_record(record)

    assert validated.company_name == "Acme Corp"
    assert validated.company_id is None

    row = _job_row(validated)
    assert row["company"] == "Acme Corp"
    assert row["company_id"] is None


def test_missing_company_does_not_raise():
    record = {"title": "Data Scientist"}
    validated = JobRaw.from_api_record(record)

    assert validated.company_name is None
    row = _job_row(validated)
    assert row["company"] is None


def test_company_is_agency_flattened_from_nested_company():
    """is_agency lives under company per the confirmed OpenAPI schema,
    not top-level on the job — this locks that in."""
    record = {
        "title": "IT Business Analyst",
        "company": {"id": 41144, "name": "Some Agency", "is_agency": True},
    }
    validated = JobRaw.from_api_record(record)

    assert validated.company_is_agency is True
    row = _job_row(validated)
    assert row["company_is_agency"] is True


def test_topics_and_tags_are_pipe_joined():
    record = {
        "title": "ML Engineer",
        "topics": [
            {"id": 1, "name": "Data, AI, and Machine Learning", "is_main": True},
            {"id": 2, "name": "Software Engineering", "is_main": False},
        ],
        "tags": [{"id": 10, "name": "python", "tag_type": "skill"}],
    }
    validated = JobRaw.from_api_record(record)

    assert validated.topic_names == "Data, AI, and Machine Learning|Software Engineering"
    assert validated.topic_ids == "1|2"
    assert validated.tag_names == "python"

    row = _job_row(validated)
    assert row["topics"] == "Data, AI, and Machine Learning|Software Engineering"
    assert row["topic_ids"] == "1|2"


def test_empty_topics_and_tags_produce_empty_string_not_error():
    record = {"title": "Data Scientist"}
    validated = JobRaw.from_api_record(record)

    assert validated.topic_names == ""
    row = _job_row(validated)
    assert row["topics"] == ""
