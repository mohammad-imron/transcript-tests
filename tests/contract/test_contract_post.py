# tests/contract/test_contract_post.py
"""
Contract tests for the /transcripts/process endpoint.
These tests verify the API adheres to the expected request/response contract:
- Accepts multipart/form-data with metadata and optional transcript parts
- Returns appropriate HTTP status codes (200/202 for success, 4xx for validation errors)
- Response body contains expected fields (job_id, transcript_id, status, errors/warnings)
- Invalid metadata (semantic errors) produce 4xx with structured error details

Notes:
- The tests expect the API under test to be reachable at the URL defined by the
  TRANSCRIPT_API_BASE environment variable or default to http://localhost:8000.
- Integration tests that exercise the live service should run in an isolated test environment.
"""

import os
import json
import time
import pytest
import requests
from jsonschema import validate, ValidationError
from pathlib import Path

BASE_URL = os.environ.get("TRANSCRIPT_API_BASE", "http://localhost:8000")
SCHEMA_DIR = Path("schemas")
FIXTURES_DIR = Path("tests/fixtures")

# Minimal expected response schema for POST /transcripts/process (contract-level)
POST_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "job_id": {"type": "string"},
        "transcript_id": {"type": "string"},
        "status": {"type": "string"},
        "message": {"type": "string"},
        "warnings": {"type": "array"},
        "errors": {"type": "array"}
    },
    "required": ["status"],
    "additionalProperties": True
}


def load_fixture(name: str):
    return json.loads((FIXTURES_DIR / name).read_text())


def post_metadata(metadata: dict, extra_files: dict = None, headers: dict = None):
    """
    Helper to POST metadata as multipart/form-data.
    metadata: dict -> sent as a metadata part
    extra_files: mapping name -> (filename, content, content_type) or simple (None, text, type)
    """
    files = {
        "metadata": (None, json.dumps(metadata), "application/json")
    }
    if extra_files:
        files.update(extra_files)
    h = headers or {}
    return requests.post(f"{BASE_URL}/transcripts/process", files=files, headers=h, timeout=30)


@pytest.mark.contract
def test_post_valid_metadata_returns_202_or_200():
    """
    POST a valid metadata payload and expect a 202 Accepted (async) or 200 OK (sync).
    Validate the response body contains the expected contract fields.
    """
    metadata = load_fixture("valid_metadata.json")
    resp = post_metadata(metadata)
    assert resp.status_code in (200, 202), f"Unexpected status: {resp.status_code} body: {resp.text}"
    body = resp.json()
    # Validate contract shape
    validate(instance=body, schema=POST_RESPONSE_SCHEMA)
    # status must be present and be one of expected values
    assert body["status"] in ("queued", "processing", "processed", "failed", "accepted", "completed", "queued")


@pytest.mark.contract
def test_post_invalid_timestamp_returns_4xx_with_details():
    """
    POST metadata with an invalid created_at value and expect a 4xx response
    with structured error details referencing the created_at field.
    """
    metadata = load_fixture("invalid_metadata_timestamp.json")
    resp = post_metadata(metadata)
    assert resp.status_code in (400, 422), f"Expected 400/422 for invalid timestamp, got {resp.status_code}"
    try:
        body = resp.json()
    except ValueError:
        pytest.fail(f"Response not JSON: {resp.text}")
    # Expect structured error details
    assert "details" in body or "errors" in body or "message" in body
    # If details present, ensure created_at is referenced
    details = body.get("details") or body.get("errors") or []
    if isinstance(details, list):
        # At least one detail should mention created_at
        assert any("created_at" in (d.get("field") or str(d)) for d in details), f"Details did not reference created_at: {details}"


@pytest.mark.contract
def test_post_missing_required_parts_returns_400():
    """
    POST without the required metadata part should return 400 Bad Request.
    """
    # Send an empty multipart (no metadata)
    files = {
        "raw_transcript": (None, "Some transcript text", "text/plain")
    }
    r = requests.post(f"{BASE_URL}/transcripts/process", files=files, timeout=10)
    assert r.status_code in (400, 422), f"Expected 400/422 for missing metadata, got {r.status_code}"


@pytest.mark.contract
def test_post_idempotency_key_returns_same_job_id():
    """
    Posting the same payload with the same Idempotency-Key should return the same job_id.
    This contract test verifies idempotency behavior.
    """
    metadata = load_fixture("valid_metadata.json")
    headers = {"Idempotency-Key": "contract-test-key-12345"}
    r1 = post_metadata(metadata, headers=headers)
    assert r1.status_code in (200, 202)
    body1 = r1.json()
    job1 = body1.get("job_id") or body1.get("transcript_id")
    assert job1, "First response did not include job_id/transcript_id"

    # Repeat request with same key
    r2 = post_metadata(metadata, headers=headers)
    assert r2.status_code in (200, 202)
    body2 = r2.json()
    job2 = body2.get("job_id") or body2.get("transcript_id")
    assert job2 == job1, f"Idempotency violated: {job1} != {job2}"


@pytest.mark.contract
def test_post_returns_warnings_for_duration_mismatch():
    """
    If the payload contains a significant audio/transcript duration mismatch,
    the API should either reject (4xx) or accept and include a 'warnings' entry
    describing the mismatch.
    """
    metadata = load_fixture("valid_metadata.json")
    # Inject a large mismatch
    metadata["audio_duration_ms"] = 630000
    metadata["transcript_duration_ms"] = 605000
    r = post_metadata(metadata)
    assert r.status_code in (202, 200, 422)
    body = r.json()
    if r.status_code in (200, 202):
        # Expect warnings array and a duration_mismatch code or message
        warnings = body.get("warnings", [])
        assert isinstance(warnings, list)
        assert any("duration" in (w.get("code") or w.get("message", "").lower()) or "mismatch" in (w.get("code") or w.get("message", "").lower()) for w in warnings), f"No duration mismatch warning found: {warnings}"
    else:
        # If rejected, ensure details mention transcript_duration_ms or audio_duration_ms
        details = body.get("details") or body.get("errors") or []
        assert any("transcript_duration" in json.dumps(d).lower() or "audio_duration" in json.dumps(d).lower() for d in details), f"Rejection did not reference durations: {details}"


@pytest.mark.contract
def test_post_response_contains_location_header_for_async():
    """
    When the API accepts a job asynchronously (202), it should include a Location header
    pointing to the job or transcript resource.
    """
    metadata = load_fixture("valid_metadata.json")
    r = post_metadata(metadata)
    if r.status_code == 202:
        assert "Location" in r.headers, "202 response missing Location header"
        # Basic sanity: Location should be a path or URL
        loc = r.headers["Location"]
        assert isinstance(loc, str) and len(loc) > 0


# Optional: skip tests that require a running worker if environment variable set
@pytest.mark.skipif(os.environ.get("SKIP_INTEGRATION", "false").lower() == "true", reason="Skipping integration-level contract tests")
def test_full_async_lifecycle_polling():
    """
    End-to-end contract check: POST -> poll GET until processed -> validate GET response contract.
    This test assumes the service will process the job within a reasonable timeout.
    """
    metadata = load_fixture("valid_metadata.json")
    r = post_metadata(metadata)
    assert r.status_code in (200, 202)
    body = r.json()
    job_id = body.get("job_id") or body.get("transcript_id")
    assert job_id, "No job identifier returned"

    # Poll GET /transcripts/:id
    timeout_seconds = 60
    interval = 2
    elapsed = 0
    final = None
    while elapsed < timeout_seconds:
        g = requests.get(f"{BASE_URL}/transcripts/{job_id}", timeout=10)
        assert g.status_code == 200
        j = g.json()
        status = j.get("status")
        if status == "processed":
            final = j
            break
        if status in ("failed", "error"):
            pytest.fail(f"Job failed: {j}")
        time.sleep(interval)
        elapsed += interval

    assert final is not None, f"Job did not reach processed state within {timeout_seconds}s"
    # Basic contract checks on final payload
    assert "transcript" in final
    assert isinstance(final["transcript"].get("text", ""), str)
