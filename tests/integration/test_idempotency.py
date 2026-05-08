# tests/integration/test_idempotency.py
"""
Integration test for Idempotency-Key behavior.
- Posts the same payload twice with identical Idempotency-Key header.
- Expects the API to return the same job_id/transcript_id for both requests.
"""

import json
import requests
from pathlib import Path

FIXTURES_DIR = Path("tests/fixtures")

def load_fixture(name: str):
    return json.loads((FIXTURES_DIR / name).read_text())

def post_metadata(api_base: str, metadata: dict, headers: dict = None):
    files = {"metadata": (None, json.dumps(metadata), "application/json")}
    return requests.post(f"{api_base}/transcripts/process", files=files, headers=headers or {}, timeout=30)

def test_idempotency_same_key_returns_same_job(api_base):
    metadata = load_fixture("valid_metadata.json")
    headers = {"Idempotency-Key": "integration-idempotency-test-001"}
    r1 = post_metadata(api_base, metadata, headers=headers)
    assert r1.status_code in (200, 202), f"First request failed: {r1.status_code} {r1.text}"
    b1 = r1.json()
    id1 = b1.get("job_id") or b1.get("transcript_id")
    assert id1, f"First response missing job id: {b1}"

    r2 = post_metadata(api_base, metadata, headers=headers)
    assert r2.status_code in (200, 202), f"Second request failed: {r2.status_code} {r2.text}"
    b2 = r2.json()
    id2 = b2.get("job_id") or b2.get("transcript_id")
    assert id2, f"Second response missing job id: {b2}"

    assert id1 == id2, f"Idempotency violated: {id1} != {id2}"
