# tests/integration/test_async_flow.py
"""
Integration test: POST a job (async) and poll GET until processed.
- Posts metadata + small raw_transcript as multipart/form-data.
- Expects 202 Accepted with job_id (or 200 with immediate result).
- Polls GET /transcripts/:id until status == "processed" or timeout.
- Validates final payload contains 'transcript' and expected fields.
"""

import json
import time
import requests
from pathlib import Path

FIXTURES_DIR = Path("tests/fixtures")

def load_fixture(name: str):
    return json.loads((FIXTURES_DIR / name).read_text())

def post_job(api_base: str, metadata: dict, extra_parts: dict = None, headers: dict = None):
    files = {
        "metadata": (None, json.dumps(metadata), "application/json")
    }
    if extra_parts:
        files.update(extra_parts)
    return requests.post(f"{api_base}/transcripts/process", files=files, headers=headers or {}, timeout=30)

def test_async_post_and_polling(api_base):
    metadata = load_fixture("valid_metadata.json")
    # small raw transcript part
    extra = {"raw_transcript": (None, "This is a short transcript used for integration testing.", "text/plain")}
    resp = post_job(api_base, metadata, extra_parts=extra)
    assert resp.status_code in (200, 202), f"Unexpected status {resp.status_code}: {resp.text}"
    body = resp.json()
    # Accept either immediate processing or queued job
    job_id = body.get("job_id") or body.get("transcript_id")
    assert job_id, f"No job identifier returned in response body: {body}"

    # Poll until processed or timeout
    timeout_seconds = 60
    interval = 2
    elapsed = 0
    final = None
    while elapsed < timeout_seconds:
        g = requests.get(f"{api_base}/transcripts/{job_id}", timeout=10)
        assert g.status_code == 200, f"GET returned {g.status_code}: {g.text}"
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
    assert "transcript" in final, f"Processed payload missing 'transcript': {final}"
    # Basic checks on transcript object
    transcript = final["transcript"]
    assert isinstance(transcript.get("text", ""), str)
    # Warnings may be present; ensure structure is list if present
    warnings = final.get("warnings")
    if warnings is not None:
        assert isinstance(warnings, list)

