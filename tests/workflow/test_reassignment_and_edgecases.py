# tests/workflow/test_reassignment_and_edgecases.py
"""
Tests covering reassignment, reviewer send-back, rollback, concurrency, and bulk operations.
These tests assume the API supports:
- PATCH /workitems/{id} to update assignee
- POST /workitems/{id}/transition with reason for backward transitions
- POST /workitems/bulk-transition for bulk operations (optional)
- ETag or versioning via If-Match header for concurrency control (optional)
Adapt endpoints as needed for your implementation.
"""

import os
import time
import uuid
import requests
import pytest

API_BASE = os.environ.get("TRANSCRIPT_API_BASE", "http://localhost:8000")
TIMEOUT = 10

def create_workitem(api_base: str, payload: dict = None):
    payload = payload or {"title": f"edge-test-{uuid.uuid4()}", "initial_state": "NEW"}
    r = requests.post(f"{api_base}/workitems", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def patch_assignee(api_base: str, item_id: str, assignee: str, actor: str = "manager"):
    headers = {"X-Actor": actor}
    r = requests.patch(f"{api_base}/workitems/{item_id}", json={"assignee": assignee}, headers=headers, timeout=TIMEOUT)
    return r

def transition(api_base: str, item_id: str, target: str, actor: str = "system", reason: str = None, headers: dict = None):
    payload = {"target_state": target}
    if reason:
        payload["reason"] = reason
    h = headers or {}
    h.update({"X-Actor": actor})
    return requests.post(f"{api_base}/workitems/{item_id}/transition", json=payload, headers=h, timeout=TIMEOUT)

def get_item(api_base: str, item_id: str):
    r = requests.get(f"{api_base}/workitems/{item_id}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

# Tests --------------------------------------------------------------------

@pytest.mark.integration
def test_reassignment_updates_assignee_and_audit():
    item = create_workitem(API_BASE)
    item_id = item["id"]

    # Assign to user A
    r = patch_assignee(API_BASE, item_id, "userA", actor="manager")
    assert r.status_code == 200
    j = get_item(API_BASE, item_id)
    assert j.get("assignee") == "userA"
    assert any(a["to"] == "ASSIGNED" or a.get("field") == "assignee" for a in j.get("audit", []))

    # Reassign to user B
    r2 = patch_assignee(API_BASE, item_id, "userB", actor="manager")
    assert r2.status_code == 200
    j2 = get_item(API_BASE, item_id)
    assert j2.get("assignee") == "userB"
    # Audit should contain previous assignee record
    assert any("userA" in str(a) or a.get("from") == "userA" for a in j2.get("audit", []))


@pytest.mark.integration
def test_reviewer_sends_back_to_transcribed_with_reason():
    item = create_workitem(API_BASE)
    item_id = item["id"]

    # Move through happy path to TRANSCRIBED
    requests.post(f"{API_BASE}/workitems/{item_id}/transition", json={"target_state": "ASSIGNED"}, headers={"X-Actor": "manager"}, timeout=TIMEOUT)
    # Attach transcript artifact (if supported)
    requests.post(f"{API_BASE}/workitems/{item_id}/artifacts", json={"type": "transcript", "content": "dummy"}, timeout=TIMEOUT)
    requests.post(f"{API_BASE}/workitems/{item_id}/transition", json={"target_state": "TRANSCRIBED"}, headers={"X-Actor": "transcriber"}, timeout=TIMEOUT)

    # Reviewer rejects and sends back to TRANSCRIBED (backward transition)
    r = transition(API_BASE, item_id, "TRANSCRIBED", actor="reviewer", reason="Missing citations in transcript")
    # Expect 200 OK for a valid send-back, or 202 if asynchronous
    assert r.status_code in (200, 202)
    j = get_item(API_BASE, item_id)
    # State should be TRANSCRIBED and review comments present in audit or review_notes
    assert j["state"] == "TRANSCRIBED"
    assert any("Missing citations" in (a.get("reason") or "") or "review" in (a.get("type") or "") for a in j.get("audit", []))


@pytest.mark.integration
def test_admin_forced_rollback_from_completed():
    item = create_workitem(API_BASE)
    item_id = item["id"]

    # Move to COMPLETED quickly (simulate)
    requests.post(f"{API_BASE}/workitems/{item_id}/transition", json={"target_state": "ASSIGNED"}, headers={"X-Actor": "manager"}, timeout=TIMEOUT)
    requests.post(f"{API_BASE}/workitems/{item_id}/transition", json={"target_state": "TRANSCRIBED"}, headers={"X-Actor": "transcriber"}, timeout=TIMEOUT)
    requests.post(f"{API_BASE}/workitems/{item_id}/transition", json={"target_state": "REVIEWED"}, headers={"X-Actor": "reviewer"}, timeout=TIMEOUT)
    requests.post(f"{API_BASE}/workitems/{item_id}/transition", json={"target_state": "COMPLETED"}, headers={"X-Actor": "admin"}, timeout=TIMEOUT)

    # Admin rollback to TRANSCRIBED with reason
    r = transition(API_BASE, item_id, "TRANSCRIBED", actor="admin", reason="Critical error found post-completion")
    assert r.status_code == 200
    j = get_item(API_BASE, item_id)
    assert j["state"] == "TRANSCRIBED"
    assert any(a.get("actor") == "admin" and "Critical error" in (a.get("reason") or "") for a in j.get("audit", []))


@pytest.mark.integration
def test_concurrent_transitions_handle_race_conditions():
    """
    Simulate two actors attempting to transition TRANSCRIBED -> REVIEWED at the same time.
    Expect one to succeed and the other to receive a conflict (409) or be a no-op.
    """
    item = create_workitem(API_BASE)
    item_id = item["id"]

    # Move to TRANSCRIBED
    requests.post(f"{API_BASE}/workitems/{item_id}/transition", json={"target_state": "ASSIGNED"}, headers={"X-Actor": "manager"}, timeout=TIMEOUT)
    requests.post(f"{API_BASE}/workitems/{item_id}/artifacts", json={"type": "transcript", "content": "dummy"}, timeout=TIMEOUT)
    requests.post(f"{API_BASE}/workitems/{item_id}/transition", json={"target_state": "TRANSCRIBED"}, headers={"X-Actor": "transcriber"}, timeout=TIMEOUT)

    # Fire two concurrent requests
    import concurrent.futures
    def do_review(actor):
        return requests.post(f"{API_BASE}/workitems/{item_id}/transition", json={"target_state": "REVIEWED"}, headers={"X-Actor": actor}, timeout=TIMEOUT)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        fut1 = ex.submit(do_review, "reviewerA")
        fut2 = ex.submit(do_review, "reviewerB")
        r1 = fut1.result()
        r2 = fut2.result()

    statuses = {r1.status_code, r2.status_code}
    # At least one should be 200; the other should be 200 or 409/422
    assert 200 in statuses
    assert statuses.issubset({200, 409, 422})


@pytest.mark.integration
def test_bulk_transition_validates_each_item_individually():
    """
    If the API supports bulk transitions, ensure items that skip steps are rejected individually,
    while valid items are processed. The response should include per-item results.
    """
    # Create two items: one valid path, one that will skip
    item_valid = create_workitem(API_BASE)
    item_skip = create_workitem(API_BASE)

    # Prepare bulk payload: attempt to move both to REVIEWED directly (invalid)
    bulk_payload = {
        "operations": [
            {"id": item_valid["id"], "target_state": "REVIEWED"},
            {"id": item_skip["id"], "target_state": "REVIEWED"}
        ]
    }
    r = requests.post(f"{API_BASE}/workitems/bulk-transition", json=bulk_payload, timeout=TIMEOUT)
    # Accept either 200 with per-item results or 207 Multi-Status
    assert r.status_code in (200, 207)
    body = r.json()
    assert isinstance(body.get("results", []), list)
    # Each result should include id and status/success flag
    ids = {res["id"] for res in body["results"]}
    assert item_valid["id"] in ids and item_skip["id"] in ids
    # At least one result should indicate failure for invalid skip
    assert any(not res.get("success", True) for res in body["results"])
