# tests/workflow/test_transitions_api.py
"""
Integration-style tests for the workflow state machine via the API.
These tests assume the service exposes endpoints to manipulate and query
a transcript/work item state, for example:
  POST /workitems            -> create work item (returns id)
  POST /workitems/{id}/transition -> apply transition with payload {"target_state": "...", "reason": "..."}
  GET  /workitems/{id}      -> fetch work item (includes "state" and audit log)

If your real API uses different routes, adapt the helper functions below.
The tests cover:
- valid forward transitions
- invalid transitions (skips)
- idempotency of transitions
- audit log presence
- precondition enforcement (e.g., TRANSCRIBED requires transcript artifact)
"""

import os
import time
import uuid
import requests
import pytest
from pathlib import Path

API_BASE = os.environ.get("TRANSCRIPT_API_BASE", "http://localhost:8000")
TIMEOUT = 10

# Helper functions ---------------------------------------------------------

def create_workitem(api_base: str, payload: dict = None):
    """Create a new work item in state NEW. Returns the created item JSON."""
    payload = payload or {"title": f"test-{uuid.uuid4()}", "initial_state": "NEW"}
    r = requests.post(f"{api_base}/workitems", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def get_workitem(api_base: str, item_id: str):
    r = requests.get(f"{api_base}/workitems/{item_id}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

def transition_workitem(api_base: str, item_id: str, target_state: str, actor: str = "system", reason: str = None, headers: dict = None):
    payload = {"target_state": target_state}
    if reason is not None:
        payload["reason"] = reason
    h = headers or {}
    h.update({"X-Actor": actor})
    r = requests.post(f"{api_base}/workitems/{item_id}/transition", json=payload, headers=h, timeout=TIMEOUT)
    return r

# Tests --------------------------------------------------------------------

@pytest.mark.integration
def test_valid_forward_transitions(api_base=API_BASE):
    """
    NEW -> ASSIGNED -> TRANSCRIBED -> REVIEWED -> COMPLETED
    Each step should succeed and produce an audit entry.
    """
    item = create_workitem(API_BASE)
    item_id = item["id"]

    # NEW -> ASSIGNED
    r = transition_workitem(API_BASE, item_id, "ASSIGNED", actor="manager")
    assert r.status_code == 200
    j = get_workitem(API_BASE, item_id)
    assert j["state"] == "ASSIGNED"
    assert any(a["to"] == "ASSIGNED" for a in j.get("audit", []))

    # ASSIGNED -> TRANSCRIBED (simulate attaching transcript artifact first)
    # Attach artifact (if API supports). If not, assume precondition satisfied by payload.
    attach = requests.post(f"{API_BASE}/workitems/{item_id}/artifacts", json={"type": "transcript", "content": "dummy"}, timeout=TIMEOUT)
    # ignore attach result if endpoint not present; proceed to transition
    r = transition_workitem(API_BASE, item_id, "TRANSCRIBED", actor="transcriber")
    assert r.status_code == 200
    j = get_workitem(API_BASE, item_id)
    assert j["state"] == "TRANSCRIBED"
    assert any(a["to"] == "TRANSCRIBED" for a in j.get("audit", []))

    # TRANSCRIBED -> REVIEWED
    r = transition_workitem(API_BASE, item_id, "REVIEWED", actor="reviewer")
    assert r.status_code == 200
    j = get_workitem(API_BASE, item_id)
    assert j["state"] == "REVIEWED"
    assert any(a["to"] == "REVIEWED" for a in j.get("audit", []))

    # REVIEWED -> COMPLETED
    r = transition_workitem(API_BASE, item_id, "COMPLETED", actor="admin")
    assert r.status_code == 200
    j = get_workitem(API_BASE, item_id)
    assert j["state"] == "COMPLETED"
    assert any(a["to"] == "COMPLETED" for a in j.get("audit", []))


@pytest.mark.integration
def test_invalid_skip_transitions_rejected(api_base=API_BASE):
    """
    Attempt to skip steps (e.g., NEW -> TRANSCRIBED) and expect 4xx rejection.
    """
    item = create_workitem(API_BASE)
    item_id = item["id"]

    # NEW -> TRANSCRIBED (skip ASSIGNED)
    r = transition_workitem(API_BASE, item_id, "TRANSCRIBED", actor="transcriber")
    assert r.status_code in (400, 409, 422), f"Expected client error for invalid skip, got {r.status_code}: {r.text}"
    # Ensure state unchanged
    j = get_workitem(API_BASE, item_id)
    assert j["state"] == "NEW"


@pytest.mark.integration
def test_idempotent_transition(api_base=API_BASE):
    """
    Re-applying the same transition should be idempotent (no duplicate audit entries or side effects).
    """
    item = create_workitem(API_BASE)
    item_id = item["id"]

    r1 = transition_workitem(API_BASE, item_id, "ASSIGNED", actor="manager")
    assert r1.status_code == 200
    j1 = get_workitem(API_BASE, item_id)
    audits_before = list(j1.get("audit", []))

    # Repeat same transition
    r2 = transition_workitem(API_BASE, item_id, "ASSIGNED", actor="manager")
    # Accept either 200 (no-op) or 409/422 indicating already in that state
    assert r2.status_code in (200, 409, 422)
    j2 = get_workitem(API_BASE, item_id)
    audits_after = list(j2.get("audit", []))

    # No new audit entries for a true idempotent no-op
    assert len(audits_after) == len(audits_before)


@pytest.mark.integration
def test_precondition_enforced_transcribed_requires_artifact(api_base=API_BASE):
    """
    TRANSCRIBED transition should require a transcript artifact to be present.
    If artifact missing, transition should be rejected.
    """
    item = create_workitem(API_BASE)
    item_id = item["id"]

    # Move to ASSIGNED first
    r = transition_workitem(API_BASE, item_id, "ASSIGNED", actor="manager")
    assert r.status_code == 200

    # Attempt TRANSCRIBED without attaching artifact
    r = transition_workitem(API_BASE, item_id, "TRANSCRIBED", actor="transcriber")
    assert r.status_code in (400, 422), f"Expected precondition failure, got {r.status_code}: {r.text}"
    j = get_workitem(API_BASE, item_id)
    assert j["state"] == "ASSIGNED"
