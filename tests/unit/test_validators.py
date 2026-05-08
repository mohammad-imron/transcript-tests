# tests/unit/test_validators.py
"""
Pytest unit tests for basic metadata and transcript validation rules.
This file complements the JSON Schema checks by exercising semantic validators:
- ISO 8601 timestamp validation (basic Zulu format)
- Duration mismatch detection
- Speaker name consistency (first/last vs full_name)
- Word-level timestamp monotonicity checks
- Basic sanity check for edge_all_open_tabs context (read-only)
"""

import json
from pathlib import Path
from datetime import datetime
import pytest
from jsonschema import validate, ValidationError

SCHEMA_PATH = Path("schemas/metadata.schema.json")
FIXTURES_DIR = Path("tests/fixtures")

SCHEMA = json.loads(SCHEMA_PATH.read_text())


# --- Helper validators -----------------------------------------------------

def is_valid_iso8601_z(dt_str: str) -> bool:
    """
    Accepts a strict subset of ISO 8601 datetimes in Zulu form:
    YYYY-MM-DDTHH:MM:SSZ
    Returns True if parseable, False otherwise.
    """
    try:
        datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except Exception:
        return False


def duration_mismatch(audio_ms: int, transcript_ms: int, pct_threshold: float = 0.05, abs_threshold_ms: int = 10000):
    """
    Returns (mismatch_bool, details)
    - mismatch_bool True when difference exceeds either percentage threshold or absolute threshold.
    """
    if audio_ms is None or transcript_ms is None:
        return False, "one or both durations missing"
    if audio_ms == 0:
        return transcript_ms != 0, "audio duration zero"
    diff = abs(audio_ms - transcript_ms)
    pct = diff / audio_ms
    if diff > abs_threshold_ms or pct > pct_threshold:
        return True, {"audio_ms": audio_ms, "transcript_ms": transcript_ms, "diff_ms": diff, "pct": pct}
    return False, {"diff_ms": diff, "pct": pct}


def speaker_name_consistent(speaker_obj: dict):
    """
    Basic consistency check:
    - If full_name present and first_name/last_name present, ensure they match.
    Returns (is_consistent, message)
    """
    full = speaker_obj.get("full_name")
    first = speaker_obj.get("first_name")
    last = speaker_obj.get("last_name")
    if not full:
        return True, "no full_name to compare"
    if first and last:
        composed = f"{first} {last}"
        if composed.strip().lower() == full.strip().lower():
            return True, "consistent"
        return False, {"expected": composed, "full_name": full}
    # If only one of first/last present, warn but not fail schema-level
    return True, "partial name info"


def word_starts_monotonic(words: list):
    """
    words: list of dicts with numeric 'start' keys (units assumed ms)
    Returns (is_monotonic, offending_indices)
    """
    last = -1
    offenders = []
    for i, w in enumerate(words):
        s = w.get("start")
        if s is None:
            offenders.append({"index": i, "reason": "missing_start"})
            continue
        if not isinstance(s, int):
            offenders.append({"index": i, "reason": "non_integer_start", "value": s})
            continue
        if s < last:
            offenders.append({"index": i, "reason": "decreasing_start", "value": s, "previous": last})
        last = s
    return (len(offenders) == 0), offenders


# --- Tests ---------------------------------------------------------------

def test_valid_metadata_schema():
    """Schema-level validation should accept the provided valid fixture."""
    payload = json.loads((FIXTURES_DIR / "valid_metadata.json").read_text())
    # Should not raise
    validate(instance=payload, schema=SCHEMA)


def test_invalid_timestamp_semantic():
    """
    Schema may accept string format but semantic validator should reject impossible hour/minute.
    Fixture contains created_at with hour 26 and minute 61.
    """
    payload = json.loads((FIXTURES_DIR / "invalid_metadata_timestamp.json").read_text())
    created_at = payload.get("created_at")
    assert created_at is not None
    assert is_valid_iso8601_z(created_at) is False


def test_duration_mismatch_detection():
    """Detects significant duration mismatches beyond thresholds."""
    # Case: audio 630000 ms, transcript 605000 ms -> diff 25000 ms (25s)
    mismatch, details = duration_mismatch(630000, 605000, pct_threshold=0.05, abs_threshold_ms=10000)
    assert mismatch is True
    assert isinstance(details, dict)
    assert details["diff_ms"] == 25000
    # Case within tolerance
    mismatch2, _ = duration_mismatch(600000, 595000, pct_threshold=0.05, abs_threshold_ms=10000)
    assert mismatch2 is False


def test_speaker_name_consistency_checks():
    """Validate detection of inconsistent speaker name fields."""
    # consistent
    s1 = {"post_asr_label": "S1", "first_name": "Jerry", "last_name": "Sellingman", "full_name": "Jerry Sellingman"}
    ok, msg = speaker_name_consistent(s1)
    assert ok is True

    # inconsistent
    s2 = {"post_asr_label": "S2", "first_name": "Terry", "last_name": "Sellingman", "full_name": "Jerry Sellingman"}
    ok2, msg2 = speaker_name_consistent(s2)
    assert ok2 is False
    assert "expected" in msg2 and "full_name" in msg2


def test_word_timestamps_monotonicity_positive():
    """Monotonic increasing starts should pass."""
    words = [{"start": 100}, {"start": 200}, {"start": 200}, {"start": 350}]
    ok, offenders = word_starts_monotonic(words)
    # equality allowed (simultaneous tokens) but decreasing would fail
    assert ok is True
    assert offenders == []


def test_word_timestamps_monotonicity_negative():
    """Non-monotonic starts should be detected."""
    words = [{"start": 100}, {"start": 90}, {"start": 200}]
    ok, offenders = word_starts_monotonic(words)
    assert ok is False
    assert len(offenders) >= 1
    assert offenders[0]["reason"] == "decreasing_start"


def test_schema_and_semantic_combined_for_fixture():
    """
    Full check: schema validation + semantic checks for the valid fixture.
    Ensures the fixture passes both layers.
    """
    payload = json.loads((FIXTURES_DIR / "valid_metadata.json").read_text())
    # schema
    validate(instance=payload, schema=SCHEMA)
    # semantic: created_at
    assert is_valid_iso8601_z(payload["created_at"])
    # durations
    audio = payload.get("audio_duration_ms")
    transcript = payload.get("transcript_duration_ms")
    mismatch, _ = duration_mismatch(audio, transcript)
    assert mismatch is False
    # speakers
    for sp in payload.get("speakers", []):
        ok, _ = speaker_name_consistent(sp)
        # allow partial name info; only fail if explicit mismatch
        assert ok is True
