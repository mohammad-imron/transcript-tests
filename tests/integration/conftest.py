# tests/integration/conftest.py
"""
Pytest fixtures for integration tests.
Provides:
- api_base fixture (reads TRANSCRIPT_API_BASE env var)
- helper to check service availability and skip tests if unreachable
- small helper to load fixtures
"""

import os
import json
import pytest
import requests
from pathlib import Path

FIXTURES_DIR = Path("tests/fixtures")
DEFAULT_BASE = "http://localhost:8000"

@pytest.fixture(scope="session")
def api_base():
    """Base URL for the transcript API under test."""
    return os.environ.get("TRANSCRIPT_API_BASE", DEFAULT_BASE)

def load_fixture(name: str):
    return json.loads((FIXTURES_DIR / name).read_text())

def is_service_up(base_url: str, timeout: float = 2.0) -> bool:
    """Lightweight health check: GET /health or root; tolerant to 404."""
    try:
        r = requests.get(base_url, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False

@pytest.fixture(autouse=True)
def require_service(api_base):
    """
    Skip integration tests if the service is not reachable.
    This prevents CI failures when the API is not started.
    """
    if not is_service_up(api_base):
        pytest.skip(f"Integration service not reachable at {api_base}")
