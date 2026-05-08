Project Overview
This repository is a pytest test scaffold for the Transcript API and workflow. It includes unit, contract, integration, and workflow tests, JSON Schemas, fixtures, and a GitHub Actions CI workflow to run tests automatically.

Prerequisites
Python 3.10+

Git and a working shell environment

Optional for integration tests: a running instance of the API under test reachable at http://localhost:8000 or a mock server (WireMock, simple Flask mock, or docker-compose).

CI secrets for staging E2E tests if you enable the e2e-staging job.

Quick Setup
Clone the repo and enter the directory:

bash
git clone <repo-url>
cd transcript-tests
Create and activate a virtual environment:

bash
python -m venv .venv
source .venv/bin/activate
Install dependencies:

bash
pip install --upgrade pip
pip install -r requirements.txt
Ensure fixtures exist under tests/fixtures and schemas under schemas.

Running Tests Locally
Unit tests

bash
pytest tests/unit -q
Contract tests

bash
pytest tests/contract -q
Integration tests

Start the API or a mock at http://localhost:8000.

Run:

bash
export TRANSCRIPT_API_BASE=http://localhost:8000
pytest tests/integration -q
Workflow tests

bash
pytest tests/workflow -q
Run all tests

bash
pytest -q
Test Structure
tests/unit: fast validators and pure logic tests (schema + semantic checks).

tests/contract: API contract checks for POST /transcripts/process and basic response shape.

tests/integration: end-to-end flows that require the service to be running; include idempotency and async polling.

tests/workflow: state machine and workflow transition tests (valid, invalid, edge cases).

tests/fixtures: sample metadata, transcripts, and small text/audio placeholders.

schemas: JSON Schema files used by contract and unit tests.

CI Integration
The repository includes .github/workflows/ci.yml. The pipeline runs:

Lint with flake8.

Unit and contract tests on every PR.

Integration tests after unit/contract pass. Integration steps assume the API is started in CI (via docker-compose or a mock).

Optional E2E on staging when merging to main using secrets for STAGING_API_BASE and STAGING_API_TOKEN.

Artifacts: test reports are uploaded as pipeline artifacts for triage.

Environment Variables
TRANSCRIPT_API_BASE default http://localhost:8000 used by integration and workflow tests.

SKIP_INTEGRATION set to true to skip long-running integration tests in CI or local runs.

STAGING_API_BASE and STAGING_API_TOKEN used by the e2e staging job in CI.

Extending Tests
Add new JSON Schemas to schemas and reference them in contract tests.

Add fixtures to tests/fixtures for negative cases: truncated wrappers, corrupted multipart, large files.

Add performance scripts under tests/performance and schedule nightly runs.

Add mocks for external services in tests/mocks and start them in CI via docker-compose.

Troubleshooting
Integration tests skipped: ensure the API is reachable at TRANSCRIPT_API_BASE or disable the require_service fixture skip.

Timeouts while polling: increase timeout_seconds in integration tests for slower staging environments.

Flaky tests: mark with @pytest.mark.flaky and quarantine until fixed; avoid retries masking real issues.

Secrets: never commit API tokens; use CI secret store.

Maintenance Notes
Keep fixtures small and synthetic to avoid large binaries in the repo. Use an object store for large audio artifacts.

Update fuzzy-match dictionaries and legal-term lists used by validators as the product evolves.

Track flaky tests and require fixes for tests that fail repeatedly.