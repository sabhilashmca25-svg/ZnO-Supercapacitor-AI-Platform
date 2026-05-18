"""Pytest fixtures for the backend test suite."""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_request():
    return {
        "scan_rate_mVs": 30.0,
        "potential_V": [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                         0.4,  0.3, 0.2, 0.1, 0.0, -0.1],
    }
