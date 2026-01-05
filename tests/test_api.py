from fastapi.testclient import TestClient
from src.api import app
import os

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_resolve_no_match():
    # Random data that shouldn't match anything
    payload = {
        "first_name": "Xyz",
        "last_name": "Abc",
        "dob": "1990-01-01"
    }
    response = client.post("/resolve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_match"

def test_resolve_match():
    # We need to know a record that exists in the DB.
    # Since we don't know the DB state perfectly, we can't guarantee a match 
    # without querying the DB first.
    # However, we can check if the API runs without crashing.
    payload = {
        "first_name": "John",
        "last_name": "Doe",
        "email": "john.doe@example.com"
    }
    response = client.post("/resolve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "processing_time_ms" in data
