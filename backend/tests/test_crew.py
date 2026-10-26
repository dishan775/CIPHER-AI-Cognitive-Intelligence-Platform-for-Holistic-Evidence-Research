"""
Test suite for CIPHER Multi-Agent Research Collaboration Pipeline
Tests health check, pipeline execution, runs listing, and run retrieval.
"""

import pytest
from fastapi.testclient import TestClient
import main
from database import SessionLocal, CrewRun, CrewAgentStep

client = TestClient(main.app)

def test_crew_health():
    resp = client.get("/api/crew/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "operational"
    assert "researcher" in data["agents"]
    assert "reviewer" in data["agents"]
    assert "experiment_designer" in data["agents"]
    assert "statistician" in data["agents"]

def test_crew_run_execution():
    topic = "Self-Supervised Vision Transformers for Medical Imaging"
    resp = client.post("/api/crew/run", json={"topic": topic})
    assert resp.status_code == 200
    data = resp.json()
    
    assert data["status"] == "completed"
    assert data["topic"] == topic
    assert len(data["agents"]) == 4
    
    roles = [a["role"] for a in data["agents"]]
    assert roles == ["researcher", "reviewer", "experiment_designer", "statistician"]
    
    # Check outputs exist for each agent
    for agent in data["agents"]:
        assert agent["status"] == "completed"
        assert agent["output"] is not None
        assert agent["confidence"] > 0
    
    # Check final synthesized report
    assert data["final_report"] is not None
    assert len(data["final_report"]) > 100
    assert "Research Intelligence Report" in data["final_report"]

def test_crew_list_runs():
    resp = client.get("/api/crew/runs")
    assert resp.status_code == 200
    runs = resp.json()
    assert isinstance(runs, list)
    assert len(runs) >= 1

def test_crew_get_run_detail():
    resp = client.get("/api/crew/runs")
    runs = resp.json()
    run_id = runs[0]["id"]
    
    resp_detail = client.get(f"/api/crew/runs/{run_id}")
    assert resp_detail.status_code == 200
    detail = resp_detail.json()
    assert detail["id"] == run_id
    assert len(detail["agents"]) == 4
