"""
CIPHER Multi-Agent Pipeline — FastAPI Routes
=============================================
REST endpoints for the multi-agent research collaboration pipeline.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional
import json

from database import get_db, CrewRun, CrewAgentStep

router = APIRouter(prefix="/api/crew", tags=["Multi-Agent Pipeline"])


# ─── Pydantic Models ──────────────────────────────────────────

class CrewRunRequest(BaseModel):
    topic: str
    project_id: Optional[int] = None


class CrewRunResponse(BaseModel):
    id: int
    topic: str
    status: str
    agents: list
    final_report: Optional[str] = None
    created_at: str


# ─── Endpoints ─────────────────────────────────────────────────

@router.get("/health")
async def crew_health():
    """Health check for the multi-agent pipeline."""
    return {
        "status": "operational",
        "agents": ["researcher", "reviewer", "experiment_designer", "statistician"],
        "pipeline": "Sequential Multi-Agent Collaboration",
    }


@router.post("/run")
async def start_crew_run(req: CrewRunRequest, db: Session = Depends(get_db)):
    """
    Execute the full multi-agent research pipeline.
    Runs all 4 agents sequentially: Researcher → Reviewer → Experiment Designer → Statistician.
    """
    if not req.topic or len(req.topic.strip()) < 3:
        raise HTTPException(status_code=400, detail="Topic must be at least 3 characters.")

    from crew_service import run_crew_pipeline
    try:
        result = await run_crew_pipeline(req.topic.strip(), req.project_id, db)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@router.get("/runs")
async def list_crew_runs(project_id: Optional[int] = None, limit: int = 20, db: Session = Depends(get_db)):
    """List all pipeline runs, optionally filtered by project."""
    query = db.query(CrewRun).order_by(CrewRun.created_at.desc())
    if project_id:
        query = query.filter(CrewRun.project_id == project_id)
    runs = query.limit(limit).all()

    result = []
    for run in runs:
        steps = db.query(CrewAgentStep).filter(
            CrewAgentStep.crew_run_id == run.id
        ).order_by(CrewAgentStep.step_order).all()

        agents = []
        for step in steps:
            output = None
            if step.output_json:
                try:
                    output = json.loads(step.output_json)
                except Exception:
                    output = {"raw": step.output_json}

            agents.append({
                "role": step.agent_role,
                "status": step.status,
                "output": output,
                "confidence": step.confidence,
                "duration_ms": step.duration_ms,
            })

        result.append({
            "id": run.id,
            "topic": run.topic,
            "status": run.status,
            "current_agent": run.current_agent,
            "agents": agents,
            "final_report": run.final_report,
            "created_at": run.created_at.isoformat() if run.created_at else None,
        })

    return result


@router.get("/runs/{run_id}")
async def get_crew_run(run_id: int, db: Session = Depends(get_db)):
    """Get details of a specific pipeline run."""
    run = db.query(CrewRun).filter(CrewRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    steps = db.query(CrewAgentStep).filter(
        CrewAgentStep.crew_run_id == run.id
    ).order_by(CrewAgentStep.step_order).all()

    agents = []
    for step in steps:
        output = None
        if step.output_json:
            try:
                output = json.loads(step.output_json)
            except Exception:
                output = {"raw": step.output_json}

        agents.append({
            "role": step.agent_role,
            "title": {
                "researcher": "Research Analyst",
                "reviewer": "Critical Reviewer",
                "experiment_designer": "Experiment Designer",
                "statistician": "Statistical Analyst",
            }.get(step.agent_role, step.agent_role),
            "icon": {"researcher": "🔬", "reviewer": "🔍", "experiment_designer": "⚗️", "statistician": "📊"}.get(step.agent_role, "🤖"),
            "color": {"researcher": "#3B82F6", "reviewer": "#F59E0B", "experiment_designer": "#10B981", "statistician": "#8B5CF6"}.get(step.agent_role, "#888"),
            "status": step.status,
            "output": output,
            "confidence": step.confidence,
            "duration_ms": step.duration_ms,
        })

    return {
        "id": run.id,
        "topic": run.topic,
        "status": run.status,
        "current_agent": run.current_agent,
        "agents": agents,
        "final_report": run.final_report,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


@router.delete("/runs/{run_id}")
async def delete_crew_run(run_id: int, db: Session = Depends(get_db)):
    """Delete a pipeline run and all associated agent steps."""
    run = db.query(CrewRun).filter(CrewRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    db.delete(run)
    db.commit()
    return {"status": "deleted", "id": run_id}
