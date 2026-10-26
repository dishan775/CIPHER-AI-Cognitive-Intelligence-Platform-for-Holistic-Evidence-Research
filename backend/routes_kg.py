"""
CIPHER — Knowledge Graph API Router
Dedicated endpoints for graph extraction, visualization data, entity inspection, evidence inspection, entity merging, and relation review.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Optional, List, Dict, Any

from database import SessionLocal, Document, Project, KGBuildJob, get_db
from knowledge_graph_service import (
    kg_service,
    ENTITY_CATEGORIES,
    RELATION_PREDICATES
)

router = APIRouter(prefix="/api/knowledge-graph", tags=["Knowledge Graph"])

# --- Pydantic Schemas ---

class KGBuildRequest(BaseModel):
    document_id: int
    project_id: Optional[int] = None
    rebuild: bool = False

class KGGenerateRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw text, article, or prompt to extract knowledge graph from")
    title: Optional[str] = Field(None, description="Optional title for the knowledge graph document")
    project_id: Optional[int] = Field(None, description="Project ID to associate with")

class MergeEntitiesRequest(BaseModel):
    source_entity_id: int
    target_entity_id: int

class ReviewRelationRequest(BaseModel):
    status: str = Field(..., description="Review status: approved, rejected, corrected, unreviewed")
    corrected_predicate: Optional[str] = None
    notes: Optional[str] = None

# --- Endpoints ---

@router.get("/health")
def kg_health():
    """Service health, engine availability, and schema configuration."""
    return {
        "status": "online",
        "service": "CIPHER Knowledge Graph Engine",
        "supported_entity_categories": list(ENTITY_CATEGORIES.keys()),
        "supported_predicates": list(RELATION_PREDICATES.keys()),
        "storage_backend": "SQLite (SQLAlchemy Relational & Evidence Store)",
        "nlp_engine": "spaCy + Scientific Phrase Lexicon + Grounded Inference"
    }

@router.post("/build")
def build_knowledge_graph(req: KGBuildRequest, db: Session = Depends(get_db)):
    """
    Extracts entities and relationships from a document and persists them into the project knowledge graph.
    """
    doc = db.query(Document).filter(Document.id == req.document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document with ID {req.document_id} not found.")

    try:
        result = kg_service.build_graph_for_document(
            db=db,
            document_id=req.document_id,
            project_id=req.project_id or doc.project_id,
            rebuild=req.rebuild
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge graph build failed: {str(e)}")

@router.post("/generate")
def generate_knowledge_graph_from_text(req: KGGenerateRequest, db: Session = Depends(get_db)):
    """
    AI-Powered Knowledge Graph Generator:
    Extracts Subject-Predicate-Object triplets, entities, and citations from raw input text
    and compiles an interactive knowledge graph.
    """
    try:
        res = kg_service.generate_graph_from_text(
            db=db,
            text=req.text,
            title=req.title,
            project_id=req.project_id
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Knowledge graph generation failed: {str(e)}")

@router.get("/graph")
def get_graph(
    project_id: Optional[int] = Query(None, description="Scope to specific project"),
    document_id: Optional[int] = Query(None, description="Filter relations to specific source document"),
    entity_type: Optional[str] = Query(None, description="Filter by entity type (MODEL, METHOD, DATASET, etc.)"),
    predicate: Optional[str] = Query(None, description="Filter by predicate (EVALUATED_ON, USES_METHOD, etc.)"),
    search: Optional[str] = Query(None, description="Search entity by name or alias"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    review_status: Optional[str] = Query(None, description="unreviewed, approved, rejected, corrected"),
    limit: int = Query(120, ge=1, le=500, description="Maximum number of relations to return"),
    include_papers: bool = Query(True, description="Whether to include PAPER nodes"),
    db: Session = Depends(get_db)
):
    """
    Retrieves filtered graph nodes and edges with server-side boundaries for interactive visualization.
    """
    try:
        data = kg_service.get_graph_data(
            db=db,
            project_id=project_id,
            document_id=document_id,
            entity_type=entity_type,
            predicate=predicate,
            search_query=search,
            min_confidence=min_confidence,
            review_status=review_status,
            limit=limit,
            include_papers=include_papers
        )
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query knowledge graph: {str(e)}")

@router.get("/entities/{entity_id}")
def get_entity_detail(entity_id: int, db: Session = Depends(get_db)):
    """
    Detailed inspector for a canonical entity, including aliases, occurrences, supporting citations, and connections.
    """
    detail = kg_service.get_entity_detail(db, entity_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Entity with ID {entity_id} not found.")
    return detail

@router.get("/relations/{relation_id}")
def get_relation_detail(relation_id: int, db: Session = Depends(get_db)):
    """
    Detailed inspector for a relationship with all traceable original source evidence snippets.
    """
    detail = kg_service.get_relation_detail(db, relation_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Relation with ID {relation_id} not found.")
    return detail

@router.get("/documents/{document_id}/status")
def get_document_kg_status(document_id: int, db: Session = Depends(get_db)):
    """
    Returns extraction job status, entity count, relation count, and evidence count for a document.
    """
    job = db.query(KGBuildJob).filter(KGBuildJob.document_id == document_id).first()
    if not job:
        return {
            "document_id": document_id,
            "status": "not_built",
            "entity_count": 0,
            "relation_count": 0,
            "evidence_count": 0
        }
    return {
        "document_id": document_id,
        "job_id": job.id,
        "status": job.status,
        "entity_count": job.entity_count,
        "relation_count": job.relation_count,
        "evidence_count": job.evidence_count,
        "error_message": job.error_message,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None
    }

@router.get("/stats")
def get_kg_stats(
    project_id: Optional[int] = Query(None, description="Scope stats to specific project"),
    db: Session = Depends(get_db)
):
    """
    High-level statistics: total entities, relations, source papers, evidence records, and category distribution.
    """
    return kg_service.get_stats(db, project_id)

@router.post("/entities/merge")
def merge_entities_endpoint(req: MergeEntitiesRequest, db: Session = Depends(get_db)):
    """
    Merges duplicate or equivalent entities, re-pointing mentions and relations while preserving source citations.
    """
    try:
        res = kg_service.merge_entities(db, req.source_entity_id, req.target_entity_id)
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to merge entities: {str(e)}")

@router.post("/relations/{relation_id}/review")
def review_relation_endpoint(
    relation_id: int,
    req: ReviewRelationRequest,
    db: Session = Depends(get_db)
):
    """
    Approve, reject, or correct the predicate of an extracted relationship.
    """
    try:
        res = kg_service.review_relation(
            db=db,
            relation_id=relation_id,
            status=req.status,
            corrected_predicate=req.corrected_predicate
        )
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to review relation: {str(e)}")

@router.get("/search")
def search_entities(
    query: str = Query(..., min_length=1, description="Entity name or keyword search"),
    project_id: Optional[int] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Search entities by query string for autocomplete and filter exploration.
    """
    graph_res = kg_service.get_graph_data(
        db=db,
        project_id=project_id,
        search_query=query,
        limit=limit
    )
    return {
        "query": query,
        "results": graph_res["nodes"]
    }
