from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query
from sqlalchemy.orm import Session
from database import SessionLocal, Project, Document, DocumentChunk, DocumentNlpArtifacts, NlpProcessingJob
from typing import List, Optional
from pydantic import BaseModel
import shutil
import os
import json
from pathlib import Path
import urllib.parse
import requests

from web_fetch_service import fetch_url_content
from document_pipeline import run_nlp_job_by_id

router = APIRouter()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    objectives: str = ""

class UrlImport(BaseModel):
    url: str
    title: str = ""

@router.get("/api/projects")
def get_projects(db: Session = Depends(get_db)):
    projects = db.query(Project).all()
    result = []
    for p in projects:
        doc_count = db.query(Document).filter(Document.project_id == p.id).count()
        chunk_count = db.query(DocumentChunk).join(Document, DocumentChunk.document_id == Document.id).filter(Document.project_id == p.id).count()
        result.append({
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "objectives": p.objectives,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "document_count": doc_count,
            "chunk_count": chunk_count
        })
    return result

@router.get("/api/projects/{project_id}")
def get_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    docs = db.query(Document).filter(Document.project_id == project_id).all()
    return {
        "id": project.id,
        "name": project.name,
        "description": project.description,
        "objectives": project.objectives,
        "created_at": project.created_at.isoformat() if project.created_at else None,
        "documents": [
            {
                "id": d.id,
                "title": d.title,
                "filename": d.filename,
                "source_type": d.source_type,
                "processing_status": d.processing_status,
                "created_at": d.created_at.isoformat() if d.created_at else None
            } for d in docs
        ]
    }

@router.post("/api/projects")
def create_project(project: ProjectCreate, db: Session = Depends(get_db)):
    db_project = Project(name=project.name, description=project.description, objectives=project.objectives)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

@router.delete("/api/projects/{project_id}")
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete(project)
    db.commit()
    return {"status": "deleted"}

@router.delete("/api/documents/{document_id}")
def delete_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(doc)
    db.commit()
    return {"status": "deleted", "id": document_id}

@router.get("/api/documents")
def list_all_documents(project_id: Optional[int] = None, db: Session = Depends(get_db)):
    q = db.query(Document)
    if project_id:
        q = q.filter(Document.project_id == project_id)
    docs = q.order_by(Document.id.desc()).all()
    results = []
    for d in docs:
        c_count = db.query(DocumentChunk).filter(DocumentChunk.document_id == d.id).count()
        results.append({
            "id": d.id,
            "project_id": d.project_id,
            "title": d.title,
            "filename": d.filename,
            "source_type": d.source_type,
            "processing_status": d.processing_status,
            "chunk_count": c_count,
            "word_count": len(d.content_text.split()) if d.content_text else 0,
            "created_at": d.created_at.isoformat() if hasattr(d, 'created_at') and d.created_at else None
        })
    return results

@router.get("/api/documents/{document_id}")
def get_document(document_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).order_by(DocumentChunk.chunk_index).all()
    return {
        "id": doc.id,
        "project_id": doc.project_id,
        "title": doc.title,
        "filename": doc.filename,
        "source_type": doc.source_type,
        "processing_status": doc.processing_status,
        "content_text": doc.content_text,
        "chunk_count": len(chunks)
    }


@router.post("/api/projects/{project_id}/documents")
async def upload_document(project_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Save file
    safe_name = Path(file.filename or "document.txt").name
    if Path(safe_name).suffix.lower() not in {".pdf", ".docx", ".txt", ".md", ".csv"}:
        raise HTTPException(status_code=415, detail="Supported files: PDF, DOCX, TXT, MD, CSV")
    upload_dir = Path(__file__).parent / "uploads" / f"project_{project_id}"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / safe_name
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    doc = Document(
        project_id=project_id,
        title=safe_name,
        filename=safe_name,
        source_type=safe_name.split('.')[-1],
        processing_status="pending"
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc

@router.get("/api/projects/{project_id}/documents")
def get_documents(project_id: int, db: Session = Depends(get_db)):
    return db.query(Document).filter(Document.project_id == project_id).all()

@router.get("/api/web/search")
@router.get("/api/projects/web-search")
def search_web_articles(
    q: str = Query(..., description="Query string for search"),
    limit: int = Query(6, ge=1, le=20),
):
    query = q.strip()
    if not query:
        return {"query": "", "total": 0, "results": []}

    results = []
    seen_urls = set()

    # 1. Search Wikipedia Open Knowledge & Research Articles
    try:
        wiki_url = (
            "https://en.wikipedia.org/w/api.php?action=query&generator=search"
            f"&gsrsearch={urllib.parse.quote(query)}&prop=pageimages|extracts"
            "&exintro=1&explaintext=1&exsentences=2&pithumbsize=400&format=json"
            f"&gsrlimit={limit}"
        )
        wiki_resp = requests.get(wiki_url, headers={"User-Agent": "CipherResearch/1.0"}, timeout=6)
        if wiki_resp.status_code == 200:
            pages = wiki_resp.json().get("query", {}).get("pages", {})
            for page in pages.values():
                title = page.get("title") or "Research Document"
                article_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                if article_url in seen_urls:
                    continue
                seen_urls.add(article_url)
                thumb = page.get("thumbnail", {}).get("source")
                extract = page.get("extract") or ""
                results.append({
                    "title": title,
                    "url": article_url,
                    "snippet": extract[:300] + ("..." if len(extract) > 300 else ""),
                    "source": "Wikipedia Knowledge Base",
                    "imageUrl": thumb,
                    "type": "encyclopedia"
                })
    except Exception as exc:
        print(f"Web search wiki error: {exc}")

    # 2. Search Curated Research Preprints & Scientific News fallback
    try:
        from routes_news import FALLBACK_RESEARCH_NEWS
        q_lower = query.lower()
        for item in FALLBACK_RESEARCH_NEWS:
            if (
                q_lower in item["title"].lower()
                or q_lower in item.get("description", "").lower()
                or any(q_lower in tag.lower() for tag in item.get("tags", []))
            ):
                if item["url"] not in seen_urls:
                    seen_urls.add(item["url"])
                    results.append({
                        "title": item["title"],
                        "url": item["url"],
                        "snippet": item["description"],
                        "source": item.get("source", "Research Publication"),
                        "imageUrl": item.get("imageUrl"),
                        "type": "journal_paper"
                    })
    except Exception as exc:
        print(f"Web search news error: {exc}")

    return {
        "query": query,
        "total": len(results),
        "results": results[:limit + 4]
    }


@router.post("/api/projects/{project_id}/import-url")
def import_url(
    project_id: int,
    payload: UrlImport,
    background_tasks: BackgroundTasks,
    run_nlp: bool = Query(True, description="Queue full NLP indexing after import"),
    db: Session = Depends(get_db),
):
    project = db.get(Project, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        fetched = fetch_url_content(payload.url.strip())
        text = fetched.get("full_text") or ""
        if len(text) < 40:
            raise ValueError("No readable content found on this page")

        doc_title = payload.title or fetched.get("title") or payload.url
        doc = Document(
            project_id=project_id,
            title=doc_title[:500],
            filename=f"web_{abs(hash(payload.url)) % 10**8}.txt",
            url=fetched.get("canonical_url") or payload.url,
            source_type="web",
            processing_status="pending",
            content_text=text[:500000],
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        web_meta = {
            "summary": fetched.get("summary", ""),
            "key_news": fetched.get("key_news", []),
            "headings": fetched.get("headings", []),
            "images": fetched.get("images", []),
            "links": fetched.get("links", []),
            "meta_description": fetched.get("meta_description", ""),
            "word_count": fetched.get("word_count", 0),
            "reading_time_min": fetched.get("reading_time_min", 1),
            "domain": fetched.get("domain", ""),
            "structured_by": fetched.get("structured_by", "nlp_local"),
        }
        structured = fetched.get("structured") or {}
        artifacts = DocumentNlpArtifacts(
            document_id=doc.id,
            web_metadata_json=json.dumps(web_meta),
            structured_outline_json=json.dumps(structured),
            intelligence_json=json.dumps(
                {
                    "summary": fetched.get("summary", "") or structured.get("summary", []),
                    "key_news": fetched.get("key_news", []),
                    "key_topics": structured.get("key_topics", []),
                    "sections": structured.get("sections", [])[:6],
                }
            ),
        )
        db.add(artifacts)

        job: Optional[NlpProcessingJob] = None
        if run_nlp:
            job = NlpProcessingJob(document_id=doc.id, project_id=project_id, status="pending", stage="queued")
            db.add(job)
        db.commit()
        if job:
            db.refresh(job)
            background_tasks.add_task(run_nlp_job_by_id, job.id)

        return {
            "id": doc.id,
            "project_id": doc.project_id,
            "title": doc.title,
            "filename": doc.filename,
            "url": doc.url,
            "source_type": doc.source_type,
            "processing_status": doc.processing_status,
            "content_preview": text[:400] + ("..." if len(text) > 400 else ""),
            "full_content": text,
            "word_count": fetched.get("word_count", 0),
            "reading_time_min": fetched.get("reading_time_min", 1),
            "summary": fetched.get("summary", ""),
            "key_news": fetched.get("key_news", []),
            "images": fetched.get("images", []),
            "domain": fetched.get("domain", ""),
            "structured_by": fetched.get("structured_by", "nlp_local"),
            "structured_outline": structured,
            "web_metadata": web_meta,
            "nlp_job_id": job.id if job else None,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not import URL: {exc}")
