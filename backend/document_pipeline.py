"""
Shared NLP ingestion pipeline for Knowledge Library and web imports.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from sqlalchemy.orm import Session

from database import (
    Document,
    DocumentChunk,
    DocumentNlpArtifacts,
    ExtractedClaim,
    ExtractedEntity,
    NlpProcessingJob,
    PaperAnalysis,
)
from nlp_service import nlp_service


def _read_file_content(doc: Document) -> str:
    file_path = os.path.join(".", "uploads", f"project_{doc.project_id}", doc.filename or "")
    if not os.path.exists(file_path):
        return doc.content_text or ""
    if (doc.filename or "").lower().endswith(".pdf"):
        import pdfplumber

        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text if text.strip() else "PDF contains scanned or non-extractable text."
    if (doc.filename or "").lower().endswith(".docx"):
        import docx

        text = ""
        doc_obj = docx.Document(file_path)
        for para in doc_obj.paragraphs:
            text += para.text + "\n"
        return text
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def build_document_intelligence(content: str, artifacts: Optional[DocumentNlpArtifacts] = None) -> dict[str, Any]:
    sample = content[:25000]
    keyphrases = nlp_service.extract_keyphrases(sample, top_n=12)
    metrics = nlp_service.text_metrics(sample)
    entities = nlp_service.extract_entities(sample[:15000])
    summary = nlp_service.extractive_summary(sample, num_sentences=4)
    keywords = [{"term": k["phrase"], "count": round(k["score"] * 100)} for k in keyphrases]
    payload = {
        "summary": summary,
        "entities": entities,
        "keywords": keywords,
        "keyphrases": keyphrases,
        "metrics": metrics,
        "preprocessing": {
            "word_count": metrics.get("words", 0),
            "sentence_count": metrics.get("sentences", 0),
        },
    }
    if artifacts and artifacts.structured_outline_json:
        try:
            payload["structured_outline"] = json.loads(artifacts.structured_outline_json)
        except Exception:
            pass
    if artifacts and artifacts.web_metadata_json:
        try:
            payload["web_metadata"] = json.loads(artifacts.web_metadata_json)
        except Exception:
            pass
    return payload


def run_document_nlp_pipeline(db: Session, doc: Document, job: Optional[NlpProcessingJob] = None) -> dict[str, Any]:
    start_t = time.perf_counter()

    def set_stage(stage: str, detail: str = ""):
        if not job:
            return
        job.stage = stage
        job.status = "running"
        if detail:
            job.result_json = json.dumps({"stage": stage, "detail": detail})
        db.commit()

    set_stage("parse", "Reading document content")
    if not doc.content_text or len(doc.content_text.strip()) < 10:
        try:
            doc.content_text = _read_file_content(doc)
        except Exception as exc:
            doc.content_text = doc.content_text or f"Error extracting document text: {exc}"

    content = doc.content_text or ""
    word_count = len(nlp_service.get_words(content))

    set_stage("chunk", "Semantic chunking and embedding")
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()

    chunk_size = 500
    overlap = 100
    stride = chunk_size - overlap
    raw_chunks = [content] if len(content) <= chunk_size else []
    if not raw_chunks:
        for i in range(0, len(content), stride):
            chunk = content[i : i + chunk_size].strip()
            if len(chunk) > 30:
                raw_chunks.append(chunk)

    chunks_created = 0
    for idx, c_text in enumerate(raw_chunks[:60]):
        emb = nlp_service.compute_embedding(c_text)
        db.add(
            DocumentChunk(
                document_id=doc.id,
                text=c_text,
                chunk_index=idx,
                page_num=(idx // 3) + 1,
                embedding=json.dumps(emb),
            )
        )
        chunks_created += 1

    set_stage("entities", "Named entity recognition")
    db.query(ExtractedEntity).filter(ExtractedEntity.document_id == doc.id).delete()
    entities = nlp_service.extract_entities(content[:15000])
    for ent in entities:
        db.add(
            ExtractedEntity(
                document_id=doc.id,
                text=ent["text"],
                label=ent["label"],
                start_char=ent["start"],
                end_char=ent["end"],
            )
        )

    set_stage("claims", "Empirical claim extraction")
    db.query(ExtractedClaim).filter(ExtractedClaim.document_id == doc.id).delete()
    claims = nlp_service.extract_claims(content[:20000])
    for cl in claims:
        db.add(
            ExtractedClaim(
                document_id=doc.id,
                claim_text=cl["claim_text"],
                confidence=cl["confidence"],
                category=cl["category"],
                context=cl["context"],
            )
        )

    set_stage("structure", "Research structure extraction")
    struct = nlp_service.extract_paper_structure(content[:30000], title=doc.title or "")
    analysis = db.query(PaperAnalysis).filter(PaperAnalysis.document_id == doc.id).first()
    if not analysis:
        analysis = PaperAnalysis(document_id=doc.id)
        db.add(analysis)
    analysis.authors = struct["authors"]
    analysis.publication_year = struct["publication_year"]
    analysis.objectives = struct["objectives"]
    analysis.methodology = struct["methodology"]
    analysis.datasets = struct["datasets"]
    analysis.metrics = struct["metrics"]
    analysis.findings = struct["findings"]
    analysis.limitations = struct["limitations"]
    analysis.future_work = struct["future_work"]

    intelligence = build_document_intelligence(content, db.query(DocumentNlpArtifacts).filter(DocumentNlpArtifacts.document_id == doc.id).first())
    artifacts = db.query(DocumentNlpArtifacts).filter(DocumentNlpArtifacts.document_id == doc.id).first()
    if not artifacts:
        artifacts = DocumentNlpArtifacts(document_id=doc.id)
        db.add(artifacts)
    artifacts.intelligence_json = json.dumps(intelligence)
    if doc.source_type == "web" and not artifacts.structured_outline_json:
        artifacts.structured_outline_json = json.dumps(nlp_service.structure_web_content(content[:30000], title=doc.title or "", url=doc.url or ""))

    doc.processing_status = "processed"
    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

    if job:
        job.status = "completed"
        job.stage = "done"
        job.result_json = json.dumps(
            {
                "chunks_created": chunks_created,
                "entities_found": len(entities),
                "claims_found": len(claims),
                "word_count": word_count,
                "latency_ms": latency_ms,
            }
        )
    db.commit()

    return {
        "status": "processed",
        "document_id": doc.id,
        "chunks_indexed": chunks_created,
        "chunks_created": chunks_created,
        "entities_found": len(entities),
        "claims_found": len(claims),
        "word_count": word_count,
        "latency_ms": latency_ms,
        "intelligence": intelligence,
    }


def run_nlp_job_by_id(job_id: int) -> None:
    from database import SessionLocal

    db = SessionLocal()
    try:
        job = db.query(NlpProcessingJob).filter(NlpProcessingJob.id == job_id).first()
        if not job or job.status in ("completed", "running"):
            return
        doc = db.query(Document).filter(Document.id == job.document_id).first()
        if not doc:
            job.status = "failed"
            job.error_message = "Document not found"
            db.commit()
            return
        job.status = "running"
        db.commit()
        run_document_nlp_pipeline(db, doc, job=job)
    except Exception as exc:
        db.rollback()
        job = db.query(NlpProcessingJob).filter(NlpProcessingJob.id == job_id).first()
        if job:
            job.status = "failed"
            job.error_message = str(exc)[:2000]
            if doc := db.query(Document).filter(Document.id == job.document_id).first():
                doc.processing_status = "failed"
            db.commit()
    finally:
        db.close()
