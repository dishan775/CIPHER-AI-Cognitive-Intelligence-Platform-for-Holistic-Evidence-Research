from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, ForeignKey, Float, DateTime, event
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.pool import NullPool
from datetime import datetime
import json
import time

SQLALCHEMY_DATABASE_URL = "sqlite:///./cipher.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={
        "check_same_thread": False,
        "timeout": 60,
    },
    poolclass=NullPool,
)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=60000;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Booking(Base):
    __tablename__ = "bookings"
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, index=True)
    details = Column(String)
    meta_info = Column(String)
    status = Column(String)
    price = Column(String)

class Project(Base):
    __tablename__ = "projects"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text)
    objectives = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    documents = relationship("Document", back_populates="project", cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    title = Column(String)
    filename = Column(String)
    url = Column(String)
    source_type = Column(String) # pdf, docx, txt, web
    processing_status = Column(String) # pending, processed, failed
    content_text = Column(Text) # Raw text
    created_at = Column(DateTime, default=datetime.utcnow)

    project = relationship("Project", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    text = Column(Text)
    page_num = Column(Integer, nullable=True)
    chunk_index = Column(Integer)
    embedding = Column(Text, nullable=True) # JSON serialized float vector
    
    document = relationship("Document", back_populates="chunks")

class ExtractedEntity(Base):
    __tablename__ = "extracted_entities"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    text = Column(String)
    label = Column(String)
    start_char = Column(Integer)
    end_char = Column(Integer)

class PaperAnalysis(Base):
    __tablename__ = "paper_analyses"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    authors = Column(String)
    publication_year = Column(String)
    objectives = Column(Text)
    methodology = Column(Text)
    datasets = Column(Text)
    metrics = Column(Text)
    findings = Column(Text)
    limitations = Column(Text)
    future_work = Column(Text)

class ExtractedClaim(Base):
    __tablename__ = "extracted_claims"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"))
    claim_text = Column(Text)
    confidence = Column(Float, default=0.85)
    category = Column(String, default="empirical") # empirical, methodology, theoretical
    context = Column(Text, nullable=True)

class DocumentNlpArtifacts(Base):
    __tablename__ = "document_nlp_artifacts"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), unique=True, index=True)
    intelligence_json = Column(Text, nullable=True)
    web_metadata_json = Column(Text, nullable=True)
    structured_outline_json = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class NlpProcessingJob(Base):
    __tablename__ = "nlp_processing_jobs"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True, nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), index=True, nullable=True)
    status = Column(String, default="pending", index=True)  # pending, running, completed, failed
    stage = Column(String, default="queued")
    result_json = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class WorkflowRun(Base):
    __tablename__ = "workflow_runs"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    workflow_type = Column(String) # deep_research, fact_checking, sop_extraction
    status = Column(String) # running, completed, failed
    current_step = Column(String)
    result_json = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class ReportDraft(Base):
    __tablename__ = "report_drafts"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    title = Column(String)
    report_type = Column(String) # literature_review, executive_brief, sop_manual, comparative_benchmark
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class ResearchComparison(Base):
    __tablename__ = "research_comparisons"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    matrix_json = Column(Text)

# --- Knowledge Graph Models ---

class KGEntity(Base):
    __tablename__ = "kg_entities"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), index=True, nullable=True)
    canonical_name = Column(String, index=True, nullable=False)
    entity_type = Column(String, index=True, nullable=False)
    normalized_key = Column(String, index=True, nullable=False)
    aliases_json = Column(Text, default="[]")
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    mentions = relationship("KGEntityMention", back_populates="entity", cascade="all, delete-orphan")
    outgoing_relations = relationship("KGRelation", foreign_keys="KGRelation.source_id", back_populates="source_entity")
    incoming_relations = relationship("KGRelation", foreign_keys="KGRelation.target_id", back_populates="target_entity")

class KGEntityMention(Base):
    __tablename__ = "kg_entity_mentions"
    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(Integer, ForeignKey("kg_entities.id"), index=True, nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True, nullable=False)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=True)
    mention_text = Column(String, nullable=False)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    sentence_text = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    entity = relationship("KGEntity", back_populates="mentions")
    document = relationship("Document")
    chunk = relationship("DocumentChunk")

class KGRelation(Base):
    __tablename__ = "kg_relations"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), index=True, nullable=True)
    source_id = Column(Integer, ForeignKey("kg_entities.id"), index=True, nullable=False)
    target_id = Column(Integer, ForeignKey("kg_entities.id"), index=True, nullable=False)
    predicate = Column(String, index=True, nullable=False)
    confidence = Column(Float, default=0.85)
    extraction_method = Column(String, default="rule_syntax")
    review_status = Column(String, default="unreviewed", index=True)
    polarity = Column(String, default="positive")
    modality = Column(String, default="factual")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    source_entity = relationship("KGEntity", foreign_keys=[source_id], back_populates="outgoing_relations")
    target_entity = relationship("KGEntity", foreign_keys=[target_id], back_populates="incoming_relations")
    evidence = relationship("KGRelationEvidence", back_populates="relation", cascade="all, delete-orphan")

class KGRelationEvidence(Base):
    __tablename__ = "kg_relation_evidence"
    id = Column(Integer, primary_key=True, index=True)
    relation_id = Column(Integer, ForeignKey("kg_relations.id"), index=True, nullable=False)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True, nullable=False)
    chunk_id = Column(Integer, ForeignKey("document_chunks.id"), nullable=True)
    excerpt = Column(Text, nullable=False)
    page_num = Column(Integer, nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    confidence = Column(Float, default=0.85)
    extraction_method = Column(String, default="rule_syntax")
    verification_status = Column(String, default="unverified")
    created_at = Column(DateTime, default=datetime.utcnow)

    relation = relationship("KGRelation", back_populates="evidence")
    document = relationship("Document")
    chunk = relationship("DocumentChunk")

class KGBuildJob(Base):
    __tablename__ = "kg_build_jobs"
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), index=True, nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), index=True, nullable=True)
    status = Column(String, default="pending", index=True)
    entity_count = Column(Integer, default=0)
    relation_count = Column(Integer, default=0)
    evidence_count = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# --- Multi-Agent Collaborative Pipeline Models ---

class CrewRun(Base):
    __tablename__ = "crew_runs"
    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), index=True, nullable=True)
    topic = Column(Text, nullable=False)
    status = Column(String, default="pending", index=True)  # pending, running, completed, failed
    current_agent = Column(String, nullable=True)
    final_report = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    steps = relationship("CrewAgentStep", back_populates="crew_run", cascade="all, delete-orphan", order_by="CrewAgentStep.step_order")

class CrewAgentStep(Base):
    __tablename__ = "crew_agent_steps"
    id = Column(Integer, primary_key=True, index=True)
    crew_run_id = Column(Integer, ForeignKey("crew_runs.id"), index=True, nullable=False)
    agent_role = Column(String, nullable=False)  # researcher, reviewer, experiment_designer, statistician
    step_order = Column(Integer, nullable=False)
    task_description = Column(Text, nullable=True)
    status = Column(String, default="pending")  # pending, running, completed, failed
    output_text = Column(Text, nullable=True)
    output_json = Column(Text, nullable=True)
    confidence = Column(Float, default=0.0)
    tokens_used = Column(Integer, default=0)
    duration_ms = Column(Float, default=0.0)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    crew_run = relationship("CrewRun", back_populates="steps")


Base.metadata.create_all(bind=engine)


