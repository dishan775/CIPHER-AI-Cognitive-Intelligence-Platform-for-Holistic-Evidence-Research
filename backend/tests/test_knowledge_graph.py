"""
Unit and Integration Tests for CIPHER Knowledge Graph Engine.
Uses synthetic fixtures with no external network or GPU requirements.
"""

import pytest
import os
import json
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import (
    Base, Project, Document, DocumentChunk,
    KGEntity, KGEntityMention, KGRelation, KGRelationEvidence, KGBuildJob, get_db
)
from knowledge_graph_service import KnowledgeGraphService
from main import app
import routes_kg

from sqlalchemy.pool import StaticPool

# In-memory SQLite for testing
TEST_DB_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DB_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)

@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[routes_kg.get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def sample_project(db_session):
    proj = Project(
        name="Acoustic Anomaly Research",
        description="Predictive maintenance and acoustic defect detection",
        objectives="Classify industrial fan degradation using DSP and ML"
    )
    db_session.add(proj)
    db_session.commit()
    db_session.refresh(proj)
    return proj

@pytest.fixture
def sample_document(db_session, sample_project):
    doc = Document(
        project_id=sample_project.id,
        title="Industrial Fan Acoustic Anomaly Detection Benchmark",
        filename="fan_study_2020.txt",
        content_text=(
            "The proposed Machine-Aware Gradient Boosting model was evaluated on the DCASE 2020 Task 2 dataset. "
            "Our system uses Short-Time Fourier Transform and Mel-Frequency Cepstral Coefficients to extract spectral features. "
            "The model achieves an overall accuracy of 87.5% and 0.9456 ROC-AUC, outperforming unsupervised autoencoders. "
            "Directional condenser microphones were mounted 0.5 meters from the ventilation duct. "
            "Acoustic vibration reduces undetected blade unbalance and motor failure. "
            "The authors hypothesize that domain shift could affect unseen factory installations."
        ),
        processing_status="processed"
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    # Add mock chunks
    chunk = DocumentChunk(
        document_id=doc.id,
        text=doc.content_text,
        chunk_index=0,
        page_num=1
    )
    db_session.add(chunk)
    db_session.commit()
    return doc

# ==========================================
# 1. ENTITY EXTRACTION TESTS
# ==========================================

def test_extract_scientific_entities():
    kg_svc = KnowledgeGraphService()
    sample = "We evaluated XGBoost on the DCASE 2020 dataset using STFT and MFCC feature extraction."
    entities = kg_svc.extract_entities_from_text(sample)

    entity_names = {e["canonical_name"] for e in entities}
    assert "XGBoost" in entity_names
    assert "DCASE 2020" in entity_names
    assert "Short-Time Fourier Transform" in entity_names
    assert "Mel-Frequency Cepstral Coefficients" in entity_names

    # Check category assignments
    for ent in entities:
        if ent["canonical_name"] == "XGBoost":
            assert ent["entity_type"] == "MODEL"
        elif ent["canonical_name"] == "DCASE 2020":
            assert ent["entity_type"] == "BENCHMARK"
        elif ent["canonical_name"] == "Short-Time Fourier Transform":
            assert ent["entity_type"] == "METHOD"

def test_entity_normalization_and_aliases():
    kg_svc = KnowledgeGraphService()
    k1 = kg_svc.normalize_key("XGBoost")
    k2 = kg_svc.normalize_key("xgboost")
    k3 = kg_svc.normalize_key("  XGBoost!  ")
    assert k1 == k2 == k3 == "xgboost"

def test_empty_and_noise_entity_handling():
    kg_svc = KnowledgeGraphService()
    assert kg_svc.extract_entities_from_text("") == []
    assert kg_svc.extract_entities_from_text("   \n\t  ") == []
    # Regular text without scientific phrases
    res = kg_svc.extract_entities_from_text("the and or but if because while")
    assert len(res) == 0

# ==========================================
# 2. RELATIONSHIP EXTRACTION TESTS
# ==========================================

def test_extract_relations_with_evidence():
    kg_svc = KnowledgeGraphService()
    sent = "The Gradient Boosting model was evaluated on the DCASE 2020 dataset."
    entities = kg_svc.extract_entities_from_text(sent)
    relations = kg_svc.extract_relations_from_sentence(sent, entities)

    assert len(relations) >= 1
    rel = relations[0]
    assert rel["predicate"] == "EVALUATED_ON"
    assert rel["polarity"] == "positive"
    assert rel["modality"] == "factual"
    assert rel["excerpt"] == sent
    assert rel["confidence"] >= 0.80

def test_negation_and_speculation_detection():
    kg_svc = KnowledgeGraphService()
    # Negative sentence
    neg_sent = "The model did not improve accuracy on the DCASE 2020 benchmark."
    neg_ents = kg_svc.extract_entities_from_text(neg_sent)
    neg_rels = kg_svc.extract_relations_from_sentence(neg_sent, neg_ents)
    if neg_rels:
        assert neg_rels[0]["polarity"] == "negative"

    # Speculative sentence
    spec_sent = "We hypothesize that STFT might improve overall accuracy on noisy sensors."
    spec_ents = kg_svc.extract_entities_from_text(spec_sent)
    spec_rels = kg_svc.extract_relations_from_sentence(spec_sent, spec_ents)
    if spec_rels:
        assert spec_rels[0]["modality"] in ("speculative", "hypothetical")

# ==========================================
# 3. PERSISTENCE & IDEMPOTENCY TESTS
# ==========================================

def test_build_graph_for_document_idempotency(db_session, sample_document):
    kg_svc = KnowledgeGraphService()

    # Initial build
    res1 = kg_svc.build_graph_for_document(db_session, sample_document.id, rebuild=False)
    assert res1["status"] == "completed"
    assert res1["entities_registered"] > 0
    assert res1["relations_created"] > 0

    ent_count_1 = db_session.query(KGEntity).count()
    rel_count_1 = db_session.query(KGRelation).count()
    ev_count_1 = db_session.query(KGRelationEvidence).count()

    # Repeated build (rebuild=True) should update without duplicating records
    res2 = kg_svc.build_graph_for_document(db_session, sample_document.id, rebuild=True)
    assert res2["status"] == "completed"

    ent_count_2 = db_session.query(KGEntity).count()
    rel_count_2 = db_session.query(KGRelation).count()
    ev_count_2 = db_session.query(KGRelationEvidence).count()

    # Must be identical counts, no duplication
    assert ent_count_2 == ent_count_1
    assert rel_count_2 == rel_count_1
    assert ev_count_2 == ev_count_1

def test_multi_document_evidence_preservation(db_session, sample_project, sample_document):
    kg_svc = KnowledgeGraphService()
    # Build first doc
    kg_svc.build_graph_for_document(db_session, sample_document.id)

    # Create second doc mentioning same entity
    doc2 = Document(
        project_id=sample_project.id,
        title="Follow-up Study on Gradient Boosting Evaluation",
        filename="followup.txt",
        content_text="The Gradient Boosting model was evaluated on the DCASE 2020 Task 2 dataset under severe ambient noise.",
        processing_status="processed"
    )
    db_session.add(doc2)
    db_session.commit()

    # Build second doc
    kg_svc.build_graph_for_document(db_session, doc2.id)

    # Check relations for Gradient Boosting -> EVALUATED_ON -> DCASE 2020 Task 2
    src_ent = db_session.query(KGEntity).filter(KGEntity.normalized_key == "gradient boosting").first()
    tgt_ent = db_session.query(KGEntity).filter(KGEntity.normalized_key == "dcase 2020 task 2").first()
    assert src_ent is not None
    assert tgt_ent is not None

    relation = db_session.query(KGRelation).filter(
        KGRelation.source_id == src_ent.id,
        KGRelation.target_id == tgt_ent.id,
        KGRelation.predicate == "EVALUATED_ON"
    ).first()

    assert relation is not None
    # Must preserve multiple evidence records from different documents!
    evidence_docs = {ev.document_id for ev in relation.evidence}
    assert sample_document.id in evidence_docs
    assert doc2.id in evidence_docs

def test_entity_merging(db_session, sample_project):
    kg_svc = KnowledgeGraphService()

    # Create two duplicate entities
    e1 = KGEntity(
        project_id=sample_project.id,
        canonical_name="XGBoost Classifier",
        entity_type="MODEL",
        normalized_key="xgboost classifier"
    )
    e2 = KGEntity(
        project_id=sample_project.id,
        canonical_name="XGBoost",
        entity_type="MODEL",
        normalized_key="xgboost"
    )
    target = KGEntity(
        project_id=sample_project.id,
        canonical_name="DCASE 2020",
        entity_type="BENCHMARK",
        normalized_key="dcase 2020"
    )
    db_session.add_all([e1, e2, target])
    db_session.commit()

    # Add relation from e1
    rel = KGRelation(
        project_id=sample_project.id,
        source_id=e1.id,
        target_id=target.id,
        predicate="EVALUATED_ON",
        confidence=0.92
    )
    db_session.add(rel)
    db_session.commit()

    e1_id = e1.id
    e2_id = e2.id
    rel_id = rel.id

    # Merge e1 into e2
    res = kg_svc.merge_entities(db_session, source_entity_id=e1_id, target_entity_id=e2_id)
    assert res["status"] == "success"

    # e1 should be deleted
    assert db_session.query(KGEntity).filter(KGEntity.id == e1_id).first() is None
    # relation should now point from e2 to target
    rel_after = db_session.query(KGRelation).filter(KGRelation.id == rel_id).first()
    assert rel_after.source_id == e2_id

# ==========================================
# 4. API ENDPOINT TESTS
# ==========================================

def test_api_kg_health(client):
    res = client.get("/api/knowledge-graph/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "MODEL" in data["supported_entity_categories"]
    assert "EVALUATED_ON" in data["supported_predicates"]

def test_api_build_and_graph_query(client, sample_document):
    # 1. Trigger Build
    build_res = client.post("/api/knowledge-graph/build", json={
        "document_id": sample_document.id,
        "rebuild": True
    })
    assert build_res.status_code == 200
    b_data = build_res.json()
    assert b_data["status"] == "completed"

    # 2. Query Graph
    graph_res = client.get(f"/api/knowledge-graph/graph?project_id={sample_document.project_id}")
    assert graph_res.status_code == 200
    g_data = graph_res.json()
    assert len(g_data["nodes"]) > 0
    assert len(g_data["edges"]) > 0

    # 3. Search query filter
    search_res = client.get(f"/api/knowledge-graph/graph?project_id={sample_document.project_id}&search=Gradient")
    assert search_res.status_code == 200
    s_data = search_res.json()
    assert any("Gradient" in n["label"] for n in s_data["nodes"])

    # 4. Entity Detail
    ent_id = g_data["nodes"][0]["id"]
    ent_res = client.get(f"/api/knowledge-graph/entities/{ent_id}")
    assert ent_res.status_code == 200
    assert ent_res.json()["canonical_name"] is not None

    # 5. Relation Detail
    edge_id = g_data["edges"][0]["id"]
    rel_res = client.get(f"/api/knowledge-graph/relations/{edge_id}")
    assert rel_res.status_code == 200
    assert len(rel_res.json()["evidence"]) > 0

    # 6. Review Relation
    rev_res = client.post(f"/api/knowledge-graph/relations/{edge_id}/review", json={
        "status": "approved",
        "notes": "Verified by investigator"
    })
    assert rev_res.status_code == 200
    assert rev_res.json()["review_status"] == "approved"

    # 7. Stats
    stats_res = client.get(f"/api/knowledge-graph/stats?project_id={sample_document.project_id}")
    assert stats_res.status_code == 200
    st_data = stats_res.json()
    assert st_data["total_entities"] > 0
    assert st_data["total_relations"] > 0
    assert st_data["total_evidence_records"] > 0

def test_api_generate_knowledge_graph_from_text(client, db_session, sample_project):
    prompt_text = (
        "Transformer architectures utilize multi-head self-attention mechanisms to model contextual dependencies. "
        "Large Language Models trained on massive text corpora demonstrate strong zero-shot generalization across benchmarks."
    )
    payload = {
        "text": prompt_text,
        "title": "Transformer Attention & Large Language Models",
        "project_id": sample_project.id
    }
    res = client.post("/api/knowledge-graph/generate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "nodes" in data
    assert "edges" in data
    assert "build_summary" in data
    assert data["build_summary"]["entities_registered"] > 0
    assert len(data["nodes"]) >= 2
    assert any("Transformer" in n["label"] or "Attention" in n["label"] or "Language" in n["label"] for n in data["nodes"])

