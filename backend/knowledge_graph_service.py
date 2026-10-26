"""
CIPHER — Knowledge Graph Service
Scientific Entity & Relationship Extraction, Normalization, Evidence Grounding, and Persistence Engine.
"""

import os
import re
import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, func

from database import (
    Document, DocumentChunk, Project,
    KGEntity, KGEntityMention, KGRelation, KGRelationEvidence, KGBuildJob
)
from nlp_service import nlp_service, sent_tokenize, word_tokenize

logger = logging.getLogger("cipher.knowledge_graph")

# --- Extensible Entity Categories ---
ENTITY_CATEGORIES = {
    "PAPER": "Scientific paper, study, preprint or publication",
    "AUTHOR": "Researcher, author, or investigator",
    "ORGANIZATION": "University, laboratory, institute, or company",
    "METHOD": "Algorithmic technique, statistical method, or workflow",
    "MODEL": "Machine learning, deep learning, or statistical model architecture",
    "DATASET": "Empirical dataset, corpus, collection, or database",
    "BENCHMARK": "Evaluation challenge, standard task, or benchmark suite",
    "TASK": "Problem definition, objective, or scientific domain task",
    "CONCEPT": "Theoretical construct, property, or domain idea",
    "METRIC": "Measurement criterion, evaluation index, or quantitative score",
    "RESULT": "Empirical outcome, finding, or quantitative observation",
    "TECHNOLOGY": "Software framework, runtime, hardware device, or instrument",
    "MATERIAL": "Physical substance, component, or mechanical apparatus",
    "CHEMICAL": "Chemical compound, reagent, or molecule",
    "GENE": "Gene, genetic sequence, or biomarker",
    "DISEASE": "Pathology, defect, symptom, or failure mode",
    "OTHER": "Unclassified scientific entity"
}

# --- Extensible Relationship Predicates ---
RELATION_PREDICATES = {
    "USES_METHOD": "Model or pipeline employs a specific method or technique",
    "EVALUATED_ON": "Model or technique evaluated on a dataset or benchmark",
    "COMPARES_WITH": "Evaluates or contrasts with another method or model",
    "REPORTS_METRIC": "Method or study evaluates a specific quantitative metric",
    "REPORTS_RESULT": "Method or study reports a specific empirical finding",
    "IMPROVES": "Method improves or outperforms a baseline or metric",
    "REDUCES": "Method or mechanism reduces an error, latency, or defect",
    "INCREASES": "Method increases a desirable property or metric",
    "ASSOCIATED_WITH": "Entity exhibits statistical or domain correlation with another",
    "CAUSES": "Entity directly produces or induces an outcome or defect",
    "TREATS": "Intervention treats or remedies a defect or pathology",
    "INHIBITS": "Entity prevents, slows down, or blocks a mechanism",
    "PART_OF": "Entity is a constituent or subsystem of a larger entity",
    "AUTHORED_BY": "Paper or discovery created by researcher or organization",
    "PUBLISHED_IN": "Paper or finding published in journal, conference, or repository",
    "MENTIONS": "Source document explicitly discusses an entity or topic",
    "CITES": "Source document refers to or references another work",
    "OTHER": "General unclassified source-supported relationship"
}

# Curated Scientific Phrase Lexicon (Surface name -> (Canonical Name, Category))
SCIENTIFIC_DICTIONARY: Dict[str, Tuple[str, str]] = {
    # Models & Architectures
    "xgboost": ("XGBoost", "MODEL"),
    "extreme gradient boosting": ("XGBoost", "MODEL"),
    "gradient boosting": ("Gradient Boosting", "MODEL"),
    "gbdt": ("Gradient Boosting Decision Trees", "MODEL"),
    "gradient boosted decision trees": ("Gradient Boosting Decision Trees", "MODEL"),
    "random forest": ("Random Forest", "MODEL"),
    "gradient boosting": ("Gradient Boosting", "MODEL"),
    "gradient boosted trees": ("Gradient Boosted Trees", "MODEL"),
    "gbdt": ("Gradient Boosting", "MODEL"),
    "lightgbm": ("LightGBM", "MODEL"),
    "catboost": ("CatBoost", "MODEL"),
    "autoencoder": ("Autoencoder", "MODEL"),
    "autoencoders": ("Autoencoder", "MODEL"),
    "dense autoencoder": ("Dense Autoencoder", "MODEL"),
    "cnn autoencoder": ("CNN Autoencoder", "MODEL"),
    "convolutional neural network": ("Convolutional Neural Network", "MODEL"),
    "cnn": ("Convolutional Neural Network", "MODEL"),
    "transformer": ("Transformer", "MODEL"),
    "bert": ("BERT", "MODEL"),
    "minilm": ("MiniLM", "MODEL"),
    "all-minilm-l6-v2": ("all-MiniLM-L6-v2", "MODEL"),
    "sentence transformers": ("SentenceTransformers", "MODEL"),
    "sentencetransformers": ("SentenceTransformers", "MODEL"),
    "lstm": ("LSTM", "MODEL"),
    "support vector machine": ("Support Vector Machine", "MODEL"),
    "svm": ("Support Vector Machine", "MODEL"),
    "decision tree": ("Decision Tree", "MODEL"),
    "logistic regression": ("Logistic Regression", "MODEL"),
    "qwen": ("Qwen", "MODEL"),
    "qwen3": ("Qwen 3", "MODEL"),
    "deep neural network": ("Deep Neural Network", "MODEL"),
    "dnn": ("Deep Neural Network", "MODEL"),
    
    # Methods & Digital Signal Processing
    "stft": ("Short-Time Fourier Transform", "METHOD"),
    "short-time fourier transform": ("Short-Time Fourier Transform", "METHOD"),
    "fourier transform": ("Fourier Transform", "METHOD"),
    "log-mel spectrogram": ("Log-Mel Spectrogram", "METHOD"),
    "mel spectrogram": ("Mel Spectrogram", "METHOD"),
    "mfcc": ("Mel-Frequency Cepstral Coefficients", "METHOD"),
    "mel-frequency cepstral coefficients": ("Mel-Frequency Cepstral Coefficients", "METHOD"),
    "spectral centroid": ("Spectral Centroid", "METHOD"),
    "spectral rolloff": ("Spectral Rolloff", "METHOD"),
    "zero-crossing rate": ("Zero-Crossing Rate", "METHOD"),
    "zcr": ("Zero-Crossing Rate", "METHOD"),
    "root mean square": ("Root Mean Square", "METHOD"),
    "rms": ("Root Mean Square", "METHOD"),
    "digital signal processing": ("Digital Signal Processing", "METHOD"),
    "dsp": ("Digital Signal Processing", "METHOD"),
    "stratified cross-validation": ("Stratified Cross-Validation", "METHOD"),
    "cross-validation": ("Cross-Validation", "METHOD"),
    "5-fold cross-validation": ("5-Fold Stratified Cross-Validation", "METHOD"),
    "feature extraction": ("Feature Extraction", "METHOD"),
    "data augmentation": ("Data Augmentation", "METHOD"),
    "transfer learning": ("Transfer Learning", "METHOD"),
    "supervised learning": ("Supervised Learning", "METHOD"),
    "unsupervised learning": ("Unsupervised Learning", "METHOD"),
    "semi-supervised learning": ("Semi-Supervised Learning", "METHOD"),
    "retrieval-augmented generation": ("Retrieval-Augmented Generation", "METHOD"),
    "rag": ("Retrieval-Augmented Generation", "METHOD"),
    "dense retrieval": ("Dense Retrieval", "METHOD"),
    "named entity recognition": ("Named Entity Recognition", "METHOD"),
    "ner": ("Named Entity Recognition", "METHOD"),
    "part-of-speech tagging": ("Part-of-Speech Tagging", "METHOD"),
    "pos tagging": ("Part-of-Speech Tagging", "METHOD"),
    "cosine similarity": ("Cosine Similarity", "METHOD"),
    "condition encoding": ("Machine-ID Condition Encoding", "METHOD"),
    "machine-id condition encoding": ("Machine-ID Condition Encoding", "METHOD"),
    "unsupervised mse reconstruction": ("Unsupervised MSE Reconstruction", "METHOD"),

    # Datasets & Benchmarks
    "dcase 2020": ("DCASE 2020", "BENCHMARK"),
    "dcase 2020 task 2": ("DCASE 2020 Task 2", "BENCHMARK"),
    "dcase": ("DCASE Challenge", "BENCHMARK"),
    "imagenet": ("ImageNet", "DATASET"),
    "glue": ("GLUE Benchmark", "BENCHMARK"),
    "squad": ("SQuAD", "DATASET"),
    "zenodo": ("Zenodo Archive", "DATASET"),
    "mimic-iii": ("MIMIC-III", "DATASET"),
    "mnist": ("MNIST", "DATASET"),
    "cifar-10": ("CIFAR-10", "DATASET"),
    "industrial fan dataset": ("DCASE 2020 Industrial Fan Dataset", "DATASET"),

    # Tasks & Problems
    "acoustic anomaly detection": ("Acoustic Anomaly Detection", "TASK"),
    "anomaly detection": ("Anomaly Detection", "TASK"),
    "fan failure prediction": ("Industrial Fan Failure Prediction", "TASK"),
    "failure prediction": ("Failure Prediction", "TASK"),
    "predictive maintenance": ("Predictive Maintenance", "TASK"),
    "condition monitoring": ("Machine Condition Monitoring", "TASK"),
    "fault diagnosis": ("Fault Diagnosis", "TASK"),
    "bearing degradation": ("Bearing Degradation Detection", "TASK"),
    "blade unbalance": ("Blade Unbalance Detection", "TASK"),
    "airflow distortion": ("Airflow Distortion Detection", "TASK"),
    "text classification": ("Text Classification", "TASK"),
    "speech recognition": ("Speech Recognition", "TASK"),

    # Metrics
    "accuracy": ("Classification Accuracy", "METRIC"),
    "overall accuracy": ("Classification Accuracy", "METRIC"),
    "roc-auc": ("ROC-AUC Score", "METRIC"),
    "area under roc curve": ("ROC-AUC Score", "METRIC"),
    "f1 score": ("F1 Score", "METRIC"),
    "f1-score": ("F1 Score", "METRIC"),
    "precision": ("Precision", "METRIC"),
    "recall": ("Recall (Sensitivity)", "METRIC"),
    "sensitivity": ("Recall (Sensitivity)", "METRIC"),
    "specificity": ("Specificity (TNR)", "METRIC"),
    "false positive rate": ("False Positive Rate", "METRIC"),
    "fpr": ("False Positive Rate", "METRIC"),
    "false negative rate": ("False Negative Rate", "METRIC"),
    "fnr": ("False Negative Rate", "METRIC"),
    "latency": ("Inference Latency", "METRIC"),
    "inference latency": ("Inference Latency", "METRIC"),
    "runtime": ("Execution Runtime", "METRIC"),
    "cpu runtime": ("CPU Runtime", "METRIC"),
    "mean squared error": ("Mean Squared Error", "METRIC"),
    "mse": ("Mean Squared Error", "METRIC"),
    "tokens per second": ("Tokens Per Second", "METRIC"),

    # Technologies, Frameworks & Hardware
    "spacy": ("spaCy", "TECHNOLOGY"),
    "nltk": ("NLTK", "TECHNOLOGY"),
    "scikit-learn": ("scikit-learn", "TECHNOLOGY"),
    "fastapi": ("FastAPI", "TECHNOLOGY"),
    "pytorch": ("PyTorch", "TECHNOLOGY"),
    "tensorflow": ("TensorFlow", "TECHNOLOGY"),
    "ollama": ("Ollama", "TECHNOLOGY"),
    "sqlite": ("SQLite", "TECHNOLOGY"),
    "sqlalchemy": ("SQLAlchemy", "TECHNOLOGY"),
    "cytoscape.js": ("Cytoscape.js", "TECHNOLOGY"),
    "cytoscape": ("Cytoscape.js", "TECHNOLOGY"),
    "microphone": ("Acoustic Microphone", "TECHNOLOGY"),
    "directional condenser microphone": ("Directional Condenser Microphone", "TECHNOLOGY"),
    "accelerometer": ("Accelerometer", "TECHNOLOGY"),
    "cpu": ("Central Processing Unit (CPU)", "TECHNOLOGY"),
    "gpu": ("Graphics Processing Unit (GPU)", "TECHNOLOGY"),

    # Materials, Parts & Apparatus
    "industrial fan": ("Industrial Ventilation Fan", "MATERIAL"),
    "industrial ventilation fan": ("Industrial Ventilation Fan", "MATERIAL"),
    "bearing": ("Mechanical Bearing", "MATERIAL"),
    "fan blade": ("Fan Blade", "MATERIAL"),
    "ventilation duct": ("Ventilation Duct", "MATERIAL"),
    "motor": ("Electric Motor", "MATERIAL"),

    # Organizations
    "hitachi": ("Hitachi Ltd.", "ORGANIZATION"),
    "hitachi ltd.": ("Hitachi Ltd.", "ORGANIZATION"),
    "google": ("Google", "ORGANIZATION"),
    "google deepmind": ("Google DeepMind", "ORGANIZATION"),
    "zenodo repository": ("Zenodo Repository", "ORGANIZATION")
}

# Compile regex search patterns sorted by length descending so multi-word matches take precedence
SORTED_PHRASES = sorted(SCIENTIFIC_DICTIONARY.keys(), key=lambda x: len(x), reverse=True)
PHRASE_REGEX = re.compile(
    r'\b(' + '|'.join(re.escape(p) for p in SORTED_PHRASES) + r')\b',
    re.IGNORECASE
)

# Negation & Uncertainty signals
NEGATION_WORDS = {"not", "no", "never", "did not", "does not", "cannot", "failed to", "fails", "without", "neither", "nor"}
SPECULATION_WORDS = {"suggest", "suggests", "hypothesize", "hypothesizes", "speculate", "might", "may", "could", "preliminary", "possible", "potential", "proposed", "expected"}


class KnowledgeGraphService:
    def __init__(self):
        self.nlp = nlp_service.nlp

    # --- Normalization Utilities ---
    @staticmethod
    def normalize_key(name: str) -> str:
        """Create a stable, case-insensitive, punctuation-stripped lookup key."""
        if not name:
            return ""
        s = name.strip().lower()
        s = re.sub(r'[^\w\s\-]', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()

    # --- Entity Extraction ---
    def extract_entities_from_text(self, text: str, document_title: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Hybrid Entity Extraction:
        1. Scientific phrase & domain terminology matching (highest precision).
        2. spaCy NER for general names (PERSON -> AUTHOR, ORG, DATE, GPE).
        3. Document Title tagged as PAPER entity if provided.
        """
        if not text:
            return []

        found_entities: List[Dict[str, Any]] = []
        matched_spans: Set[Tuple[int, int]] = set()

        # 1. Scientific Domain Lexicon Extraction
        for match in PHRASE_REGEX.finditer(text):
            start, end = match.start(), match.end()
            raw_text = match.group(0)
            lookup_key = raw_text.lower()
            if lookup_key in SCIENTIFIC_DICTIONARY:
                canon_name, category = SCIENTIFIC_DICTIONARY[lookup_key]
                found_entities.append({
                    "raw_text": raw_text,
                    "canonical_name": canon_name,
                    "entity_type": category,
                    "normalized_key": self.normalize_key(canon_name),
                    "start_char": start,
                    "end_char": end,
                    "confidence": 0.96
                })
                matched_spans.add((start, end))

        # 2. General spaCy NER (if spaCy available)
        if self.nlp:
            try:
                # Cap sample for NER performance
                doc_nlp = self.nlp(text[:50000])
                for ent in doc_nlp.ents:
                    ent_start = ent.start_char
                    ent_end = ent.end_char

                    # Avoid duplicate overlap with scientific dictionary
                    overlaps = any(
                        not (ent_end <= s or ent_start >= e)
                        for (s, e) in matched_spans
                    )
                    if overlaps:
                        continue

                    label = ent.label_
                    category = "OTHER"
                    if label == "PERSON":
                        category = "AUTHOR"
                    elif label == "ORG":
                        category = "ORGANIZATION"
                    elif label in ("PRODUCT", "LAW"):
                        category = "TECHNOLOGY"
                    elif label in ("DATE", "TIME"):
                        category = "CONCEPT"
                    elif label in ("GPE", "LOC"):
                        category = "CONCEPT"

                    clean_name = ent.text.strip()
                    if len(clean_name) > 2 and clean_name.lower() not in {"this", "that", "these", "those"}:
                        found_entities.append({
                            "raw_text": clean_name,
                            "canonical_name": clean_name,
                            "entity_type": category,
                            "normalized_key": self.normalize_key(clean_name),
                            "start_char": ent_start,
                            "end_char": ent_end,
                            "confidence": 0.88
                        })
                        matched_spans.add((ent_start, ent_end))

                # 3. Informative Concept & Noun Phrase Extraction (for AI Knowledge Graph generation)
                for chunk in doc_nlp.noun_chunks:
                    if chunk.root.is_stop or all(t.is_stop or t.is_punct or t.is_space for t in chunk):
                        continue
                    c_start = chunk.start_char
                    c_end = chunk.end_char
                    raw = chunk.text.strip()
                    clean_name = re.sub(r'^(a|an|the|our|their|its|this|that|these|those)\s+', '', raw, flags=re.I).strip()
                    if not clean_name:
                        continue
                    words = clean_name.split()
                    if (
                        len(clean_name) >= 3
                        and len(words) <= 4
                        and clean_name.lower() not in {"it", "they", "we", "them", "this", "that", "these", "those", "something", "anything", "nothing", "everything", "the", "and", "or", "but", "while", "because"}
                    ):
                        overlaps = any(not (c_end <= s or c_start >= e) for (s, e) in matched_spans)
                        if not overlaps:
                            cl = clean_name.lower()
                            cat = "CONCEPT"
                            if any(k in cl for k in ["model", "network", "net", "transformer", "bert", "gpt", "autoencoder", "classifier", "regressor"]):
                                cat = "MODEL"
                            elif any(k in cl for k in ["method", "algorithm", "technique", "approach", "framework", "protocol", "pipeline", "strategy"]):
                                cat = "METHOD"
                            elif any(k in cl for k in ["dataset", "corpus", "benchmark", "database", "collection"]):
                                cat = "DATASET"
                            elif any(k in cl for k in ["metric", "accuracy", "f1", "loss", "latency", "score", "rate", "index"]):
                                cat = "METRIC"
                            elif any(k in cl for k in ["software", "system", "hardware", "device", "platform", "tool", "api", "gpu", "cpu"]):
                                cat = "TECHNOLOGY"

                            found_entities.append({
                                "raw_text": clean_name,
                                "canonical_name": clean_name.title(),
                                "entity_type": cat,
                                "normalized_key": self.normalize_key(clean_name),
                                "start_char": c_start,
                                "end_char": c_end,
                                "confidence": 0.85
                            })
                            matched_spans.add((c_start, c_end))

            except Exception as e:
                logger.warning(f"spaCy NER pass failed: {e}")

        # 4. Document Title as a PAPER entity (if given)
        if document_title:
            doc_key = self.normalize_key(document_title)
            found_entities.append({
                "raw_text": document_title,
                "canonical_name": document_title,
                "entity_type": "PAPER",
                "normalized_key": doc_key,
                "start_char": 0,
                "end_char": len(document_title),
                "confidence": 0.99
            })

        return found_entities

    # --- Relationship Extraction ---
    def extract_relations_from_sentence(
        self,
        sentence: str,
        entities_in_sentence: List[Dict[str, Any]],
        doc_title: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Syntactic & Linguistic Pattern Relationship Extractor.
        Matches entity pairs co-occurring in the sentence, checks directional verbal cues,
        negation polarity, and speculative modality.
        """
        if len(entities_in_sentence) < 2:
            return []

        relations: List[Dict[str, Any]] = []
        sent_lower = sentence.lower()

        # Check Polarity (Negation)
        is_negative = any(re.search(r'\b' + re.escape(w) + r'\b', sent_lower) for w in NEGATION_WORDS)
        polarity = "negative" if is_negative else "positive"

        # Check Modality (Speculation)
        is_speculative = any(re.search(r'\b' + re.escape(w) + r'\b', sent_lower) for w in SPECULATION_WORDS)
        modality = "speculative" if is_speculative else "factual"

        # Pairwise examination
        for i in range(len(entities_in_sentence)):
            for j in range(len(entities_in_sentence)):
                if i == j:
                    continue

                e1 = entities_in_sentence[i]
                e2 = entities_in_sentence[j]

                # Prevent self-relation with same canonical entity
                if e1["normalized_key"] == e2["normalized_key"]:
                    continue

                t1 = e1["entity_type"]
                t2 = e2["entity_type"]

                # Extract inter-entity text snippet
                idx1 = sentence.find(e1["raw_text"])
                idx2 = sentence.find(e2["raw_text"])
                between_text = ""
                if idx1 != -1 and idx2 != -1 and idx1 < idx2:
                    between_text = sentence[idx1 + len(e1["raw_text"]):idx2].lower()

                predicate: Optional[str] = None
                confidence = 0.85

                # 1. EVALUATED_ON: Model/Method tested on Dataset/Benchmark
                if t1 in ("MODEL", "METHOD") and t2 in ("DATASET", "BENCHMARK"):
                    if any(w in sent_lower for w in ["evaluated on", "tested on", "benchmarked on", "trained on", "dataset", "development dataset", "applied to", "experiment"]):
                        predicate = "EVALUATED_ON"
                        confidence = 0.94
                    else:
                        predicate = "EVALUATED_ON"
                        confidence = 0.82

                # 2. REPORTS_METRIC / REPORTS_RESULT: Model/Method reports Metric
                elif t1 in ("MODEL", "METHOD", "PAPER") and t2 in ("METRIC", "RESULT"):
                    if any(w in sent_lower for w in ["achieve", "achieved", "attains", "yields", "reports", "accuracy of", "precision of", "exceeds", "scored"]):
                        predicate = "REPORTS_METRIC"
                        confidence = 0.92
                    else:
                        predicate = "REPORTS_METRIC"
                        confidence = 0.78

                # 3. USES_METHOD: Model/Method employs technique
                elif (t1 in ("MODEL", "METHOD", "PAPER") and t2 in ("METHOD", "TECHNOLOGY")) or (t1 == "METHOD" and t2 == "METHOD"):
                    if any(w in between_text for w in ["uses", "using", "utilizes", "employs", "incorporating", "based on", "via", "applying", "computed with"]):
                        predicate = "USES_METHOD"
                        confidence = 0.91
                    elif any(w in sent_lower for w in ["incorporate", "using", "extract", "pipeline", "features", "transform"]):
                        predicate = "USES_METHOD"
                        confidence = 0.83

                # 4. IMPROVES / OUTPERFORMS: Model improves metric or outperforms another model
                elif any(w in sent_lower for w in ["outperform", "outperformed", "outperforms", "improves", "surpasses", "exceeds", "boosts"]):
                    if t2 == "METRIC":
                        predicate = "IMPROVES"
                        confidence = 0.90
                    elif t1 == "MODEL" and t2 == "MODEL":
                        predicate = "IMPROVES"
                        confidence = 0.89

                # 5. REDUCES: Reduces latency, failure, defect
                elif any(w in sent_lower for w in ["reduces", "reduced", "lowers", "minimizes", "mitigates", "eliminates", "decreases"]):
                    predicate = "REDUCES"
                    confidence = 0.88

                # 6. INCREASES:
                elif any(w in sent_lower for w in ["increases", "increased", "elevates", "amplifies", "magnifies"]):
                    predicate = "INCREASES"
                    confidence = 0.86

                # 7. COMPARES_WITH:
                elif any(w in sent_lower for w in ["compared to", "in comparison with", "versus", "vs.", "against", "benchmarked against"]):
                    if t1 in ("MODEL", "METHOD") and t2 in ("MODEL", "METHOD"):
                        predicate = "COMPARES_WITH"
                        confidence = 0.89

                # 8. AUTHORED_BY / PUBLISHED_IN:
                elif t2 in ("AUTHOR", "ORGANIZATION") and t1 in ("PAPER", "MODEL", "METHOD"):
                    if any(w in sent_lower for w in ["by", "proposed by", "authored by", "published", "sourced from"]):
                        predicate = "AUTHORED_BY"
                        confidence = 0.93

                # 9. PART_OF: Component of a larger assembly or dataset
                elif any(w in sent_lower for w in ["part of", "component of", "subsystem of", "consists of", "element of"]):
                    predicate = "PART_OF"
                    confidence = 0.87

                # 10. Dynamic Verbal Predicate from between_text or sentence dependency
                elif not predicate and between_text:
                    verbs = re.findall(r'\b(structure|structures|structured|represent|represents|represented|integrate|integrates|integrated|combine|combines|combined|extract|extracts|extracted|generate|generates|generated|predict|predicts|predicted|optimize|optimizes|optimized|enable|enables|enabled|modify|modifies|modified|regulate|regulates|regulated|activate|activates|activated|inhibit|inhibits|inhibited|detect|detects|detected|analyze|analyzes|analyzed|process|processes|processed|evaluate|evaluates|evaluated|support|supports|supported|lead to|leads to|result in|results in|depend on|depends on)\b', between_text, flags=re.I)
                    if verbs:
                        predicate = verbs[0].upper().replace(" ", "_")
                        confidence = 0.86
                    elif len(between_text.split()) < 8 and len(between_text.strip()) > 0:
                        predicate = "ASSOCIATED_WITH"
                        confidence = 0.70

                if predicate:
                    relations.append({
                        "source_entity": e1,
                        "target_entity": e2,
                        "predicate": predicate,
                        "confidence": confidence,
                        "polarity": polarity,
                        "modality": modality,
                        "excerpt": sentence.strip(),
                        "extraction_method": "rule_syntax"
                    })

        return relations

    # --- Core Document Graph Builder ---
    def build_graph_for_document(
        self,
        db: Session,
        document_id: int,
        project_id: Optional[int] = None,
        rebuild: bool = False
    ) -> Dict[str, Any]:
        """
        End-to-End Pipeline:
        1. Fetch Document and text chunks.
        2. Sentence segmentation.
        3. Scientific NER + phrase mapping.
        4. Relationship extraction + evidence grounding.
        5. Idempotent persistence to SQLite tables (kg_entities, kg_entity_mentions, kg_relations, kg_relation_evidence).
        6. Updates kg_build_jobs.
        """
        doc = db.query(Document).filter(Document.id == document_id).first()
        if not doc:
            raise ValueError(f"Document with ID {document_id} not found.")

        active_project_id = project_id or doc.project_id

        # Track or create build job
        job = db.query(KGBuildJob).filter(KGBuildJob.document_id == document_id).first()
        if not job:
            job = KGBuildJob(
                document_id=document_id,
                project_id=active_project_id,
                status="processing"
            )
            db.add(job)
            db.commit()
            db.refresh(job)
        else:
            job.status = "processing"
            job.error_message = None
            db.commit()

        start_time = time.perf_counter()

        try:
            content_text = doc.content_text or ""
            # If text is empty, check chunks
            if not content_text.strip():
                chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).order_by(DocumentChunk.chunk_index).all()
                content_text = "\n\n".join([c.text for c in chunks if c.text])
                doc.content_text = content_text
                db.commit()

            if not content_text.strip():
                raise ValueError("Document has no readable text content to build a knowledge graph.")

            # Idempotency / Rebuild logic:
            if rebuild:
                # Remove prior mentions and evidence from this document
                db.query(KGEntityMention).filter(KGEntityMention.document_id == document_id).delete()
                
                # Delete evidence from this document
                evidences = db.query(KGRelationEvidence).filter(KGRelationEvidence.document_id == document_id).all()
                rel_ids_to_check = {ev.relation_id for ev in evidences}
                db.query(KGRelationEvidence).filter(KGRelationEvidence.document_id == document_id).delete()
                
                # If a relation now has 0 evidence records across all documents, clean it up unless approved by user
                for r_id in rel_ids_to_check:
                    remaining_ev = db.query(func.count(KGRelationEvidence.id)).filter(KGRelationEvidence.relation_id == r_id).scalar()
                    if remaining_ev == 0:
                        rel_obj = db.query(KGRelation).filter(KGRelation.id == r_id).first()
                        if rel_obj and rel_obj.review_status != "approved":
                            db.delete(rel_obj)
                
                db.commit()

            # Retrieve Document Chunks for passage offset tracking
            db_chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).order_by(DocumentChunk.chunk_index).all()
            
            # Sentence segmentation
            sentences = sent_tokenize(content_text)
            if not sentences:
                sentences = [s.strip() for s in re.split(r'\n+', content_text) if s.strip()]

            # Entity registry cache to resolve entities in memory before DB lookups
            entity_cache: Dict[str, KGEntity] = {}

            # Pre-populate entity cache with existing project entities
            existing_entities = db.query(KGEntity).filter(KGEntity.project_id == active_project_id).all()
            for ent in existing_entities:
                entity_cache[ent.normalized_key] = ent

            entity_mention_count = 0
            relation_created_count = 0
            evidence_count = 0

            # Process sentence-by-sentence to keep strict evidence provenance
            for sent_idx, sentence in enumerate(sentences):
                sent_clean = sentence.strip()
                if len(sent_clean) < 15:
                    continue

                # Find which chunk this sentence belongs to
                matched_chunk_id = None
                matched_page_num = None
                for chk in db_chunks:
                    if sent_clean[:40] in (chk.text or ""):
                        matched_chunk_id = chk.id
                        matched_page_num = chk.page_num
                        break

                # Extract entities from sentence
                sent_entities = self.extract_entities_from_text(sent_clean, document_title=None)
                if not sent_entities:
                    continue

                # Ensure entity records exist in DB
                for e_data in sent_entities:
                    n_key = e_data["normalized_key"]
                    if n_key not in entity_cache:
                        # Create new canonical entity
                        new_entity = KGEntity(
                            project_id=active_project_id,
                            canonical_name=e_data["canonical_name"],
                            entity_type=e_data["entity_type"],
                            normalized_key=n_key,
                            aliases_json=json.dumps([e_data["raw_text"]]),
                            description=f"Extracted from {doc.title}"
                        )
                        db.add(new_entity)
                        db.commit()
                        db.refresh(new_entity)
                        entity_cache[n_key] = new_entity
                    else:
                        # Append alias if new surface form
                        curr_ent = entity_cache[n_key]
                        try:
                            aliases = json.loads(curr_ent.aliases_json or "[]")
                        except Exception:
                            aliases = []
                        if e_data["raw_text"] not in aliases and len(aliases) < 20:
                            aliases.append(e_data["raw_text"])
                            curr_ent.aliases_json = json.dumps(aliases)
                            db.commit()

                    ent_obj = entity_cache[n_key]

                    # Record mention
                    mention = KGEntityMention(
                        entity_id=ent_obj.id,
                        document_id=document_id,
                        chunk_id=matched_chunk_id,
                        mention_text=e_data["raw_text"],
                        start_char=e_data["start_char"],
                        end_char=e_data["end_char"],
                        sentence_text=sent_clean
                    )
                    db.add(mention)
                    entity_mention_count += 1

                # Extract pairwise relationships within this sentence
                extracted_relations = self.extract_relations_from_sentence(
                    sent_clean, sent_entities, doc_title=doc.title
                )

                for rel_data in extracted_relations:
                    src_key = rel_data["source_entity"]["normalized_key"]
                    tgt_key = rel_data["target_entity"]["normalized_key"]
                    pred = rel_data["predicate"]

                    src_obj = entity_cache.get(src_key)
                    tgt_obj = entity_cache.get(tgt_key)

                    if not src_obj or not tgt_obj or src_obj.id == tgt_obj.id:
                        continue

                    # Check if relation already exists in project
                    existing_rel = db.query(KGRelation).filter(
                        KGRelation.project_id == active_project_id,
                        KGRelation.source_id == src_obj.id,
                        KGRelation.target_id == tgt_obj.id,
                        KGRelation.predicate == pred
                    ).first()

                    if not existing_rel:
                        existing_rel = KGRelation(
                            project_id=active_project_id,
                            source_id=src_obj.id,
                            target_id=tgt_obj.id,
                            predicate=pred,
                            confidence=rel_data["confidence"],
                            extraction_method=rel_data["extraction_method"],
                            review_status="unreviewed",
                            polarity=rel_data["polarity"],
                            modality=rel_data["modality"]
                        )
                        db.add(existing_rel)
                        db.commit()
                        db.refresh(existing_rel)
                        relation_created_count += 1
                    else:
                        # Boost confidence with multiple evidence observations
                        existing_rel.confidence = min(0.99, existing_rel.confidence + 0.02)
                        db.commit()

                    # Record evidence snippet
                    # Check if identical excerpt already logged
                    existing_evidence = db.query(KGRelationEvidence).filter(
                        KGRelationEvidence.relation_id == existing_rel.id,
                        KGRelationEvidence.document_id == document_id,
                        KGRelationEvidence.excerpt == rel_data["excerpt"]
                    ).first()

                    if not existing_evidence:
                        ev = KGRelationEvidence(
                            relation_id=existing_rel.id,
                            document_id=document_id,
                            chunk_id=matched_chunk_id,
                            excerpt=rel_data["excerpt"],
                            page_num=matched_page_num,
                            confidence=rel_data["confidence"],
                            extraction_method=rel_data["extraction_method"],
                            verification_status="unverified"
                        )
                        db.add(ev)
                        evidence_count += 1

            db.commit()

            # Also create a canonical PAPER entity representing this document and link to key methods
            doc_paper_key = self.normalize_key(doc.title)
            if doc_paper_key not in entity_cache:
                doc_paper_ent = KGEntity(
                    project_id=active_project_id,
                    canonical_name=doc.title,
                    entity_type="PAPER",
                    normalized_key=doc_paper_key,
                    description=f"Source Document: {doc.filename or doc.title}"
                )
                db.add(doc_paper_ent)
                db.commit()
                db.refresh(doc_paper_ent)
                entity_cache[doc_paper_key] = doc_paper_ent
            else:
                doc_paper_ent = entity_cache[doc_paper_key]

            # Connect PAPER to top mentioned models/methods
            top_entities = db.query(
                KGEntity, func.count(KGEntityMention.id).label("m_count")
            ).join(KGEntityMention, KGEntity.id == KGEntityMention.entity_id)\
             .filter(KGEntityMention.document_id == document_id)\
             .filter(KGEntity.entity_type.in_(["MODEL", "METHOD", "DATASET", "BENCHMARK"]))\
             .group_by(KGEntity.id)\
             .order_by(func.count(KGEntityMention.id).desc())\
             .limit(5).all()

            for top_ent, _ in top_entities:
                if top_ent.id == doc_paper_ent.id:
                    continue
                pred = "EVALUATED_ON" if top_ent.entity_type in ("DATASET", "BENCHMARK") else "USES_METHOD"
                p_rel = db.query(KGRelation).filter(
                    KGRelation.project_id == active_project_id,
                    KGRelation.source_id == doc_paper_ent.id,
                    KGRelation.target_id == top_ent.id,
                    KGRelation.predicate == pred
                ).first()
                if not p_rel:
                    p_rel = KGRelation(
                        project_id=active_project_id,
                        source_id=doc_paper_ent.id,
                        target_id=top_ent.id,
                        predicate=pred,
                        confidence=0.95,
                        extraction_method="document_provenance",
                        review_status="unreviewed",
                        polarity="positive",
                        modality="factual"
                    )
                    db.add(p_rel)
                    db.commit()
                    db.refresh(p_rel)
                    relation_created_count += 1

                    ev = KGRelationEvidence(
                        relation_id=p_rel.id,
                        document_id=document_id,
                        excerpt=f"Document '{doc.title}' investigates and benchmarks {top_ent.canonical_name}.",
                        confidence=0.95,
                        extraction_method="document_provenance",
                        verification_status="verified"
                    )
                    db.add(ev)
                    evidence_count += 1
            db.commit()

            # Finalize Job
            job.status = "completed"
            job.entity_count = len(entity_cache)
            job.relation_count = relation_created_count
            job.evidence_count = evidence_count
            db.commit()

            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return {
                "status": "completed",
                "document_id": document_id,
                "document_title": doc.title,
                "project_id": active_project_id,
                "entities_registered": len(entity_cache),
                "mentions_extracted": entity_mention_count,
                "relations_created": relation_created_count,
                "evidence_snippets_saved": evidence_count,
                "latency_ms": latency_ms
            }

        except Exception as e:
            db.rollback()
            job.status = "failed"
            job.error_message = str(e)
            db.commit()
            logger.error(f"Knowledge Graph build failed for document {document_id}: {e}", exc_info=True)
            raise e

    # --- Query & Serialization Engine ---
    def get_graph_data(
        self,
        db: Session,
        project_id: Optional[int] = None,
        document_id: Optional[int] = None,
        entity_type: Optional[str] = None,
        predicate: Optional[str] = None,
        search_query: Optional[str] = None,
        min_confidence: float = 0.0,
        review_status: Optional[str] = None,
        limit: int = 120,
        include_papers: bool = True
    ) -> Dict[str, Any]:
        """
        Query graph nodes and edges with server-side filters, confidence boundaries,
        and safe truncation limits.
        """
        # Base query for relations
        rel_query = db.query(KGRelation)

        if project_id:
            rel_query = rel_query.filter(KGRelation.project_id == project_id)

        if predicate:
            rel_query = rel_query.filter(KGRelation.predicate == predicate)

        if min_confidence > 0.0:
            rel_query = rel_query.filter(KGRelation.confidence >= min_confidence)

        if review_status:
            rel_query = rel_query.filter(KGRelation.review_status == review_status)

        # Filter by document if specified
        if document_id:
            rel_query = rel_query.join(KGRelationEvidence, KGRelation.id == KGRelationEvidence.relation_id)\
                                 .filter(KGRelationEvidence.document_id == document_id)

        # If search query provided, find matching entities and their 1-hop connected relations
        if search_query:
            sq = f"%{search_query.strip().lower()}%"
            matched_nodes = db.query(KGEntity).filter(
                or_(
                    func.lower(KGEntity.canonical_name).like(sq),
                    func.lower(KGEntity.normalized_key).like(sq),
                    func.lower(KGEntity.aliases_json).like(sq)
                )
            )
            if project_id:
                matched_nodes = matched_nodes.filter(KGEntity.project_id == project_id)
            matched_ids = [e.id for e in matched_nodes.all()]

            rel_query = rel_query.filter(
                or_(
                    KGRelation.source_id.in_(matched_ids or [-1]),
                    KGRelation.target_id.in_(matched_ids or [-1])
                )
            )
            relations = rel_query.order_by(KGRelation.confidence.desc()).limit(limit).all()
            active_entity_ids: Set[int] = set(matched_ids)
            for r in relations:
                active_entity_ids.add(r.source_id)
                active_entity_ids.add(r.target_id)

            ent_query = db.query(KGEntity).filter(KGEntity.id.in_(active_entity_ids or [-1]))
        else:
            relations = rel_query.order_by(KGRelation.confidence.desc()).limit(limit).all()
            active_entity_ids = set()
            for r in relations:
                active_entity_ids.add(r.source_id)
                active_entity_ids.add(r.target_id)

            ent_query = db.query(KGEntity)
            if project_id:
                ent_query = ent_query.filter(KGEntity.project_id == project_id)

            if not include_papers:
                ent_query = ent_query.filter(KGEntity.entity_type != "PAPER")

            if entity_type:
                ent_query = ent_query.filter(KGEntity.entity_type == entity_type)

        if active_entity_ids and not entity_type:
            ent_query = ent_query.filter(KGEntity.id.in_(active_entity_ids))

        entities = ent_query.limit(limit * 2).all()
        entity_map = {e.id: e for e in entities}

        # Calculate mention and document counts per entity
        mention_counts: Dict[int, int] = dict(
            db.query(KGEntityMention.entity_id, func.count(KGEntityMention.id))
              .filter(KGEntityMention.entity_id.in_(list(entity_map.keys()) or [0]))
              .group_by(KGEntityMention.entity_id).all()
        )

        doc_counts: Dict[int, int] = dict(
            db.query(KGEntityMention.entity_id, func.count(func.distinct(KGEntityMention.document_id)))
              .filter(KGEntityMention.entity_id.in_(list(entity_map.keys()) or [0]))
              .group_by(KGEntityMention.entity_id).all()
        )

        # Build Cytoscape/JSON-compatible Nodes
        nodes = []
        for e in entities:
            try:
                aliases = json.loads(e.aliases_json or "[]")
            except Exception:
                aliases = []

            nodes.append({
                "id": str(e.id),
                "label": e.canonical_name,
                "type": e.entity_type,
                "normalized_key": e.normalized_key,
                "mention_count": mention_counts.get(e.id, 1),
                "document_count": doc_counts.get(e.id, 1),
                "aliases": aliases
            })

        # Build Edges (only if both source and target nodes exist in the filtered node set)
        node_id_set = {str(e.id) for e in entities}
        edges = []
        for r in relations:
            src_str = str(r.source_id)
            tgt_str = str(r.target_id)
            if src_str in node_id_set and tgt_str in node_id_set:
                ev_count = len(r.evidence)
                edges.append({
                    "id": str(r.id),
                    "source": src_str,
                    "target": tgt_str,
                    "label": r.predicate,
                    "confidence": round(r.confidence, 2),
                    "polarity": r.polarity,
                    "modality": r.modality,
                    "review_status": r.review_status,
                    "evidence_count": ev_count
                })

        return {
            "nodes": nodes,
            "edges": edges,
            "meta": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "truncated": len(nodes) >= limit or len(edges) >= limit,
                "project_id": project_id,
                "document_id": document_id
            }
        }

    # --- Details & Inspection ---
    def get_entity_detail(self, db: Session, entity_id: int) -> Optional[Dict[str, Any]]:
        ent = db.query(KGEntity).filter(KGEntity.id == entity_id).first()
        if not ent:
            return None

        # Fetch mentions with documents
        mentions = db.query(KGEntityMention, Document.title)\
                     .join(Document, KGEntityMention.document_id == Document.id)\
                     .filter(KGEntityMention.entity_id == entity_id)\
                     .limit(20).all()

        mention_records = []
        doc_citations = set()
        for m, doc_title in mentions:
            doc_citations.add(doc_title)
            mention_records.append({
                "id": m.id,
                "document_id": m.document_id,
                "document_title": doc_title,
                "mention_text": m.mention_text,
                "sentence_text": m.sentence_text,
                "created_at": m.created_at.isoformat() if m.created_at else None
            })

        # Outgoing relations
        out_rels = db.query(KGRelation, KGEntity.canonical_name, KGEntity.entity_type)\
                     .join(KGEntity, KGRelation.target_id == KGEntity.id)\
                     .filter(KGRelation.source_id == entity_id).all()
        outgoing = [{
            "id": r.id,
            "predicate": r.predicate,
            "target_id": r.target_id,
            "target_name": t_name,
            "target_type": t_type,
            "confidence": r.confidence,
            "review_status": r.review_status
        } for r, t_name, t_type in out_rels]

        # Incoming relations
        in_rels = db.query(KGRelation, KGEntity.canonical_name, KGEntity.entity_type)\
                    .join(KGEntity, KGRelation.source_id == KGEntity.id)\
                    .filter(KGRelation.target_id == entity_id).all()
        incoming = [{
            "id": r.id,
            "predicate": r.predicate,
            "source_id": r.source_id,
            "source_name": s_name,
            "source_type": s_type,
            "confidence": r.confidence,
            "review_status": r.review_status
        } for r, s_name, s_type in in_rels]

        try:
            aliases = json.loads(ent.aliases_json or "[]")
        except Exception:
            aliases = []

        return {
            "id": ent.id,
            "canonical_name": ent.canonical_name,
            "entity_type": ent.entity_type,
            "normalized_key": ent.normalized_key,
            "aliases": aliases,
            "description": ent.description,
            "created_at": ent.created_at.isoformat() if ent.created_at else None,
            "mention_count": len(mentions),
            "citations": list(doc_citations),
            "mentions": mention_records,
            "outgoing_relations": outgoing,
            "incoming_relations": incoming
        }

    def get_relation_detail(self, db: Session, relation_id: int) -> Optional[Dict[str, Any]]:
        rel = db.query(KGRelation).filter(KGRelation.id == relation_id).first()
        if not rel:
            return None

        src = db.query(KGEntity).filter(KGEntity.id == rel.source_id).first()
        tgt = db.query(KGEntity).filter(KGEntity.id == rel.target_id).first()

        evidences = db.query(KGRelationEvidence, Document.title)\
                      .join(Document, KGRelationEvidence.document_id == Document.id)\
                      .filter(KGRelationEvidence.relation_id == relation_id).all()

        evidence_list = []
        for ev, doc_title in evidences:
            evidence_list.append({
                "id": ev.id,
                "document_id": ev.document_id,
                "document_title": doc_title,
                "chunk_id": ev.chunk_id,
                "excerpt": ev.excerpt,
                "sentence_text": ev.excerpt,
                "page_num": ev.page_num,
                "confidence": ev.confidence,
                "extraction_method": ev.extraction_method,
                "verification_status": ev.verification_status,
                "created_at": ev.created_at.isoformat() if ev.created_at else None
            })

        return {
            "id": rel.id,
            "source_id": rel.source_id,
            "source_name": src.canonical_name if src else "Unknown",
            "source_type": src.entity_type if src else "OTHER",
            "target_id": rel.target_id,
            "target_name": tgt.canonical_name if tgt else "Unknown",
            "target_type": tgt.entity_type if tgt else "OTHER",
            "predicate": rel.predicate,
            "confidence": rel.confidence,
            "polarity": rel.polarity,
            "modality": rel.modality,
            "review_status": rel.review_status,
            "evidence_count": len(evidence_list),
            "evidence": evidence_list
        }

    # --- Entity Merging & Deduplication ---
    def merge_entities(self, db: Session, source_entity_id: int, target_entity_id: int) -> Dict[str, Any]:
        """
        Merge source_entity into target_entity:
        - Updates all mentions to target_entity.
        - Appends source aliases to target aliases.
        - Reroutes all relations to target_entity, merging duplicate edges and preserving evidence.
        - Deletes source_entity.
        """
        if source_entity_id == target_entity_id:
            raise ValueError("Cannot merge an entity into itself.")

        src = db.query(KGEntity).filter(KGEntity.id == source_entity_id).first()
        tgt = db.query(KGEntity).filter(KGEntity.id == target_entity_id).first()

        if not src or not tgt:
            raise ValueError("Source or target entity not found.")

        # 1. Update mentions
        db.query(KGEntityMention).filter(KGEntityMention.entity_id == source_entity_id)\
          .update({KGEntityMention.entity_id: target_entity_id})

        # 2. Merge aliases
        try:
            tgt_aliases = json.loads(tgt.aliases_json or "[]")
        except Exception:
            tgt_aliases = []
        try:
            src_aliases = json.loads(src.aliases_json or "[]")
        except Exception:
            src_aliases = []

        all_aliases = list(set(tgt_aliases + src_aliases + [src.canonical_name]))
        tgt.aliases_json = json.dumps(all_aliases)

        # 3. Reroute outgoing relations
        out_rels = db.query(KGRelation).filter(KGRelation.source_id == source_entity_id).all()
        for r in out_rels:
            if r.target_id == target_entity_id:
                # Self-loop would be created, move evidence then delete relation
                for ev in r.evidence:
                    db.delete(ev)
                db.delete(r)
                continue

            duplicate = db.query(KGRelation).filter(
                KGRelation.project_id == r.project_id,
                KGRelation.source_id == target_entity_id,
                KGRelation.target_id == r.target_id,
                KGRelation.predicate == r.predicate
            ).first()

            if duplicate:
                for ev in r.evidence:
                    ev.relation_id = duplicate.id
                db.delete(r)

        # 4. Reroute incoming relations
        in_rels = db.query(KGRelation).filter(KGRelation.target_id == source_entity_id).all()
        for r in in_rels:
            if r.source_id == target_entity_id:
                for ev in r.evidence:
                    db.delete(ev)
                db.delete(r)
                continue

            duplicate = db.query(KGRelation).filter(
                KGRelation.project_id == r.project_id,
                KGRelation.source_id == r.source_id,
                KGRelation.target_id == target_entity_id,
                KGRelation.predicate == r.predicate
            ).first()

            if duplicate:
                for ev in r.evidence:
                    ev.relation_id = duplicate.id
                db.delete(r)

        db.flush()

        # Update remaining relations to point to target_entity_id
        db.query(KGRelation).filter(KGRelation.source_id == source_entity_id).update({KGRelation.source_id: target_entity_id})
        db.query(KGRelation).filter(KGRelation.target_id == source_entity_id).update({KGRelation.target_id: target_entity_id})

        # Expire session cache and delete source entity cleanly
        db.expire_all()
        db.query(KGEntity).filter(KGEntity.id == source_entity_id).delete()
        db.commit()

        return {
            "status": "success",
            "merged_source_id": source_entity_id,
            "target_entity_id": target_entity_id,
            "target_canonical_name": tgt.canonical_name,
            "total_aliases": len(all_aliases)
        }

    # --- Review & Curation ---
    def review_relation(
        self,
        db: Session,
        relation_id: int,
        status: str,
        corrected_predicate: Optional[str] = None
    ) -> Dict[str, Any]:
        """Approve, reject, or correct relationship."""
        rel = db.query(KGRelation).filter(KGRelation.id == relation_id).first()
        if not rel:
            raise ValueError(f"Relation ID {relation_id} not found.")

        if status not in ("approved", "rejected", "corrected", "unreviewed"):
            raise ValueError("Invalid review status. Must be approved, rejected, corrected, or unreviewed.")

        rel.review_status = status
        if corrected_predicate and corrected_predicate in RELATION_PREDICATES:
            rel.predicate = corrected_predicate
            rel.review_status = "corrected"

        db.commit()
        return {
            "relation_id": rel.id,
            "predicate": rel.predicate,
            "review_status": rel.review_status
        }

    # --- Overview & Stats ---
    def get_stats(self, db: Session, project_id: Optional[int] = None) -> Dict[str, Any]:
        ent_q = db.query(func.count(KGEntity.id))
        rel_q = db.query(func.count(KGRelation.id))
        ev_q = db.query(func.count(KGRelationEvidence.id))
        review_q = db.query(func.count(KGRelation.id)).filter(KGRelation.review_status == "unreviewed")
        paper_q = db.query(func.count(func.distinct(KGEntityMention.document_id)))

        if project_id:
            ent_q = ent_q.filter(KGEntity.project_id == project_id)
            rel_q = rel_q.filter(KGRelation.project_id == project_id)
            review_q = review_q.filter(KGRelation.project_id == project_id)

        # Entity type distribution
        dist_q = db.query(KGEntity.entity_type, func.count(KGEntity.id))
        if project_id:
            dist_q = dist_q.filter(KGEntity.project_id == project_id)
        type_distribution = dict(dist_q.group_by(KGEntity.entity_type).all())

        # Predicate distribution
        pred_q = db.query(KGRelation.predicate, func.count(KGRelation.id))
        if project_id:
            pred_q = pred_q.filter(KGRelation.project_id == project_id)
        predicate_distribution = dict(pred_q.group_by(KGRelation.predicate).all())

        return {
            "total_entities": ent_q.scalar() or 0,
            "total_relations": rel_q.scalar() or 0,
            "total_evidence_records": ev_q.scalar() or 0,
            "papers_represented": paper_q.scalar() or 0,
            "pending_reviews": review_q.scalar() or 0,
            "type_distribution": type_distribution,
            "predicate_distribution": predicate_distribution
        }

    # --- AI Prompt / Free-form Text Knowledge Graph Generator ---
    def generate_graph_from_text(
        self,
        db: Session,
        text: str,
        title: Optional[str] = None,
        project_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        AI-Powered Knowledge Graph Generator from raw prompt or freeform text:
        1. Ensures a Project exists.
        2. Creates a Document record from the raw text.
        3. Executes the full NLP extraction pipeline (entities + SPO triplets + grounding).
        4. Returns the resulting nodes, edges, meta, and summary for immediate visualization.
        """
        if not text or len(text.strip()) < 10:
            raise ValueError("Input text must contain at least 10 characters.")

        # Determine project
        active_project_id = project_id
        if not active_project_id:
            first_project = db.query(Project).first()
            if first_project:
                active_project_id = first_project.id
            else:
                new_project = Project(
                    name="Default Workspace",
                    description="Primary Research & Knowledge Graph Workspace"
                )
                db.add(new_project)
                db.commit()
                db.refresh(new_project)
                active_project_id = new_project.id

        # Generate clean title if not provided
        doc_title = title
        if not doc_title or not doc_title.strip():
            first_line = text.strip().split("\n")[0][:60].strip()
            doc_title = first_line if len(first_line) > 5 else "AI Generated Knowledge Graph"

        # Create Document
        doc = Document(
            project_id=active_project_id,
            title=doc_title,
            filename=f"ai_kg_{int(time.time())}.txt",
            source_type="ai_prompt",
            processing_status="processed",
            content_text=text.strip()
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)

        # Build knowledge graph
        build_res = self.build_graph_for_document(
            db=db,
            document_id=doc.id,
            project_id=active_project_id,
            rebuild=True
        )

        # Fetch resulting graph elements for this generated document
        graph_data = self.get_graph_data(
            db=db,
            project_id=active_project_id,
            document_id=doc.id,
            limit=250
        )

        return {
            "status": "success",
            "document_id": doc.id,
            "document_title": doc.title,
            "project_id": active_project_id,
            "nodes": graph_data["nodes"],
            "edges": graph_data["edges"],
            "meta": graph_data["meta"],
            "build_summary": build_res
        }



# Global singleton instance
kg_service = KnowledgeGraphService()
